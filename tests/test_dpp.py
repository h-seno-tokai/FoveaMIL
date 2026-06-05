"""微分可能 k-DPP 選択コントローラと多様性正則化のユニット

植え込みデータ（k 個のタイトなクラスタに高品質タイルが分散）で DPP が各クラスタを
被覆し top-k が 1 クラスタへ潰れることを確認しつつ，形状・one-hot 性・勾配・k クランプ・
多様性正則化のゲート・後方互換を回帰ガードする
"""

import torch

from foveamil.models import FoveaMIL, build_selection_controller
from foveamil.models.regularizers import iter_active_regularizers
from foveamil.models.regularizers.base import ForwardContext
from foveamil.models.regularizers.dpp_diversity import DPPDiversityRegularizer
from foveamil.training.config import TrainConfig

# 植え込みクラスタ数（= k）
PLANTED_CLUSTERS = 4
# 各クラスタのタイル数
TILES_PER_CLUSTER = 15
# クラスタ中心のスケール（クラスタ間を遠く離す）
CENTER_SCALE = 5.0
# クラスタ内のジッタ
JITTER = 0.05
# 緩和温度（推論の貪欲 MAP を鋭くする）
SHARP_TEMP = 0.1


def _planted_clusters(seed: int = 0):
    """k クラスタに高品質タイルが分散する植え込みデータを作る

    最初のクラスタだけ品質を高くし，top-k が偏りやすい状況を作る

    Returns:
        ``(scores[1, N], features[1, N, D], cluster_of[N])``
    """
    torch.manual_seed(seed)
    dim = 16
    centers = torch.randn(PLANTED_CLUSTERS, dim) * CENTER_SCALE
    feats, raw_scores, cluster_of = [], [], []
    for c in range(PLANTED_CLUSTERS):
        for _ in range(TILES_PER_CLUSTER):
            feats.append(centers[c] + JITTER * torch.randn(dim))
            raw_scores.append(2.0 if c == 0 else 0.5)
            cluster_of.append(c)
    features = torch.stack(feats).unsqueeze(0)
    scores = torch.softmax(torch.tensor(raw_scores).unsqueeze(0), dim=-1)
    return scores, features, cluster_of


def _covered_clusters(selection, cluster_of):
    """選択行列の argmax がどのクラスタを被覆したかの集合を返す"""
    idx = selection.argmax(dim=-1)[0].tolist()
    return {cluster_of[i] for i in idx}


def test_dpp_covers_clusters_topk_collapses():
    scores, features, cluster_of = _planted_clusters()

    dpp = build_selection_controller(
        "dpp", k=PLANTED_CLUSTERS, similarity="cosine", temperature=SHARP_TEMP
    )
    dpp.eval()
    dpp_cov = _covered_clusters(dpp.select(scores, features), cluster_of)

    topk = build_selection_controller("topk", k=PLANTED_CLUSTERS, topk_method="perturbed")
    topk.eval()
    topk_cov = _covered_clusters(topk.select(scores, features), cluster_of)

    # DPP は全クラスタを被覆し top-k は最も顕著な 1 クラスタへ潰れる
    assert len(dpp_cov) == PLANTED_CLUSTERS
    assert len(topk_cov) < len(dpp_cov)
    assert len(topk_cov) == 1


def test_dpp_rbf_also_covers_clusters():
    scores, features, cluster_of = _planted_clusters()
    dpp = build_selection_controller(
        "dpp", k=PLANTED_CLUSTERS, similarity="rbf", temperature=SHARP_TEMP
    )
    dpp.eval()
    cov = _covered_clusters(dpp.select(scores, features), cluster_of)
    assert len(cov) >= PLANTED_CLUSTERS - 1


def test_dpp_selection_shape_and_eval_one_hot():
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller("dpp", k=PLANTED_CLUSTERS, temperature=SHARP_TEMP)
    dpp.eval()
    sel = dpp.select(scores, features)
    n = features.shape[1]
    assert sel.shape == (1, PLANTED_CLUSTERS, n)
    # 推論時の各行は one-hot（和=1，最大値≈1）
    assert torch.allclose(sel.sum(dim=-1), torch.ones(1, PLANTED_CLUSTERS), atol=1e-5)
    assert float(sel.max(dim=-1).values.min()) > 0.99


def test_dpp_train_soft_shape():
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller("dpp", k=3, temperature=1.0)
    dpp.train()
    sel = dpp.select(scores, features)
    assert sel.shape == (1, 3, features.shape[1])


def test_dpp_k_clamps_to_n():
    scores = torch.softmax(torch.randn(1, 5), dim=-1)
    features = torch.randn(1, 5, 8)
    dpp = build_selection_controller("dpp", k=100, similarity="cosine")
    dpp.eval()
    sel = dpp.select(scores, features)
    assert sel.shape == (1, 5, 5)


def test_dpp_gradients_flow_to_scores_and_features():
    _, features, _ = _planted_clusters()
    scores = torch.softmax(torch.randn(1, features.shape[1]), dim=-1)
    dpp = build_selection_controller("dpp", k=PLANTED_CLUSTERS, temperature=1.0)
    dpp.train()
    s = scores.clone().requires_grad_(True)
    f = features.clone().requires_grad_(True)
    out = dpp.select(s, f)
    # 各行は和=1 で sum は不変なので，下流が使う重み付き和で勾配を測る
    torch.manual_seed(1)
    target = torch.randn_like(out)
    (out * target).sum().backward()
    assert s.grad is not None and f.grad is not None
    assert torch.isfinite(s.grad).all() and torch.isfinite(f.grad).all()
    assert float(s.grad.abs().sum()) > 0.0
    assert float(f.grad.abs().sum()) > 0.0


def test_dpp_deterministic_with_seed():
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller(
        "dpp", k=PLANTED_CLUSTERS, use_gumbel=True, temperature=1.0, seed=123
    )
    dpp.train()
    a = dpp.select(scores, features)
    b = dpp.select(scores, features)
    assert torch.allclose(a, b)


def test_dpp_train_mode_distinct_argmax_covers_clusters():
    # 学習（soft）モードでも k 個の argmax は相異なり全クラスタを被覆する（D1 ガード）
    scores, features, cluster_of = _planted_clusters()
    dpp = build_selection_controller(
        "dpp", k=PLANTED_CLUSTERS, similarity="cosine", temperature=1.0
    )
    dpp.train()
    sel = dpp.select(scores, features)
    idx = sel.argmax(dim=-1)[0].tolist()
    assert len(set(idx)) == PLANTED_CLUSTERS
    assert _covered_clusters(sel, cluster_of) == set(range(PLANTED_CLUSTERS))


def test_dpp_pop_log_det_finite_in_train_mode():
    # 学習モードの forward 後でも pop_log_det が有限（D2 ガード）
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller("dpp", k=PLANTED_CLUSTERS, temperature=1.0)
    dpp.train()
    dpp.select(scores, features)
    log_det = dpp.pop_log_det()
    assert log_det is not None
    assert torch.isfinite(log_det).all()


def test_dpp_select_does_not_perturb_global_rng():
    # seed 付き select は global RNG を汚さない（D3 ガード）
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller(
        "dpp", k=PLANTED_CLUSTERS, use_gumbel=True, temperature=1.0, seed=123
    )
    dpp.train()
    torch.manual_seed(999)
    before = torch.rand(5)
    dpp.select(scores, features)
    torch.manual_seed(999)
    after = torch.rand(5)
    assert torch.allclose(before, after)


def test_dpp_log_det_decreases_with_similarity():
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller("dpp", k=PLANTED_CLUSTERS, temperature=SHARP_TEMP)
    dpp.eval()

    dpp.select(scores, features)
    log_det_diverse = float(dpp.pop_log_det())
    # 全タイルを同一特徴にすると多様性が消え log-det が下がる
    same = features[:, :1, :].repeat(1, features.shape[1], 1)
    dpp.select(scores, same)
    log_det_similar = float(dpp.pop_log_det())
    assert log_det_diverse > log_det_similar


def test_dpp_pop_log_det_consumes():
    scores, features, _ = _planted_clusters()
    dpp = build_selection_controller("dpp", k=2, temperature=SHARP_TEMP)
    dpp.eval()
    dpp.select(scores, features)
    assert dpp.pop_log_det() is not None
    assert dpp.pop_log_det() is None


# --- 多様性正則化（dpp_diversity）---


def _cfg(**overrides):
    config = TrainConfig()
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_dpp_diversity_registered_and_default_off():
    # 既定（weight=0）では from_config が None を返す
    assert iter_active_regularizers(TrainConfig()) == []


def test_dpp_diversity_off_when_weight_zero():
    config = _cfg(selector="dpp", magnifications=[1.25, 2.5], dpp_diversity_weight=0.0)
    assert DPPDiversityRegularizer.from_config(config) is None


def test_dpp_diversity_off_when_selector_not_dpp():
    config = _cfg(selector="topk", magnifications=[1.25, 2.5], dpp_diversity_weight=0.1)
    assert DPPDiversityRegularizer.from_config(config) is None


def test_dpp_diversity_off_for_single_magnification():
    config = _cfg(selector="dpp", magnifications=[40], dpp_diversity_weight=0.1)
    assert DPPDiversityRegularizer.from_config(config) is None


def test_dpp_diversity_active_when_dpp_multi_mag_weighted():
    config = _cfg(selector="dpp", magnifications=[1.25, 2.5], dpp_diversity_weight=0.1)
    reg = DPPDiversityRegularizer.from_config(config)
    assert isinstance(reg, DPPDiversityRegularizer)
    assert reg.weight == 0.1


def test_dpp_diversity_penalty_increases_with_similarity():
    reg = DPPDiversityRegularizer(1.0)
    label = torch.tensor([0])
    diverse = ForwardContext(m_list=[], dpp_log_dets=[torch.tensor(-0.1)])
    similar = ForwardContext(m_list=[], dpp_log_dets=[torch.tensor(-5.0)])
    # 類似な選択（log-det 小）ほど -log-det（罰則）が大きい
    assert float(reg(similar, label)) > float(reg(diverse, label))


def test_dpp_diversity_zero_without_log_dets():
    reg = DPPDiversityRegularizer(1.0)
    label = torch.tensor([0])
    assert float(reg(ForwardContext(m_list=[]), label)) == 0.0


# --- モデル統合と後方互換 ---


def test_foveamil_dpp_forward_backward_smoke():
    torch.manual_seed(0)
    model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=2,
        topk_method="perturbed",
        selector="dpp",
        selector_kwargs={"similarity": "cosine", "temperature": 0.5},
    )
    model.train()
    M, idx, weight, aux = model.forward_layer(torch.randn(1, 20, 8), layer_idx=0)
    assert M.shape == (1, 1, 12)
    assert idx.shape == (1, 4) and weight.shape == (1, 4)
    assert (idx[:, 1:] >= idx[:, :-1]).all()
    (weight.sum() + M.sum()).backward()
    grad_sum = sum(
        float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
    )
    assert grad_sum > 0.0


def test_selector_kwargs_forwards_all_dpp_knobs():
    # config の DPP ノブが selector_kwargs 経由でコントローラへ渡る（D4 ガード）
    from foveamil.training.trainer import _selector_kwargs

    config = _cfg(
        selector="dpp",
        magnifications=[1.25, 2.5],
        dpp_similarity="rbf",
        dpp_temperature=0.3,
        dpp_quality_beta=2.0,
        dpp_rbf_gamma=0.5,
        dpp_use_gumbel=True,
    )
    kwargs = _selector_kwargs(config)
    assert kwargs == {
        "similarity": "rbf",
        "temperature": 0.3,
        "quality_beta": 2.0,
        "rbf_gamma": 0.5,
        "use_gumbel": True,
    }
    dpp = build_selection_controller("dpp", k=2, **kwargs)
    assert dpp.quality_beta == 2.0
    assert dpp.rbf_gamma == 0.5
    assert dpp.use_gumbel is True


def test_selector_kwargs_empty_for_topk():
    from foveamil.training.trainer import _selector_kwargs

    assert _selector_kwargs(_cfg(selector="topk", magnifications=[1.25, 2.5])) == {}


def test_default_selector_topk_bit_identical():
    # selector 既定（topk）は DPP 追加前と同一の選択を出す
    torch.manual_seed(7)
    model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=2,
        topk_method="perturbed",
        selector="topk",
    )
    model.eval()
    x_fc = model.projections[0](torch.randn(2, 10, 8))
    raw, _ = model.aux_attentions[0](x_fc)
    scores = model.aux_norm(raw.squeeze(dim=-1))

    from foveamil.models.topk import build_topk

    ref = build_topk("perturbed", k=4)
    ref.eval()
    assert torch.allclose(model.selector.select(scores, x_fc), ref(scores))
