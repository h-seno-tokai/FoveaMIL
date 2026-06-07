"""models のコア部品（微分可能 top-k・アテンション・融合・組立）のユニット"""

import pytest
import torch

from foveamil.models import (
    FoveaMIL,
    GatedAttention,
    InstanceClusteringLoss,
    LinearClassifierHead,
    MLPClassifierHead,
    build_fusion,
    build_head,
    build_topk,
)


# --- top-k レジストリ ---


def test_build_topk_registry_types():
    from foveamil.models.topk import FastSparseTopK, PerturbedTopK

    assert isinstance(build_topk("perturbed", k=3), PerturbedTopK)
    assert isinstance(build_topk("fast_sparse", k=3), FastSparseTopK)


def test_build_topk_unknown_raises():
    with pytest.raises(KeyError):
        build_topk("does_not_exist", k=3)


# --- hard 選択（推論時）---


def test_hard_select_is_index_sorted_one_hot():
    sel = build_topk("perturbed", k=2)
    sel.eval()
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.7, 0.2]])
    out = sel(scores)
    assert out.shape == (1, 2, 5)
    # 各行は one-hot（和 1）
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, 2))
    # 選択 index は上位 k を昇順に並べたもの（0.9->idx1, 0.7->idx3）
    assert out.argmax(dim=-1).tolist() == [[1, 3]]


def test_topk_clamps_k_to_n():
    sel = build_topk("perturbed", k=5)
    sel.eval()
    scores = torch.randn(1, 3)
    out = sel(scores)
    # k>N は min(N, k) に丸まり 全要素を選ぶ
    assert out.shape == (1, 3, 3)
    assert sorted(out.argmax(dim=-1)[0].tolist()) == [0, 1, 2]


# --- soft 選択（学習時）の形状 ---


@pytest.mark.parametrize("method,kwargs", [("perturbed", {}), ("fast_sparse", {})])
def test_soft_select_shape(method, kwargs):
    sel = build_topk(method, k=3, **kwargs)
    sel.train()
    out = sel(torch.randn(2, 8))
    assert out.shape == (2, 3, 8)


# --- 勾配伝播（選択経路に勾配が流れることの回帰ガード）---


def test_perturbed_topk_propagates_gradient():
    torch.manual_seed(0)
    sel = build_topk("perturbed", k=3)
    sel.train()
    scores = torch.randn(2, 10, requires_grad=True)
    out = sel(scores)
    out.sum().backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() > 0


def test_fast_sparse_topk_propagates_gradient():
    torch.manual_seed(0)
    # epsilon を大きくすると射影が soft になり内部に勾配が流れる
    sel = build_topk("fast_sparse", k=3, epsilon=1.0)
    sel.train()
    scores = torch.randn(2, 10, requires_grad=True)
    out = sel(scores)
    out.sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0


# --- アテンション ---


@pytest.mark.parametrize("n_cls", [1, 3])
def test_gated_attention_shape(n_cls):
    att = GatedAttention(L=16, D=8, dropout=None, n_cls=n_cls)
    A, passthrough = att(torch.randn(4, 10, 16))
    assert A.shape == (4, 10, n_cls)
    assert passthrough.shape == (4, 10, 16)


# --- 融合 ---


def test_sum_fusion_out_dim_and_value():
    fus = build_fusion("sum", dim=8, num_layers=3)
    assert fus.out_dim == 8
    m_list = [torch.randn(2, 1, 8) for _ in range(3)]
    out = fus(m_list)
    assert out.shape == (2, 8)
    expected = sum(m.squeeze(1) for m in m_list)
    assert torch.allclose(out, expected, atol=1e-6)


def test_build_fusion_unknown_raises():
    with pytest.raises(KeyError):
        build_fusion("does_not_exist", dim=8, num_layers=2)


# --- 識別器ヘッド ---


def test_linear_head_shape():
    head = LinearClassifierHead(in_dim=8, n_cls=4)
    assert head(torch.randn(2, 8)).shape == (2, 4)


def test_build_head_registry_types():
    assert isinstance(build_head("linear", in_dim=8, n_cls=4), LinearClassifierHead)
    assert isinstance(build_head("mlp", in_dim=8, n_cls=4), MLPClassifierHead)


def test_build_head_unknown_raises():
    with pytest.raises(KeyError):
        build_head("does_not_exist", in_dim=8, n_cls=4)


def test_build_head_linear_bit_identical_to_direct():
    # build_head("linear") は LinearClassifierHead と RNG 描画順を保ち数値一致する
    torch.manual_seed(0)
    direct = LinearClassifierHead(in_dim=8, n_cls=4)
    torch.manual_seed(0)
    built = build_head("linear", in_dim=8, n_cls=4)
    assert torch.equal(direct.fc.weight, built.fc.weight)
    assert torch.equal(direct.fc.bias, built.fc.bias)


def test_mlp_head_shape():
    head = build_head("mlp", in_dim=8, n_cls=4, hidden_dim=16)
    assert head(torch.randn(2, 8)).shape == (2, 4)


def test_mlp_head_layernorm_placement():
    # 構成は Linear→LayerNorm→ReLU→(Dropout)→Linear の順
    head = MLPClassifierHead(in_dim=8, n_cls=4, hidden_dim=16, dropout=0.5)
    types = [type(m) for m in head.mlp]
    assert types == [
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.ReLU,
        torch.nn.Dropout,
        torch.nn.Linear,
    ]
    # dropout=None なら Dropout を挟まない
    no_drop = MLPClassifierHead(in_dim=8, n_cls=4, hidden_dim=16, dropout=None)
    assert [type(m) for m in no_drop.mlp] == [
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]


def test_mlp_head_propagates_gradient():
    head = build_head("mlp", in_dim=8, n_cls=4, hidden_dim=16)
    x = torch.randn(3, 8, requires_grad=True)
    logits = head(x)
    logits.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    # 全パラメタへ勾配が流れる
    for p in head.parameters():
        assert p.grad is not None


# --- FoveaMIL 組立 ---


def _build_model(num_layers, n_cls=3, k_sample=4, out_feat_dim=12):
    return FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=out_feat_dim,
        k_sample=k_sample,
        n_cls=n_cls,
        num_layers=num_layers,
        topk_method="perturbed",
        fusion="sum",
    )


def test_forward_layer_non_final_returns_selection():
    model = _build_model(num_layers=2, k_sample=4, out_feat_dim=12)
    model.eval()
    M, idx, weight, aux = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    assert M.shape == (2, 1, 12)
    assert idx.shape == (2, 4)
    assert weight.shape == (2, 4)
    # 補助アテンションは正規化済みで和 1
    assert aux.shape == (2, 10)
    assert torch.allclose(aux.sum(dim=-1), torch.ones(2), atol=1e-6)
    # 選択 index は昇順
    assert (idx[:, 1:] >= idx[:, :-1]).all()


def test_forward_layer_final_has_no_selection():
    model = _build_model(num_layers=2)
    model.eval()
    M, idx, weight, aux = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    assert M.shape == (2, 1, 12)
    assert idx is None and weight is None and aux is None


def test_forward_final_outputs():
    model = _build_model(num_layers=2, n_cls=3, out_feat_dim=12)
    model.eval()
    M0, _, _, _ = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    M1, _, _, _ = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    logits, Y_hat, Y_prob = model.forward_final([M0, M1])
    assert logits.shape == (2, 3)
    assert Y_hat.shape == (2, 1)
    assert Y_prob.shape == (2, 3)
    # softmax 確率は和 1
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)
    # Y_hat は最大ロジットのクラス
    assert (Y_hat.squeeze(-1) == logits.argmax(dim=-1)).all()
    # 予測クラスは範囲内
    assert int(Y_hat.min()) >= 0 and int(Y_hat.max()) < 3


def test_default_head_is_linear_and_bit_identical():
    # 既定 head_type は linear で，明示しない既定モデルと全パラメタが数値一致する
    torch.manual_seed(0)
    default_model = _build_model(num_layers=2, n_cls=3, out_feat_dim=12)
    torch.manual_seed(0)
    explicit_model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=2,
        topk_method="perturbed",
        fusion="sum",
        head_type="linear",
    )
    assert isinstance(default_model.head, LinearClassifierHead)
    for (n_a, p_a), (n_b, p_b) in zip(
        default_model.named_parameters(), explicit_model.named_parameters()
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b)


def test_mlp_head_model_forward_and_gradient():
    # head_type="mlp" でも出力契約 [B, n_cls] を保ち head へ勾配が流れる
    model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=2,
        topk_method="perturbed",
        fusion="sum",
        head_type="mlp",
        head_hidden_dim=16,
    )
    assert isinstance(model.head, MLPClassifierHead)
    model.train()
    M0, _, _, _ = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    M1, _, _, _ = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    logits, _, _ = model.forward_final([M0, M1])
    assert logits.shape == (2, 3)
    logits.sum().backward()
    for p in model.head.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0


def test_single_magnification_is_final_only():
    # 単一倍率（ズーム無し ABMIL 相当）は補助アテンション・選択を持たない
    model = _build_model(num_layers=1, n_cls=3)
    model.eval()
    M, idx, weight, aux = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    assert idx is None and weight is None and aux is None
    logits, _, Y_prob = model.forward_final([M])
    assert logits.shape == (2, 3)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)


# --- インスタンス補助損失 ---


def test_instance_loss_shape_and_gradient():
    loss = InstanceClusteringLoss(in_dim=12, n_cls=3, k=4, subtyping=True)
    h = torch.randn(1, 50, 12, requires_grad=True)
    attention = torch.softmax(torch.randn(1, 50), dim=-1)
    out = loss(h, attention, torch.tensor([1]))
    assert out.dim() == 0
    out.backward()
    assert h.grad is not None and h.grad.abs().sum().item() > 0


def test_instance_loss_clamps_k_for_small_bags():
    # パッチ数 < 2k なら k を縮め，パッチが 1 枚以下なら 0 を返す
    loss = InstanceClusteringLoss(in_dim=8, n_cls=2, k=8, subtyping=False)
    small = loss(torch.randn(1, 3, 8), torch.softmax(torch.randn(1, 3), -1), torch.tensor([0]))
    assert torch.isfinite(small)
    single = loss(torch.randn(1, 1, 8), torch.softmax(torch.randn(1, 1), -1), torch.tensor([0]))
    assert single.item() == 0.0


def test_instance_loss_subtyping_uses_all_classifiers():
    # subtyping=True は非正解クラスの分類器にも勾配を流す
    loss = InstanceClusteringLoss(in_dim=8, n_cls=3, k=4, subtyping=True)
    h = torch.randn(1, 40, 8)
    attention = torch.softmax(torch.randn(1, 40), dim=-1)
    loss(h, attention, torch.tensor([0])).backward()
    grads = [c.weight.grad for c in loss.classifiers]
    assert all(g is not None and g.abs().sum().item() > 0 for g in grads)


def test_model_builds_instance_module_only_when_enabled():
    off = _build_model(num_layers=1, n_cls=3)
    assert off.instance_module is None
    # 無効時は補助損失 None で bag forward は従来どおり
    logits, _, _, inst = off.forward_with_instance_loss(
        torch.randn(1, 10, 8), torch.tensor([0])
    )
    assert inst is None and logits.shape == (1, 3)

    on = FoveaMIL(
        in_feat_dim=8, hidden_feat_dim=16, out_feat_dim=12, n_cls=3,
        num_layers=1, instance_loss=True, inst_k=4,
    )
    assert on.instance_module is not None
    logits, _, _, inst = on.forward_with_instance_loss(
        torch.randn(1, 30, 8), torch.tensor([1])
    )
    assert logits.shape == (1, 3)
    assert inst.dim() == 0 and torch.isfinite(inst)


def test_forward_with_instance_loss_shares_one_forward():
    # bag forward と補助損失は同一の射影・主アテンションを共有する（dropout 無しでは一致確認）
    model = FoveaMIL(
        in_feat_dim=8, hidden_feat_dim=16, out_feat_dim=12, n_cls=3,
        num_layers=1, instance_loss=True, inst_k=4,
    )
    model.train()
    x = torch.randn(1, 40, 8)
    logits, _, _, inst = model.forward_with_instance_loss(x, torch.tensor([0]))
    # forward_layer のプーリングと同じ M から logits が出る（dropout=None なので決定的）
    M, _, _, _ = model.forward_layer(x, 0)
    ref_logits, _, _ = model.forward_final([M])
    assert torch.allclose(logits, ref_logits, atol=1e-6)
    # 結合損失の backward が bag ヘッドと instance 分類器の双方へ勾配を流す
    (0.7 * logits.sum() + 0.3 * inst).backward()
    assert model.head.fc.weight.grad is not None
    assert model.instance_module.classifiers[0].weight.grad is not None


def test_instance_loss_requires_single_magnification():
    with pytest.raises(ValueError, match="single magnification"):
        FoveaMIL(in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2, instance_loss=True)
