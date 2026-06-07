"""models のコア部品（微分可能 top-k・アテンション・融合・組立）のユニット"""

import pytest
import torch

from foveamil.models import (
    FoveaMIL,
    GatedAttention,
    InstanceClusteringLoss,
    LinearClassifierHead,
    build_fusion,
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


@pytest.mark.parametrize("name", ["sum", "gated", "scale_attention"])
def test_fusion_out_dim_contract_and_shape(name):
    # 全融合は out_dim == dim を保ちヘッドを不変にする
    fus = build_fusion(name, dim=8, num_layers=3)
    assert fus.out_dim == 8
    m_list = [torch.randn(2, 1, 8) for _ in range(3)]
    out = fus(m_list)
    assert out.shape == (2, 8)


@pytest.mark.parametrize("name", ["sum", "gated", "scale_attention"])
def test_fusion_single_layer_degenerates(name):
    # L=1 では縮退安全（形状契約を保つ）
    fus = build_fusion(name, dim=8, num_layers=1)
    out = fus([torch.randn(2, 1, 8)])
    assert out.shape == (2, 8)


def test_gated_fusion_single_layer_is_identity():
    # softmax over 1 スケール = 1 なので加重和は M をそのまま返す
    fus = build_fusion("gated", dim=8, num_layers=1)
    m = torch.randn(2, 1, 8)
    out = fus([m])
    assert torch.allclose(out, m.squeeze(1), atol=1e-6)


@pytest.mark.parametrize("name", ["gated", "scale_attention"])
def test_fusion_propagates_gradient(name):
    torch.manual_seed(0)
    fus = build_fusion(name, dim=8, num_layers=3)
    fus.train()
    m_list = [torch.randn(2, 1, 8, requires_grad=True) for _ in range(3)]
    out = fus(m_list)
    # 重み付き和（スカラ化）で勾配を確認する
    (out * torch.randn_like(out)).sum().backward()
    for m in m_list:
        assert m.grad is not None
        assert torch.isfinite(m.grad).all()
    assert any(m.grad.abs().sum().item() > 0 for m in m_list)


@pytest.mark.parametrize("name", ["sum", "gated", "scale_attention"])
def test_fusion_is_deterministic(name):
    torch.manual_seed(0)
    fus = build_fusion(name, dim=8, num_layers=3)
    fus.eval()
    m_list = [torch.randn(2, 1, 8) for _ in range(3)]
    first = fus(m_list)
    second = fus(m_list)
    assert torch.allclose(first, second, atol=1e-6)


def test_gated_fusion_weights_sum_to_one():
    # ゲート重みはスケール軸で softmax = 加重和の重み総和は 1
    fus = build_fusion("gated", dim=4, num_layers=3)
    fus.eval()
    tokens = torch.randn(2, 3, 4)
    weights = torch.softmax(fus.gate(tokens).squeeze(-1), dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)


@pytest.mark.parametrize("name", ["gated", "scale_attention"])
def test_fusion_default_sum_unaffected(name):
    # 新融合の登録で既定 sum の数値が変わらない（bit 互換のガード）
    torch.manual_seed(0)
    m_list = [torch.randn(2, 1, 8) for _ in range(3)]
    sum_out = build_fusion("sum", dim=8, num_layers=3)(m_list)
    expected = sum(m.squeeze(1) for m in m_list)
    assert torch.equal(sum_out, expected)


# --- 識別器ヘッド ---


def test_linear_head_shape():
    head = LinearClassifierHead(in_dim=8, n_cls=4)
    assert head(torch.randn(2, 8)).shape == (2, 4)


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


@pytest.mark.parametrize("fusion", ["sum", "gated", "scale_attention"])
def test_model_assembles_with_fusion(fusion):
    # 融合方式を変えても out_dim 契約でヘッド形状は不変（logits は n_cls 次元）
    model = FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=2,
        topk_method="perturbed",
        fusion=fusion,
    )
    model.eval()
    M0, _, _, _ = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    M1, _, _, _ = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    logits, _, Y_prob = model.forward_final([M0, M1])
    assert logits.shape == (2, 3)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)


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
