"""models のコア部品（微分可能 top-k・アテンション・融合・組立）のユニット"""

import pytest
import torch

from foveamil.models import (
    FoveaMIL,
    GatedAttention,
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
    M, idx, weight = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    assert M.shape == (2, 1, 12)
    assert idx.shape == (2, 4)
    assert weight.shape == (2, 4)
    # 選択 index は昇順
    assert (idx[:, 1:] >= idx[:, :-1]).all()


def test_forward_layer_final_has_no_selection():
    model = _build_model(num_layers=2)
    model.eval()
    M, idx, weight = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    assert M.shape == (2, 1, 12)
    assert idx is None and weight is None


def test_forward_final_outputs():
    model = _build_model(num_layers=2, n_cls=3, out_feat_dim=12)
    model.eval()
    M0, _, _ = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    M1, _, _ = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
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


def test_single_magnification_is_final_only():
    # 単一倍率（ズーム無し ABMIL 相当）は補助アテンション・選択を持たない
    model = _build_model(num_layers=1, n_cls=3)
    model.eval()
    M, idx, weight = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    assert idx is None and weight is None
    logits, _, Y_prob = model.forward_final([M])
    assert logits.shape == (2, 3)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)
