"""集約器（aggregator）のユニット

既定 abmil の bit 互換・self_attn の出力契約／勾配／Nyström 近似の妥当性・縮退安全・
決定性，および集約器差し替えが選択経路へ波及しないこと（直交性）を検証する
"""

import pytest
import torch
import torch.nn.functional as F

from foveamil.models import FoveaMIL, build_aggregator
from foveamil.models.aggregator import available_aggregators
from foveamil.models.aggregator.self_attn import SelfAttentionAggregator
from foveamil.models.attention import GatedAttention


# --- レジストリ ---


def test_registry_lists_both_aggregators():
    names = available_aggregators()
    assert "abmil" in names and "self_attn" in names


def test_build_aggregator_unknown_raises():
    with pytest.raises(KeyError):
        build_aggregator("does_not_exist", dim=8, hidden_dim=8)


# --- 既定 abmil の bit 互換 ---


def test_abmil_matches_legacy_gated_attention_pooling():
    # abmil 集約器は従来の softmax(GatedAttention)·x を bit 単位で再現する
    torch.manual_seed(0)
    agg = build_aggregator("abmil", dim=12, hidden_dim=16, dropout=None)
    agg.eval()
    x = torch.randn(3, 9, 12)
    M, A = agg(x)
    # 集約器自身のアテンションで従来式を手計算し torch.equal で一致を確認する
    raw, _ = agg.attention(x)
    A_ref = F.softmax(raw.permute(0, 2, 1), dim=-1)
    M_ref = A_ref @ x
    assert torch.equal(A, A_ref)
    assert torch.equal(M, M_ref)


def test_abmil_output_contract():
    agg = build_aggregator("abmil", dim=12, hidden_dim=16)
    agg.eval()
    M, A = agg(torch.randn(2, 7, 12))
    assert M.shape == (2, 1, 12)
    assert A.shape == (2, 1, 7)
    assert torch.allclose(A.sum(dim=-1), torch.ones(2, 1), atol=1e-6)


def test_foveamil_default_aggregator_is_abmil_bit_compatible():
    # 既定モデルのプーリング表現は abmil の手計算式と数値一致する（既定の数値不変ガード）
    torch.manual_seed(0)
    model = FoveaMIL(
        in_feat_dim=8, hidden_feat_dim=16, out_feat_dim=12, n_cls=3, num_layers=2
    )
    model.eval()
    x = torch.randn(2, 10, 8)
    M, _, A_primary = model._project_and_pool(x, 0)
    x_fc = model.projections[0](x)
    raw, _ = model.aggregators[0].attention(x_fc)
    A_ref = F.softmax(raw.permute(0, 2, 1), dim=-1)
    assert torch.equal(A_primary, A_ref)
    assert torch.equal(M, A_ref @ x_fc)


# --- self_attn の出力契約 ---


@pytest.mark.parametrize("n", [1, 2, 5, 50])
def test_self_attn_output_contract(n):
    agg = build_aggregator(
        "self_attn", dim=12, hidden_dim=16, num_heads=4, num_landmarks=8
    )
    agg.eval()
    M, A = agg(torch.randn(1, n, 12))
    assert M.shape == (1, 1, 12)
    assert A.shape == (1, 1, n)
    # プーリング重みは非負・和 1
    assert (A >= 0).all()
    assert torch.allclose(A.sum(dim=-1), torch.ones(1, 1), atol=1e-5)


def test_self_attn_requires_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        build_aggregator("self_attn", dim=10, hidden_dim=16, num_heads=4)


# --- 勾配（simplex 出力の罠を避け M 経由で確認）---


def test_self_attn_propagates_gradient_through_pooled_representation():
    torch.manual_seed(0)
    agg = build_aggregator(
        "self_attn", dim=12, hidden_dim=16, num_heads=4, num_landmarks=8
    )
    agg.train()
    x = torch.randn(1, 40, 12, requires_grad=True)
    M, _ = agg(x)
    M.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0


# --- Nyström 近似の妥当性（厳密注意との差・landmark で改善）---


def test_self_attn_exact_path_when_n_le_landmarks():
    # N <= landmark のとき Nyström を使わず厳密注意へ縮退する
    torch.manual_seed(0)
    agg = SelfAttentionAggregator(
        dim=16, hidden_dim=16, num_heads=4, num_landmarks=32
    )
    agg.eval()
    x = agg.norm1(torch.randn(1, 20, 16))
    qkv = agg.qkv(x).chunk(3, dim=-1)
    q, k, v = (agg._split_heads(t) for t in qkv)
    exact = agg._exact_attention(q, k, v)
    nys = agg._nystrom_attention(q, k, v)
    # N(=20) <= landmark(=32) なら segment_means は要素ごとに分かれ近似は厳密に一致する
    # （ゼロ詰め landmark の擬似逆で生じる微小誤差のみ許容）
    assert torch.allclose(exact, nys, atol=1e-3)


def test_nystrom_approximation_error_decreases_with_landmarks():
    # landmark を増やすほど Nyström 近似は厳密注意へ近づく
    torch.manual_seed(0)
    agg = SelfAttentionAggregator(
        dim=16, hidden_dim=16, num_heads=4, num_landmarks=8
    )
    agg.eval()
    x = agg.norm1(torch.randn(1, 128, 16))
    qkv = agg.qkv(x).chunk(3, dim=-1)
    q, k, v = (agg._split_heads(t) for t in qkv)
    exact = agg._exact_attention(q, k, v)

    def rel_err(num_landmarks):
        agg.num_landmarks = num_landmarks
        nys = agg._nystrom_attention(q, k, v)
        return (exact - nys).norm() / exact.norm()

    coarse = rel_err(8)
    fine = rel_err(64)
    # 近似誤差は有界で，landmark を増やすと縮む
    assert coarse < 0.5
    assert fine < coarse


# --- 縮退安全（空に近い・N 小・単一パッチ）---


@pytest.mark.parametrize("n", [1, 2, 3])
def test_self_attn_degenerate_small_bags(n):
    torch.manual_seed(0)
    agg = build_aggregator(
        "self_attn", dim=12, hidden_dim=16, num_heads=4, num_landmarks=8
    )
    agg.eval()
    M, A = agg(torch.randn(1, n, 12))
    assert torch.isfinite(M).all() and torch.isfinite(A).all()
    assert torch.allclose(A.sum(dim=-1), torch.ones(1, 1), atol=1e-5)


# --- 決定性 ---


def test_self_attn_is_deterministic_in_eval():
    torch.manual_seed(0)
    agg = build_aggregator(
        "self_attn", dim=12, hidden_dim=16, num_heads=4, num_landmarks=8
    )
    agg.eval()
    x = torch.randn(1, 70, 12)
    M1, A1 = agg(x)
    M2, A2 = agg(x)
    assert torch.equal(M1, M2)
    assert torch.equal(A1, A2)


# --- 選択経路の不変性（集約器の差し替えと探索系の直交）---


def _aux_attention_outputs(model, x, layer_idx=0):
    """補助アテンション（選択経路の入口）の重みを取り出す"""
    _, A_aux = model.layer_attention(x, layer_idx)
    return A_aux


def test_aggregator_swap_leaves_selection_path_modules_intact():
    # 集約器を差し替えても補助アテンション・正規化器・選択コントローラは同型のまま
    base = FoveaMIL(in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2)
    swapped = FoveaMIL(
        in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2,
        aggregator="self_attn",
        aggregator_kwargs={"num_heads": 4, "num_landmarks": 8},
    )
    assert len(base.aux_attentions) == len(swapped.aux_attentions)
    for a, b in zip(base.aux_attentions, swapped.aux_attentions):
        assert isinstance(a, GatedAttention) and isinstance(b, GatedAttention)
    assert type(base.selector) is type(swapped.selector)
    assert type(base.aux_norm) is type(swapped.aux_norm)


def test_aggregator_does_not_alter_selection_for_fixed_aux_weights():
    # 補助アテンション重みを固定すれば，集約器が何であれ選択 index は同一になる
    # （選択は A_aux と x_fc のみに依存し，主プーリング集約器とは独立）
    torch.manual_seed(0)
    abmil_model = FoveaMIL(
        in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2, k_sample=3
    )
    sa_model = FoveaMIL(
        in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2, k_sample=3,
        aggregator="self_attn",
        aggregator_kwargs={"num_heads": 4, "num_landmarks": 8},
    )
    # 射影・補助アテンション・選択器の重みを揃える（集約器のみ差がある状態を作る）
    sa_model.projections.load_state_dict(abmil_model.projections.state_dict())
    sa_model.aux_attentions.load_state_dict(abmil_model.aux_attentions.state_dict())
    abmil_model.eval()
    sa_model.eval()
    x = torch.randn(2, 12, 8)
    _, idx_a, _, aux_a = abmil_model.forward_layer(x, 0)
    _, idx_b, _, aux_b = sa_model.forward_layer(x, 0)
    # 補助アテンション（選択経路の入口）は集約器に依らず一致する
    assert torch.allclose(aux_a, aux_b, atol=1e-6)
    # 選択 index も一致する（選択経路が集約器と直交する証左）
    assert torch.equal(idx_a, idx_b)


def test_self_attn_model_full_forward_contract():
    # self_attn 集約器でも段階 forward の出力契約は abmil と同型
    model = FoveaMIL(
        in_feat_dim=8, out_feat_dim=12, n_cls=3, num_layers=2, k_sample=3,
        aggregator="self_attn",
        aggregator_kwargs={"num_heads": 4, "num_landmarks": 8},
    )
    model.eval()
    M0, idx, weight, aux = model.forward_layer(torch.randn(2, 100, 8), 0)
    assert M0.shape == (2, 1, 12)
    assert idx.shape == (2, 3) and weight.shape == (2, 3)
    assert aux.shape == (2, 100)
    M1, _, _, _ = model.forward_layer(torch.randn(2, 30, 8), 1)
    logits, Y_hat, Y_prob = model.forward_final([M0, M1])
    assert logits.shape == (2, 3)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)
