"""foundation の拡張シーム（正規化器・選択コントローラ・正則化フック）のユニット

既定（softmax 正規化・topk 選択・正則化なし）が従来挙動と一致することを回帰ガードする
"""

import pytest
import torch
import torch.nn.functional as F

from foveamil.models import (
    FoveaMIL,
    ForwardContext,
    build_attention_norm,
    build_selection_controller,
    build_topk,
    iter_active_regularizers,
)
from foveamil.models.attention_norm import available_attention_norms
from foveamil.models.regularizers import available_regularizers
from foveamil.models.selection import available_selection_controllers
from foveamil.training.config import TrainConfig


# --- アテンション正規化器 ---


def test_softmax_norm_equals_f_softmax():
    norm = build_attention_norm("softmax")
    scores = torch.randn(3, 17)
    assert torch.allclose(norm(scores), F.softmax(scores, dim=-1), atol=1e-7)


def test_softmax_is_registered():
    assert "softmax" in available_attention_norms()


def test_build_attention_norm_unknown_raises():
    with pytest.raises(KeyError):
        build_attention_norm("does_not_exist")


# --- 選択コントローラ ---


def test_topk_controller_registered_and_unknown_raises():
    assert "topk" in available_selection_controllers()
    with pytest.raises(KeyError):
        build_selection_controller("does_not_exist", k=3)


def test_topk_controller_eval_matches_build_topk():
    ctrl = build_selection_controller("topk", k=2, topk_method="perturbed")
    ctrl.eval()
    ref = build_topk("perturbed", k=2)
    ref.eval()
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.7, 0.2]])
    feats = torch.randn(1, 5, 8)
    out = ctrl.select(scores, feats)
    assert out.shape == (1, 2, 5)
    assert torch.allclose(out, ref(scores))


def test_topk_controller_ignores_features():
    ctrl = build_selection_controller("topk", k=2, topk_method="perturbed")
    ctrl.eval()
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.7, 0.2]])
    out_a = ctrl.select(scores, torch.randn(1, 5, 8))
    out_b = ctrl.select(scores, torch.randn(1, 5, 8))
    assert torch.allclose(out_a, out_b)


def test_topk_controller_train_soft_shape():
    ctrl = build_selection_controller("topk", k=3, topk_method="perturbed")
    ctrl.train()
    out = ctrl.select(torch.randn(2, 9), torch.randn(2, 9, 8))
    assert out.shape == (2, 3, 9)


# --- 正則化フック ---


def test_foundation_has_no_active_regularizers():
    # foundation は具体正則化項を出荷しない 既定設定では空
    assert iter_active_regularizers(TrainConfig()) == []
    assert available_regularizers() == []


def test_forward_context_holds_m_list():
    m_list = [torch.randn(1, 1, 4) for _ in range(3)]
    ctx = ForwardContext(m_list=m_list)
    assert ctx.m_list is m_list
    assert ctx.extra_losses == {}
    assert ctx.layer_aux == []
    assert ctx.selections == []


# --- 既定挙動の後方互換（softmax・topk） ---


def _model(num_layers, aux_norm="softmax", selector="topk"):
    return FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=num_layers,
        topk_method="perturbed",
        aux_norm=aux_norm,
        selector=selector,
        fusion="sum",
    )


def test_layer_attention_aux_uses_softmax_default():
    model = _model(num_layers=2)
    model.eval()
    x = torch.randn(2, 10, 8)
    x_fc = model.projections[0](x)
    raw, _ = model.aux_attentions[0](x_fc)
    expected = F.softmax(raw.squeeze(dim=-1), dim=-1)
    _, aux = model.layer_attention(x, 0)
    assert torch.allclose(aux, expected, atol=1e-6)


def test_seams_add_no_state_dict_keys():
    # 正規化器・選択コントローラはパラメータを持たず checkpoint 互換を保つ
    model = _model(num_layers=3)
    extra = [
        key
        for key in model.state_dict()
        if key.startswith("aux_norm") or key.startswith("selector")
    ]
    assert extra == []


def test_default_model_forward_layer_contract_unchanged():
    model = _model(num_layers=2)
    model.eval()
    M, idx, weight = model.forward_layer(torch.randn(2, 10, 8), layer_idx=0)
    assert M.shape == (2, 1, 12)
    assert idx.shape == (2, 4) and weight.shape == (2, 4)
    assert (idx[:, 1:] >= idx[:, :-1]).all()
    M_final, idx_f, w_f = model.forward_layer(torch.randn(2, 7, 8), layer_idx=1)
    assert idx_f is None and w_f is None
    logits, Y_hat, Y_prob = model.forward_final([M, M_final])
    assert logits.shape == (2, 3)
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(2), atol=1e-6)
