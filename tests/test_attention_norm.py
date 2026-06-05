"""スパース・温度付きアテンション正規化器（temperature / sparsemax / entmax）のユニット

各正規化器が単体上の分布（非負・和 1）を返し，スパース系が鋭い入力で厳密に 0 を出し，
温度・α の極限挙動が softmax / sparsemax に一致し，勾配が有限かつ非零であることを検算する
補助アテンションへ組み込んだ FoveaMIL の後方互換と forward/backward の健全性も確認する
"""

import pytest
import torch
import torch.nn.functional as F

from foveamil.models import FoveaMIL, build_attention_norm
from foveamil.models.attention_norm import available_attention_norms
from foveamil.training.config import TrainConfig
from foveamil.training.trainer import _aux_norm_kwargs, build_foveamil_from_config

# 単体上の和の許容誤差
_SUM_TOL = 1e-5
# softmax / sparsemax 一致の許容誤差
_LIMIT_TOL = 1e-3
# 厳密 0 とみなす上限
_ZERO_TOL = 1e-6
# 二分探索の極限一致に十分な反復回数
_FINE_ITER = 200

ALL_NORMS = ["temperature", "sparsemax", "entmax"]


def _norm(name):
    """名前から正規化器を構築する（既定引数）"""
    return build_attention_norm(name)


# --- レジストリ ---


def test_all_norms_registered():
    available = available_attention_norms()
    for name in ALL_NORMS + ["softmax"]:
        assert name in available


# --- 単体性（非負・和 1）---


@pytest.mark.parametrize("name", ALL_NORMS)
def test_outputs_nonnegative_and_sum_to_one(name):
    torch.manual_seed(0)
    scores = torch.randn(4, 13)
    out = _norm(name)(scores)
    assert (out >= -_ZERO_TOL).all()
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=_SUM_TOL)


# --- スパース性（鋭い入力で厳密 0）---


@pytest.mark.parametrize("name", ["sparsemax", "entmax"])
def test_peaked_input_is_exactly_sparse(name):
    peaked = torch.tensor([[6.0, 0.1, 0.0, -4.0, 0.2]])
    out = _norm(name)(peaked)
    n_zeros = (out <= _ZERO_TOL).sum().item()
    assert n_zeros > 0
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=_SUM_TOL)


# --- 温度の極限挙動 ---


def test_temperature_one_equals_softmax():
    torch.manual_seed(1)
    scores = torch.randn(5, 9)
    out = build_attention_norm("temperature", temperature=1.0)(scores)
    assert torch.allclose(out, F.softmax(scores, dim=-1), atol=1e-7)


def test_temperature_sharpens_and_flattens():
    scores = torch.tensor([[2.0, 1.0, 0.0]])
    sharp = build_attention_norm("temperature", temperature=0.1)(scores)
    flat = build_attention_norm("temperature", temperature=10.0)(scores)
    # 鋭いほど最大要素へ集中し平坦なほど一様へ近づく
    assert sharp.max() > F.softmax(scores, dim=-1).max()
    assert flat.max() < F.softmax(scores, dim=-1).max()


def test_temperature_nonpositive_raises():
    with pytest.raises(ValueError):
        build_attention_norm("temperature", temperature=0.0)


# --- entmax の極限挙動 ---


def test_entmax_alpha_near_one_approximates_softmax():
    torch.manual_seed(2)
    scores = torch.randn(4, 11)
    out = build_attention_norm("entmax", alpha=1.0001, max_iter=_FINE_ITER)(scores)
    assert torch.allclose(out, F.softmax(scores, dim=-1), atol=_LIMIT_TOL)


def test_entmax_alpha_one_is_exactly_softmax():
    torch.manual_seed(3)
    scores = torch.randn(4, 11)
    out = build_attention_norm("entmax", alpha=1.0)(scores)
    assert torch.allclose(out, F.softmax(scores, dim=-1), atol=1e-6)


def test_entmax_alpha_two_approximates_sparsemax():
    torch.manual_seed(4)
    scores = torch.randn(4, 11)
    out = build_attention_norm("entmax", alpha=2.0, max_iter=_FINE_ITER)(scores)
    ref = build_attention_norm("sparsemax")(scores)
    assert torch.allclose(out, ref, atol=_LIMIT_TOL)


def test_entmax_alpha_below_one_raises():
    with pytest.raises(ValueError):
        build_attention_norm("entmax", alpha=0.5)


def test_entmax_alpha_above_two_raises():
    with pytest.raises(ValueError):
        build_attention_norm("entmax", alpha=2.5)


# --- 勾配の正しさ（解析 backward vs 数値微分）---


@pytest.mark.parametrize("alpha", [1.3, 1.5, 1.8, 2.0])
def test_entmax_backward_matches_numerical_gradient(alpha):
    # 全要素が台に乗る内点（kink を避ける）で解析 Jacobian を数値微分と照合する
    scores = torch.tensor(
        [[0.5, 0.45, 0.55, 0.48, 0.52]], dtype=torch.double, requires_grad=True
    )
    norm = build_attention_norm("entmax", alpha=alpha, max_iter=_FINE_ITER)
    out = norm(scores)
    # 内点であること（台の外＝0 がないこと）を前提に確認する
    assert (out > _ZERO_TOL).all()
    assert torch.autograd.gradcheck(
        lambda z: norm(z), (scores,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


# --- 勾配（有限・非零）---


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("temperature", {"temperature": 0.7}),
        ("sparsemax", {}),
        ("entmax", {"alpha": 1.5}),
        ("entmax", {"alpha": 2.0}),
    ],
)
def test_gradient_is_finite_and_nonzero(name, kwargs):
    torch.manual_seed(5)
    # 単体制約で sum 還元は勾配を消すため要素ごとの重み付き還元を使う
    weight = torch.randn(10)
    scores = torch.randn(3, 10, requires_grad=True)
    (build_attention_norm(name, **kwargs)(scores) * weight).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum().item() > 0


# --- FoveaMIL 組込（後方互換・forward/backward 健全性）---


def _model(aux_norm="softmax", aux_norm_kwargs=None, num_layers=2):
    return FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=num_layers,
        topk_method="perturbed",
        aux_norm=aux_norm,
        aux_norm_kwargs=aux_norm_kwargs,
        fusion="sum",
    )


def test_softmax_aux_backward_compatible():
    # 既定 softmax の補助アテンションは生スコアの F.softmax と一致する（回帰ガード）
    torch.manual_seed(6)
    model = _model(aux_norm="softmax")
    model.eval()
    x = torch.randn(2, 10, 8)
    x_fc = model.projections[0](x)
    raw, _ = model.aux_attentions[0](x_fc)
    expected = F.softmax(raw.squeeze(dim=-1), dim=-1)
    _, aux = model.layer_attention(x, 0)
    assert torch.allclose(aux, expected, atol=1e-6)


@pytest.mark.parametrize(
    "aux_norm,aux_norm_kwargs",
    [
        ("temperature", {"temperature": 0.5}),
        ("sparsemax", {}),
        ("entmax", {"alpha": 1.5}),
    ],
)
def test_model_aux_norm_keeps_distribution(aux_norm, aux_norm_kwargs):
    torch.manual_seed(7)
    model = _model(aux_norm=aux_norm, aux_norm_kwargs=aux_norm_kwargs)
    model.eval()
    _, aux = model.layer_attention(torch.randn(2, 12, 8), 0)
    assert (aux >= -_ZERO_TOL).all()
    assert torch.allclose(aux.sum(dim=-1), torch.ones(2), atol=_SUM_TOL)


def test_aux_norm_adds_no_state_dict_keys():
    # パラメータを持たないため checkpoint 互換を保つ
    for aux_norm, kw in [
        ("temperature", {"temperature": 0.5}),
        ("sparsemax", {}),
        ("entmax", {"alpha": 1.5}),
    ]:
        model = _model(aux_norm=aux_norm, aux_norm_kwargs=kw, num_layers=3)
        extra = [k for k in model.state_dict() if k.startswith("aux_norm")]
        assert extra == []


# --- config 経由の組立と forward/backward smoke ---


@pytest.mark.parametrize(
    "aux_norm",
    ["softmax", "temperature", "sparsemax", "entmax"],
)
def test_build_from_config_forward_backward_smoke(aux_norm):
    torch.manual_seed(8)
    config = TrainConfig(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=3,
        n_cls=3,
        aux_norm=aux_norm,
        aux_norm_temperature=0.5,
        aux_norm_alpha=1.5,
    )
    model = build_foveamil_from_config(config, num_layers=2)
    model.train()
    M0, idx, weight, aux = model.forward_layer(torch.randn(1, 10, 8), layer_idx=0)
    # 選択重みを子特徴へ掛けて補助アテンションへ勾配を流す経路を模す
    x_child = torch.randn(1, weight.shape[1], 8) * weight.unsqueeze(-1)
    M1, _, _, _ = model.forward_layer(x_child, layer_idx=1)
    logits, _, Y_prob = model.forward_final([M0, M1])
    assert torch.isfinite(logits).all()
    assert torch.allclose(Y_prob.sum(dim=-1), torch.ones(1), atol=1e-6)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


# --- trainer の kwargs 写像 ---


def test_aux_norm_kwargs_mapping():
    assert _aux_norm_kwargs(TrainConfig(aux_norm="softmax")) == {}
    assert _aux_norm_kwargs(
        TrainConfig(aux_norm="temperature", aux_norm_temperature=0.3)
    ) == {"temperature": 0.3}
    assert _aux_norm_kwargs(
        TrainConfig(aux_norm="entmax", aux_norm_alpha=1.7)
    ) == {"alpha": 1.7}
