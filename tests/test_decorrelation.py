"""倍率間冗長性の正則化項と診断ロジックのユニット

正則化項は直交表現で 0・共線表現で大，無効化条件，勾配の有限性，スカラ出力を確かめる
診断ロジックは既知の共線/直交構造で類似度・CKA・実効ランクが正しく並ぶことを確かめる
"""

import numpy as np
import pytest
import torch

from foveamil.evaluation.redundancy import (
    aggregate_redundancy,
    cosine_similarity_matrix,
    effective_rank,
    linear_cka,
    mean_pairwise_cosine,
)
from foveamil.models import FoveaMIL, ForwardContext
from foveamil.models.regularizers import iter_active_regularizers
from foveamil.models.regularizers.decorrelation import (
    METHOD_COSINE,
    METHOD_COVARIANCE,
    DecorrelationRegularizer,
)
from foveamil.training.config import TrainConfig


# --- 正則化項 ---


def _context_from_rows(rows):
    """``[L, D]`` 配列を ``m_list``（各 ``[1, 1, D]``）の ForwardContext にする"""
    m_list = [torch.tensor(r, dtype=torch.float32).reshape(1, 1, -1) for r in rows]
    return ForwardContext(m_list=m_list)


def test_cosine_orthogonal_vectors_near_zero():
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COSINE)
    rows = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    out = reg(_context_from_rows(rows), torch.tensor([0]))
    assert out.dim() == 0
    assert out.item() < 1e-4


def test_covariance_uncorrelated_vectors_near_zero():
    # 共分散法は次元方向の標準化後の相関を見るため，高次元の独立乱数で 0 に近づく
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COVARIANCE)
    rng = np.random.default_rng(0)
    rows = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]
    out = reg(_context_from_rows(rows), torch.tensor([0]))
    assert out.dim() == 0
    assert out.item() < 1e-2


@pytest.mark.parametrize("method", [METHOD_COSINE, METHOD_COVARIANCE])
def test_collinear_vectors_large(method):
    reg = DecorrelationRegularizer(weight=1.0, method=method)
    base = np.array([0.3, -0.7, 1.1, 0.5], dtype=np.float32)
    collinear = [base, base * 2.0, base * -1.5]
    orthogonal = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    coll = reg(_context_from_rows(collinear), torch.tensor([0]))
    orth = reg(_context_from_rows(orthogonal), torch.tensor([0]))
    assert coll.item() > 0.5
    assert coll.item() > orth.item()


def test_cosine_collinear_near_one():
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COSINE)
    base = np.array([0.3, -0.7, 1.1, 0.5], dtype=np.float32)
    rows = [base, base * 2.0, base * 4.0]
    out = reg(_context_from_rows(rows), torch.tensor([0]))
    # 完全共線なら余弦 2 乗の非対角平均は 1
    assert out.item() == pytest.approx(1.0, abs=1e-5)


def test_single_layer_returns_zero():
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COSINE)
    ctx = ForwardContext(m_list=[torch.randn(1, 1, 4)])
    out = reg(ctx, torch.tensor([0]))
    assert out.dim() == 0
    assert out.item() == 0.0


def test_finite_nonzero_gradient_flows_to_inputs():
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COSINE)
    m_list = [torch.randn(1, 1, 6, requires_grad=True) for _ in range(3)]
    ctx = ForwardContext(m_list=m_list)
    out = reg(ctx, torch.tensor([0]))
    out.backward()
    for m in m_list:
        assert m.grad is not None
        assert torch.isfinite(m.grad).all()
    assert sum(float(m.grad.abs().sum()) for m in m_list) > 0.0


def test_covariance_gradient_finite():
    reg = DecorrelationRegularizer(weight=1.0, method=METHOD_COVARIANCE)
    m_list = [torch.randn(2, 1, 8, requires_grad=True) for _ in range(3)]
    ctx = ForwardContext(m_list=m_list)
    out = reg(ctx, torch.tensor([0, 1]))
    out.backward()
    for m in m_list:
        assert m.grad is not None and torch.isfinite(m.grad).all()


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        DecorrelationRegularizer(weight=1.0, method="nope")


# --- from_config / 自動探索 ---


def _multi_config(**kwargs):
    return TrainConfig(magnifications=[1.25, 2.5], **kwargs)


def test_from_config_inactive_when_weight_zero():
    assert DecorrelationRegularizer.from_config(_multi_config()) is None
    assert (
        DecorrelationRegularizer.from_config(
            _multi_config(decorrelation_weight=0.0)
        )
        is None
    )


def test_from_config_inactive_single_magnification():
    config = TrainConfig(magnifications=[1.25], decorrelation_weight=1.0)
    assert DecorrelationRegularizer.from_config(config) is None


def test_from_config_active_multi_magnification():
    reg = DecorrelationRegularizer.from_config(
        _multi_config(decorrelation_weight=0.5, decorrelation_method=METHOD_COVARIANCE)
    )
    assert isinstance(reg, DecorrelationRegularizer)
    assert reg.weight == 0.5
    assert reg.method == METHOD_COVARIANCE


def test_auto_discovered_via_iter_active():
    active = iter_active_regularizers(
        _multi_config(decorrelation_weight=1.0)
    )
    assert any(isinstance(r, DecorrelationRegularizer) for r in active)
    # 無効設定では現れない
    assert iter_active_regularizers(TrainConfig(magnifications=[1.25, 2.5])) == []


# --- 診断ロジック ---


def _collinear_set(num_slides=12, num_layers=3, dim=16, seed=0):
    """全倍率が共線（倍率ごとに固定スケール）のスライド集合を作る

    倍率ごとのスケールは全スライド共通にするため，倍率対は単一スケールの線形写像で
    結ばれる（CKA は 1，余弦は 1，実効ランクは 1 に近づく）
    """
    rng = np.random.default_rng(seed)
    scales = rng.uniform(0.5, 2.0, size=num_layers).astype(np.float32)
    slides = []
    for _ in range(num_slides):
        base = rng.standard_normal(dim).astype(np.float32)
        slides.append(np.stack([base * s for s in scales], axis=0))
    return slides


def _orthogonal_set(num_slides=12, num_layers=3, dim=16, seed=1):
    """倍率ごとに独立な部分空間を使う（ほぼ直交）スライド集合を作る"""
    rng = np.random.default_rng(seed)
    block = dim // num_layers
    slides = []
    for _ in range(num_slides):
        rows = []
        for i in range(num_layers):
            v = np.zeros(dim, dtype=np.float32)
            seg = slice(i * block, (i + 1) * block)
            v[seg] = rng.standard_normal(block).astype(np.float32)
            rows.append(v)
        slides.append(np.stack(rows, axis=0))
    return slides


def test_cosine_orders_collinear_above_orthogonal():
    coll = _collinear_set()
    orth = _orthogonal_set()
    coll_cos = np.mean([mean_pairwise_cosine(v) for v in coll])
    orth_cos = np.mean([mean_pairwise_cosine(v) for v in orth])
    assert coll_cos > orth_cos
    assert orth_cos < 0.2


def test_effective_rank_lower_for_collinear():
    coll = _collinear_set()
    orth = _orthogonal_set()
    coll_rank = np.mean([effective_rank(v) for v in coll])
    orth_rank = np.mean([effective_rank(v) for v in orth])
    assert coll_rank < orth_rank
    # 完全共線の実効ランクは 1 に近い
    assert coll_rank == pytest.approx(1.0, abs=0.1)


def test_cosine_matrix_symmetric_unit_diag():
    v = _orthogonal_set(num_slides=1)[0]
    matrix = cosine_similarity_matrix(v)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_linear_cka_high_for_collinear_low_for_orthogonal():
    coll = _collinear_set()
    orth = _orthogonal_set()
    coll_stack = np.stack(coll, axis=0)
    orth_stack = np.stack(orth, axis=0)
    cka_coll = linear_cka(coll_stack[:, 0, :], coll_stack[:, 1, :])
    cka_orth = linear_cka(orth_stack[:, 0, :], orth_stack[:, 1, :])
    assert cka_coll > cka_orth
    assert cka_coll == pytest.approx(1.0, abs=0.1)


def test_aggregate_orders_redundancy():
    coll = aggregate_redundancy(_collinear_set())
    orth = aggregate_redundancy(_orthogonal_set())
    assert coll["n_slides"] == 12 and coll["n_layers"] == 3
    assert coll["mean_cosine"] > orth["mean_cosine"]
    assert coll["mean_cka"] > orth["mean_cka"]
    assert coll["mean_effective_rank"] < orth["mean_effective_rank"]
    assert np.shape(coll["cka_matrix"]) == (3, 3)
    assert np.shape(coll["pearson_matrix"]) == (3, 3)


def test_max_effective_rank_is_min_layers_dim():
    # 倍率数 L が次元 D を上回る場合は上限が D になる
    rng = np.random.default_rng(7)
    slides = [rng.standard_normal((5, 2)).astype(np.float32) for _ in range(4)]
    summary = aggregate_redundancy(slides)
    assert summary["max_effective_rank"] == min(5, 2)
    # 実効ランクは上限 min(L, D) を超えない
    assert summary["mean_effective_rank"] <= summary["max_effective_rank"] + 1e-9


def test_aggregate_edge_cases():
    assert aggregate_redundancy([]) == {"n_slides": 0, "n_layers": 0}
    single = aggregate_redundancy([np.ones((1, 8), dtype=np.float32)])
    assert single["n_layers"] == 1
    assert "mean_cosine" not in single


def test_aggregate_degenerate_constant_slides_no_warning():
    # 全スライドが同一だと CKA は標本分散ゼロで nan になるが警告は出さず JSON 安全に返す
    import warnings

    same = np.random.default_rng(3).standard_normal((2, 16)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary = aggregate_redundancy([same, same, same])
    # 標本方向の分散が無い CKA は nan（中心化でゼロ行列になる）
    assert np.isnan(summary["mean_cka"])
    # 余弦・実効ランクはスライド単位で計算でき有限
    assert np.isfinite(summary["mean_cosine"])
    assert np.isfinite(summary["mean_effective_rank"])


# --- 統合スモーク（実モデルの m_list で backward）---


def _model(num_layers):
    return FoveaMIL(
        in_feat_dim=8,
        hidden_feat_dim=16,
        out_feat_dim=12,
        k_sample=4,
        n_cls=3,
        num_layers=num_layers,
        topk_method="perturbed",
        fusion="sum",
    )


@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("method", [METHOD_COSINE, METHOD_COVARIANCE])
def test_integration_smoke_backward(num_layers, method):
    torch.manual_seed(0)
    model = _model(num_layers)
    model.train()
    m_list = []
    for layer_idx in range(num_layers):
        M, _, _ = model.forward_layer(torch.randn(1, 10, 8), layer_idx)
        m_list.append(M)
    ctx = ForwardContext(m_list=m_list)
    reg = DecorrelationRegularizer(weight=0.3, method=method)
    loss = reg.weight * reg(ctx, torch.tensor([0]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0
