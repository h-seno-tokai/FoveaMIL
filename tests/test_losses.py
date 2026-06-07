"""不均衡対応損失のユニット

各損失の数式が小例で手計算と一致し勾配が logits へ流れることを確かめる``plain`` が
素 cross-entropy と数値一致すること（後方互換）クラス頻度の自動算出が件数と一致する
ことを確かめる
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from foveamil.training.losses import (
    LOSS_CLASS_BALANCED,
    LOSS_LDAM,
    LOSS_LOGIT_ADJUSTED,
    LOSS_PLAIN,
    ClassBalancedLoss,
    LDAMLoss,
    LogitAdjustedLoss,
    build_loss,
)
from foveamil.training.trainer import _class_frequencies


# --- plain（後方互換） ---


def test_plain_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(5, 3)
    target = torch.tensor([0, 1, 2, 1, 0])
    loss = build_loss(LOSS_PLAIN, [10, 20, 30])
    expected = F.cross_entropy(logits, target)
    assert torch.allclose(loss(logits, target), expected)


def test_logit_adjusted_tau_zero_equals_plain():
    # τ=0 では補正項が消えて素 CE と一致する
    torch.manual_seed(1)
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 2, 1, 0])
    loss = LogitAdjustedLoss([5, 50, 100], tau=0.0)
    assert torch.allclose(loss(logits, target), F.cross_entropy(logits, target))


# --- logit-adjusted CE の数式 ---


def test_logit_adjusted_matches_manual_formula():
    freqs = [10, 30, 60]
    tau = 1.5
    logits = torch.tensor([[0.5, -0.2, 1.0]])
    target = torch.tensor([2])
    loss = LogitAdjustedLoss(freqs, tau=tau)

    total = sum(freqs)
    log_priors = torch.tensor([math.log(f / total) for f in freqs])
    adjusted = logits + tau * log_priors
    expected = F.cross_entropy(adjusted, target)
    assert torch.allclose(loss(logits, target), expected)


def test_logit_adjusted_gradient_flows_to_logits():
    logits = torch.randn(3, 4, requires_grad=True)
    target = torch.tensor([0, 1, 3])
    out = LogitAdjustedLoss([1, 2, 3, 4], tau=1.0)(logits, target)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


# --- LDAM の数式 ---


def test_ldam_matches_manual_formula():
    freqs = [16, 81, 256]
    max_margin = 0.5
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 0.1]])
    target = torch.tensor([0, 2])
    loss = LDAMLoss(freqs, max_margin=max_margin)

    inv_quartic = np.array([1.0 / (f ** 0.25) for f in freqs])
    margins = inv_quartic * (max_margin / inv_quartic.max())
    adjusted = logits.clone()
    for row, cls in enumerate(target.tolist()):
        adjusted[row, cls] -= float(margins[cls])
    expected = F.cross_entropy(adjusted, target)
    assert torch.allclose(loss(logits, target), expected)


def test_ldam_max_margin_on_minority_class():
    # 最小頻度クラスのマージンが max_margin と一致する
    freqs = [16, 81, 256]
    max_margin = 0.7
    loss = LDAMLoss(freqs, max_margin=max_margin)
    assert math.isclose(float(loss.margins.max()), max_margin, rel_tol=1e-6)
    assert int(loss.margins.argmax()) == 0  # 最小頻度クラス


def test_ldam_gradient_flows_to_logits():
    logits = torch.randn(4, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2, 1])
    out = LDAMLoss([5, 10, 20], max_margin=0.5)(logits, target)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


# --- class-balanced の数式 ---


def test_class_balanced_matches_manual_formula():
    freqs = [10, 100, 1000]
    beta = 0.99
    logits = torch.tensor([[0.5, 0.1, -0.3], [0.0, 1.0, 0.2]])
    target = torch.tensor([0, 1])
    loss = ClassBalancedLoss(freqs, beta=beta)

    effective = np.array([(1.0 - beta ** f) / (1.0 - beta) for f in freqs])
    weight = 1.0 / effective
    weight = weight / weight.sum() * len(freqs)
    expected = F.cross_entropy(
        logits, target, weight=torch.tensor(weight, dtype=torch.float32)
    )
    assert torch.allclose(loss(logits, target), expected, atol=1e-6)


def test_class_balanced_weight_normalized_to_num_classes():
    # 重みは平均 1（総和 = クラス数）に正規化される
    loss = ClassBalancedLoss([10, 100, 1000], beta=0.999)
    assert math.isclose(float(loss.weight.sum()), 3.0, rel_tol=1e-6)
    # 少数クラスの重みが多数クラスより大きい
    assert loss.weight[0] > loss.weight[2]


def test_class_balanced_gradient_flows_to_logits():
    logits = torch.randn(3, 3, requires_grad=True)
    target = torch.tensor([0, 1, 2])
    out = ClassBalancedLoss([5, 50, 500], beta=0.99)(logits, target)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


# --- factory ---


@pytest.mark.parametrize(
    "loss_type,cls",
    [
        (LOSS_LOGIT_ADJUSTED, LogitAdjustedLoss),
        (LOSS_LDAM, LDAMLoss),
        (LOSS_CLASS_BALANCED, ClassBalancedLoss),
    ],
)
def test_build_loss_dispatches(loss_type, cls):
    assert isinstance(build_loss(loss_type, [1, 2, 3]), cls)


def test_build_loss_plain_is_cross_entropy():
    assert isinstance(build_loss(LOSS_PLAIN, [1, 2, 3]), torch.nn.CrossEntropyLoss)


def test_build_loss_rejects_unknown():
    with pytest.raises(ValueError):
        build_loss("nope", [1, 2, 3])


# --- クラス頻度の自動算出 ---


class _FakeDataset:
    def __init__(self, labels):
        self._labels = labels

    def __len__(self):
        return len(self._labels)

    def get_label(self, idx):
        return self._labels[idx]


def test_class_frequencies_counts_labels():
    ds = _FakeDataset([0, 0, 1, 2, 2, 2])
    assert _class_frequencies(ds, 3) == [2, 1, 3]


def test_class_frequencies_includes_absent_classes():
    # 出現しないクラスも 0 件で並びに含む（頻度ベクトルが n_cls 長）
    ds = _FakeDataset([0, 0, 1])
    assert _class_frequencies(ds, 4) == [2, 1, 0, 0]
