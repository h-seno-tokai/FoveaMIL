"""クラス不均衡に対応する分類損失

いずれも ``(logits[B, C], target[B])`` を取り平均損失スカラを返す
:class:`torch.nn.CrossEntropyLoss` と同じ呼び出し規約に揃える素 cross-entropy
（``plain``）に加え logit-adjusted CE（``logit_adjusted``）・LDAM（``ldam``）・
class-balanced 重み付き CE（``class_balanced``）を提供するクラス頻度は学習 split から
算出した値を渡す（このモジュールはハードコードしない）``build_loss`` が損失種別名と
頻度から該当する :class:`torch.nn.Module` を構築する
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# 損失種別名
LOSS_PLAIN = "plain"
LOSS_LOGIT_ADJUSTED = "logit_adjusted"
LOSS_LDAM = "ldam"
LOSS_CLASS_BALANCED = "class_balanced"
# 損失種別名の集合
LOSS_TYPES = (LOSS_PLAIN, LOSS_LOGIT_ADJUSTED, LOSS_LDAM, LOSS_CLASS_BALANCED)
# 頻度 0 のクラスを事前確率の対数で扱う際の下限（log(0) 回避）
PRIOR_FLOOR = 1e-12


def _class_frequencies_tensor(
    class_frequencies: Sequence[int], device: torch.device
) -> Tensor:
    """クラスごとのサンプル件数を ``float`` テンソルへ整える"""
    return torch.as_tensor(
        [float(c) for c in class_frequencies], dtype=torch.float32, device=device
    )


class LogitAdjustedLoss(nn.Module):
    """logit-adjusted cross-entropy

    クラス事前確率 ``π_y = n_y / Σ n``（学習 split の頻度比）の対数を logit へ
    ``τ·log π_y`` で加算してから cross-entropy を取る少数クラスへ実効的なマージンを
    与え argmax 閾値の偏りを補正する

    Args:
        class_frequencies: クラスごとのサンプル件数
        tau: 補正強度 τ（0 で素 CE と一致）
    """

    def __init__(self, class_frequencies: Sequence[int], tau: float = 1.0) -> None:
        super().__init__()
        counts = _class_frequencies_tensor(class_frequencies, torch.device("cpu"))
        priors = counts / counts.sum()
        log_priors = torch.log(priors.clamp_min(PRIOR_FLOOR))
        self.tau = tau
        self.register_buffer("log_priors", log_priors)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        adjusted = logits + self.tau * self.log_priors
        return F.cross_entropy(adjusted, target)


class LDAMLoss(nn.Module):
    """label-distribution-aware margin（LDAM）損失

    クラス頻度 ``n_y`` に応じたマージン ``Δ_y = C / n_y^{1/4}`` を正解クラスの logit から
    引いてから cross-entropy を取る少数クラスほど大きなマージンを課す定数 ``C`` は
    最大マージンが ``max_margin`` になるよう ``Δ`` を正規化して決める

    Args:
        class_frequencies: クラスごとのサンプル件数
        max_margin: 最大マージン（最小頻度クラスのマージン）
        weight: クラス重み（``None`` で無し``CrossEntropyLoss`` の ``weight`` と同義）
    """

    def __init__(
        self,
        class_frequencies: Sequence[int],
        max_margin: float = 0.5,
        weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        counts = _class_frequencies_tensor(class_frequencies, torch.device("cpu"))
        inv_quartic = 1.0 / counts.clamp_min(1.0).pow(0.25)
        margins = inv_quartic * (max_margin / inv_quartic.max())
        self.register_buffer("margins", margins)
        self.register_buffer(
            "weight", weight if weight is not None else torch.tensor([])
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        margin_per_sample = self.margins[target]
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0)), target] -= margin_per_sample
        weight = self.weight if self.weight.numel() > 0 else None
        return F.cross_entropy(adjusted, target, weight=weight)


class ClassBalancedLoss(nn.Module):
    """class-balanced 重み付き cross-entropy

    有効標本数 ``E_y = (1 - β^{n_y}) / (1 - β)`` の逆数をクラス重みとし クラス数で
    平均が 1 になるよう正規化してから重み付き cross-entropy を取る``β→0`` で重み無し，
    ``β→1`` で頻度逆数重みに近づく

    Args:
        class_frequencies: クラスごとのサンプル件数
        beta: 有効標本数の β（``[0, 1)``）
    """

    def __init__(self, class_frequencies: Sequence[int], beta: float = 0.999) -> None:
        super().__init__()
        counts = _class_frequencies_tensor(class_frequencies, torch.device("cpu"))
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        weight = 1.0 / effective_num.clamp_min(PRIOR_FLOOR)
        weight = weight / weight.sum() * weight.numel()
        self.register_buffer("weight", weight)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, target, weight=self.weight)


def build_loss(
    loss_type: str,
    class_frequencies: Sequence[int],
    tau: float = 1.0,
    beta: float = 0.999,
    ldam_max_margin: float = 0.5,
) -> nn.Module:
    """損失種別名とクラス頻度から損失モジュールを構築する

    ``plain`` は :class:`torch.nn.CrossEntropyLoss`（素 CE）を返し既定挙動と一致する
    他の種別はクラス頻度に依存する補正を持つ損失を返す

    Args:
        loss_type: 損失種別名（:data:`LOSS_TYPES` のいずれか）
        class_frequencies: クラスごとのサンプル件数（学習 split から算出した値）
        tau: ``logit_adjusted`` の補正強度
        beta: ``class_balanced`` の有効標本数 β
        ldam_max_margin: ``ldam`` の最大マージン

    Returns:
        ``(logits, target)`` を取る損失モジュール

    Raises:
        ValueError: 未知の損失種別名
    """
    if loss_type == LOSS_PLAIN:
        return nn.CrossEntropyLoss()
    if loss_type == LOSS_LOGIT_ADJUSTED:
        return LogitAdjustedLoss(class_frequencies, tau=tau)
    if loss_type == LOSS_LDAM:
        return LDAMLoss(class_frequencies, max_margin=ldam_max_margin)
    if loss_type == LOSS_CLASS_BALANCED:
        return ClassBalancedLoss(class_frequencies, beta=beta)
    raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got '{loss_type}'")
