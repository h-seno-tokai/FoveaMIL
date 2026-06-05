"""α-entmax アテンション正規化器

スコア ``[B, N]`` を Tsallis α-エントロピー正則化下の確率単体への射影で正規化する
出力は非負・和 1 で，``alpha>1`` では鋭い入力に対して厳密に 0 を含むスパースな分布
になる``p = clamp((alpha-1)*z/tau* - 1, min=0)^(1/(alpha-1))`` の形を取り，和が 1 に
なる閾値 ``tau*`` を二分探索で求める閾値経由の clamp/べき乗のみで構成されるため
autograd が勾配を流す``alpha=1`` で softmax，``alpha=2`` で sparsemax に一致する
Peters et al., Sparse Sequence-to-Sequence Models (2019) に基づく
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention_norm import register_attention_norm

# 正規化を取る軸
_NORM_AXIS = -1
# 確率単体の和の目標値
_SIMPLEX_SUM = 1.0
# clamp の下限
_FLOOR = 0.0
# 既定の α（1.5-entmax）
DEFAULT_ALPHA = 1.5
# 二分探索の既定反復回数
DEFAULT_MAX_ITER = 50
# softmax へ落とす α の近接判定閾値
_ALPHA_SOFTMAX_TOL = 1e-4


def _entmax_threshold(scores: Tensor, alpha: float, max_iter: int) -> Tensor:
    """``sum_i p_i(tau) = 1`` を満たす閾値 ``tau`` を二分探索で求める ``[B, 1]``

    ``p_i(tau) = clamp((alpha-1)*z_i - tau, min=0)^(1/(alpha-1))`` の和は ``tau`` に
    ついて単調減少するため ``[tau_lo, tau_hi]`` を半分に詰める下限 ``tau_lo`` は
    全要素が台に入る値，上限 ``tau_hi`` は最大要素のみが台に残る値で初期化する
    """
    exponent = 1.0 / (alpha - 1.0)
    scaled = (alpha - 1.0) * scores
    z_max = scaled.max(dim=_NORM_AXIS, keepdim=True).values
    # 和 S(tau) は tau について単調減少する tau_lo で S>=1，tau_hi で S<=1 を保つ
    # tau_lo では最大要素のみで p_max=1 となり和は 1 以上
    tau_lo = z_max - _SIMPLEX_SUM ** (alpha - 1.0)
    # tau_hi では最大要素も台から外れ和は 0
    tau_hi = z_max

    for _ in range(max_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=_FLOOR) ** exponent
        too_small = p.sum(dim=_NORM_AXIS, keepdim=True) < _SIMPLEX_SUM
        tau_hi = torch.where(too_small, tau, tau_hi)
        tau_lo = torch.where(too_small, tau_lo, tau)
    return (tau_lo + tau_hi) / 2.0


class Entmax(nn.Module):
    """α-entmax で正規化する（パラメータなし）

    Args:
        alpha: Tsallis エントロピーの次数``1`` で softmax，``2`` で sparsemax に一致する
        max_iter: 二分探索の反復回数

    Raises:
        ValueError: ``alpha`` が 1 未満の場合
    """

    def __init__(
        self, alpha: float = DEFAULT_ALPHA, max_iter: int = DEFAULT_MAX_ITER
    ) -> None:
        super().__init__()
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1, got {alpha}")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)

    def forward(self, scores: Tensor) -> Tensor:
        """スコア ``[B, N]`` を α-entmax 正規化した ``[B, N]`` を返す"""
        if abs(self.alpha - 1.0) < _ALPHA_SOFTMAX_TOL:
            return F.softmax(scores, dim=_NORM_AXIS)
        tau = _entmax_threshold(scores, self.alpha, self.max_iter)
        scaled = (self.alpha - 1.0) * scores
        exponent = 1.0 / (self.alpha - 1.0)
        p = torch.clamp(scaled - tau, min=_FLOOR) ** exponent
        return p / p.sum(dim=_NORM_AXIS, keepdim=True)


@register_attention_norm("entmax")
def _build_entmax(
    alpha: float = DEFAULT_ALPHA, max_iter: int = DEFAULT_MAX_ITER
) -> Entmax:
    return Entmax(alpha, max_iter)
