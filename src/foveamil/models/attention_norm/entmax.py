"""α-entmax アテンション正規化器

スコア ``[B, N]`` を Tsallis α-エントロピー正則化下の確率単体への射影で正規化する
出力は非負・和 1 で，``1<alpha<=2`` では鋭い入力に対して厳密に 0 を含むスパースな
分布になる``p = clamp((alpha-1)*z - tau, min=0)^(1/(alpha-1))`` の形を取り，和が 1 に
なる閾値 ``tau`` を二分探索で求める閾値は ``torch.no_grad`` 下で解き，逆伝播は
α-entmax の解析 Jacobian-vector 積を ``torch.autograd.Function`` で与える
``alpha=1`` で softmax，``alpha=2`` で sparsemax の極限に一致する
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
# サポートする α の下限・上限（softmax / sparsemax が両端の極限を覆う）
_ALPHA_MIN = 1.0
_ALPHA_MAX = 2.0


def _entmax_threshold(scaled: Tensor, alpha: float, max_iter: int) -> Tensor:
    """``sum_i p_i(tau) = 1`` を満たす閾値 ``tau`` を二分探索で求める ``[..., 1]``

    ``p_i(tau) = clamp(scaled_i - tau, min=0)^(1/(alpha-1))`` の和は ``tau`` について
    単調減少するため ``[tau_lo, tau_hi]`` を半分に詰める``scaled = (alpha-1)*z``
    tau_lo では最大要素のみで p_max=1 となり和は 1 以上，tau_hi では全要素が台から
    外れ和は 0 になる
    """
    exponent = 1.0 / (alpha - 1.0)
    z_max = scaled.max(dim=_NORM_AXIS, keepdim=True).values
    tau_lo = z_max - _SIMPLEX_SUM ** (alpha - 1.0)
    tau_hi = z_max

    for _ in range(max_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=_FLOOR) ** exponent
        too_small = p.sum(dim=_NORM_AXIS, keepdim=True) < _SIMPLEX_SUM
        tau_hi = torch.where(too_small, tau, tau_hi)
        tau_lo = torch.where(too_small, tau_lo, tau)
    return (tau_lo + tau_hi) / 2.0


class _EntmaxFunction(torch.autograd.Function):
    """α-entmax の前向き射影と解析 Jacobian-vector 積

    forward は閾値を ``torch.no_grad`` 下で解いて ``p`` を返し，backward は台集合 S 上で
    ``s_i = p_i^(2-alpha)`` を重みに ``ds_i = s_i*(g_i - <s,g>/sum_S s)`` を返す
    （台の外は 0）
    """

    @staticmethod
    def forward(ctx, scores: Tensor, alpha: float, max_iter: int) -> Tensor:
        with torch.no_grad():
            scaled = (alpha - 1.0) * scores
            tau = _entmax_threshold(scaled, alpha, max_iter)
            exponent = 1.0 / (alpha - 1.0)
            p = torch.clamp(scaled - tau, min=_FLOOR) ** exponent
            p = p / p.sum(dim=_NORM_AXIS, keepdim=True)
        ctx.alpha = alpha
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (p,) = ctx.saved_tensors
        alpha = ctx.alpha
        # 台集合上の重み s_i = p_i^(2-alpha)（台の外は p=0 ゆえ s=0）
        s = p ** (2.0 - alpha)
        s_sum = s.sum(dim=_NORM_AXIS, keepdim=True)
        weighted = (s * grad_output).sum(dim=_NORM_AXIS, keepdim=True) / s_sum
        grad_scores = s * (grad_output - weighted)
        return grad_scores, None, None


class Entmax(nn.Module):
    """α-entmax で正規化する（パラメータなし）

    Args:
        alpha: Tsallis エントロピーの次数``(1, 2]`` を取り ``2`` で sparsemax に一致する
            （``1`` の極限は softmax，``2`` は sparsemax）
        max_iter: 二分探索の反復回数

    Raises:
        ValueError: ``alpha`` が ``(1, 2]`` の外（``alpha=1`` は softmax として許す）
    """

    def __init__(
        self, alpha: float = DEFAULT_ALPHA, max_iter: int = DEFAULT_MAX_ITER
    ) -> None:
        super().__init__()
        alpha = float(alpha)
        is_softmax = abs(alpha - _ALPHA_MIN) < _ALPHA_SOFTMAX_TOL
        if not is_softmax and not (_ALPHA_MIN < alpha <= _ALPHA_MAX):
            raise ValueError(
                f"alpha must be in (1, 2], got {alpha} "
                "(alpha=1 is softmax, alpha=2 is sparsemax)"
            )
        self.alpha = alpha
        self.max_iter = int(max_iter)

    def forward(self, scores: Tensor) -> Tensor:
        """スコア ``[B, N]`` を α-entmax 正規化した ``[B, N]`` を返す"""
        if scores.shape[_NORM_AXIS] == 0:
            return scores
        if abs(self.alpha - _ALPHA_MIN) < _ALPHA_SOFTMAX_TOL:
            return F.softmax(scores, dim=_NORM_AXIS)
        return _EntmaxFunction.apply(scores, self.alpha, self.max_iter)


@register_attention_norm("entmax")
def _build_entmax(
    alpha: float = DEFAULT_ALPHA, max_iter: int = DEFAULT_MAX_ITER
) -> Entmax:
    return Entmax(alpha, max_iter)
