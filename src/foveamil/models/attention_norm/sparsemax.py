"""sparsemax アテンション正規化器

スコア ``[B, N]`` を確率単体への Euclidean 射影で正規化する出力は非負・和 1 で，
鋭い入力に対しては厳密に 0 を含むスパースな分布になるソート・累積和による
閾値 ``tau`` を求め ``clamp(scores - tau, min=0)`` を返す残った台集合上での
gather により autograd が正しい劣勾配を流す
Martins and Astudillo, From Softmax to Sparsemax (2016) に基づく
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from foveamil.models.attention_norm import register_attention_norm

# 正規化を取る軸
_NORM_AXIS = -1
# 確率単体の和の目標値
_SIMPLEX_SUM = 1.0
# clamp の下限
_FLOOR = 0.0


def _threshold_and_support(scores: Tensor) -> Tensor:
    """sparsemax の閾値 ``tau`` を最終軸ごとに求める ``[B, 1]``

    降順ソートと累積和から台集合サイズ ``k(z)`` を定め，
    ``tau = (cumsum_k - 1) / k`` を返す
    """
    sorted_scores, _ = torch.sort(scores, dim=_NORM_AXIS, descending=True)
    cumulative = sorted_scores.cumsum(dim=_NORM_AXIS)
    rank = torch.arange(
        1, scores.shape[_NORM_AXIS] + 1, device=scores.device, dtype=scores.dtype
    )
    support = rank * sorted_scores > (cumulative - _SIMPLEX_SUM)
    k = support.sum(dim=_NORM_AXIS, keepdim=True)
    cumulative_k = cumulative.gather(_NORM_AXIS, k - 1)
    return (cumulative_k - _SIMPLEX_SUM) / k.to(scores.dtype)


class Sparsemax(nn.Module):
    """確率単体への Euclidean 射影で正規化する（パラメータなし）"""

    def forward(self, scores: Tensor) -> Tensor:
        """スコア ``[B, N]`` を sparsemax 正規化した ``[B, N]`` を返す"""
        tau = _threshold_and_support(scores)
        return torch.clamp(scores - tau, min=_FLOOR)


@register_attention_norm("sparsemax")
def _build_sparsemax() -> Sparsemax:
    return Sparsemax()
