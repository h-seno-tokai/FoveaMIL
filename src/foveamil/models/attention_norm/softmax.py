"""softmax アテンション正規化器

スコア ``[B, N]`` を最終軸の softmax で正規化する密な既定正規化器スパース系
正規化器（温度付き softmax，sparsemax，entmax 等）と同一インタフェースを持つ
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention_norm import register_attention_norm

# 正規化を取る軸
_NORM_AXIS = -1


class SoftmaxNorm(nn.Module):
    """最終軸の softmax で正規化する（パラメータなし）"""

    def forward(self, scores: Tensor) -> Tensor:
        """スコア ``[B, N]`` を softmax 正規化した ``[B, N]`` を返す"""
        return F.softmax(scores, dim=_NORM_AXIS)


@register_attention_norm("softmax")
def _build_softmax() -> SoftmaxNorm:
    return SoftmaxNorm()
