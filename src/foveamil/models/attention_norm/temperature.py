"""温度付き softmax アテンション正規化器

スコア ``[B, N]`` を温度 ``temperature`` で割ってから最終軸の softmax を取る
``temperature`` が小さいほど分布が鋭くなり大きいほど平坦になる``temperature=1``
で密な softmax と一致する
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention_norm import register_attention_norm

# 正規化を取る軸
_NORM_AXIS = -1
# 既定の温度（1 で密な softmax と一致）
DEFAULT_TEMPERATURE = 1.0


class TemperatureSoftmax(nn.Module):
    """温度でスケールした softmax で正規化する（パラメータなし）

    Args:
        temperature: スコアを割る温度小さいほど鋭く大きいほど平坦になる

    Raises:
        ValueError: ``temperature`` が正でない場合
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(self, scores: Tensor) -> Tensor:
        """スコア ``[B, N]`` を温度付き softmax で正規化した ``[B, N]`` を返す"""
        return F.softmax(scores / self.temperature, dim=_NORM_AXIS)


@register_attention_norm("temperature")
def _build_temperature(
    temperature: float = DEFAULT_TEMPERATURE,
) -> TemperatureSoftmax:
    return TemperatureSoftmax(temperature)
