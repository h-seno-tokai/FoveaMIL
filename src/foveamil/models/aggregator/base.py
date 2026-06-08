"""インスタンス集約器の抽象基底

射影済み特徴 ``[B, N, D]`` を 1 つのプーリング表現 ``M = [B, 1, D]`` に集約する
共通インタフェースを定義する併せて後段（インスタンス補助損失・可視化）が参照する
プーリング重み ``A = [B, 1, N]`` を返す各倍率の pooling を差し替える境界であり，
選択経路（補助アテンション→セレクタ）とは独立に扱う

出力契約:
- ``M`` は ``[B, 1, D]``（``D`` は入力特徴次元と同一）
- ``A`` は ``[B, 1, N]`` の非負・最終軸で和 1 のプーリング重み
- ``batch=1`` ・``N`` 可変で動き，``N`` が小でも縮退せず決定的に振る舞う
"""

from __future__ import annotations

import abc
from typing import Tuple

import torch.nn as nn
from torch import Tensor


class Aggregator(nn.Module, abc.ABC):
    """射影特徴をプーリング表現へ集約する基底

    Args:
        dim: 入力特徴次元（出力 ``M`` の次元も同一）
        hidden_dim: 内部の中間次元
        dropout: Dropout 率``None`` なら Dropout を挟まない
    """

    def __init__(self, dim: int, hidden_dim: int, dropout=None) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @abc.abstractmethod
    def forward(self, x_fc: Tensor) -> Tuple[Tensor, Tensor]:
        """射影特徴 ``[B, N, D]`` から ``(M, A)`` を返す

        Args:
            x_fc: 射影済み特徴 ``[B, N, D]``

        Returns:
            ``(M, A)``M はプーリング表現 ``[B, 1, D]``，A はプーリング重み
            ``[B, 1, N]``（非負・最終軸で和 1）
        """
