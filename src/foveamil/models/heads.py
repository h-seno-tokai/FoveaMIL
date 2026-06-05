"""識別器ヘッド

融合済み特徴 ``[B, in_dim]`` をクラスロジット ``[B, n_cls]`` へ写す
融合と分離した独立部品とする
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class LinearClassifierHead(nn.Module):
    """線形識別器ヘッド

    Args:
        in_dim: 入力特徴次元
        n_cls: 出力クラス数
    """

    def __init__(self, in_dim: int, n_cls: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """ロジット ``[B, n_cls]`` を返す"""
        return self.fc(x)
