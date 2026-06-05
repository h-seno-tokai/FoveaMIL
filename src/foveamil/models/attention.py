"""ゲート付きアテンション機構

入力特徴 ``[B, N, L]`` から各要素のアテンションスコア ``A`` を計算する
``attention_a`` の Tanh 枝と ``attention_b`` の Sigmoid 枝を要素積し，
``attention_c`` で ``n_cls`` 次元へ写すスコアは正規化前の生値を返す
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch.nn as nn
from torch import Tensor


class GatedAttention(nn.Module):
    """ゲート付きアテンション

    Args:
        L: 入力特徴次元
        D: 中間層の特徴次元
        dropout: Dropout 率``None`` なら Dropout を挟まない
        n_cls: 出力アテンションのクラス数

    Attributes:
        attention_a: Tanh ゲート枝（``Linear(L, D)`` + Tanh ``(+ Dropout)``）
        attention_b: Sigmoid ゲート枝（``Linear(L, D)`` + Sigmoid ``(+ Dropout)``）
        attention_c: スコア射影（``Linear(D, n_cls)``）
    """

    def __init__(self, L: int, D: int, dropout: Optional[float] = None, n_cls: int = 1) -> None:
        super().__init__()
        layers_a: List[nn.Module] = [nn.Linear(L, D), nn.Tanh()]
        layers_b: List[nn.Module] = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout is not None:
            layers_a.append(nn.Dropout(dropout))
            layers_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*layers_a)
        self.attention_b = nn.Sequential(*layers_b)
        self.attention_c = nn.Linear(D, n_cls)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """アテンションスコアと入力をそのまま返す

        Args:
            x: 入力特徴 ``[B, N, L]``

        Returns:
            ``(A, x)``A はアテンションスコア ``[B, N, n_cls]``（正規化前）
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        gated = a.mul(b)
        A = self.attention_c(gated)
        return A, x
