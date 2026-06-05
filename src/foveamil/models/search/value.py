"""ズーム木の状態価値ネットワーク

現倍率の射影特徴 ``[B, N, D]`` から，現在の部分選択のもとでスライドがどれだけ
よく分類されるかのスカラ推定 ``v(state)`` を返すゲート付きアテンションで状態を
プーリングし，小 MLP でスカラへ写す実現報酬（探索表現の負分類損失など）への
回帰で学習し，探索の葉評価に用いる

価値ネットは価値回帰項からのみ勾配を受ける学習されるのは
``value_loss_weight > 0`` のときに限り 0 なら重みは初期値のまま固定される
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention import GatedAttention

# プーリング用アテンションのクラス数（1 スコア/要素）
_POOL_N_CLS = 1
# プーリング重みを正規化する軸（要素 N 軸）
_POOL_AXIS = 1
# スカラ価値の出力次元
_VALUE_DIM = 1
# スカラ価値を squeeze する末尾軸
_SQUEEZE_AXIS = -1


class ValueNetwork(nn.Module):
    """部分選択状態のスカラ価値 ``v(state)`` を返すネットワーク

    Args:
        feat_dim: 入力射影特徴の次元 D
        hidden_dim: 中間次元
        dropout: Dropout 率``None`` なら Dropout なし
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.pool_attention = GatedAttention(
            feat_dim, hidden_dim, dropout, n_cls=_POOL_N_CLS
        )
        head: list = [nn.Linear(feat_dim, hidden_dim), nn.ReLU()]
        if dropout is not None:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden_dim, _VALUE_DIM))
        self.head = nn.Sequential(*head)

    def forward(self, features: Tensor) -> Tensor:
        """部分選択状態のスカラ価値 ``[B]`` を返す

        Args:
            features: 射影特徴 ``[B, N, D]``

        Returns:
            スカラ価値 ``[B]``
        """
        scores, _ = self.pool_attention(features)
        weights = F.softmax(scores.permute(0, 2, 1), dim=_POOL_AXIS)
        pooled = (weights @ features).squeeze(dim=_POOL_AXIS)
        return self.head(pooled).squeeze(dim=_SQUEEZE_AXIS)
