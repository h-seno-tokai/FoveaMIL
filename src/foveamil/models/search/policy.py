"""ズーム木の事前方策ネットワーク

現倍率の射影特徴 ``[B, N, D]`` から，次倍率へ展開する候補親 N 個上の事前分布
``π(a|state)`` を返すゲート付きアテンションを状態符号化器に流用し，要素ごとの
スコアを softmax で正規化して方策にする容量を基線の補助アテンションと揃え，探索の
事前として用いる
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.attention import GatedAttention

# 方策スコアのクラス数（1 スコア/候補）
_SCORE_N_CLS = 1
# 候補軸を畳む対象軸（``[B, N, 1]`` の末尾）
_SCORE_AXIS = -1
# 方策を正規化する軸（候補 N 軸）
_NORM_AXIS = -1


class PolicyNetwork(nn.Module):
    """候補親上の事前方策 ``π(a|state)`` を返すネットワーク

    Args:
        feat_dim: 入力射影特徴の次元 D
        hidden_dim: ゲート付きアテンション中間次元
        dropout: Dropout 率``None`` なら Dropout なし
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.scorer = GatedAttention(
            feat_dim, hidden_dim, dropout, n_cls=_SCORE_N_CLS
        )

    def logits(self, features: Tensor) -> Tensor:
        """候補ごとの正規化前スコア ``[B, N]`` を返す

        Args:
            features: 射影特徴 ``[B, N, D]``

        Returns:
            候補ごとのスコア ``[B, N]``（正規化前）
        """
        scores, _ = self.scorer(features)
        return scores.squeeze(dim=_SCORE_AXIS)

    def forward(self, features: Tensor) -> Tensor:
        """候補親上の事前方策 ``[B, N]`` を返す（候補軸 softmax）

        Args:
            features: 射影特徴 ``[B, N, D]``

        Returns:
            事前方策 ``[B, N]``（候補軸の和が 1）
        """
        return F.softmax(self.logits(features), dim=_NORM_AXIS)
