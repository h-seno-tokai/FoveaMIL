"""ゲート付きアテンションによる既定の集約器（ABMIL 相当）

各要素を独立にスコアリングするゲート付きアテンションで softmax 重み付き和を取り，
プーリング表現 ``M`` を作るパッチ間コンテキストを見ない従来挙動を再現する既定実装
で，スコアリング・正規化・重み付き和の順序は従来の ``_project_and_pool`` と数値一致
する
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn.functional as F
from torch import Tensor

from foveamil.models.aggregator.base import Aggregator
from foveamil.models.aggregator.registry import register_aggregator
from foveamil.models.attention import GatedAttention

# プーリング用アテンションのクラス数（1 スコア/要素）
_ATTENTION_N_CLS = 1


@register_aggregator("abmil")
class GatedAttentionAggregator(Aggregator):
    """ゲート付きアテンションプーリング集約器

    Args:
        dim: 入力特徴次元（出力 ``M`` の次元も同一）
        hidden_dim: アテンション中間次元
        dropout: Dropout 率``None`` なら Dropout を挟まない
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: Optional[float] = None) -> None:
        super().__init__(dim, hidden_dim, dropout)
        self.attention = GatedAttention(dim, hidden_dim, dropout, n_cls=_ATTENTION_N_CLS)

    def forward(self, x_fc: Tensor) -> Tuple[Tensor, Tensor]:
        """ゲート付きアテンションで softmax 重み付き和を取り ``(M, A)`` を返す"""
        A, _ = self.attention(x_fc)
        A = F.softmax(A.permute(0, 2, 1), dim=-1)
        M = A @ x_fc
        return M, A
