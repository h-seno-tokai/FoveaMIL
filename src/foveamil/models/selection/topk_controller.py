"""微分可能 top-k による選択コントローラ

補助アテンションスコアのみから :class:`TopKSelector` で選択行列を作る射影特徴は
使わないスコアの上位 k を選ぶ既定の選択方式で，多様性を考慮しない
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from foveamil.models.selection import register_selection_controller
from foveamil.models.selection.base import SelectionController
from foveamil.models.topk import build_topk


@register_selection_controller("topk")
class TopKSelectionController(SelectionController):
    """補助アテンションスコアの上位 k を取る選択コントローラ

    Args:
        k: 選択する要素数
        topk_method: top-k 手法名（``build_topk`` のレジストリ）
        topk_kwargs: top-k セレクタへ渡す追加引数
    """

    def __init__(
        self,
        k: int,
        topk_method: str = "perturbed",
        topk_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(k)
        self.topk = build_topk(topk_method, k, **(topk_kwargs or {}))

    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """スコア ``[B, N]`` の上位 k から選択行列 ``[B, k, N]`` を返す（特徴は未使用）"""
        return self.topk(scores)
