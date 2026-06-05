"""パッチ選択コントローラの抽象基底

補助アテンションの正規化スコア ``[B, N]`` と射影特徴 ``[B, N, D]`` から，次倍率へ
ズームするパッチの選択行列 ``[B, k, N]`` を作る共通インタフェースを定義するスコア
のみで選ぶ手法は特徴を無視し，多様性を見る手法（DPP 等）は特徴を使う学習時は soft，
推論時は hard な選択行列を返すかは各コントローラの責任とする
"""

from __future__ import annotations

import abc

import torch.nn as nn
from torch import Tensor


class SelectionController(nn.Module, abc.ABC):
    """選択行列を作るコントローラの基底

    Args:
        k: 選択する要素数
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    @abc.abstractmethod
    def select(self, scores: Tensor, features: Tensor) -> Tensor:
        """選択行列 ``[B, k, N]`` を返す

        Args:
            scores: 正規化済み補助アテンション ``[B, N]``
            features: 射影特徴 ``[B, N, D]``

        Returns:
            選択行列 ``[B, k, N]``（学習時 soft / 推論時 hard）
        """
