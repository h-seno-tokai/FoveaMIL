"""微分可能 top-k セレクタの抽象基底

スコア ``[B, N]`` から選択行列 ``[B, k, N]`` を作る共通インタフェースを定義する
学習時（``self.training=True``）は各サブクラスが実装する soft な選択行列を返し，
推論時は基底が提供する hard 実装（``torch.topk`` で上位 k を取り one-hot 化）を返す
``k`` が ``N`` を超える場合は ``min(N, k)`` に丸める
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn
from torch import Tensor


class TopKSelector(nn.Module, abc.ABC):
    """スコアから選択行列を作る微分可能 top-k の基底

    Args:
        k: 選択する要素数
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def effective_k(self, num_elements: int) -> int:
        """要素数 ``num_elements`` に対する実効 k（``min(num_elements, k)``）を返す"""
        return min(num_elements, self.k)

    @abc.abstractmethod
    def soft_select(self, scores: Tensor, k: int) -> Tensor:
        """学習時の soft な選択行列を返す

        Args:
            scores: 入力スコア ``[B, N]``
            k: 実効 k

        Returns:
            soft な選択行列 ``[B, k, N]``
        """

    def hard_select(self, scores: Tensor, k: int) -> Tensor:
        """推論時の hard な選択行列を返す（上位 k を index 昇順で one-hot 化）

        Args:
            scores: 入力スコア ``[B, N]``
            k: 実効 k

        Returns:
            hard な選択行列 ``[B, k, N]``（float）
        """
        num_elements = scores.shape[-1]
        indices = torch.topk(scores, k=k, dim=-1, sorted=False).indices
        indices = torch.sort(indices, dim=-1).values
        return torch.nn.functional.one_hot(indices, num_classes=num_elements).float()

    def forward(self, scores: Tensor) -> Tensor:
        """選択行列 ``[B, k, N]`` を返す（学習時 soft / 推論時 hard）

        Args:
            scores: 入力スコア ``[B, N]``

        Returns:
            選択行列 ``[B, k, N]``
        """
        k = self.effective_k(scores.shape[-1])
        if self.training:
            return self.soft_select(scores, k)
        return self.hard_select(scores, k)
