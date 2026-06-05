"""L2 正則化 top-k（Permutahedron 射影）型の微分可能 top-k

スコアを Permutahedron へ Euclidean 射影し，``sum=k``・``0<=y<=1`` を満たす
スパースなマスク ``[B, N]`` を二分探索で求める射影値を上位 k 要素について
index 昇順に並べ，``scatter_`` で soft な選択行列 ``[B, k, N]`` を構築する
clamp と四則演算のみで構成されるため自動微分で勾配が流れる
Sander et al., Fast, Differentiable and Sparse Top-k (2023) に基づく
"""

from __future__ import annotations

import torch
from torch import Tensor

from foveamil.models.topk.base import TopKSelector

# 既定の正則化強度（温度パラメータ）
DEFAULT_EPSILON = 0.002
# 二分探索の既定反復回数
DEFAULT_MAX_ITER = 50
# 二分探索の下限初期値で min から引くマージン
_LOWER_MARGIN = 1.0


def soft_topk_projection(
    scores: Tensor, k: float, epsilon: float, max_iter: int
) -> Tensor:
    """``min ||y - scores/epsilon||^2 s.t. sum(y)=k, 0<=y<=1`` を二分探索で解く

    解は閾値 ``nu`` を用いて ``y = clamp(scores/epsilon - nu, 0, 1)`` で表され，
    ``sum(y)=k`` を満たす ``nu`` を二分探索で求める

    Args:
        scores: 入力スコア ``[B, N]``
        k: 和の目標値
        epsilon: 正則化強度小さいほどスパースになる
        max_iter: 二分探索の反復回数

    Returns:
        スパースマスク ``[B, N]``（``0<=y<=1``，``sum(y)≈k``）
    """
    scaled = scores / epsilon
    lower = scaled.min(dim=-1, keepdim=True).values - _LOWER_MARGIN
    upper = scaled.max(dim=-1, keepdim=True).values

    for _ in range(max_iter):
        nu = (lower + upper) / 2.0
        y = torch.clamp(scaled - nu, 0.0, 1.0)
        too_large = y.sum(dim=-1, keepdim=True) > k
        lower = torch.where(too_large, nu, lower)
        upper = torch.where(too_large, upper, nu)

    nu_final = (lower + upper) / 2.0
    return torch.clamp(scaled - nu_final, 0.0, 1.0)


class FastSparseTopK(TopKSelector):
    """Permutahedron 射影型の微分可能 top-k セレクタ

    Args:
        k: 選択する要素数
        epsilon: 正則化強度小さいほどスパースになる
        max_iter: 二分探索の反復回数
    """

    def __init__(
        self,
        k: int,
        epsilon: float = DEFAULT_EPSILON,
        max_iter: int = DEFAULT_MAX_ITER,
    ) -> None:
        super().__init__(k)
        self.epsilon = epsilon
        self.max_iter = max_iter

    def soft_select(self, scores: Tensor, k: int) -> Tensor:
        """射影マスクの上位 k 値から soft な選択行列 ``[B, k, N]`` を構築する"""
        mask = soft_topk_projection(scores, float(k), self.epsilon, self.max_iter)

        batch_size, num_elements = mask.shape
        indices = torch.topk(mask, k=k, dim=-1).indices
        indices = torch.sort(indices, dim=-1).values
        values = torch.gather(mask, 1, indices)

        selection = torch.zeros(batch_size, k, num_elements, device=mask.device)
        selection.scatter_(2, indices.unsqueeze(2), values.unsqueeze(2))
        return selection
