"""摂動サンプリング型の微分可能 top-k

スコアにガウス雑音を加えて多数の摂動標本を作り，各標本で hard top-k を取って
one-hot 化し標本平均を取ることで soft な選択行列 ``[B, k, N]`` を得る
勾配はガウス雑音を用いた期待値推定で逆伝播する
Cordonnier et al., Differentiable Patch Selection (CVPR 2021) に基づく
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from foveamil.models.topk.base import TopKSelector

# 既定の摂動標本数
DEFAULT_NUM_SAMPLES = 100
# 既定の摂動標準偏差
DEFAULT_SIGMA = 0.002


class _PerturbedTopKFunction(torch.autograd.Function):
    """摂動 top-k の前向き計算と期待値勾配を定義する autograd 関数"""

    @staticmethod
    def forward(ctx, scores: Tensor, k: int, num_samples: int, sigma: float) -> Tensor:
        batch_size, num_elements = scores.shape
        noise = torch.normal(
            mean=0.0, std=1.0, size=(batch_size, num_samples, num_elements)
        ).to(scores.device)

        perturbed = scores[:, None, :] + noise * sigma
        indices = torch.topk(perturbed, k=k, dim=-1, sorted=False).indices
        indices = torch.sort(indices, dim=-1).values

        perturbed_output = F.one_hot(indices, num_classes=num_elements).float()
        indicators = perturbed_output.mean(dim=1)

        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.save_for_backward(perturbed_output, noise)
        return indicators

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if grad_output is None:
            return (None, None, None, None)
        perturbed_output, noise = ctx.saved_tensors
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", perturbed_output, noise)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input, None, None, None)


class PerturbedTopK(TopKSelector):
    """摂動サンプリング型の微分可能 top-k セレクタ

    Args:
        k: 選択する要素数
        num_samples: 摂動標本数
        sigma: 摂動の標準偏差
    """

    def __init__(
        self,
        k: int,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        sigma: float = DEFAULT_SIGMA,
    ) -> None:
        super().__init__(k)
        self.num_samples = num_samples
        self.sigma = sigma

    def soft_select(self, scores: Tensor, k: int) -> Tensor:
        """摂動標本平均による soft な選択行列 ``[B, k, N]`` を返す"""
        return _PerturbedTopKFunction.apply(scores, k, self.num_samples, self.sigma)
