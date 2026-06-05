"""多解像度プーリング表現の融合

各倍率のプーリング表現 ``[B, 1, dim]`` のリストを 1 つの ``[B, out_dim]`` に
融合する共通インタフェースを定義する融合は識別器と分離し，``out_dim`` 属性で
後段の入力次元を宣言する
"""

from __future__ import annotations

import abc
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class Fusion(nn.Module, abc.ABC):
    """多解像度表現を融合する基底

    Attributes:
        out_dim: 融合後の特徴次元
    """

    out_dim: int

    @abc.abstractmethod
    def forward(self, M_list: Sequence[Tensor]) -> Tensor:
        """各倍率の表現 ``[B, 1, dim]`` のリストを ``[B, out_dim]`` に融合する"""


class SumFusion(Fusion):
    """各倍率の表現を要素ごとに総和して融合する

    Args:
        dim: 各倍率の表現次元
        num_layers: 倍率数
    """

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.out_dim = dim
        self.num_layers = num_layers

    def forward(self, M_list: Sequence[Tensor]) -> Tensor:
        """総和して ``[B, out_dim]`` に squeeze する"""
        fused = torch.stack(list(M_list), dim=0).sum(dim=0)
        return fused.squeeze(dim=1)


FUSION_METHODS = {
    "sum": SumFusion,
}


def build_fusion(name: str, dim: int, num_layers: int) -> Fusion:
    """名前から融合器を構築する

    Args:
        name: ``FUSION_METHODS`` に登録された融合名
        dim: 各倍率の表現次元
        num_layers: 倍率数

    Returns:
        構築した :class:`Fusion`

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    if name not in FUSION_METHODS:
        raise KeyError(
            f"unknown fusion method '{name}'; available: {sorted(FUSION_METHODS)}"
        )
    return FUSION_METHODS[name](dim=dim, num_layers=num_layers)
