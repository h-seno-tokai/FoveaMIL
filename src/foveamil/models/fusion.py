"""多解像度プーリング表現の融合

各倍率のプーリング表現 ``[B, 1, dim]`` のリストを 1 つの ``[B, out_dim]`` に
融合する共通インタフェースを定義する融合は識別器と分離し，``out_dim`` 属性で
後段の入力次元を宣言する全融合は ``out_dim == dim`` を保ちヘッドを不変にする
"""

from __future__ import annotations

import abc
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

# スケール自己アテンションの既定層数（軽量集約）
DEFAULT_SCALE_ATTENTION_LAYERS = 2
# スケール自己アテンションの既定ヘッド数
DEFAULT_SCALE_ATTENTION_HEADS = 4
# スケール自己アテンションの FFN 拡大率
_FFN_EXPANSION = 2
# スケール軸（M_list を stack した次元）
_SCALE_AXIS = 1
# プーリング表現の単位スケール軸（``[B, 1, dim]`` の 1 の軸）
_UNIT_AXIS = 1


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
        return fused.squeeze(dim=_UNIT_AXIS)


def _stack_scales(M_list: Sequence[Tensor]) -> Tensor:
    """``[B, 1, dim]`` のリストを ``[B, L, dim]`` のスケールトークン列へ並べる"""
    return torch.cat(list(M_list), dim=_SCALE_AXIS)


class GatedWeightedFusion(Fusion):
    """スライド依存のゲート重みでスケールを加重和して融合する

    各スケール表現 ``M_i`` から 1 スカラのゲートスコアを線形に作り，スケール軸で
    softmax して重みとし加重和を取る重みは入力内容（スライド）に依存し，総和は
    1 に正規化される単一倍率では softmax over 1 要素が 1 となり ``M`` をそのまま返す

    Args:
        dim: 各倍率の表現次元
        num_layers: 倍率数
    """

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.out_dim = dim
        self.num_layers = num_layers
        self.gate = nn.Linear(dim, 1)

    def forward(self, M_list: Sequence[Tensor]) -> Tensor:
        """ゲート重みで加重和し ``[B, out_dim]`` を返す"""
        tokens = _stack_scales(M_list)
        gate_logits = self.gate(tokens).squeeze(dim=-1)
        weights = torch.softmax(gate_logits, dim=-1)
        fused = (weights.unsqueeze(dim=-1) * tokens).sum(dim=_SCALE_AXIS)
        return fused


class _ScaleAttentionLayer(nn.Module):
    """スケールトークン列への 1 層の自己アテンション（pre-norm + FFN）

    Args:
        dim: トークン次元
        num_heads: アテンションヘッド数
        dropout: Dropout 率``None`` なら 0
    """

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * _FFN_EXPANSION),
            nn.GELU(),
            nn.Linear(dim * _FFN_EXPANSION, dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """``[B, L, dim]`` を更新して同形で返す"""
        normed = self.norm_attn(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens


class ScaleSelfAttentionFusion(Fusion):
    """スケールトークン列へ軽量な自己アテンションをかけ平均集約して融合する

    L 個のスケール表現を ``[B, L, dim]`` のトークン列とみなし，少数層の自己
    アテンションでスケール間の相互作用を通したのちスケール軸で平均し ``[B, dim]``
    を返す単一倍率ではトークン 1 個の自己アテンションを経て ``M`` を返す（縮退安全）

    Args:
        dim: 各倍率の表現次元
        num_layers: 倍率数
        attention_layers: 自己アテンション層数
        num_heads: アテンションヘッド数
        dropout: Dropout 率``None`` なら 0

    Raises:
        ValueError: ``dim`` が ``num_heads`` で割り切れない場合
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        attention_layers: int = DEFAULT_SCALE_ATTENTION_LAYERS,
        num_heads: int = DEFAULT_SCALE_ATTENTION_HEADS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        heads = num_heads if dim % num_heads == 0 else 1
        self.out_dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            _ScaleAttentionLayer(dim, heads, dropout)
            for _ in range(attention_layers)
        )

    def forward(self, M_list: Sequence[Tensor]) -> Tensor:
        """自己アテンション後にスケール軸で平均し ``[B, out_dim]`` を返す"""
        tokens = _stack_scales(M_list)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.mean(dim=_SCALE_AXIS)


FUSION_METHODS = {
    "sum": SumFusion,
    "gated": GatedWeightedFusion,
    "scale_attention": ScaleSelfAttentionFusion,
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
