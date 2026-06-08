"""識別器ヘッド

融合済み特徴 ``[B, in_dim]`` をクラスロジット ``[B, n_cls]`` へ写す
融合と分離した独立部品とする線形ヘッドと小 MLP ヘッドを名前で選べる
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

# 小 MLP ヘッドの既定中間次元
DEFAULT_MLP_HIDDEN_DIM = 512


class LinearClassifierHead(nn.Module):
    """線形識別器ヘッド

    Args:
        in_dim: 入力特徴次元
        n_cls: 出力クラス数
    """

    def __init__(self, in_dim: int, n_cls: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """ロジット ``[B, n_cls]`` を返す"""
        return self.fc(x)


class MLPClassifierHead(nn.Module):
    """小 MLP 識別器ヘッド

    ``Linear → LayerNorm → ReLU → Dropout → Linear`` の 2 層構成で容量を増やす
    LayerNorm は非線形の前に置き，活性前の分布を安定させる

    Args:
        in_dim: 入力特徴次元
        n_cls: 出力クラス数
        hidden_dim: 中間次元（``None`` なら ``DEFAULT_MLP_HIDDEN_DIM``）
        dropout: Dropout 率（``None`` なら Dropout なし）
    """

    def __init__(
        self,
        in_dim: int,
        n_cls: int,
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else DEFAULT_MLP_HIDDEN_DIM
        layers: list = [
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        ]
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, n_cls))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """ロジット ``[B, n_cls]`` を返す"""
        return self.mlp(x)


HEAD_METHODS = {
    "linear": LinearClassifierHead,
    "mlp": MLPClassifierHead,
}


def build_head(
    name: str,
    in_dim: int,
    n_cls: int,
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
) -> nn.Module:
    """名前から識別器ヘッドを構築する

    出力契約は ``[B, n_cls]`` で全ヘッド共通``linear`` は ``hidden_dim`` /
    ``dropout`` を受け取らず既定の線形ヘッドと数値一致する

    Args:
        name: ``HEAD_METHODS`` に登録されたヘッド名
        in_dim: 入力特徴次元
        n_cls: 出力クラス数
        hidden_dim: 小 MLP ヘッドの中間次元（``mlp`` のときのみ有効）
        dropout: 小 MLP ヘッドの Dropout 率（``mlp`` のときのみ有効）

    Returns:
        構築した識別器ヘッド

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    if name not in HEAD_METHODS:
        raise KeyError(
            f"unknown head method '{name}'; available: {sorted(HEAD_METHODS)}"
        )
    if name == "linear":
        return LinearClassifierHead(in_dim, n_cls)
    return MLPClassifierHead(in_dim, n_cls, hidden_dim=hidden_dim, dropout=dropout)
