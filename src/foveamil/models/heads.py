"""識別器ヘッド

融合済み特徴 ``[B, in_dim]`` をクラスロジット ``[B, n_cls]`` へ写す
融合と分離した独立部品とする
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Type


class ClassifierHead(nn.Module):
    """識別器ヘッドの基底クラス"""
    pass


class LinearClassifierHead(ClassifierHead):
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


class MLPClassifierHead(ClassifierHead):
    """小 MLP 識別器ヘッド

    Linear -> LayerNorm -> ReLU -> Dropout -> Linear の 2 層構成

    Args:
        in_dim: 入力特徴次元
        n_cls: 出力クラス数
        hidden_dim: 中間次元 (既定: in_dim // 2)
        dropout: Dropout 率
    """

    def __init__(
        self,
        in_dim: int,
        n_cls: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim // 2, n_cls)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_cls),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# HEAD レジストリ
_HEAD_REGISTRY: Dict[str, Type[ClassifierHead]] = {
    "linear": LinearClassifierHead,
    "mlp": MLPClassifierHead,
}


def build_classifier_head(
    name: str, in_dim: int, n_cls: int, **kwargs
) -> ClassifierHead:
    """レジストリから識別器ヘッドを構築する

    Args:
        name: ヘッド名 ("linear" | "mlp")
        in_dim: 入力次元
        n_cls: クラス数
        kwargs: 各ヘッドへ渡す追加引数

    Returns:
        構築された識別器ヘッド
    """
    if name not in _HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head type '{name}'; available: {list(_HEAD_REGISTRY.keys())}"
        )
    return _HEAD_REGISTRY[name](in_dim, n_cls, **kwargs)


def available_heads() -> list[str]:
    """利用可能なヘッド名の一覧を返す"""
    return list(_HEAD_REGISTRY.keys())
