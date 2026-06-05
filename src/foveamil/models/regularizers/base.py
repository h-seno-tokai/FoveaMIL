"""補助損失（正則化項）の抽象基底と forward 文脈

:class:`ForwardContext` は段階 forward が集めた中間量を運ぶ容器で，各倍率のプーリング
表現 ``m_list``，各層の正規化補助アテンション ``layer_aux``，各層の選択 ``selections``，
コンポーネントが寄与する名前付きスカラ損失 ``extra_losses`` を持つ
:class:`Regularizer` はこの文脈と正解ラベルからスカラ損失を返す補助損失の共通インタ
フェースで，``weight`` を持ち，``from_config`` で設定に応じて有効/無効を決める
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch import Tensor


@dataclass
class ForwardContext:
    """段階 forward が集めた中間量の容器

    Attributes:
        m_list: 各倍率のプーリング表現 ``[B, 1, out_dim]`` のリスト
        extra_losses: コンポーネントが寄与する名前付きスカラ損失
        layer_aux: 各層の正規化補助アテンション ``[B, N]``（最終層は ``None``）
        selections: 各層の選択情報（``select_indices`` / ``select_weight`` 等）
    """

    m_list: List[Tensor]
    extra_losses: Dict[str, Tensor] = field(default_factory=dict)
    layer_aux: List[Optional[Tensor]] = field(default_factory=list)
    selections: List[Optional[Dict[str, Tensor]]] = field(default_factory=list)


class Regularizer(abc.ABC):
    """文脈とラベルからスカラ補助損失を返す正則化項の基底

    学習対象パラメータを持たない（最適化器はモデルのパラメータのみを更新し，正則化項の
    パラメータは登録されない）パラメータを伴う補助損失はモデル側の部品として持たせる

    Attributes:
        name: レジストリ上の名前
        weight: 総損失へ加える際の係数
    """

    name: str = ""

    def __init__(self, weight: float) -> None:
        self.weight = weight

    @abc.abstractmethod
    def __call__(self, context: ForwardContext, label: Tensor) -> Tensor:
        """スカラ補助損失を返す

        Args:
            context: 段階 forward の中間量
            label: 正解クラス ``[B]``

        Returns:
            スカラ損失
        """

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config) -> "Optional[Regularizer]":
        """設定から有効な正則化項を作る無効なら ``None`` を返す

        Args:
            config: ``TrainConfig``

        Returns:
            構築した正則化項，または無効時 ``None``
        """
