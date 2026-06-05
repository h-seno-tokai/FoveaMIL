"""選択集合の多様性を促す DPP 補助損失

各層の選択部分カーネル ``L_S`` の負の log-det を加える項多様（互いに非類似）な
選択ほど log-det が大きく，この負号により損失が小さくなるため，多様な選択へ誘導する
``config.selector=="dpp"`` かつ多倍率かつ ``config.dpp_diversity_weight>0`` のときのみ
有効で，``ForwardContext.dpp_log_dets`` に積まれた各層の log-det を平均して用いる
Kulesza & Taskar 2012 の DPP 尤度（部分集合確率 ∝ det(L_S)）に基づく
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from foveamil.models.regularizers import register_regularizer
from foveamil.models.regularizers.base import ForwardContext, Regularizer

# 既定の多様性正則化重み（0 で無効）
DEFAULT_DPP_DIVERSITY_WEIGHT = 0.0
# 単一倍率を表す倍率数
_SINGLE_MAG = 1
# DPP 選択コントローラ名
_DPP_SELECTOR = "dpp"


@register_regularizer
class DPPDiversityRegularizer(Regularizer):
    """選択部分カーネルの負 log-det を加える多様性正則化

    Args:
        weight: 総損失へ加える際の係数
    """

    name = "dpp_diversity"

    def __call__(self, context: ForwardContext, label: Tensor) -> Tensor:
        """各層の選択 log-det の平均に負号を付けたスカラを返す

        ``dpp_log_dets`` が空のとき（DPP 未使用や最終層のみ）は 0 を返す

        Args:
            context: 段階 forward の中間量
            label: 正解クラス ``[B]``（未使用）

        Returns:
            スカラ補助損失（``-mean(log-det)``）
        """
        log_dets = context.dpp_log_dets
        if not log_dets:
            return torch.zeros((), device=label.device)
        stacked = torch.stack([ld for ld in log_dets])
        return -stacked.mean()

    @classmethod
    def from_config(cls, config) -> "Optional[Regularizer]":
        """設定から有効な多様性正則化を作る

        ``selector=="dpp"`` かつ多倍率かつ ``dpp_diversity_weight>0`` のときのみ有効で，
        それ以外は ``None`` を返す（既定では無効）

        Args:
            config: ``TrainConfig``

        Returns:
            構築した正則化項，または無効時 ``None``
        """
        weight = getattr(config, "dpp_diversity_weight", DEFAULT_DPP_DIVERSITY_WEIGHT)
        if weight <= 0.0:
            return None
        if getattr(config, "selector", None) != _DPP_SELECTOR:
            return None
        mags = getattr(config, "magnifications", None)
        if not mags or len(mags) <= _SINGLE_MAG:
            return None
        return cls(weight)
