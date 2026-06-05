"""倍率間表現の冗長性を抑える補助損失（相関の罰則）

段階 forward が集めた各倍率のプーリング表現 ``m_list`` をサンプルごとに ``[L, D]`` の
行列とみなし，倍率間の相関を罰則化する``"cosine"`` は倍率ベクトル対の余弦類似度の
2 乗を非対角で平均し，``"covariance"`` は各倍率ベクトルを次元方向に標準化した上で
作る ``L×L`` 相互相関の非対角 2 乗和を罰則化する直交で 0，共線で増大する
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from foveamil.models.regularizers import register_regularizer
from foveamil.models.regularizers.base import ForwardContext, Regularizer

# 余弦類似度の手法名
METHOD_COSINE = "cosine"
# 相互相関（Barlow-Twins / VICReg 系）の手法名
METHOD_COVARIANCE = "covariance"
# 選べる手法の一覧
METHODS = (METHOD_COSINE, METHOD_COVARIANCE)
# 倍率数がこれ未満なら罰則を 0 とする（対が作れない）
_MIN_LAYERS = 2
# ノルム・分散の数値安定化 eps
_EPS = 1e-8
# m_list 各要素のスライド軸（バッチサイズ 1 前提でない一般形の squeeze 対象）
_POOL_AXIS = 1


def _stack_m_list(m_list: List[Tensor]) -> Tensor:
    """``m_list``（各 ``[B, 1, D]``）をサンプルごとの ``[B, L, D]`` に積む"""
    return torch.stack([m.squeeze(_POOL_AXIS) for m in m_list], dim=1)


def _cosine_redundancy(stacked: Tensor) -> Tensor:
    """倍率ベクトル対の余弦類似度 2 乗の非対角平均を返す

    ``stacked`` は ``[B, L, D]``各サンプルで ``[L, D]`` を L2 正規化し ``L×L`` の
    グラム行列を作り，非対角の 2 乗をバッチ・対で平均する直交で 0，共線で 1 に近づく

    Args:
        stacked: 倍率ベクトル ``[B, L, D]``

    Returns:
        スカラ罰則
    """
    normalized = stacked / stacked.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    gram = normalized @ normalized.transpose(-1, -2)
    num_layers = gram.shape[-1]
    eye = torch.eye(num_layers, device=gram.device, dtype=gram.dtype)
    off_diag = gram * (1.0 - eye)
    num_pairs = num_layers * (num_layers - 1)
    return (off_diag.pow(2).sum(dim=(-1, -2)) / num_pairs).mean()


def _covariance_redundancy(stacked: Tensor) -> Tensor:
    """標準化倍率ベクトルの相互相関の非対角 2 乗和を返す

    ``stacked`` は ``[B, L, D]``各サンプルで各倍率ベクトルを次元方向に零平均・単位
    分散へ標準化し，``(1/D) Z Zᵀ`` で ``L×L`` 相互相関を作り，非対角 2 乗和を
    ``L(L-1)`` で正規化してバッチ平均する（Barlow-Twins / VICReg の冗長性項に準ずる）

    Args:
        stacked: 倍率ベクトル ``[B, L, D]``

    Returns:
        スカラ罰則
    """
    mean = stacked.mean(dim=-1, keepdim=True)
    std = stacked.std(dim=-1, keepdim=True, unbiased=False).clamp_min(_EPS)
    z = (stacked - mean) / std
    dim = stacked.shape[-1]
    cross = (z @ z.transpose(-1, -2)) / dim
    num_layers = cross.shape[-1]
    eye = torch.eye(num_layers, device=cross.device, dtype=cross.dtype)
    off_diag = cross * (1.0 - eye)
    denom = num_layers * (num_layers - 1)
    return (off_diag.pow(2).sum(dim=(-1, -2)) / denom).mean()


_DISPATCH = {
    METHOD_COSINE: _cosine_redundancy,
    METHOD_COVARIANCE: _covariance_redundancy,
}


@register_regularizer
class DecorrelationRegularizer(Regularizer):
    """倍率間表現の冗長性を抑える補助損失

    Args:
        weight: 総損失へ加える係数
        method: 罰則手法（``"cosine"`` / ``"covariance"``）

    Raises:
        ValueError: ``method`` が未知の場合
    """

    name = "decorrelation"

    def __init__(self, weight: float, method: str = METHOD_COSINE) -> None:
        super().__init__(weight)
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {METHODS}, got '{method}'"
            )
        self.method = method

    def __call__(self, context: ForwardContext, label: Tensor) -> Tensor:
        """各倍率のプーリング表現から冗長性罰則を返す

        倍率数が ``_MIN_LAYERS`` 未満なら 0 スカラを返す

        Args:
            context: 段階 forward の中間量
            label: 正解クラス ``[B]``（本項では未使用）

        Returns:
            スカラ罰則
        """
        m_list = context.m_list
        if len(m_list) < _MIN_LAYERS:
            return m_list[0].new_zeros(())
        stacked = _stack_m_list(m_list)
        return _DISPATCH[self.method](stacked)

    @classmethod
    def from_config(cls, config) -> "Optional[DecorrelationRegularizer]":
        """設定から有効な冗長性罰則を作る

        ``decorrelation_weight > 0`` かつ多倍率（``magnifications`` が 2 つ以上）の
        ときのみ構築する手法は ``decorrelation_method`` で選ぶ

        Args:
            config: ``TrainConfig``

        Returns:
            構築した罰則，または無効時 ``None``
        """
        weight = getattr(config, "decorrelation_weight", 0.0)
        magnifications = getattr(config, "magnifications", None) or []
        if weight <= 0.0 or len(magnifications) < _MIN_LAYERS:
            return None
        method = getattr(config, "decorrelation_method", METHOD_COSINE)
        return cls(weight=weight, method=method)
