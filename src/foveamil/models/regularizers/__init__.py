from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

from foveamil.models.regularizers.base import ForwardContext, Regularizer

# 正則化項名 → クラス
REGULARIZERS: Dict[str, Type[Regularizer]] = {}
_DISCOVERED = False
# 自動探索の対象外にするサブモジュール名
_SKIP_MODULES = frozenset({"base"})


def register_regularizer(cls: Type[Regularizer]) -> Type[Regularizer]:
    """正則化項クラスをレジストリへ登録するデコレータ（``cls.name`` をキーにする）"""
    REGULARIZERS[cls.name] = cls
    return cls


def _discover() -> None:
    """パッケージ内のサブモジュールを import し登録を発火させる"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for module in pkgutil.iter_modules(__path__):
        if module.name in _SKIP_MODULES:
            continue
        importlib.import_module(f"{__name__}.{module.name}")


def iter_active_regularizers(config) -> List[Regularizer]:
    """設定から有効な正則化項の一覧を作る

    登録済みの各クラスの ``from_config`` を呼び，``None`` でないものを集める

    Args:
        config: ``TrainConfig``

    Returns:
        有効な :class:`Regularizer` のリスト（無ければ空）
    """
    _discover()
    active: List[Regularizer] = []
    for cls in REGULARIZERS.values():
        instance = cls.from_config(config)
        if instance is not None:
            active.append(instance)
    return active


def available_regularizers():
    """登録済みの正則化項名一覧を返す"""
    _discover()
    return sorted(REGULARIZERS)


__all__ = [
    "ForwardContext",
    "Regularizer",
    "register_regularizer",
    "iter_active_regularizers",
    "available_regularizers",
    "REGULARIZERS",
]
