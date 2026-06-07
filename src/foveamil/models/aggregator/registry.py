"""集約器のレジストリと構築関数

集約器名 → クラスの対応を保持し，名前から :class:`Aggregator` を構築する各実装
モジュールは ``register_aggregator`` デコレータで登録し，``_discover`` がパッケージ内
を import して登録を発火させる
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Type

from foveamil.models.aggregator.base import Aggregator

# 集約器名 → クラス
AGGREGATORS: Dict[str, Type[Aggregator]] = {}
_DISCOVERED = False
# 自動探索の対象外にするサブモジュール名
_SKIP_MODULES = frozenset({"base", "registry"})


def register_aggregator(name: str):
    """集約器をレジストリへ登録するデコレータ"""

    def _decorator(cls: Type[Aggregator]) -> Type[Aggregator]:
        AGGREGATORS[name] = cls
        return cls

    return _decorator


def _discover() -> None:
    """パッケージ内のサブモジュールを import し登録を発火させる"""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package = __name__.rsplit(".", 1)[0]
    module = importlib.import_module(package)
    for info in pkgutil.iter_modules(module.__path__):
        if info.name in _SKIP_MODULES:
            continue
        importlib.import_module(f"{package}.{info.name}")


def build_aggregator(
    name: str,
    dim: int,
    hidden_dim: int,
    dropout: Optional[float] = None,
    **kwargs,
) -> Aggregator:
    """名前から集約器を構築する

    Args:
        name: ``AGGREGATORS`` に登録された集約器名
        dim: 入力特徴次元（出力 ``M`` の次元も同一）
        hidden_dim: 内部の中間次元
        dropout: Dropout 率``None`` なら Dropout を挟まない
        **kwargs: 各集約器固有の追加引数

    Returns:
        構築した :class:`Aggregator`

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    _discover()
    if name not in AGGREGATORS:
        raise KeyError(
            f"unknown aggregator '{name}'; available: {sorted(AGGREGATORS)}"
        )
    return AGGREGATORS[name](dim=dim, hidden_dim=hidden_dim, dropout=dropout, **kwargs)


def available_aggregators():
    """登録済みの集約器名一覧を返す"""
    _discover()
    return sorted(AGGREGATORS)
