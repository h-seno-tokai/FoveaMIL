from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict

from torch import Tensor

# 正規化器名 → ファクトリ（``**kwargs`` を受けて callable を返す）
ATTENTION_NORMS: Dict[str, Callable[..., Callable[[Tensor], Tensor]]] = {}
_DISCOVERED = False
# 自動探索の対象外にするサブモジュール名
_SKIP_MODULES = frozenset({"base"})


def register_attention_norm(name: str):
    """正規化器ファクトリをレジストリへ登録するデコレータ"""

    def _decorator(factory):
        ATTENTION_NORMS[name] = factory
        return factory

    return _decorator


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


def build_attention_norm(name: str, **kwargs) -> Callable[[Tensor], Tensor]:
    """名前からアテンション正規化器を構築する

    Args:
        name: ``ATTENTION_NORMS`` に登録された正規化器名
        **kwargs: 各正規化器固有の追加引数

    Returns:
        スコア ``[B, N]`` を最終軸で正規化する callable

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    _discover()
    if name not in ATTENTION_NORMS:
        raise KeyError(
            f"unknown attention norm '{name}'; available: {sorted(ATTENTION_NORMS)}"
        )
    return ATTENTION_NORMS[name](**kwargs)


def available_attention_norms():
    """登録済みの正規化器名一覧を返す"""
    _discover()
    return sorted(ATTENTION_NORMS)
