from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Optional, Type

from foveamil.models.selection.base import SelectionController

# コントローラ名 → クラス
SELECTION_CONTROLLERS: Dict[str, Type[SelectionController]] = {}
_DISCOVERED = False
# 自動探索の対象外にするサブモジュール名
_SKIP_MODULES = frozenset({"base"})


def register_selection_controller(name: str):
    """選択コントローラをレジストリへ登録するデコレータ"""

    def _decorator(cls: Type[SelectionController]) -> Type[SelectionController]:
        SELECTION_CONTROLLERS[name] = cls
        return cls

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


def build_selection_controller(
    name: str,
    k: int,
    topk_method: str = "perturbed",
    topk_kwargs: Optional[dict] = None,
    **kwargs,
) -> SelectionController:
    """名前から選択コントローラを構築する

    Args:
        name: ``SELECTION_CONTROLLERS`` に登録されたコントローラ名
        k: 選択する要素数
        topk_method: 既定 top-k コントローラの top-k 手法名
        topk_kwargs: 既定 top-k コントローラへ渡す追加引数
        **kwargs: 各コントローラ固有の追加引数

    Returns:
        構築した :class:`SelectionController`

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    _discover()
    if name not in SELECTION_CONTROLLERS:
        raise KeyError(
            f"unknown selection controller '{name}'; "
            f"available: {sorted(SELECTION_CONTROLLERS)}"
        )
    if name == "topk":
        return SELECTION_CONTROLLERS[name](
            k, topk_method=topk_method, topk_kwargs=topk_kwargs
        )
    return SELECTION_CONTROLLERS[name](k, **kwargs)


def available_selection_controllers():
    """登録済みのコントローラ名一覧を返す"""
    _discover()
    return sorted(SELECTION_CONTROLLERS)


__all__ = [
    "SelectionController",
    "register_selection_controller",
    "build_selection_controller",
    "available_selection_controllers",
    "SELECTION_CONTROLLERS",
]
