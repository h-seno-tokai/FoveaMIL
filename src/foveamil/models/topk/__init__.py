from __future__ import annotations

from typing import Dict, Type

from foveamil.models.topk.base import TopKSelector
from foveamil.models.topk.perturbed import PerturbedTopK
from foveamil.models.topk.sparse import FastSparseTopK

TOPK_METHODS: Dict[str, Type[TopKSelector]] = {
    "perturbed": PerturbedTopK,
    "fast_sparse": FastSparseTopK,
}


def build_topk(name: str, k: int, **kwargs) -> TopKSelector:
    """名前から top-k セレクタを構築する

    Args:
        name: ``TOPK_METHODS`` に登録された手法名
        k: 選択する要素数
        **kwargs: 各セレクタ固有の追加引数

    Returns:
        構築した :class:`TopKSelector`

    Raises:
        KeyError: ``name`` が未登録の場合
    """
    if name not in TOPK_METHODS:
        raise KeyError(
            f"unknown topk method '{name}'; available: {sorted(TOPK_METHODS)}"
        )
    return TOPK_METHODS[name](k=k, **kwargs)


__all__ = [
    "TopKSelector",
    "PerturbedTopK",
    "FastSparseTopK",
    "TOPK_METHODS",
    "build_topk",
]
