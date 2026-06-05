"""親パッチから子パッチへの index 計算

ある倍率で選んだ親パッチに対し，1 段上の倍率の子パッチ index を求める座標は
連続 2x の階層細分化（親→2x2=4 子）で生成され，子 global index は subdivide 順に
連続する（親 g の子が ``g*4 .. g*4+3``）倍率比が 2 のべき ``r=2^d`` のとき，親の
子は ``r*r`` 個で global index は連続ブロック ``g*r^2 .. g*r^2 + r^2-1`` になる
（2x を d 段合成した結果連続性が保たれるため）アテンションは順序不変なので
ブロック内の空間並びは結果に影響しない
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# 連続 2x（親→2x2）の子パッチ数
CHILDREN_PER_PARENT = 4
# 倍率比として許す最小値（昇順かつ拡大を要求）
MIN_RATIO = 2
# 浮動小数の倍率比比較の許容誤差
RATIO_TOLERANCE = 1e-6
# index 配列の dtype
INDEX_DTYPE = np.int64


def _is_power_of_two(value: int) -> bool:
    """``value`` が 2 のべき（>=1）か返す"""
    return value >= 1 and (value & (value - 1)) == 0


def children_per_parent(parent_mag: float, child_mag: float) -> int:
    """親→子の倍率比から子パッチ数（``比^2``）を返す

    倍率比は 2 のべき（``2, 4, 8, ...``）でなければならない比 ``r`` のとき，親 1 つは
    ``r*r`` 個の子を持つ（連続 2x を ``log2(r)`` 段合成した数）

    Args:
        parent_mag: 親倍率
        child_mag: 子倍率（``parent_mag`` より高い）

    Returns:
        子パッチ数 ``r^2``

    Raises:
        ValueError: 倍率比が 2 のべき（>=2）でない場合
    """
    ratio = child_mag / parent_mag
    rounded = int(round(ratio))
    if (
        rounded < MIN_RATIO
        or abs(ratio - rounded) > RATIO_TOLERANCE
        or not _is_power_of_two(rounded)
    ):
        raise ValueError(
            "magnification ratio must be an integer power of 2 (>= 2); "
            f"got {parent_mag} -> {child_mag} (ratio {ratio:.4g})"
        )
    return rounded * rounded


def validate_magnification_hierarchy(magnifications: Sequence[float]) -> None:
    """学習に使う倍率列が階層として妥当か検証する

    単一倍率（長さ 1）は許す（ズーム無しの attention pooling のみ）2 倍率以上は
    昇順かつ隣接比が 2 のべきであることを要求する（``[2.5, 10, 40]`` や
    ``[1.25, 2.5, 5, 40]`` のような飛ばし組も許す ``[1.25, 3.75]`` 等は不可）

    Args:
        magnifications: 低 → 高の順の倍率列

    Raises:
        ValueError: 空列，または隣接比が 2 のべきでない場合
    """
    if len(magnifications) < 1:
        raise ValueError("need at least 1 magnification")
    for parent, child in zip(magnifications, magnifications[1:]):
        children_per_parent(parent, child)


def compute_child_indices(
    parent_local_indices: np.ndarray,
    parent_global_indices: Optional[np.ndarray] = None,
    children: int = CHILDREN_PER_PARENT,
) -> np.ndarray:
    """親の local index から子の global index を計算する

    ``parent_global_indices`` があれば ``parent_global_indices[parent_local]`` で
    local→global へ変換し，無ければ ``parent_local`` を global とみなす変換後の各
    global 親に対し ``global * children + [0 .. children-1]`` を並べた ``(children*k,)``
    を返す``children`` は親→子の倍率比から :func:`children_per_parent` で得た値
    （連続 2x なら 4）

    Args:
        parent_local_indices: 選択された親パッチの local index ``(k,)``
        parent_global_indices: 現倍率パッチセットの global index ``(N,)````None``
            のとき ``parent_local_indices`` を global とみなす
        children: 1 親あたりの子パッチ数（``比^2``既定は連続 2x の 4）

    Returns:
        子パッチの global index ``(children*k,)``（各親の子が連続）
    """
    parent_local = np.asarray(parent_local_indices, dtype=INDEX_DTYPE)
    if parent_global_indices is not None:
        global_parents = np.asarray(parent_global_indices, dtype=INDEX_DTYPE)[
            parent_local
        ]
    else:
        global_parents = parent_local

    base = global_parents * children
    offsets = np.arange(children, dtype=INDEX_DTYPE)
    child_indices = (base[:, np.newaxis] + offsets).reshape(-1)
    return child_indices.astype(INDEX_DTYPE)
