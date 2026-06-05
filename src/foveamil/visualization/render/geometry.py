"""可視化の座標・寸法計算（純関数）

level-0 ピクセル空間と各倍率のピクセル空間，拡大ビューのセル矩形の間の変換を担う
パッチの level-0 footprint・親→子のオフセット・子数 r² は ``coordinates`` と
``hierarchy`` の式に一致させる（子数は :func:`hierarchy.children_per_parent` へ委譲し
ここで再導出しない）画像 I/O もモデルも扱わない
"""

from __future__ import annotations

from typing import List, Tuple

from foveamil.training.hierarchy import children_per_parent

# パッチの一辺（全倍率で共通の画素数 coords/features の DEFAULT_PATCH_SIZE と一致）
DEFAULT_PATCH_SIZE = 224


def patch_footprint_level0(
    magnification: float, actual_max_mag: int, patch_size: int = DEFAULT_PATCH_SIZE
) -> int:
    """倍率パッチの level-0 上の一辺（画素）を返す

    ``coordinates`` の ``parent_level0_patch_size = int(patch_size * max/mag)`` と一致する
    """
    return int(patch_size * (actual_max_mag / magnification))


def ratio_and_children(parent_mag: float, child_mag: float) -> Tuple[int, int]:
    """親→子の倍率比 ``r`` と子数 ``cpp=r²`` を返す（比は 2 のべき）

    子数は :func:`hierarchy.children_per_parent` に委譲する（再導出しない）
    """
    cpp = children_per_parent(parent_mag, child_mag)
    ratio = int(round((cpp) ** 0.5))
    return ratio, cpp


def child_offset_level0(
    parent_mag: float,
    actual_max_mag: int,
    ratio: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> int:
    """親パッチを r×r に割るときの子オフセット（level-0 画素）を返す

    ``parent_footprint // ratio``連続 2x（r=2）では ``coordinates`` の
    ``child_offset = parent_level0_patch_size // 2`` と一致する
    """
    return patch_footprint_level0(parent_mag, actual_max_mag, patch_size) // ratio


def slot_to_cell(slot: int, ratio: int) -> Tuple[int, int]:
    """子 slot を ``(row, col)`` セル位置に変換する

    子 global index は ``subdivide_coordinates`` が連続 2x2 を ``log2(r)`` 段ネストした
    順（Morton/Z-order）で生成されるので，slot を粗い段から 2bit ずつ ``(dy, dx)`` に
    分解して位置を組む``r=2`` では ``(slot//2, slot%2)`` と一致する
    """
    d = ratio.bit_length() - 1
    row = col = 0
    for level in reversed(range(d)):
        quad = (slot >> (2 * level)) & 3
        row = row * 2 + (quad >> 1)
        col = col * 2 + (quad & 1)
    return row, col


def child_local_offsets(ratio: int) -> List[Tuple[int, int]]:
    """r×r 子の ``(row, col)`` を slot 順（ネスト 2x2）で返す"""
    return [slot_to_cell(slot, ratio) for slot in range(ratio * ratio)]


def child_slot(child_global: int, parent_global: int, cpp: int) -> int:
    """子 global index を親内 slot（0..cpp-1）へ変換する

    ``hierarchy.compute_child_indices`` の ``child = parent*cpp + slot`` の逆変換
    """
    return int(child_global) - int(parent_global) * cpp


def level0_to_mag_px(
    xy: Tuple[int, int], magnification: float, actual_max_mag: int
) -> Tuple[float, float]:
    """level-0 座標を指定倍率のピクセル座標へ写す（``xy * mag/max``）"""
    scale = magnification / actual_max_mag
    return xy[0] * scale, xy[1] * scale


def patch_rect_at_mag(
    xy: Tuple[int, int],
    magnification: float,
    actual_max_mag: int,
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> Tuple[float, float, int, int]:
    """level-0 左上座標のパッチを指定倍率ピクセルの ``(x, y, w, h)`` にする

    その倍率での一辺は定義上 ``patch_size``左上は :func:`level0_to_mag_px`
    """
    x, y = level0_to_mag_px(xy, magnification, actual_max_mag)
    return x, y, patch_size, patch_size


def child_cell_in_view(slot: int, ratio: int, zoom_px: int) -> Tuple[int, int, int, int]:
    """拡大ビュー（``zoom_px`` 正方）内の子セル矩形 ``(x, y, w, h)`` を返す"""
    cell = zoom_px // ratio
    row, col = slot_to_cell(slot, ratio)
    return col * cell, row * cell, cell, cell
