"""visualization.render.geometry のユニット"""

import pytest

from foveamil.visualization.render.geometry import (
    child_cell_in_view,
    child_local_offsets,
    child_offset_level0,
    child_slot,
    level0_to_mag_px,
    patch_footprint_level0,
    patch_rect_at_mag,
    ratio_and_children,
    slot_to_cell,
)


def test_patch_footprint_level0():
    assert patch_footprint_level0(20, 40, 224) == 448
    assert patch_footprint_level0(40, 40, 224) == 224
    assert patch_footprint_level0(1.25, 40, 224) == 224 * 32


@pytest.mark.parametrize(
    "parent,child,r,cpp",
    [(20, 40, 2, 4), (10, 40, 4, 16), (5, 40, 8, 64), (1.25, 2.5, 2, 4)],
)
def test_ratio_and_children(parent, child, r, cpp):
    assert ratio_and_children(parent, child) == (r, cpp)


def test_child_offset_level0_matches_subdivide():
    # 連続 2x: coordinates の child_offset = parent_footprint // 2
    assert child_offset_level0(20, 40, 2, 224) == 224  # 448 // 2
    # 飛ばし 4x: 896 // 4
    assert child_offset_level0(10, 40, 4, 224) == 224


def _true_cell_nested(slot, ratio):
    """連続 2x2 を log2(r) 段合成した真の (row, col)（subdivide と同順）"""
    d = ratio.bit_length() - 1
    row = col = 0
    for level in reversed(range(d)):
        quad = (slot >> (2 * level)) & 3
        row = row * 2 + (quad >> 1)
        col = col * 2 + (quad & 1)
    return row, col


def test_slot_to_cell_r2_is_rowmajor():
    assert child_local_offsets(2) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_slot_to_cell_morton_for_skip_ratios():
    # r=4: slot4 はネスト順では (0,2)（row-major の (1,0) ではない）
    assert slot_to_cell(4, 4) == (0, 2)
    assert slot_to_cell(0, 4) == (0, 0)
    assert slot_to_cell(3, 4) == (1, 1)
    for ratio in (2, 4, 8):
        for slot in range(ratio * ratio):
            assert slot_to_cell(slot, ratio) == _true_cell_nested(slot, ratio)


def test_child_slot_inverse_of_compute_child_indices():
    # compute_child_indices: child = parent*cpp + slot
    parent_global, cpp = 7, 4
    for slot in range(cpp):
        child_global = parent_global * cpp + slot
        assert child_slot(child_global, parent_global, cpp) == slot


def test_level0_to_mag_px():
    assert level0_to_mag_px((448, 896), 20, 40) == (224.0, 448.0)


def test_patch_rect_at_mag():
    x, y, w, h = patch_rect_at_mag((448, 0), 20, 40, 224)
    assert (x, y, w, h) == (224.0, 0.0, 224, 224)


def test_child_cell_in_view():
    # zoom_px=896, r=2 -> cell=448; slot 3 = (row,col)=(1,1)
    assert child_cell_in_view(0, 2, 896) == (0, 0, 448, 448)
    assert child_cell_in_view(3, 2, 896) == (448, 448, 448, 448)
    # r=4, zoom_px=896 -> cell=224; slot 4 = (row,col)=(0,2) -> x=448,y=0
    assert child_cell_in_view(4, 4, 896) == (448, 0, 224, 224)
