"""hierarchy の倍率階層・子 index 計算のユニット"""

import numpy as np
import pytest

from foveamil.training.hierarchy import (
    children_per_parent,
    compute_child_indices,
    validate_magnification_hierarchy,
)


@pytest.mark.parametrize(
    "parent,child,expected",
    [
        (1.25, 2.5, 4),     # 2x  -> 4
        (2.5, 10.0, 16),    # 4x  -> 16
        (5.0, 40.0, 64),    # 8x  -> 64
        (1.25, 20.0, 256),  # 16x -> 256
    ],
)
def test_children_per_parent_power_of_two(parent, child, expected):
    assert children_per_parent(parent, child) == expected


@pytest.mark.parametrize("parent,child", [(1.25, 3.75), (2.5, 2.5), (5.0, 2.5), (1.25, 5.1)])
def test_children_per_parent_rejects_non_power_of_two(parent, child):
    with pytest.raises(ValueError):
        children_per_parent(parent, child)


def test_validate_hierarchy_allows_single_mag():
    validate_magnification_hierarchy([40.0])  # ズーム無し（ABMIL 相当）は許す


@pytest.mark.parametrize(
    "mags",
    [
        [1.25, 2.5, 5.0, 10.0],   # 連続 2x
        [2.5, 10.0, 40.0],        # 4x 飛ばし
        [1.25, 2.5, 5.0, 40.0],   # 変則（末尾 8x）
    ],
)
def test_validate_hierarchy_allows_power_of_two_subsets(mags):
    validate_magnification_hierarchy(mags)


@pytest.mark.parametrize("mags", [[], [1.25, 3.75], [10.0, 5.0], [5.0, 5.0]])
def test_validate_hierarchy_rejects_invalid(mags):
    with pytest.raises(ValueError):
        validate_magnification_hierarchy(mags)


def test_compute_child_indices_default_is_2x():
    # 親 local [0, 2] を global とみなし，各 4 子が連続
    out = compute_child_indices(np.array([0, 2]))
    assert out.tolist() == [0, 1, 2, 3, 8, 9, 10, 11]


def test_compute_child_indices_4x_block_of_16():
    cpp = children_per_parent(2.5, 10.0)  # 16
    out = compute_child_indices(np.array([0, 1]), children=cpp)
    assert out.tolist() == list(range(0, 16)) + list(range(16, 32))


def test_compute_child_indices_local_to_global():
    # local [0,1] を global [5, 7] へ写し，2x の 4 子ブロック
    out = compute_child_indices(
        np.array([0, 1]), parent_global_indices=np.array([5, 7]), children=4
    )
    assert out.tolist() == [20, 21, 22, 23, 28, 29, 30, 31]


def test_compute_child_indices_contiguous_blocks_per_parent():
    cpp = children_per_parent(5.0, 40.0)  # 8x -> 64
    out = compute_child_indices(np.array([3]), children=cpp)
    assert out.tolist() == list(range(3 * 64, 3 * 64 + 64))
