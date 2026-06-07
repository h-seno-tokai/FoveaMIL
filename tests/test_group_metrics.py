"""group-F1（指定クラス集合の非加重平均 F1）のユニット"""

import math

import pytest

from foveamil.evaluation.group_metrics import (
    class_f1_key,
    group_f1_from_fold,
    group_f1_per_fold,
    group_f1_summary,
)


def _fold(class_f1):
    return {class_f1_key(i): v for i, v in class_f1.items()}


def test_group_f1_from_fold_is_unweighted_mean():
    fold = _fold({0: 0.8, 1: 0.6, 2: 0.4})
    assert group_f1_from_fold(fold, [0, 1]) == pytest.approx(0.7)
    assert group_f1_from_fold(fold, [0, 2]) == pytest.approx(0.6)
    assert group_f1_from_fold(fold, [2]) == pytest.approx(0.4)


def test_group_f1_empty_set_is_nan():
    fold = _fold({0: 0.8, 1: 0.6})
    assert math.isnan(group_f1_from_fold(fold, []))


def test_group_f1_missing_classes_skipped_not_raised():
    fold = _fold({0: 0.8})  # class 1,2 欠損
    # 存在する 0 のみで平均
    assert group_f1_from_fold(fold, [0, 1, 2]) == pytest.approx(0.8)
    # 全クラス欠損は nan
    assert math.isnan(group_f1_from_fold(fold, [5, 6]))


def test_group_f1_per_fold_order():
    per_fold = [
        _fold({0: 0.8, 1: 0.6}),
        _fold({0: 0.6, 1: 0.4}),
    ]
    assert group_f1_per_fold(per_fold, [0, 1]) == pytest.approx([0.7, 0.5])


def test_group_f1_summary_aggregates_and_lists_per_class():
    per_fold = [
        _fold({0: 0.8, 1: 0.6, 2: 0.2}),
        _fold({0: 0.6, 1: 0.4, 2: 0.4}),
    ]
    out = group_f1_summary(per_fold, [0, 1])
    assert out["per_fold"] == pytest.approx([0.7, 0.5])
    assert out["mean"] == pytest.approx(0.6)
    assert out["n"] == 2
    # 集合内 per-class の fold 平均
    assert out["per_class"][0] == pytest.approx(0.7)
    assert out["per_class"][1] == pytest.approx(0.5)
    assert out["class_indices"] == [0, 1]


def test_group_f1_summary_partial_missing_excludes_nan_folds():
    per_fold = [
        _fold({0: 0.8}),  # class 1 欠損 → group-F1 = 0.8
        {},  # 全欠損 → nan fold
    ]
    out = group_f1_summary(per_fold, [0, 1])
    assert math.isnan(out["per_fold"][1])
    assert out["mean"] == pytest.approx(0.8)
    assert out["n"] == 1
    assert math.isnan(out["per_class"][1])
