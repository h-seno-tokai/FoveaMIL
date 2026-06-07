"""group-F1（指定クラス集合の非加重平均 F1）のユニット"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from foveamil.evaluation.group_metrics import (
    class_f1_key,
    group_f1_from_fold,
    group_f1_per_fold,
    group_f1_summary,
    pool_combo_predictions,
    pooled_group_f1,
    pooled_group_f1_from_predictions,
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


def test_pooled_group_f1_matches_sklearn_subset_mean():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    # 集合 {0,1} の per-class F1 の非加重平均と一致
    per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    assert pooled_group_f1(y_true, y_pred, [0, 1]) == pytest.approx(
        float(np.mean(per_class))
    )
    # 完全一致なら 1.0
    assert pooled_group_f1(y_true, y_true, [0, 1, 2]) == pytest.approx(1.0)


def test_pooled_group_f1_not_equal_to_fold_mean():
    # プールは fold ごとの F1 平均とは一般に異なる（クラス不均衡で顕著）
    # fold0: class1 が test に無く F1=0／fold1: class1 のみ
    y0_t, y0_p = np.array([0, 0]), np.array([0, 1])
    y1_t, y1_p = np.array([1, 1]), np.array([1, 0])
    pooled = pooled_group_f1(
        np.concatenate([y0_t, y1_t]), np.concatenate([y0_p, y1_p]), [0, 1]
    )
    fold_mean = np.mean(
        [pooled_group_f1(y0_t, y0_p, [0, 1]), pooled_group_f1(y1_t, y1_p, [0, 1])]
    )
    assert not math.isclose(pooled, fold_mean)


def test_pooled_group_f1_empty_set_or_empty_sample_is_nan():
    y = np.array([0, 1, 2])
    assert math.isnan(pooled_group_f1(y, y, []))
    assert math.isnan(pooled_group_f1(np.array([]), np.array([]), [0, 1]))


def _write_fold_csv(combo_dir, fold_idx, slide_ids, y_true, y_pred, split="test"):
    fdir = os.path.join(combo_dir, f"fold{fold_idx}")
    os.makedirs(fdir)
    df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_0": [0.5] * len(y_true),
            "prob_1": [0.5] * len(y_true),
        }
    )
    df.to_csv(os.path.join(fdir, f"predictions_{split}.csv"), index=False)


def test_pool_combo_predictions_concats_all_folds(tmp_path):
    combo = str(tmp_path / "combo_000")
    os.makedirs(combo)
    _write_fold_csv(combo, 0, ["a", "b"], [0, 1], [0, 1])
    _write_fold_csv(combo, 1, ["c", "d"], [0, 1], [1, 1])
    df = pool_combo_predictions([combo], "test")
    assert len(df) == 4
    assert set(df["slide_id"]) == {"a", "b", "c", "d"}
    assert pooled_group_f1_from_predictions(df, [0, 1]) == pytest.approx(
        pooled_group_f1(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), [0, 1])
    )


def test_pool_combo_predictions_spans_multiple_combos(tmp_path):
    # 複数 out_root / seed の同一手法を 1 つに束ねる
    c1 = str(tmp_path / "r1" / "combo_000")
    c2 = str(tmp_path / "r2" / "combo_000")
    os.makedirs(c1)
    os.makedirs(c2)
    _write_fold_csv(c1, 0, ["a"], [0], [0])
    _write_fold_csv(c2, 0, ["b"], [1], [1])
    df = pool_combo_predictions([c1, c2], "test")
    assert len(df) == 2


def test_pool_combo_predictions_missing_returns_none(tmp_path):
    assert pool_combo_predictions([str(tmp_path / "nope")], "test") is None


def test_pooled_group_f1_from_predictions_missing_cols_is_nan():
    df = pd.DataFrame({"slide_id": ["a"], "foo": [1]})
    assert math.isnan(pooled_group_f1_from_predictions(df, [0, 1]))
    assert math.isnan(pooled_group_f1_from_predictions(None, [0, 1]))
