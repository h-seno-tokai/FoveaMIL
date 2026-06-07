"""学習安定性診断（振動 std・最小後上昇・best epoch・分散比 bootstrap）のユニット"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from foveamil.evaluation.stability import (
    best_epoch,
    combo_stability,
    fold_stability,
    per_fold_tail_std,
    post_min_rise,
    tail_std,
    variance_ratio_bootstrap,
)


def _history(val_loss, weighted_f1=None):
    """合成 history DataFrame を作る（epoch 昇順）"""
    n = len(val_loss)
    data = {
        "epoch": list(range(n)),
        "train_loss": [1.0] * n,
        "val_loss": list(val_loss),
    }
    if weighted_f1 is not None:
        data["val_weighted_f1"] = list(weighted_f1)
    return pd.DataFrame(data)


def test_tail_std_matches_last_n():
    f1 = [0.5, 0.6, 0.70, 0.72, 0.68, 0.74]
    h = _history([1.0] * 6, weighted_f1=f1)
    expected = float(np.std(np.array(f1[-3:]), ddof=1))
    assert tail_std(h, "weighted_f1", tail=3) == pytest.approx(expected)
    # val_ 接頭辞付きでも同じ列を引く
    assert tail_std(h, "val_weighted_f1", tail=3) == pytest.approx(expected)


def test_tail_std_missing_metric_is_nan():
    h = _history([1.0, 0.9, 0.8])
    assert math.isnan(tail_std(h, "macro_auc", tail=2))


def test_post_min_rise_after_minimum():
    # 最小 0.5 は index2 以後 0.7,0.9 → 上昇量 = 0.9-0.5 = 0.4
    h = _history([1.0, 0.8, 0.5, 0.7, 0.9])
    assert post_min_rise(h) == pytest.approx(0.4)


def test_post_min_rise_monotone_decrease_is_zero():
    h = _history([1.0, 0.8, 0.6, 0.4])
    assert post_min_rise(h) == pytest.approx(0.0)


def test_best_epoch_is_argmin_val_loss():
    h = _history([1.0, 0.8, 0.5, 0.7, 0.9])
    assert best_epoch(h) == pytest.approx(2.0)


def test_fold_stability_bundles_three_quantities():
    f1 = [0.5, 0.6, 0.7, 0.72, 0.68]
    h = _history([1.0, 0.8, 0.5, 0.7, 0.9], weighted_f1=f1)
    out = fold_stability(h, "weighted_f1", tail=3)
    assert out["best_epoch"] == pytest.approx(2.0)
    assert out["post_min_rise"] == pytest.approx(0.4)
    assert out["tail_std"] == pytest.approx(float(np.std(np.array(f1[-3:]), ddof=1)))


def _write_fold(combo_dir, fold, history):
    fold_dir = os.path.join(combo_dir, f"fold{fold}")
    os.makedirs(fold_dir)
    history.to_csv(os.path.join(fold_dir, "history.csv"), index=False)


def test_combo_stability_averages_over_folds(tmp_path):
    combo = str(tmp_path / "combo")
    os.makedirs(combo)
    _write_fold(combo, 1, _history([1.0, 0.5, 0.8], weighted_f1=[0.4, 0.6, 0.7]))
    _write_fold(combo, 2, _history([1.0, 0.4, 0.6], weighted_f1=[0.3, 0.7, 0.8]))
    out = combo_stability(combo, "weighted_f1", tail=2)
    assert out["n_folds"] == 2
    # best epoch: fold1=1, fold2=1 → 平均 1.0
    assert out["mean"]["best_epoch"] == pytest.approx(1.0)
    # post_min_rise: fold1=0.3, fold2=0.2 → 平均 0.25
    assert out["mean"]["post_min_rise"] == pytest.approx(0.25)


def test_per_fold_tail_std_collects_values(tmp_path):
    combo = str(tmp_path / "combo")
    os.makedirs(combo)
    _write_fold(combo, 1, _history([1.0, 0.9], weighted_f1=[0.5, 0.6]))
    _write_fold(combo, 2, _history([1.0, 0.9], weighted_f1=[0.4, 0.8]))
    values = per_fold_tail_std(combo, "weighted_f1", tail=2)
    assert len(values) == 2
    assert values[0] == pytest.approx(float(np.std(np.array([0.5, 0.6]), ddof=1)))


def test_variance_ratio_bootstrap_reproducible_and_correct():
    a = [0.10, 0.20, 0.30, 0.05, 0.25]
    b = [0.02, 0.03, 0.04, 0.05, 0.01]
    out1 = variance_ratio_bootstrap(a, b, n_boot=2000, seed=0)
    out2 = variance_ratio_bootstrap(a, b, n_boot=2000, seed=0)
    assert out1 == out2  # 決定的
    expected = float(np.var(a, ddof=1) / np.var(b, ddof=1))
    assert out1["ratio"] == pytest.approx(expected)
    assert out1["ci_low"] <= out1["ratio"] <= out1["ci_high"]


def test_variance_ratio_degenerate_is_nan():
    out = variance_ratio_bootstrap([0.1], [0.2, 0.3], n_boot=100, seed=0)
    assert math.isnan(out["ratio"])
    # var(b)==0 も縮退
    zero_b = variance_ratio_bootstrap([0.1, 0.2], [0.5, 0.5], n_boot=100, seed=0)
    assert math.isnan(zero_b["ratio"])
