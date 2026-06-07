"""学習曲線・指標要約図（per-epoch 集約・best epoch・mean±CI 棒・per-class F1）のユニット"""

import math
import os

import numpy as np
import pandas as pd
import pytest

from foveamil.evaluation.curves import (
    BAND_MINMAX,
    BAND_STD,
    epoch_curve,
    per_class_f1_bars,
    plot_curves,
    plot_per_class_f1,
    plot_summary_bars,
    summary_bars,
)
from foveamil.evaluation.curves_cli import main as curves_main


def _history(epochs, val_loss, macro_f1):
    """合成 history DataFrame を作る（epoch・val_loss・val_macro_f1）"""
    return pd.DataFrame({
        "epoch": list(epochs),
        "train_loss": [1.0] * len(epochs),
        "val_loss": list(val_loss),
        "val_macro_f1": list(macro_f1),
    })


def _write_fold(combo_dir, fold, history):
    fold_dir = os.path.join(combo_dir, f"fold{fold}")
    os.makedirs(fold_dir)
    history.to_csv(os.path.join(fold_dir, "history.csv"), index=False)


def _make_combo(tmp_path, name="combo"):
    combo = str(tmp_path / name)
    os.makedirs(combo)
    return combo


def test_epoch_curve_minmax_mean_and_band(tmp_path):
    combo = _make_combo(tmp_path)
    _write_fold(combo, 1, _history([0, 1, 2], [1.0, 0.5, 0.8], [0.4, 0.6, 0.5]))
    _write_fold(combo, 2, _history([0, 1, 2], [1.0, 0.4, 0.6], [0.2, 0.8, 0.7]))
    out = epoch_curve(combo, "macro_f1", band=BAND_MINMAX)

    assert out["n_folds"] == 2
    assert out["epochs"] == [0.0, 1.0, 2.0]
    # epoch 0: mean(0.4,0.2)=0.3, low=0.2, high=0.4
    assert out["mean"][0] == pytest.approx(0.3)
    assert out["low"][0] == pytest.approx(0.2)
    assert out["high"][0] == pytest.approx(0.4)
    # best epoch = fold 平均 val_loss 最小 epoch 1（mean(0.5,0.4)=0.45）
    assert out["best_epoch"] == pytest.approx(1.0)
    # best_value = epoch1 の macro_f1 平均 mean(0.6,0.8)=0.7
    assert out["best_value"] == pytest.approx(0.7)


def test_epoch_curve_std_band(tmp_path):
    combo = _make_combo(tmp_path)
    _write_fold(combo, 1, _history([0, 1], [1.0, 0.5], [0.4, 0.6]))
    _write_fold(combo, 2, _history([0, 1], [1.0, 0.4], [0.2, 0.8]))
    out = epoch_curve(combo, "macro_f1", band=BAND_STD)
    # epoch 0: mean 0.3, std(population)=0.1 → low 0.2 high 0.4
    assert out["low"][0] == pytest.approx(0.2)
    assert out["high"][0] == pytest.approx(0.4)


def test_epoch_curve_val_prefix_optional(tmp_path):
    combo = _make_combo(tmp_path)
    _write_fold(combo, 1, _history([0, 1], [1.0, 0.5], [0.4, 0.6]))
    a = epoch_curve(combo, "macro_f1")
    b = epoch_curve(combo, "val_macro_f1")
    assert a["mean"] == b["mean"]


def test_epoch_curve_aligns_on_common_epochs(tmp_path):
    combo = _make_combo(tmp_path)
    # fold1 は epoch 0..2，fold2 は epoch 1..3 → 共通は 1,2
    _write_fold(combo, 1, _history([0, 1, 2], [1.0, 0.5, 0.8], [0.4, 0.6, 0.5]))
    _write_fold(combo, 2, _history([1, 2, 3], [0.4, 0.6, 0.9], [0.8, 0.7, 0.3]))
    out = epoch_curve(combo, "macro_f1", band=BAND_MINMAX)
    assert out["epochs"] == [1.0, 2.0]
    assert out["mean"][0] == pytest.approx(0.7)  # mean(0.6,0.8)


def test_epoch_curve_empty_combo_is_safe(tmp_path):
    combo = _make_combo(tmp_path)
    out = epoch_curve(combo, "macro_f1")
    assert out["n_folds"] == 0
    assert out["epochs"] == []
    assert math.isnan(out["best_epoch"])


def test_epoch_curve_missing_metric_is_empty(tmp_path):
    combo = _make_combo(tmp_path)
    _write_fold(combo, 1, _history([0, 1], [1.0, 0.5], [0.4, 0.6]))
    out = epoch_curve(combo, "macro_auc")
    assert out["n_folds"] == 0
    assert out["epochs"] == []


def test_epoch_curve_invalid_band_raises(tmp_path):
    combo = _make_combo(tmp_path)
    with pytest.raises(ValueError):
        epoch_curve(combo, "macro_f1", band="iqr")


def test_summary_bars_mean_ci_matches_aggregate():
    per_fold_a = [{"macro_f1": 0.5}, {"macro_f1": 0.6}, {"macro_f1": 0.7}]
    per_fold_b = [{"macro_f1": 0.3}, {"macro_f1": 0.4}]
    records = summary_bars(
        [("A", per_fold_a), ("B", per_fold_b)], "macro_f1"
    )
    assert records[0]["label"] == "A"
    assert records[0]["mean"] == pytest.approx(0.6)
    assert records[0]["n"] == 3
    assert records[0]["ci_low"] <= records[0]["mean"] <= records[0]["ci_high"]
    assert records[1]["mean"] == pytest.approx(0.35)


def test_summary_bars_missing_metric_is_nan():
    records = summary_bars([("A", [{"accuracy": 0.5}])], "macro_f1")
    assert math.isnan(records[0]["mean"])
    assert records[0]["n"] == 0


def test_per_class_f1_bars_subset_average():
    per_fold = [
        {"class_4_f1": 0.5, "class_5_f1": 0.7, "class_6_f1": 0.3},
        {"class_4_f1": 0.7, "class_5_f1": 0.9, "class_6_f1": 0.5},
    ]
    bars = per_class_f1_bars(per_fold, [4, 5, 6])
    # per-class 平均: c4=0.6, c5=0.8, c6=0.4
    assert bars["per_class"][4] == pytest.approx(0.6)
    assert bars["per_class"][5] == pytest.approx(0.8)
    # group-F1 = fold ごとの非加重平均の fold 平均
    # fold0 mean(0.5,0.7,0.3)=0.5, fold1 mean(0.7,0.9,0.5)=0.7 → 0.6
    assert bars["group_mean"] == pytest.approx(0.6)
    assert bars["n"] == 2


def test_per_class_f1_bars_missing_class_is_nan():
    bars = per_class_f1_bars([{"class_4_f1": 0.5}], [4, 99])
    assert bars["per_class"][4] == pytest.approx(0.5)
    assert math.isnan(bars["per_class"][99])


def test_plot_curves_writes_png(tmp_path):
    combo = _make_combo(tmp_path)
    _write_fold(combo, 1, _history([0, 1], [1.0, 0.5], [0.4, 0.6]))
    _write_fold(combo, 2, _history([0, 1], [1.0, 0.4], [0.2, 0.8]))
    curve = epoch_curve(combo, "macro_f1")
    out_png = str(tmp_path / "curves.png")
    assert plot_curves([("c", curve)], "macro_f1", out_png) is True
    assert os.path.exists(out_png) and os.path.getsize(out_png) > 0


def test_plot_curves_no_data_is_false(tmp_path):
    empty = epoch_curve(_make_combo(tmp_path), "macro_f1")
    out_png = str(tmp_path / "curves.png")
    assert plot_curves([("c", empty)], "macro_f1", out_png) is False
    assert not os.path.exists(out_png)


def test_plot_summary_and_per_class_write_png(tmp_path):
    records = summary_bars(
        [("A", [{"macro_f1": 0.5}, {"macro_f1": 0.7}])], "macro_f1"
    )
    summary_png = str(tmp_path / "summary.png")
    assert plot_summary_bars(records, "macro_f1", summary_png) is True
    assert os.path.exists(summary_png)

    bars = per_class_f1_bars([{"class_4_f1": 0.5, "class_5_f1": 0.6}], [4, 5])
    pc_png = str(tmp_path / "per_class.png")
    assert plot_per_class_f1(bars, pc_png, label="A") is True
    assert os.path.exists(pc_png)


def test_plot_summary_all_nan_is_false(tmp_path):
    records = summary_bars([("A", [{"accuracy": 0.5}])], "macro_f1")
    out_png = str(tmp_path / "summary.png")
    assert plot_summary_bars(records, "macro_f1", out_png) is False


def _make_sweep(tmp_path):
    """単一 combo（cv_summary + history）の sweep 出力もどきを作る"""
    root = str(tmp_path / "sweep")
    combo_dir = os.path.join(root, "combo_000")
    os.makedirs(combo_dir)
    _write_fold(combo_dir, 1, _history([0, 1, 2], [1.0, 0.5, 0.8], [0.4, 0.6, 0.5]))
    _write_fold(combo_dir, 2, _history([0, 1, 2], [1.0, 0.4, 0.6], [0.2, 0.8, 0.7]))
    import json

    cv_summary = {
        "test": {"per_fold": [
            {"macro_f1": 0.5, "class_4_f1": 0.4, "class_5_f1": 0.6},
            {"macro_f1": 0.7, "class_4_f1": 0.5, "class_5_f1": 0.8},
        ]},
        "val": {"per_fold": [{"macro_f1": 0.45}, {"macro_f1": 0.55}]},
    }
    with open(os.path.join(combo_dir, "cv_summary.json"), "w", encoding="utf-8") as h:
        json.dump(cv_summary, h)
    summary = {"combos": [{"name": "combo_000", "out_dir": combo_dir}]}
    with open(os.path.join(root, "sweep_summary.json"), "w", encoding="utf-8") as h:
        json.dump(summary, h)
    return root


def test_cli_end_to_end_writes_outputs(tmp_path):
    root = _make_sweep(tmp_path)
    out = str(tmp_path / "out")
    rc = curves_main([
        "--in", root, "--out", out, "--metric", "macro_f1",
        "--summary-metric", "macro_f1", "--split", "test", "--classes", "4,5",
    ])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "curves.json"))
    assert os.path.exists(os.path.join(out, "curves_macro_f1.png"))
    assert os.path.exists(os.path.join(out, "summary_macro_f1.png"))
    assert os.path.exists(os.path.join(out, "per_class_f1_combo_000.png"))


def test_cli_no_plots_skips_figures(tmp_path):
    root = _make_sweep(tmp_path)
    out = str(tmp_path / "out")
    rc = curves_main(["--in", root, "--out", out, "--no-plots"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "curves.json"))
    assert not os.path.exists(os.path.join(out, "curves_macro_f1.png"))


def test_cli_missing_combo_is_safe(tmp_path):
    root = _make_sweep(tmp_path)
    out = str(tmp_path / "out")
    rc = curves_main([
        "--in", root, "--out", out, "--combo", "nonexistent", "--no-plots",
    ])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "curves.json"))
