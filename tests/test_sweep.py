"""sweep 展開部・val 選定のユニット"""

import json
import os

import pandas as pd
import pytest

from foveamil.training.resolve import ResolvedPaths
from foveamil.training.sweep import (
    SWEEP_DETAILED_CSV,
    Combo,
    SweepRunner,
    expand_combos,
    varying_axis_keys,
)

ENCODERS_4 = ["ResNet50", "UNI2-h", "Virchow2", "Virchow2-mini-dinov2"]
FEATURE_TYPES_3 = ["mean", "cls", "concat"]


def _resolved():
    return ResolvedPaths(
        n_cls=3,
        folds=10,
        labels_csv="/cohort/labels/labels_3class.csv",
        splits_dir="/cohort/splits/3class/cv10",
        feature_root_base="/feat",
    )


def _base_sweep(**overrides):
    sweep = {
        "encoder": ENCODERS_4,
        "feature_type": FEATURE_TYPES_3,
        "magnifications": [[1.25, 2.5]],
    }
    sweep.update(overrides)
    return sweep


def test_constrained_join_yields_ten_combos():
    combos = expand_combos(_base_sweep(), {}, _resolved())
    assert len(combos) == 10  # 3*3 + 1*1
    pairs = {(c.config["encoder"], c.config["feature_type"]) for c in combos}
    assert ("ResNet50", "mean") in pairs
    assert ("ResNet50", "cls") not in pairs
    assert ("ResNet50", "concat") not in pairs
    for ft in FEATURE_TYPES_3:
        assert ("Virchow2", ft) in pairs


def test_resolved_in_feat_dim_per_combo():
    combos = expand_combos(_base_sweep(), {}, _resolved())
    dim = {(c.config["encoder"], c.config["feature_type"]): c.config["in_feat_dim"]
           for c in combos}
    assert dim[("ResNet50", "mean")] == 1024
    assert dim[("UNI2-h", "mean")] == 1536
    assert dim[("Virchow2", "concat")] == 2560
    assert dim[("Virchow2-mini-dinov2", "cls")] == 384


def test_resolved_paths_carried_into_config():
    combos = expand_combos(_base_sweep(), {}, _resolved())
    for c in combos:
        assert c.config["feature_root"] == "/feat"
        assert c.config["labels_csv"].endswith("labels_3class.csv")
        assert c.config["n_cls"] == 3
        assert c.config["magnifications"] == [1.25, 2.5]


def test_product_with_other_axes():
    combos = expand_combos(_base_sweep(lr=[1e-4, 1e-3]), {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 lr
    assert {c.config["lr"] for c in combos} == {1e-4, 1e-3}


def test_magnification_sets_are_an_axis():
    sweep = _base_sweep(magnifications=[[1.25, 2.5], [1.25, 2.5, 5.0]])
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 mag sets
    mag_sets = {tuple(c.config["magnifications"]) for c in combos}
    assert mag_sets == {(1.25, 2.5), (1.25, 2.5, 5.0)}


def test_magnification_suffix_normalized():
    sweep = _base_sweep(magnifications=["1.25x", "2.5x"])
    combos = expand_combos(sweep, {}, _resolved())
    assert combos[0].config["magnifications"] == [1.25, 2.5]


def test_auto_resolved_key_in_sweep_rejected():
    with pytest.raises(ValueError, match="in_feat_dim"):
        expand_combos(_base_sweep(in_feat_dim=[1280]), {}, _resolved())


def test_auto_resolved_key_in_fixed_rejected():
    with pytest.raises(ValueError, match="feature_root"):
        expand_combos(_base_sweep(), {"feature_root": "/x"}, _resolved())


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        expand_combos(_base_sweep(), {"not_a_field": 1}, _resolved())


def test_missing_required_axis_rejected():
    sweep = {"encoder": ENCODERS_4, "feature_type": FEATURE_TYPES_3}
    with pytest.raises(ValueError, match="magnifications"):
        expand_combos(sweep, {}, _resolved())


def test_combo_names_unique_and_path_safe():
    combos = expand_combos(_base_sweep(lr=[1e-4, 1e-3]), {}, _resolved())
    names = [c.name for c in combos]
    assert len(set(names)) == len(names)
    for name in names:
        assert "/" not in name and " " not in name


def test_varying_axis_keys():
    combos = expand_combos(_base_sweep(lr=[1e-4, 1e-3]), {}, _resolved())
    keys = varying_axis_keys(combos)
    assert "encoder" in keys
    assert "feature_type" in keys
    assert "lr" in keys
    assert "fusion" not in keys  # not present at all


def _write_fold(combo_dir, fold, val_auc, test_auc):
    fold_dir = os.path.join(combo_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, "test_metrics.json"), "w") as h:
        json.dump({"macro_auc": test_auc, "accuracy": test_auc, "fold": f"fold{fold}"}, h)
    with open(os.path.join(fold_dir, "metrics_val.json"), "w") as h:
        json.dump({"macro_auc": val_auc, "accuracy": val_auc, "fold": f"fold{fold}"}, h)


def test_val_selection_and_test_oracle(tmp_path):
    # combo A: val 高 / test 低，combo B: val 低 / test 高
    combos = [
        Combo(index=0, name="combo_000__A", config={}, axis_values={"lr": 1e-4}),
        Combo(index=1, name="combo_001__B", config={}, axis_values={"lr": 1e-3}),
    ]
    out = str(tmp_path)
    for fold in (1, 2):
        _write_fold(os.path.join(out, "combo_000__A"), fold, val_auc=0.95, test_auc=0.80)
        _write_fold(os.path.join(out, "combo_001__B"), fold, val_auc=0.85, test_auc=0.90)

    runner = SweepRunner(
        combos=combos,
        split_files=["split_fold1.csv", "split_fold2.csv"],
        out_root=out,
        weights_root=out,
    )
    results = [runner._collect_combo(c, []) for c in combos]
    summary = runner._summarize(results)
    runner._write_detailed_csv(results)

    # val 選定は A（val 0.95）, test oracle は B（test 0.90）
    assert summary["best_by_val"]["name"] == "combo_000__A"
    assert summary["oracle_by_test"]["name"] == "combo_001__B"
    assert summary["selection_split"] == "val"
    # best の test は報告値（A の test 0.80）
    assert summary["best_by_val"]["test"]["macro_auc"]["mean"] == 0.80
    # cv_summary が val/test 両方を持つ
    cv = json.load(open(os.path.join(out, "combo_000__A", "cv_summary.json")))
    assert "val" in cv and "test" in cv
    # detailed CSV は combo×fold×split = 2×2×2 = 8 行
    df = pd.read_csv(os.path.join(out, SWEEP_DETAILED_CSV))
    assert len(df) == 8
    assert set(df["split"]) == {"val", "test"}
