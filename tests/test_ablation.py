"""アブレーション集計（手法タグ付け・ベースライン比 Δ 表）のユニット"""

import json
import os

import yaml

from foveamil.evaluation.ablation import (
    BASELINE_LABEL,
    collect_ablation,
    format_markdown,
    tag_combo,
)


def test_tag_combo_single_magnification_abmil_clam():
    assert tag_combo({"magnifications": [40.0], "instance_loss": False}) == (
        "single-40x",
        "ABMIL",
    )
    assert tag_combo({"magnifications": [20.0], "instance_loss": True}) == (
        "single-20x",
        "CLAM",
    )


def test_tag_combo_multi_magnification_methods():
    base = {"magnifications": [10.0, 20.0, 40.0], "zoom_driver": "differentiable"}
    assert tag_combo(base)[1] == BASELINE_LABEL
    assert tag_combo({**base, "decorrelation_weight": 0.1})[1] == "ZoomMIL+A"
    assert tag_combo({**base, "aux_norm": "entmax"})[1] == "ZoomMIL+B"
    assert tag_combo({**base, "selector": "dpp"})[1] == "ZoomMIL+D"
    assert (
        tag_combo({**base, "decorrelation_weight": 0.1, "aux_norm": "entmax", "selector": "dpp"})[1]
        == "ZoomMIL+ABD"
    )
    assert tag_combo({**base, "zoom_driver": "mcts"})[1] == "ZoomMIL+MCTS(C)"
    # 倍率レジームは同じ多倍率セットで一致する
    assert tag_combo(base)[0] == tag_combo({**base, "selector": "dpp"})[0]


def _write_combo(root, name, config, weighted_f1_mean):
    combo_dir = os.path.join(root, name)
    os.makedirs(combo_dir)
    with open(os.path.join(combo_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    summary = {
        "test": {
            "aggregate": {
                "weighted_f1": {
                    "mean": weighted_f1_mean,
                    "std": 0.02,
                    "n": 10,
                    "ci_t_low": weighted_f1_mean - 0.01,
                    "ci_t_high": weighted_f1_mean + 0.01,
                }
            }
        }
    }
    with open(os.path.join(combo_dir, "cv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)


def test_collect_and_format_delta_vs_baseline(tmp_path):
    root = str(tmp_path / "abd")
    os.makedirs(root)
    mags = [10.0, 20.0, 40.0]
    _write_combo(root, "combo_000__m3", {"magnifications": mags, "zoom_driver": "differentiable"}, 0.80)
    _write_combo(root, "combo_001__m3", {"magnifications": mags, "zoom_driver": "differentiable", "selector": "dpp"}, 0.83)
    _write_combo(root, "combo_002__m1", {"magnifications": [40.0], "instance_loss": False}, 0.75)

    rows = collect_ablation([root], "weighted_f1", "test")
    assert len(rows) == 3
    table = format_markdown(rows, "weighted_f1", "test")
    # baseline と +D の差分 Δ が出る（0.83 - 0.80 = +0.0300）
    assert "+0.0300" in table
    # baseline 自身の Δ は空欄
    assert "ZoomMIL(baseline)" in table
    # 単一倍率レジームも別セクションで出る
    assert "single-40x" in table and "ABMIL" in table


def test_collect_skips_combos_without_metric(tmp_path):
    root = str(tmp_path / "x")
    os.makedirs(root)
    combo_dir = os.path.join(root, "combo_000__m3")
    os.makedirs(combo_dir)
    with open(os.path.join(combo_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"magnifications": [10.0, 20.0], "zoom_driver": "differentiable"}, fh)
    with open(os.path.join(combo_dir, "cv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"test": {"aggregate": {"macro_auc": {"mean": 0.9, "std": 0.0}}}}, fh)
    # weighted_f1 が無い combo は飛ばす
    assert collect_ablation([root], "weighted_f1", "test") == []


def test_format_empty_rows():
    table = format_markdown([], "weighted_f1", "test")
    assert "no combos found" in table
