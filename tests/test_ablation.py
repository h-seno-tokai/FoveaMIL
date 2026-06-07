"""アブレーション集計（手法タグ付け・ベースライン比 Δ 表）のユニット"""

import json
import math
import os

import pytest
import yaml

import numpy as np
import pandas as pd

from foveamil.evaluation.ablation import (
    BASELINE_LABEL,
    GROUP_F1_METRIC,
    collect_ablation,
    collect_ablation_rows,
    compare_to_baseline,
    format_markdown,
    format_markdown_compare,
    format_markdown_pooled,
    pooled_group_f1_compare,
    tag_combo,
)
from foveamil.evaluation.group_metrics import pooled_group_f1
from foveamil.evaluation.stats import adjust_pvalues, nadeau_bengio_corrected_t


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
    assert BASELINE_LABEL == "FoveaMIL(no-A/B/C/D)"
    assert tag_combo({**base, "decorrelation_weight": 0.1})[1] == "FoveaMIL+A"
    assert tag_combo({**base, "aux_norm": "entmax"})[1] == "FoveaMIL+B"
    assert tag_combo({**base, "selector": "dpp"})[1] == "FoveaMIL+D"
    assert (
        tag_combo({**base, "decorrelation_weight": 0.1, "aux_norm": "entmax", "selector": "dpp"})[1]
        == "FoveaMIL+ABD"
    )
    assert tag_combo({**base, "zoom_driver": "mcts"})[1] == "FoveaMIL+MCTS(C)"
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
    assert "FoveaMIL(no-A/B/C/D)" in table
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


def _write_combo_per_fold(root, name, config, per_fold):
    """per_fold（fold ごとの指標辞書）を持つ combo を書く"""
    combo_dir = os.path.join(root, name)
    os.makedirs(combo_dir)
    with open(os.path.join(combo_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    summary = {"test": {"per_fold": per_fold, "aggregate": {}}}
    with open(os.path.join(combo_dir, "cv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)


def test_compare_to_baseline_delta_p_and_adjusted(tmp_path):
    root = str(tmp_path / "abd")
    os.makedirs(root)
    mags = [10.0, 20.0, 40.0]
    base_pf = [{"weighted_f1": v} for v in (0.80, 0.81, 0.79, 0.80, 0.80)]
    d_pf = [{"weighted_f1": v} for v in (0.83, 0.84, 0.82, 0.83, 0.83)]
    b_pf = [{"weighted_f1": v} for v in (0.81, 0.82, 0.80, 0.81, 0.81)]
    _write_combo_per_fold(root, "combo_000__m3",
                          {"magnifications": mags, "zoom_driver": "differentiable"}, base_pf)
    _write_combo_per_fold(root, "combo_001__m3",
                          {"magnifications": mags, "zoom_driver": "differentiable",
                           "selector": "dpp"}, d_pf)
    _write_combo_per_fold(root, "combo_002__m3",
                          {"magnifications": mags, "zoom_driver": "differentiable",
                           "aux_norm": "entmax"}, b_pf)

    rows = collect_ablation_rows([root], "weighted_f1", "test")
    assert len(rows) == 3
    n_train, n_test = 900, 100
    enriched = compare_to_baseline(rows, n_train, n_test)

    by_label = {r["label"]: r for r in enriched}
    # baseline 自身は Δ/p なし
    assert by_label[BASELINE_LABEL]["delta"] is None
    assert math.isnan(by_label[BASELINE_LABEL]["pvalue"])

    d_row = by_label["FoveaMIL+D"]
    base = [m["weighted_f1"] for m in base_pf]
    d = [m["weighted_f1"] for m in d_pf]
    diffs = [d[i] - base[i] for i in range(5)]
    assert d_row["delta"] == pytest.approx(sum(diffs) / 5)
    expect_p = nadeau_bengio_corrected_t(diffs, n_train, n_test)["pvalue"]
    assert d_row["pvalue"] == pytest.approx(expect_p)

    # 補正後 p は同レジームの 2 method（D・B）に Holm をかけた結果と一致
    raw = [by_label["FoveaMIL+D"]["pvalue"], by_label["FoveaMIL+B"]["pvalue"]]
    adj = adjust_pvalues(raw, method="holm")["adjusted"]
    assert by_label["FoveaMIL+D"]["pvalue_adj"] == pytest.approx(adj[0])
    assert by_label["FoveaMIL+B"]["pvalue_adj"] == pytest.approx(adj[1])

    table = format_markdown_compare(enriched, "weighted_f1", "test")
    assert "p (NB)" in table and "p (holm)" in table


def test_group_f1_metric_in_collect_and_compare(tmp_path):
    root = str(tmp_path / "g")
    os.makedirs(root)
    mags = [10.0, 20.0]
    # class 0,1 の F1 を fold ごとに持つ
    base_pf = [
        {"class_0_f1": 0.6, "class_1_f1": 0.4},
        {"class_0_f1": 0.7, "class_1_f1": 0.5},
    ]
    d_pf = [
        {"class_0_f1": 0.8, "class_1_f1": 0.6},
        {"class_0_f1": 0.9, "class_1_f1": 0.7},
    ]
    _write_combo_per_fold(root, "combo_000__m2",
                          {"magnifications": mags, "zoom_driver": "differentiable"}, base_pf)
    _write_combo_per_fold(root, "combo_001__m2",
                          {"magnifications": mags, "zoom_driver": "differentiable",
                           "selector": "dpp"}, d_pf)

    rows = collect_ablation_rows([root], GROUP_F1_METRIC, "test", group_classes=[0, 1])
    by_label = {r["label"]: r for r in rows}
    # group-F1 = 各 fold の class0,1 F1 平均
    assert by_label[BASELINE_LABEL]["per_fold"] == pytest.approx([0.5, 0.6])
    assert by_label["FoveaMIL+D"]["per_fold"] == pytest.approx([0.7, 0.8])

    enriched = compare_to_baseline(rows, 900, 100)
    d_row = {r["label"]: r for r in enriched}["FoveaMIL+D"]
    # Δ = (0.7-0.5 + 0.8-0.6)/2 = 0.2
    assert d_row["delta"] == pytest.approx(0.2)


def test_collect_ablation_rows_skips_missing_metric(tmp_path):
    root = str(tmp_path / "x")
    os.makedirs(root)
    _write_combo_per_fold(root, "combo_000__m2",
                          {"magnifications": [10.0, 20.0], "zoom_driver": "differentiable"},
                          [{"macro_auc": 0.9}])
    assert collect_ablation_rows([root], "weighted_f1", "test") == []


def _write_combo_with_preds(root, name, config, folds, split="test"):
    """per_fold の per-class F1 と予測 CSV を持つ combo を書く

    ``folds`` は ``[(slide_ids, y_true, y_pred), ...]`` の fold 列
    """
    combo_dir = os.path.join(root, name)
    os.makedirs(combo_dir)
    with open(os.path.join(combo_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    per_fold = []
    for fi, (slide_ids, y_true, y_pred) in enumerate(folds):
        fdir = os.path.join(combo_dir, f"fold{fi}")
        os.makedirs(fdir)
        df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_0": [0.4] * len(y_true),
                "prob_1": [0.3] * len(y_true),
                "prob_2": [0.3] * len(y_true),
            }
        )
        df.to_csv(os.path.join(fdir, f"predictions_{split}.csv"), index=False)
        per_fold.append({"class_0_f1": 0.5, "class_1_f1": 0.5, "class_2_f1": 0.5})
    summary = {split: {"per_fold": per_fold, "aggregate": {}}}
    with open(os.path.join(combo_dir, "cv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)


def test_pooled_group_f1_compare_delta_perm_ci(tmp_path):
    root = str(tmp_path / "p")
    os.makedirs(root)
    mags = [10.0, 20.0]
    # 同一スライド集合を 2 fold に分割（slide_id で baseline と対応付く）
    base_folds = [
        (["a", "b", "c"], [0, 1, 2], [0, 0, 2]),  # class1 を誤り
        (["d", "e", "f"], [0, 1, 2], [1, 1, 2]),  # class0 を誤り
    ]
    meth_folds = [
        (["a", "b", "c"], [0, 1, 2], [0, 1, 2]),  # perfect
        (["d", "e", "f"], [0, 1, 2], [0, 1, 2]),
    ]
    _write_combo_with_preds(root, "combo_000__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable"},
                            base_folds)
    _write_combo_with_preds(root, "combo_001__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable",
                             "selector": "dpp"}, meth_folds)

    rows = collect_ablation_rows([root], GROUP_F1_METRIC, "test", group_classes=[0, 1])
    enriched = pooled_group_f1_compare(rows, [0, 1], n_perm=1000, n_boot=1000, seed=0)
    by_label = {r["label"]: r for r in enriched}

    # baseline 自身は pooled gF1 のみ Δ/p/CI なし
    base = by_label[BASELINE_LABEL]
    assert base["pooled_delta"] is None
    assert math.isnan(base["perm_pvalue"])

    # method の pooled gF1 と Δ が予測から正しく計算される
    yt = [0, 1, 2, 0, 1, 2]
    y_m = [0, 1, 2, 0, 1, 2]
    y_b = [0, 0, 2, 1, 1, 2]
    expect_m = pooled_group_f1(np.array(yt), np.array(y_m), [0, 1])
    expect_b = pooled_group_f1(np.array(yt), np.array(y_b), [0, 1])
    d = by_label["FoveaMIL+D"]
    assert d["pooled_gf1"] == pytest.approx(expect_m)
    assert d["pooled_delta"] == pytest.approx(expect_m - expect_b)
    assert 0.0 <= d["perm_pvalue"] <= 1.0
    assert d["boot_ci_low"] <= d["pooled_delta"] <= d["boot_ci_high"]

    table = format_markdown_pooled(enriched, "test")
    assert "pooled group-F1" in table and "p (perm)" in table


def test_pooled_compare_no_baseline_leaves_delta_none(tmp_path):
    root = str(tmp_path / "nb")
    os.makedirs(root)
    mags = [10.0, 20.0]
    # baseline 不在（D のみ）
    _write_combo_with_preds(root, "combo_000__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable",
                             "selector": "dpp"},
                            [(["a", "b"], [0, 1], [0, 1])])
    rows = collect_ablation_rows([root], GROUP_F1_METRIC, "test", group_classes=[0, 1])
    enriched = pooled_group_f1_compare(rows, [0, 1], n_perm=100, n_boot=100)
    d = {r["label"]: r for r in enriched}["FoveaMIL+D"]
    # baseline が無いので Δ/p/CI は付かないが pooled gF1 は出る
    assert d["pooled_delta"] is None
    assert math.isnan(d["perm_pvalue"])
    assert not math.isnan(d["pooled_gf1"])


def test_pooled_compare_deterministic(tmp_path):
    root = str(tmp_path / "det")
    os.makedirs(root)
    mags = [10.0, 20.0]
    base = [(["a", "b", "c", "d"], [0, 0, 1, 1], [0, 1, 1, 0])]
    meth = [(["a", "b", "c", "d"], [0, 0, 1, 1], [0, 0, 1, 1])]
    _write_combo_with_preds(root, "combo_000__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable"}, base)
    _write_combo_with_preds(root, "combo_001__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable",
                             "selector": "dpp"}, meth)
    rows = collect_ablation_rows([root], GROUP_F1_METRIC, "test", group_classes=[0, 1])
    a = pooled_group_f1_compare(rows, [0, 1], n_perm=500, n_boot=500, seed=42)
    b = pooled_group_f1_compare(rows, [0, 1], n_perm=500, n_boot=500, seed=42)
    ad = {r["label"]: r for r in a}["FoveaMIL+D"]
    bd = {r["label"]: r for r in b}["FoveaMIL+D"]
    assert ad["perm_pvalue"] == bd["perm_pvalue"]
    assert ad["boot_ci_low"] == bd["boot_ci_low"]
    assert ad["boot_ci_high"] == bd["boot_ci_high"]


def test_format_markdown_pooled_empty():
    assert "no combos found" in format_markdown_pooled([], "test")


def test_cli_pooled_writes_table(tmp_path):
    from foveamil.evaluation.ablation_cli import main as ablation_main

    root = str(tmp_path / "cli")
    os.makedirs(root)
    mags = [10.0, 20.0]
    base = [(["a", "b", "c"], [0, 1, 2], [0, 0, 2])]
    meth = [(["a", "b", "c"], [0, 1, 2], [0, 1, 2])]
    _write_combo_with_preds(root, "combo_000__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable"}, base)
    _write_combo_with_preds(root, "combo_001__m2",
                            {"magnifications": mags, "zoom_driver": "differentiable",
                             "selector": "dpp"}, meth)
    out_md = str(tmp_path / "table.md")
    rc = ablation_main([
        "--in", root, "--pooled", "--baseline", BASELINE_LABEL,
        "--group-classes", "0", "1", "--n-perm", "300", "--n-boot", "300",
        "--out", out_md,
    ])
    assert rc == 0
    text = open(out_md, encoding="utf-8").read()
    assert "pooled group-F1" in text and "FoveaMIL+D" in text


def test_cli_pooled_requires_baseline_and_group_classes(tmp_path):
    from foveamil.evaluation.ablation_cli import main as ablation_main

    root = str(tmp_path / "cli2")
    os.makedirs(root)
    with pytest.raises(SystemExit):
        ablation_main(["--in", root, "--pooled", "--group-classes", "0"])
    with pytest.raises(SystemExit):
        ablation_main(["--in", root, "--pooled", "--baseline", BASELINE_LABEL])
