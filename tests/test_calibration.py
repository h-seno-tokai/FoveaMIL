"""事後較正（temperature scaling・クラス別ロジット補正 δ_c）のユニット"""

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from foveamil.evaluation.calibration import (
    IDENTITY_TEMPERATURE,
    apply_calibration,
    calibrate_val_to_test,
    evaluate_predictions,
    extract_logits,
    fit_class_deltas,
    fit_temperature,
)
from foveamil.evaluation.calibration_cli import main as calibrate_main


def _df_from_logits(logits, y_true, with_logit_cols=True):
    """ロジットと正解から予測 DataFrame を作る prob は softmax で導出"""
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / exp.sum(axis=1, keepdims=True)
    data = {
        "slide_id": [f"s{i}" for i in range(len(y_true))],
        "y_true": list(y_true),
        "y_pred": prob.argmax(axis=1).tolist(),
    }
    for i in range(logits.shape[1]):
        data[f"prob_{i}"] = prob[:, i]
    if with_logit_cols:
        for i in range(logits.shape[1]):
            data[f"logit_{i}"] = logits[:, i]
    return pd.DataFrame(data)


# ---- temperature scaling ----


def test_fit_temperature_cools_overconfident_logits():
    # 大振幅の正解ロジット＝過信 T>1 で鋭さを下げるはず
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 3, size=n)
    logits = np.full((n, 3), -5.0)
    logits[np.arange(n), y] = 5.0  # ほぼ確実に正解だが確率は飽和し過信
    # 1 割をわざと誤らせて過信を作る
    flip = rng.choice(n, size=n // 10, replace=False)
    for i in flip:
        wrong = (y[i] + 1) % 3
        logits[i, y[i]] = -5.0
        logits[i, wrong] = 5.0
    t = fit_temperature(logits, y)
    assert t > 1.0


def test_fit_temperature_does_not_change_argmax():
    # 温度は argmax を変えない＝分類予測は不変
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(50, 4))
    y = logits.argmax(axis=1)
    t = fit_temperature(logits, y)
    _, pred = apply_calibration(logits, t)
    assert np.array_equal(pred, logits.argmax(axis=1))


def test_fit_temperature_is_deterministic():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(80, 5))
    y = rng.integers(0, 5, size=80)
    assert fit_temperature(logits, y) == fit_temperature(logits, y)


def test_fit_temperature_degenerate_single_class_is_identity():
    logits = np.random.default_rng(3).normal(size=(10, 3))
    y = np.zeros(10, dtype=int)  # 1 クラスのみ
    assert fit_temperature(logits, y) == IDENTITY_TEMPERATURE
    assert fit_temperature(np.zeros((0, 3)), np.array([])) == IDENTITY_TEMPERATURE


# ---- per-class delta ----


def test_fit_class_deltas_recovers_minority_with_macro_f1():
    # クラス 2 を恒常的に過小評価するバイアスを δ で補正できる
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, 3, size=n)
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 2.0
    # クラス 2 のロジットだけ常に押し下げる＝recall 低下
    logits[:, 2] -= 3.0
    base_pred = logits.argmax(axis=1)
    from sklearn.metrics import f1_score

    base_macro = f1_score(y, base_pred, labels=[0, 1, 2], average="macro")
    deltas = fit_class_deltas(logits, y, n_classes=3)
    cal_pred = (logits + deltas).argmax(axis=1)
    cal_macro = f1_score(y, cal_pred, labels=[0, 1, 2], average="macro")
    # クラス 2 に正の δ がつき macro-F1 が改善する
    assert deltas[2] > 0
    assert cal_macro > base_macro


def test_fit_class_deltas_l2_shrinks_toward_zero():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.integers(0, 3, size=n)
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 1.5
    logits[:, 2] -= 2.0
    weak = fit_class_deltas(logits, y, 3, l2=1e-4)
    strong = fit_class_deltas(logits, y, 3, l2=1.0)
    # 強い L2 ほど δ のノルムは小さい
    assert np.sum(strong ** 2) <= np.sum(weak ** 2)


def test_fit_class_deltas_is_deterministic():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 4, size=150)
    logits = rng.normal(size=(150, 4))
    a = fit_class_deltas(logits, y, 4)
    b = fit_class_deltas(logits, y, 4)
    assert np.array_equal(a, b)


def test_fit_class_deltas_degenerate_returns_zero():
    logits = np.random.default_rng(7).normal(size=(8, 3))
    y = np.zeros(8, dtype=int)
    assert np.array_equal(fit_class_deltas(logits, y, 3), np.zeros(3))


def test_fit_class_deltas_group_objective_targets_subset():
    # group-F1 目的だと対象クラス集合の recall を優先して補正する
    rng = np.random.default_rng(8)
    n = 300
    y = rng.integers(0, 3, size=n)
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 2.0
    logits[:, 1] -= 3.0  # クラス 1 を過小評価
    deltas = fit_class_deltas(logits, y, 3, group_classes=[1])
    assert deltas[1] > 0


# ---- apply + extract ----


def test_apply_calibration_matches_manual_softmax():
    logits = np.array([[1.0, 2.0, 0.0]])
    prob, pred = apply_calibration(logits, temperature=2.0, deltas=np.array([0.0, 0.0, 1.0]))
    scaled = logits / 2.0 + np.array([0.0, 0.0, 1.0])
    exp = np.exp(scaled - scaled.max())
    expected = exp / exp.sum()
    assert prob == pytest.approx(expected)
    assert pred[0] == int(expected.argmax())


def test_extract_logits_prefers_logit_cols_else_log_prob():
    logits = np.array([[2.0, -1.0, 0.5], [0.0, 1.0, -1.0]])
    df = _df_from_logits(logits, [0, 1], with_logit_cols=True)
    got = extract_logits(df)
    assert got == pytest.approx(logits)
    # logit 列が無ければ log(prob) で復元（softmax 差は定数だが argmax は一致）
    df_noco = _df_from_logits(logits, [0, 1], with_logit_cols=False)
    recovered = extract_logits(df_noco)
    assert recovered.argmax(axis=1).tolist() == logits.argmax(axis=1).tolist()


def test_extract_logits_none_when_no_cols():
    df = pd.DataFrame({"slide_id": ["a"], "y_true": [0], "y_pred": [0]})
    assert extract_logits(df) is None


# ---- evaluate before/after ----


def test_evaluate_predictions_macro_and_recall():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2])
    out = evaluate_predictions(y_true, y_pred, n_classes=3, group_classes=[2])
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["group_f1"] == pytest.approx(1.0)
    # クラス 2 は少数（support 1 < 中央値 2）
    assert 2 in out["minority_classes"]
    assert out["per_class_recall"][2] == pytest.approx(1.0)


def test_evaluate_predictions_absent_class_recall_nan():
    # test に出現しないクラスの recall は nan（0 でない）
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    out = evaluate_predictions(y_true, y_pred, n_classes=3)
    assert math.isnan(out["per_class_recall"][2])


def test_evaluate_top_confusions_reports_main_outflow():
    # クラス 2（少数）が主に 0 へ流出
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    out = evaluate_predictions(y_true, y_pred, n_classes=3, top_confusions=2)
    flows = out["top_confusions"][2]
    assert flows[0]["to"] == 0
    assert flows[0]["count"] == 2
    assert flows[0]["rate"] == pytest.approx(1.0)


# ---- end-to-end val->test ----


def test_calibrate_val_to_test_stages_and_marginal():
    rng = np.random.default_rng(9)
    n = 400
    y_val = rng.integers(0, 3, size=n)
    logits_val = np.zeros((n, 3))
    logits_val[np.arange(n), y_val] = 2.0
    logits_val[:, 2] -= 3.0  # クラス 2 を過小評価
    val_df = _df_from_logits(logits_val, y_val)

    y_test = rng.integers(0, 3, size=n)
    logits_test = np.zeros((n, 3))
    logits_test[np.arange(n), y_test] = 2.0
    logits_test[:, 2] -= 3.0
    test_df = _df_from_logits(logits_test, y_test)

    res = calibrate_val_to_test(val_df, test_df)
    assert set(res["stages"]) == {"baseline", "temperature", "temperature_delta"}
    # temperature 段は argmax を変えない＝baseline と macro-F1 一致
    assert res["stages"]["temperature"]["macro_f1"] == pytest.approx(
        res["stages"]["baseline"]["macro_f1"]
    )
    # δ 段で macro-F1 が改善（同じバイアスを test も持つため）
    assert (
        res["stages"]["temperature_delta"]["macro_f1"]
        >= res["stages"]["baseline"]["macro_f1"]
    )
    # 限界効用の分解が整合（T 寄与＋δ 寄与＝合計）
    mm = res["marginal"]["macro_f1"]
    assert mm["temperature"] + mm["delta"] == pytest.approx(mm["total"])
    assert res["temperature"] > 0
    assert res["logit_source"] == "logit"


def test_calibrate_val_to_test_is_deterministic():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 3, size=120)
    logits = rng.normal(size=(120, 3))
    val_df = _df_from_logits(logits, y)
    test_df = _df_from_logits(logits, y)
    a = calibrate_val_to_test(val_df, test_df)
    b = calibrate_val_to_test(val_df, test_df)
    assert a["temperature"] == b["temperature"]
    assert a["deltas"] == b["deltas"]


def test_calibrate_degenerate_no_logits_falls_back():
    # logit も prob も無い＝較正不能 baseline のみ返し例外なし
    val_df = pd.DataFrame({"slide_id": ["a"], "y_true": [0], "y_pred": [0]})
    test_df = pd.DataFrame({"slide_id": ["b"], "y_true": [0], "y_pred": [0]})
    res = calibrate_val_to_test(val_df, test_df)
    assert res["temperature"] == IDENTITY_TEMPERATURE
    assert res["stages"]["baseline"]["macro_f1"] is not None
    assert "temperature_delta" not in res["stages"]


def test_calibrate_single_class_val_keeps_identity():
    # val が 1 クラスのみ＝T も δ も恒等
    logits = np.random.default_rng(11).normal(size=(20, 3))
    val_df = _df_from_logits(logits, np.zeros(20, dtype=int))
    test_df = _df_from_logits(logits, np.random.default_rng(12).integers(0, 3, size=20))
    res = calibrate_val_to_test(val_df, test_df)
    assert res["temperature"] == IDENTITY_TEMPERATURE
    assert res["deltas"] == [0.0, 0.0, 0.0]


# ---- CLI end-to-end on a sweep-output fixture ----


def _write_fold(combo_dir, fold_name, val_df, test_df, classes):
    fold_dir = os.path.join(combo_dir, fold_name)
    os.makedirs(fold_dir)
    val_df.to_csv(os.path.join(fold_dir, "predictions_val.csv"), index=False)
    test_df.to_csv(os.path.join(fold_dir, "predictions_test.csv"), index=False)
    meta = {"data": {"classes": classes, "class_breakdown": {"train": {}, "test": {}}}}
    with open(os.path.join(fold_dir, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def test_calibrate_cli_writes_report(tmp_path):
    rng = np.random.default_rng(13)
    root = str(tmp_path / "out")
    combo_name = "combo_000__m3"
    combo_dir = os.path.join(root, combo_name)
    classes = ["A", "B", "C"]

    for f in range(2):
        n = 150
        y_val = rng.integers(0, 3, size=n)
        lv = np.zeros((n, 3))
        lv[np.arange(n), y_val] = 2.0
        lv[:, 2] -= 3.0
        y_test = rng.integers(0, 3, size=n)
        lt = np.zeros((n, 3))
        lt[np.arange(n), y_test] = 2.0
        lt[:, 2] -= 3.0
        _write_fold(
            combo_dir, f"fold{f}",
            _df_from_logits(lv, y_val), _df_from_logits(lt, y_test), classes,
        )

    summary = {
        "best_by_val": {"name": combo_name, "out_dir": combo_dir},
        "combos": [{"name": combo_name, "out_dir": combo_dir}],
    }
    with open(os.path.join(root, "sweep_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    rc = calibrate_main(["--in", root, "--split", "test"])
    assert rc == 0
    out_json = os.path.join(root, "calibration", "calibration.json")
    out_md = os.path.join(root, "calibration", "calibration.md")
    assert os.path.exists(out_json) and os.path.exists(out_md)
    with open(out_json, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    assert len(results) == 1
    assert results[0]["combo"] == combo_name
    assert "temperature_delta" in results[0]["stages"]


# ---- present-only F1 規約（不在クラスを誤減点しない）----


def test_macro_f1_excludes_absent_class():
    # クラス 2 が y_true 不在＝present-only なら誤減点せず macro-F1=1.0
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    out = evaluate_predictions(y_true, y_pred, n_classes=3)
    # 不在クラス 2 を F1=0 で算入すると 2/3 になるが present-only なので 1.0
    assert out["macro_f1"] == pytest.approx(1.0)


def test_macro_f1_matches_present_only_sklearn():
    # present クラスだけを labels に渡した sklearn macro-F1 と一致
    from sklearn.metrics import f1_score

    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    out = evaluate_predictions(y_true, y_pred, n_classes=5)
    present = sorted(set(y_true.tolist()))
    expected = f1_score(y_true, y_pred, labels=present, average="macro")
    assert out["macro_f1"] == pytest.approx(expected)


def test_group_f1_excludes_absent_class_in_subset():
    # group 集合 {1,2} のうち 2 は y_true 不在＝present の 1 のみで平均
    from sklearn.metrics import f1_score

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    out = evaluate_predictions(y_true, y_pred, n_classes=3, group_classes=[1, 2])
    expected = f1_score(y_true, y_pred, labels=[1], average="macro")
    assert out["group_f1"] == pytest.approx(expected)
    assert out["group_f1"] == pytest.approx(1.0)


def test_group_f1_all_absent_is_nan():
    # group 集合が全て y_true 不在なら nan（0 でない）
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    out = evaluate_predictions(y_true, y_pred, n_classes=4, group_classes=[2, 3])
    assert math.isnan(out["group_f1"])


def test_delta_objective_not_diluted_by_val_absent_class():
    # val に不在のクラス 2 が δ_c 目的を希釈しない＝present-only で当てられる
    rng = np.random.default_rng(20)
    n = 200
    y = rng.integers(0, 2, size=n)  # クラス 2 は val 不在
    logits = np.zeros((n, 3))
    logits[np.arange(n), y] = 2.0
    logits[:, 1] -= 3.0  # クラス 1 を過小評価
    deltas = fit_class_deltas(logits, y, n_classes=3)
    # present な少数クラス 1 を引き上げる δ がつく（不在クラス 2 の 0 算入に薄められない）
    assert deltas[1] > 0


# ---- temperature の効用は NLL/ECE で観測（F1 でなく）----


def test_temperature_stage_reports_nll_and_ece():
    # 段階指標に NLL・ECE が出る T 段は argmax 不変で F1 は baseline と一致
    rng = np.random.default_rng(21)
    n = 300
    y = rng.integers(0, 3, size=n)
    logits = np.full((n, 3), -4.0)
    logits[np.arange(n), y] = 6.0  # 過信
    flip = rng.choice(n, size=n // 8, replace=False)
    for i in flip:
        wrong = (y[i] + 1) % 3
        logits[i, y[i]] = -4.0
        logits[i, wrong] = 6.0
    df = _df_from_logits(logits, y)
    res = calibrate_val_to_test(df, df)
    base = res["stages"]["baseline"]
    temp = res["stages"]["temperature"]
    # NLL/ECE が数値で出る
    assert not math.isnan(base["nll"]) and not math.isnan(base["ece"])
    assert not math.isnan(temp["nll"]) and not math.isnan(temp["ece"])
    # T 段は分類指標を変えない（argmax 不変）
    assert temp["macro_f1"] == pytest.approx(base["macro_f1"])
    # 過信を冷ます＝T 段で NLL が改善（marginal の T 寄与が負）
    assert res["temperature"] > 1.0
    assert res["marginal"]["nll"]["temperature"] < 0
    # marginal に nll/ece が含まれる
    assert set(res["marginal"]) >= {"macro_f1", "nll", "ece"}


def test_evaluate_predictions_nll_ece_nan_without_prob():
    # prob 未指定なら NLL/ECE は nan（後方互換＝既存呼び出しは prob なし）
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    out = evaluate_predictions(y_true, y_pred, n_classes=3)
    assert math.isnan(out["nll"]) and math.isnan(out["ece"])


def test_calibrate_stages_nll_ece_deterministic():
    rng = np.random.default_rng(22)
    y = rng.integers(0, 3, size=120)
    logits = rng.normal(size=(120, 3))
    df = _df_from_logits(logits, y)
    a = calibrate_val_to_test(df, df)
    b = calibrate_val_to_test(df, df)
    assert a["stages"]["temperature"]["nll"] == b["stages"]["temperature"]["nll"]
    assert a["stages"]["temperature"]["ece"] == b["stages"]["temperature"]["ece"]


def test_calibrate_cli_group_classes(tmp_path):
    rng = np.random.default_rng(14)
    root = str(tmp_path / "out")
    combo_name = "combo_000__m2"
    combo_dir = os.path.join(root, combo_name)
    n = 200
    y_val = rng.integers(0, 3, size=n)
    lv = np.zeros((n, 3))
    lv[np.arange(n), y_val] = 2.0
    y_test = rng.integers(0, 3, size=n)
    lt = np.zeros((n, 3))
    lt[np.arange(n), y_test] = 2.0
    _write_fold(
        combo_dir, "fold0",
        _df_from_logits(lv, y_val), _df_from_logits(lt, y_test), ["A", "B", "C"],
    )
    summary = {
        "best_by_val": {"name": combo_name, "out_dir": combo_dir},
        "combos": [{"name": combo_name, "out_dir": combo_dir}],
    }
    with open(os.path.join(root, "sweep_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    rc = calibrate_main(
        ["--in", root, "--split", "test", "--group-classes", "1,2", "--all-combos"]
    )
    assert rc == 0
    with open(
        os.path.join(root, "calibration", "calibration.json"), "r", encoding="utf-8"
    ) as fh:
        results = json.load(fh)
    assert results[0]["stages"]["baseline"]["group_f1"] is not None
    assert "group_f1" in results[0]["marginal"]
