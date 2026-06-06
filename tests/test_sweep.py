"""sweep 展開部・val 選定のユニット"""

import json
import os

import pandas as pd
import pytest

from foveamil.training.resolve import ResolvedPaths
from foveamil.training.sweep import (
    FOLD_RESULT_JSON,
    SWEEP_DETAILED_CSV,
    Combo,
    SweepRunner,
    _result_complete,
    _write_json,
    expand_combos,
    run_jobs_on_gpu_memory_pool,
    run_jobs_on_gpu_pool,
    varying_axis_keys,
)


def test_result_complete_rejects_corrupt_and_atomic_write(tmp_path):
    # 中断で半端に書かれた結果 JSON を完了扱いせず再実行対象にする（silent fold loss 防止）
    good = str(tmp_path / "good.json")
    _write_json(good, {"weighted_f1": 0.5})
    assert _result_complete(good) is True
    bad = str(tmp_path / "bad.json")
    with open(bad, "w", encoding="utf-8") as handle:
        handle.write('{"weighted_f1": 0.5')   # 切れた JSON
    assert _result_complete(bad) is False
    assert _result_complete(str(tmp_path / "absent.json")) is False
    # アトミック書き込みは .tmp を残さない
    import glob

    assert glob.glob(str(tmp_path / "*.tmp")) == []

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


# --- 構成に無関係なパラメータの統合（sweep 健全化）---


def test_instance_loss_single_mag_only_constrained_join():
    # instance_loss[true,false] x mags[[1.25,5.0],[10]] -> 多倍率は false のみ，単一倍率は両方
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[1.25, 5.0], [10]],
        "instance_loss": [True, False],
    }
    combos = expand_combos(sweep, {}, _resolved())
    seen = {(tuple(c.config["magnifications"]), c.config["instance_loss"]) for c in combos}
    assert seen == {((1.25, 5.0), False), ((10.0,), True), ((10.0,), False)}
    assert len(combos) == 3
    # 多倍率 + instance_loss=True は無い
    assert ((1.25, 5.0), True) not in seen


def test_single_mag_collapses_zoom_params():
    # 単一倍率では k_sample / k_sigma は無関係なので畳んで統合する（pair ぶんのみ残る）
    sweep = _base_sweep(
        magnifications=[[40]], k_sample=[8, 15, 25], k_sigma=[0.002, 0.005]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ zoom 系は畳まれる
    keys = varying_axis_keys(combos)
    assert "k_sample" not in keys and "k_sigma" not in keys
    for c in combos:
        assert c.config["k_sample"] == 12  # DEFAULT_K_SAMPLE
        assert c.config["k_sigma"] == 0.002  # DEFAULT_K_SIGMA


def test_multi_mag_keeps_zoom_params():
    sweep = _base_sweep(magnifications=[[1.25, 2.5]], k_sample=[8, 25])
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 k_sample
    assert {c.config["k_sample"] for c in combos} == {8, 25}
    assert "k_sample" in varying_axis_keys(combos)


def test_instance_params_collapse_when_loss_off():
    # instance_loss 既定 False -> bag_weight / inst_k は無関係なので畳む
    sweep = _base_sweep(magnifications=[[40]], bag_weight=[0.7, 0.9], inst_k=[8, 16])
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10
    for c in combos:
        assert c.config["bag_weight"] == 0.7  # DEFAULT_BAG_WEIGHT
        assert c.config["inst_k"] == 8  # DEFAULT_INST_K


def test_instance_params_kept_when_loss_on():
    sweep = _base_sweep(
        magnifications=[[40]], instance_loss=[True], bag_weight=[0.7, 0.9]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 bag_weight
    assert {c.config["bag_weight"] for c in combos} == {0.7, 0.9}
    for c in combos:
        assert c.config["instance_loss"] is True


def test_dedup_merges_type_divergent_instance_loss():
    # 単一倍率で instance_loss を True と 1 で書いても同一視して統合し bool へ正規化する
    sweep = _base_sweep(magnifications=[[40]], instance_loss=[True, 1])
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs（True/1 は 1 つに統合）
    for c in combos:
        assert c.config["instance_loss"] is True


def test_dedup_merges_int_and_float_param():
    # 数値の型違い（1 と 1.0）は同値として統合する
    sweep = _base_sweep(
        magnifications=[[40]], instance_loss=[True], bag_weight=[1, 1.0]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # bag_weight 1 と 1.0 は同一


def test_mixed_single_and_multi_mag_with_zoom_axis():
    # 単一 [40] は zoom 系を畳んで 1 件，多倍率 [1.25,2.5] は 2 件 -> 1 pair で計 3
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[40], [1.25, 2.5]],
        "k_sample": [8, 25],
    }
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 3
    single = [c for c in combos if len(c.config["magnifications"]) == 1]
    multi = [c for c in combos if len(c.config["magnifications"]) == 2]
    assert len(single) == 1 and len(multi) == 2
    assert single[0].config["k_sample"] == 12  # 畳まれて既定
    assert {c.config["k_sample"] for c in multi} == {8, 25}


def test_single_mag_collapses_decorrelation_params():
    # 単一倍率では decorrelation 系は無関係なので畳んで統合する
    sweep = _base_sweep(
        magnifications=[[40]],
        decorrelation_weight=[0.0, 0.5],
        decorrelation_method=["cosine", "covariance"],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ decorrelation 系は畳まれる
    keys = varying_axis_keys(combos)
    assert "decorrelation_weight" not in keys
    assert "decorrelation_method" not in keys
    for c in combos:
        assert c.config["decorrelation_weight"] == 0.0
        assert c.config["decorrelation_method"] == "cosine"


def test_multi_mag_keeps_decorrelation_weight():
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]], decorrelation_weight=[0.1, 0.5]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 weight
    assert {c.config["decorrelation_weight"] for c in combos} == {0.1, 0.5}
    assert "decorrelation_weight" in varying_axis_keys(combos)


def test_decorrelation_method_collapses_when_weight_zero():
    # weight=0 では method は無関係なので畳む
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]],
        decorrelation_method=["cosine", "covariance"],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # method は畳まれて 10 pairs のみ
    assert "decorrelation_method" not in varying_axis_keys(combos)
    for c in combos:
        assert c.config["decorrelation_method"] == "cosine"


def test_decorrelation_method_kept_when_weight_positive():
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]],
        decorrelation_weight=[0.5],
        decorrelation_method=["cosine", "covariance"],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 method
    assert {c.config["decorrelation_method"] for c in combos} == {
        "cosine", "covariance"
    }


def test_single_mag_collapses_aux_norm_params():
    # 単一倍率では補助アテンションを持たないため aux_norm 系は無関係で畳む
    sweep = _base_sweep(
        magnifications=[[40]],
        aux_norm=["softmax", "temperature", "entmax"],
        aux_norm_temperature=[0.5, 2.0],
        aux_norm_alpha=[1.2, 1.8],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ aux_norm 系は畳まれる
    keys = varying_axis_keys(combos)
    for key in ("aux_norm", "aux_norm_temperature", "aux_norm_alpha"):
        assert key not in keys
    for c in combos:
        assert c.config["aux_norm"] == "softmax"  # DEFAULT_AUX_NORM
        assert c.config["aux_norm_temperature"] == 1.0
        assert c.config["aux_norm_alpha"] == 1.5


def test_aux_norm_temperature_relevant_only_for_temperature():
    # aux_norm=temperature のときのみ aux_norm_temperature が軸として残る
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[1.25, 2.5]],
        "aux_norm": ["softmax", "temperature"],
        "aux_norm_temperature": [0.5, 2.0],
    }
    combos = expand_combos(sweep, {}, _resolved())
    # softmax: temp 畳んで 1，temperature: temp 2 値で 2 -> 計 3
    assert len(combos) == 3
    soft = [c for c in combos if c.config["aux_norm"] == "softmax"]
    temp = [c for c in combos if c.config["aux_norm"] == "temperature"]
    assert len(soft) == 1 and soft[0].config["aux_norm_temperature"] == 1.0
    assert {c.config["aux_norm_temperature"] for c in temp} == {0.5, 2.0}


def test_aux_norm_alpha_relevant_only_for_entmax():
    # aux_norm=entmax のときのみ aux_norm_alpha が軸として残る
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[1.25, 2.5]],
        "aux_norm": ["softmax", "entmax"],
        "aux_norm_alpha": [1.2, 1.8],
    }
    combos = expand_combos(sweep, {}, _resolved())
    # softmax: alpha 畳んで 1，entmax: alpha 2 値で 2 -> 計 3
    assert len(combos) == 3
    soft = [c for c in combos if c.config["aux_norm"] == "softmax"]
    ent = [c for c in combos if c.config["aux_norm"] == "entmax"]
    assert len(soft) == 1 and soft[0].config["aux_norm_alpha"] == 1.5
    assert {c.config["aux_norm_alpha"] for c in ent} == {1.2, 1.8}


def test_temperature_alpha_dont_cross_contaminate():
    # temperature 値は alpha 軸を，entmax 値は temperature 軸を増やさない
    sweep = {
        "encoder": ["ResNet50"],
        "feature_type": ["mean"],
        "magnifications": [[1.25, 2.5]],
        "aux_norm": ["temperature", "entmax"],
        "aux_norm_temperature": [0.5, 2.0],
        "aux_norm_alpha": [1.2, 1.8],
    }
    combos = expand_combos(sweep, {}, _resolved())
    # temperature: temp 2 値 * alpha 畳む = 2，entmax: alpha 2 値 * temp 畳む = 2 -> 計 4
    assert len(combos) == 4
    temp = [c for c in combos if c.config["aux_norm"] == "temperature"]
    ent = [c for c in combos if c.config["aux_norm"] == "entmax"]
    assert {c.config["aux_norm_temperature"] for c in temp} == {0.5, 2.0}
    assert all(c.config["aux_norm_alpha"] == 1.5 for c in temp)
    assert {c.config["aux_norm_alpha"] for c in ent} == {1.2, 1.8}
    assert all(c.config["aux_norm_temperature"] == 1.0 for c in ent)


def test_multi_mag_keeps_aux_norm_axis():
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]], aux_norm=["softmax", "sparsemax"]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert {c.config["aux_norm"] for c in combos} == {"softmax", "sparsemax"}
    assert "aux_norm" in varying_axis_keys(combos)


def test_dpp_params_collapse_when_selector_not_dpp():
    # selector 既定 topk -> dpp 系は無関係なので畳んで統合する
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]],
        dpp_temperature=[0.5, 1.0],
        dpp_similarity=["cosine", "rbf"],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ dpp 系は畳まれる
    keys = varying_axis_keys(combos)
    assert "dpp_temperature" not in keys and "dpp_similarity" not in keys
    for c in combos:
        assert c.config["dpp_temperature"] == 1.0  # DEFAULT_DPP_TEMPERATURE
        assert c.config["dpp_similarity"] == "cosine"  # DEFAULT_DPP_SIMILARITY


def test_dpp_params_kept_when_dpp_multi_mag():
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]], selector=["dpp"], dpp_temperature=[0.5, 1.0]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 dpp_temperature
    assert {c.config["dpp_temperature"] for c in combos} == {0.5, 1.0}
    assert "dpp_temperature" in varying_axis_keys(combos)


def test_dpp_params_collapse_for_single_magnification():
    # 単一倍率では選択自体が無効なので dpp 系も畳む
    sweep = _base_sweep(
        magnifications=[[40]], selector=["dpp"], dpp_diversity_weight=[0.0, 0.1]
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10
    keys = varying_axis_keys(combos)
    assert "dpp_diversity_weight" not in keys
    for c in combos:
        assert c.config["dpp_diversity_weight"] == 0.0  # DEFAULT_DPP_DIVERSITY_WEIGHT


def test_selector_collapses_for_single_magnification():
    # 単一倍率では選択が走らず topk/dpp は挙動同一なので selector 軸を畳む
    sweep = _base_sweep(magnifications=[[40]], selector=["topk", "dpp"])
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10
    assert "selector" not in varying_axis_keys(combos)
    for c in combos:
        assert c.config["selector"] == "topk"  # DEFAULT_SELECTOR


def test_selector_kept_as_axis_for_multi_magnification():
    # 多倍率では selector を on/off 軸として保持する
    sweep = _base_sweep(magnifications=[[1.25, 2.5]], selector=["topk", "dpp"])
    combos = expand_combos(sweep, {}, _resolved())
    assert {c.config["selector"] for c in combos} == {"topk", "dpp"}
    assert "selector" in varying_axis_keys(combos)


# --- MCTS（探索駆動）パラメータの統合 ---


def test_mcts_params_collapse_when_driver_not_mcts():
    # 既定 zoom_driver=differentiable では MCTS 系は無関係なので畳む
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]],
        mcts_simulations=[8, 16],
        policy_loss_weight=[0.5, 1.0],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ MCTS 系は畳まれる
    keys = varying_axis_keys(combos)
    assert "mcts_simulations" not in keys and "policy_loss_weight" not in keys
    for c in combos:
        assert c.config["mcts_simulations"] == 16  # DEFAULT_MCTS_SIMULATIONS
        assert c.config["policy_loss_weight"] == 1.0  # DEFAULT_POLICY_LOSS_WEIGHT


def test_mcts_params_kept_when_driver_mcts():
    sweep = _base_sweep(
        magnifications=[[1.25, 2.5]],
        zoom_driver=["mcts"],
        mcts_simulations=[8, 16],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 20  # 10 pairs * 2 simulations
    assert {c.config["mcts_simulations"] for c in combos} == {8, 16}
    assert "mcts_simulations" in varying_axis_keys(combos)
    for c in combos:
        assert c.config["zoom_driver"] == "mcts"


def test_zoom_driver_collapses_for_single_mag():
    # 単一倍率はズーム自体が無いため zoom_driver も MCTS 系も無関係 -> 畳む
    sweep = _base_sweep(
        magnifications=[[40]],
        zoom_driver=["differentiable", "mcts"],
        mcts_simulations=[8, 16],
    )
    combos = expand_combos(sweep, {}, _resolved())
    assert len(combos) == 10  # 10 pairs のみ
    keys = varying_axis_keys(combos)
    assert "zoom_driver" not in keys and "mcts_simulations" not in keys
    for c in combos:
        assert c.config["zoom_driver"] == "differentiable"  # DEFAULT_ZOOM_DRIVER


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


# --- 動的 GPU キュー（スロット割り当て） ---


def _make_jobs(tmp_path, n):
    jobs = []
    for i in range(n):
        d = tmp_path / f"job{i}"
        d.mkdir()
        jobs.append({"fold_dir": str(d), "combo_index": i, "fold": 1})
    return jobs


def test_dynamic_pool_respects_per_gpu_capacity_and_uses_all_gpus(tmp_path):
    import collections
    import threading
    import time

    gpu_ids = [0, 1]
    jobs_per_gpu = 2
    running = collections.Counter()
    peak = collections.Counter()
    lock = threading.Lock()
    global_cur = [0]
    global_peak = [0]

    def fake_run(job, gpu):
        with lock:
            running[gpu] += 1
            global_cur[0] += 1
            peak[gpu] = max(peak[gpu], running[gpu])
            global_peak[0] = max(global_peak[0], global_cur[0])
        time.sleep(0.02)
        with lock:
            running[gpu] -= 1
            global_cur[0] -= 1
        return 0

    jobs = _make_jobs(tmp_path, 12)
    results = run_jobs_on_gpu_pool(jobs, gpu_ids, jobs_per_gpu, fake_run)

    assert len(results) == 12
    assert all(rc == 0 for rc in results.values())
    # GPU あたり同時実行は jobs_per_gpu を超えない
    for gpu in gpu_ids:
        assert peak[gpu] <= jobs_per_gpu
    # 全スロットが埋まる（GPU が遊ばない）
    assert global_peak[0] == len(gpu_ids) * jobs_per_gpu
    # 両 GPU が使われる
    assert peak[0] > 0 and peak[1] > 0


def test_dynamic_pool_skips_done_jobs_without_acquiring_gpu(tmp_path):
    done = tmp_path / "done"
    done.mkdir()
    (done / FOLD_RESULT_JSON).write_text("{}", encoding="utf-8")
    todo = tmp_path / "todo"
    todo.mkdir()

    called = []

    def fake_run(job, gpu):
        called.append(job["fold_dir"])
        return 0

    jobs = [{"fold_dir": str(done)}, {"fold_dir": str(todo)}]
    results = run_jobs_on_gpu_pool(jobs, [0], 1, fake_run)

    assert results == {0: 0, 1: 0}
    # 既存結果のある job は run_fn を呼ばない
    assert called == [str(todo)]


def test_dynamic_pool_records_failures_without_aborting(tmp_path):
    def fake_run(job, gpu):
        if job["fold_dir"].endswith("bad"):
            raise RuntimeError("boom")
        return 7 if job["fold_dir"].endswith("rc") else 0

    ok = tmp_path / "ok"
    ok.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    rc = tmp_path / "rc"
    rc.mkdir()
    jobs = [{"fold_dir": str(ok)}, {"fold_dir": str(bad)}, {"fold_dir": str(rc)}]
    results = run_jobs_on_gpu_pool(jobs, [0], 1, fake_run)

    assert results[0] == 0
    assert results[1] == 1  # 例外は終了コード 1 に正規化
    assert results[2] == 7  # 非ゼロ終了コードを保持


# --- メモリ動的 GPU スケジューラ ---


def test_memory_pool_respects_reserved_memory(tmp_path):
    import collections
    import threading
    import time

    free = {0: 10000, 1: 10000}
    running = collections.Counter()
    peak = collections.Counter()
    lock = threading.Lock()

    def fake(job, gpu):
        with lock:
            running[gpu] += 1
            peak[gpu] = max(peak[gpu], running[gpu])
        time.sleep(0.03)
        with lock:
            running[gpu] -= 1
        return 0

    jobs = _make_jobs(tmp_path, 18)
    results = run_jobs_on_gpu_memory_pool(
        jobs, [0, 1], fake, per_job_mem_mb=3000, headroom_mb=1000,
        poll_interval=0.01, mem_query=lambda: dict(free),
    )
    assert len(results) == 18 and all(rc == 0 for rc in results.values())
    # 空き 10000 - headroom 1000 = 9000 / 3000 = GPU あたり最大 3
    for gpu in (0, 1):
        assert peak[gpu] <= 3
    assert peak[0] > 0 and peak[1] > 0
    assert max(peak.values()) == 3  # 充填上限まで使う


def test_memory_pool_max_per_gpu_caps_concurrency(tmp_path):
    import collections
    import threading
    import time

    running = collections.Counter()
    peak = collections.Counter()
    lock = threading.Lock()

    def fake(job, gpu):
        with lock:
            running[gpu] += 1
            peak[gpu] = max(peak[gpu], running[gpu])
        time.sleep(0.02)
        with lock:
            running[gpu] -= 1
        return 0

    jobs = _make_jobs(tmp_path, 12)
    # メモリは潤沢でも max_per_gpu=3 で同時数を抑える
    run_jobs_on_gpu_memory_pool(
        jobs, [0], fake, per_job_mem_mb=100, headroom_mb=0, poll_interval=0.01,
        max_per_gpu=3, mem_query=lambda: {0: 100000},
    )
    assert peak[0] == 3


def test_memory_pool_skips_done_jobs(tmp_path):
    done = tmp_path / "done"
    done.mkdir()
    (done / FOLD_RESULT_JSON).write_text("{}", encoding="utf-8")
    todo = tmp_path / "todo"
    todo.mkdir()
    called = []

    def fake(job, gpu):
        called.append(job["fold_dir"])
        return 0

    jobs = [{"fold_dir": str(done)}, {"fold_dir": str(todo)}]
    results = run_jobs_on_gpu_memory_pool(
        jobs, [0], fake, per_job_mem_mb=100, headroom_mb=0, poll_interval=0.01,
        mem_query=lambda: {0: 100000},
    )
    assert results == {0: 0, 1: 0}
    assert called == [str(todo)]


def test_memory_pool_records_failures(tmp_path):
    def fake(job, gpu):
        if job["fold_dir"].endswith("bad"):
            raise RuntimeError("boom")
        return 0

    ok = tmp_path / "ok"
    ok.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    results = run_jobs_on_gpu_memory_pool(
        [{"fold_dir": str(ok)}, {"fold_dir": str(bad)}], [0], fake,
        per_job_mem_mb=100, headroom_mb=0, poll_interval=0.01,
        mem_query=lambda: {0: 100000},
    )
    assert results[0] == 0 and results[1] == 1


def test_memory_pool_serializes_when_only_one_fits(tmp_path):
    import collections
    import threading
    import time

    running = collections.Counter()
    peak = collections.Counter()
    lock = threading.Lock()

    def fake(job, gpu):
        with lock:
            running[gpu] += 1
            peak[gpu] = max(peak[gpu], running[gpu])
        time.sleep(0.02)
        with lock:
            running[gpu] -= 1
        return 0

    jobs = _make_jobs(tmp_path, 5)
    # 空き 5000 - headroom 1000 = 4000 / per_job 4000 = GPU あたり 1（直列化）
    run_jobs_on_gpu_memory_pool(
        jobs, [0], fake, per_job_mem_mb=4000, headroom_mb=1000, poll_interval=0.01,
        mem_query=lambda: {0: 5000},
    )
    assert peak[0] == 1


def test_memory_pool_fails_job_that_fits_no_gpu(tmp_path):
    # per_job がどの GPU 空きより大きい誤設定 -> ハングせず失敗で進める
    called = []

    def fake(job, gpu):
        called.append(gpu)
        return 0

    jobs = _make_jobs(tmp_path, 2)
    results = run_jobs_on_gpu_memory_pool(
        jobs, [0], fake, per_job_mem_mb=100000, headroom_mb=0, poll_interval=0.01,
        mem_query=lambda: {0: 10000},
    )
    assert results == {0: 1, 1: 1}  # 配置不能なジョブは失敗扱い
    assert called == []  # run_fn は呼ばれない
