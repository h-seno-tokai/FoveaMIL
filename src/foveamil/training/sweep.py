"""sweep 設定を combo へ展開し fold 並列で実行してランキングする

設定は ``sweep``（リスト→展開する軸）/ ``fixed``（全 combo 共通スカラ）/ ``resolve``
（解決の起点）/ ``parallel``（制御）の 4 ブロックからなる``encoder`` と ``feature_type``
は直積でなく妥当な組合せのみ残す（cls/concat は ``has_cls=True`` のエンコーダのみ）
他の軸は直積展開する各 combo に解決済みの ``in_feat_dim`` / ``feature_root`` /
``labels_csv`` / ``n_cls`` を載せ，``job=(combo, fold)`` を ``foveamil-train --split`` の
サブプロセスとして GPU へ割り当て並列実行する``test_metrics.json`` 既存の job は
スキップ（resume），失敗は記録して継続し，combo ごとに CV 集計してランキングする
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from foveamil.encoders import ENCODERS
from foveamil.training.accessor import (
    FEATURE_TYPE_MEAN,
    FEATURE_TYPES,
    FeatureAccessor,
    _format_mag,
)
from foveamil.training.config import TrainConfig
from foveamil.training.cv import (
    CV_SCHEMA_VERSION,
    CV_SUMMARY_JSON,
    SUMMARY_METRICS,
    aggregate_folds_ci,
)
from foveamil.training.resolve import (
    ResolvedPaths,
    _fold_number,
    normalize_mags,
    resolve_in_feat_dim,
)

logger = logging.getLogger(__name__)

# sweep 軸のうち特別扱いするキー
ENCODER_KEY = "encoder"
FEATURE_TYPE_KEY = "feature_type"
MAGNIFICATIONS_KEY = "magnifications"
# 自動解決されるため設定に手書きを許さないキー
AUTO_RESOLVED_KEYS = ("in_feat_dim", "feature_root", "labels_csv", "n_cls", "classes")
# combo ディレクトリ名の接頭辞
COMBO_DIR_PREFIX = "combo_"
# combo 名でパス非対応文字を置換する文字
NAME_SANITIZE_SUB = "_"
# combo ごとの解決済み設定ファイル名
COMBO_CONFIG_NAME = "config.yaml"
# 単一 fold の test 指標ファイル名（foveamil-train の出力，skip-done 判定にも使う）
FOLD_RESULT_JSON = "test_metrics.json"
# 単一 fold の val 指標ファイル名（foveamil-train の出力）
METRICS_VAL_JSON = "metrics_val.json"
# fold ディレクトリ名の接頭辞
FOLD_DIR_PREFIX = "fold"
# combo ランキングに使う split（model selection は val）
SELECTION_SPLIT = "val"
# 報告に使う split
REPORT_SPLIT = "test"
# sweep 要約の保存ファイル名（機械可読）
SWEEP_SUMMARY_JSON = "sweep_summary.json"
# sweep 要約の保存ファイル名（人間可読）
SWEEP_SUMMARY_MD = "sweep_summary.md"
# sweep の詳細フラット CSV ファイル名（全 combo×fold×split）
SWEEP_DETAILED_CSV = "sweep_detailed.csv"
# 既定の使用 GPU 一覧
DEFAULT_GPU_IDS = (0,)
# 既定の GPU あたり並列ジョブ数
DEFAULT_JOBS_PER_GPU = 1
# ランキングの主要指標（優先順位順に最初に見つかったものを使う）
RANK_METRICS = ("macro_auc", "weighted_f1", "macro_f1", "accuracy", "kappa")


@dataclass
class Combo:
    """1 combo の展開結果

    Attributes:
        index: combo 連番
        name: パス安全な combo 名（出力ディレクトリ名）
        config: ``TrainConfig`` 互換の設定辞書（解決済み値を含む）
        axis_values: この combo の各 sweep 軸の値（表示用パスを含めない）
    """

    index: int
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    axis_values: Dict[str, Any] = field(default_factory=dict)


def _train_config_fields() -> frozenset:
    """``TrainConfig`` のフィールド名集合を返す"""
    return frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _reject_auto_resolved(block_name: str, block: Dict[str, Any]) -> None:
    """自動解決キーが設定に手書きされていればエラーにする"""
    for key in AUTO_RESOLVED_KEYS:
        if key in block:
            raise ValueError(
                f"'{key}' must not be set in '{block_name}'; it is resolved "
                "automatically from resolve.n_cls / encoder / feature_type"
            )


def _valid_pair(encoder: str, feature_type: str) -> bool:
    """``(encoder, feature_type)`` が有効か返す

    ``mean`` は全エンコーダで有効``cls`` / ``concat`` は ``has_cls=True`` の
    エンコーダのみ有効（``has_cls=False`` の h5 には cls 特徴が無い）
    """
    if feature_type == FEATURE_TYPE_MEAN:
        return True
    return ENCODERS[encoder].has_cls


def _encoder_feature_pairs(
    encoders: Sequence[str], feature_types: Sequence[str]
) -> List[tuple]:
    """有効な ``(encoder, feature_type)`` ペアのみを列挙する（直積にしない）"""
    for enc in encoders:
        if enc not in ENCODERS:
            raise KeyError(
                f"unknown encoder '{enc}'; available: {sorted(ENCODERS)}"
            )
    for ft in feature_types:
        if ft not in FEATURE_TYPES:
            raise ValueError(
                f"unknown feature_type '{ft}'; available: {sorted(FEATURE_TYPES)}"
            )
    return [
        (enc, ft)
        for enc in encoders
        for ft in feature_types
        if _valid_pair(enc, ft)
    ]


def _as_mag_sets(value: Any) -> List[List[float]]:
    """``magnifications`` の値を倍率セットの列に正規化する

    要素がすべて list なら倍率セットの列（軸），そうでなければ単一セットとみなす

    Args:
        value: 単一セット（``[1.25, 2.5]``）または複数セット（``[[...], [...]]``）

    Returns:
        正規化済みの倍率セット列（``[[float, ...], ...]``）
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"magnifications must be a non-empty list, got {value!r}")
    if all(isinstance(item, list) for item in value):
        return [normalize_mags(item) for item in value]
    return [normalize_mags(value)]


def _listify(value: Any) -> List[Any]:
    """スカラを 1 要素 list に包む（既に list ならそのまま）"""
    return value if isinstance(value, list) else [value]


def _sanitize(text: str) -> str:
    """combo 名でパス非対応文字を安全な文字へ置換する"""
    return re.sub(r"[^0-9A-Za-z._-]", NAME_SANITIZE_SUB, str(text))


def _combo_name(index: int, encoder: str, feature_type: str, n_mags: int) -> str:
    """combo の決定的でパス安全な名前を作る"""
    enc = _sanitize(encoder)
    return f"{COMBO_DIR_PREFIX}{index:03d}__{enc}_{feature_type}_m{n_mags}"


def expand_combos(
    sweep: Dict[str, Any], fixed: Dict[str, Any], resolved: ResolvedPaths
) -> List[Combo]:
    """sweep / fixed / 解決済みパスから combo 一覧を生成する

    ``(encoder, feature_type)`` は制約付き join，他の軸は ``ParameterGrid`` で直積展開し，
    各 combo に解決済み ``in_feat_dim`` / ``feature_root`` / ``labels_csv`` / ``n_cls`` を
    載せる``config`` のキーは ``TrainConfig`` フィールドに限る

    Args:
        sweep: 展開対象の軸辞書（``encoder`` / ``feature_type`` / ``magnifications`` を含む）
        fixed: 全 combo 共通スカラ辞書（``TrainConfig`` フィールド）
        resolved: 解決済みパス群

    Returns:
        combo の列（連番・名前・設定・軸値を持つ）

    Raises:
        ValueError: 必須軸の欠落，自動解決キーの手書き，未知の設定キー
        KeyError: 未知のエンコーダ名
    """
    _reject_auto_resolved("sweep", sweep)
    _reject_auto_resolved("fixed", fixed)

    for required in (ENCODER_KEY, FEATURE_TYPE_KEY, MAGNIFICATIONS_KEY):
        if required not in sweep:
            raise ValueError(f"sweep is missing required axis '{required}'")

    pairs = _encoder_feature_pairs(
        _listify(sweep[ENCODER_KEY]), _listify(sweep[FEATURE_TYPE_KEY])
    )
    mag_sets = _as_mag_sets(sweep[MAGNIFICATIONS_KEY])

    other_axes = {
        key: _listify(value)
        for key, value in sweep.items()
        if key not in (ENCODER_KEY, FEATURE_TYPE_KEY, MAGNIFICATIONS_KEY)
    }
    other_combos = list(ParameterGrid(other_axes)) if other_axes else [{}]

    known = _train_config_fields()
    combos: List[Combo] = []
    index = 0
    for encoder, feature_type in pairs:
        for mags in mag_sets:
            for other in other_combos:
                config: Dict[str, Any] = dict(fixed)
                config.update(other)
                config[ENCODER_KEY] = encoder
                config[FEATURE_TYPE_KEY] = feature_type
                config[MAGNIFICATIONS_KEY] = mags
                config["in_feat_dim"] = resolve_in_feat_dim(encoder, feature_type)
                config["feature_root"] = resolved.feature_root_base
                config["labels_csv"] = resolved.labels_csv
                config["n_cls"] = resolved.n_cls

                unknown = set(config) - known
                if unknown:
                    raise ValueError(
                        f"unknown config keys (not TrainConfig fields): "
                        f"{sorted(unknown)}"
                    )

                axis_values: Dict[str, Any] = {
                    ENCODER_KEY: encoder,
                    FEATURE_TYPE_KEY: feature_type,
                    MAGNIFICATIONS_KEY: mags,
                    **other,
                }
                combos.append(
                    Combo(
                        index=index,
                        name=_combo_name(index, encoder, feature_type, len(mags)),
                        config=config,
                        axis_values=axis_values,
                    )
                )
                index += 1
    return combos


def varying_axis_keys(combos: Sequence[Combo]) -> List[str]:
    """combo 間で値が複数に分かれた軸キーを返す（結果表の列・通知用）

    パス系の固定値を含まないため通知・表に載せても安全
    """
    seen: Dict[str, set] = {}
    for combo in combos:
        for key, value in combo.axis_values.items():
            seen.setdefault(key, set()).add(repr(value))
    return sorted(key for key, values in seen.items() if len(values) > 1)


def verify_feature_dims(combos: Sequence[Combo]) -> None:
    """各 combo の解決 ``in_feat_dim`` を実 h5 の特徴次元と突き合わせる

    ``(feature_root, encoder, feature_type, base_mag)`` ごとに 1 スライドだけ開き，
    :meth:`FeatureAccessor.feature_dim` と一致するか確認する特徴の不在や cls 欠落も
    ここで検出する

    Raises:
        ValueError: 特徴が見つからない，または次元が解決値と一致しない場合
    """
    import glob

    checked: set = set()
    for combo in combos:
        encoder = combo.config[ENCODER_KEY]
        feature_type = combo.config[FEATURE_TYPE_KEY]
        feature_root = combo.config["feature_root"]
        base_mag = combo.config[MAGNIFICATIONS_KEY][0]
        key = (feature_root, encoder, feature_type, base_mag)
        if key in checked:
            continue
        checked.add(key)

        mag_dir = os.path.join(feature_root, encoder, _format_mag(base_mag))
        slides = sorted(glob.glob(os.path.join(mag_dir, "*.h5")))
        if not slides:
            raise ValueError(
                f"no feature h5 found for encoder={encoder} mag={base_mag} "
                f"under {mag_dir}"
            )
        slide_id = os.path.splitext(os.path.basename(slides[0]))[0]
        accessor = FeatureAccessor(feature_root, encoder, slide_id, feature_type)
        try:
            actual = accessor.feature_dim(base_mag)
        finally:
            accessor.close()
        expected = combo.config["in_feat_dim"]
        if actual != expected:
            raise ValueError(
                f"in_feat_dim mismatch for encoder={encoder} "
                f"feature_type={feature_type}: resolved {expected} but h5 has "
                f"{actual} (slide {slide_id})"
            )


@dataclass
class ComboResult:
    """1 combo の実行・集計結果

    Attributes:
        index: combo 連番
        name: combo 名
        axis_values: sweep 軸の値（表示用）
        out_dir: この combo の出力先
        n_folds_total: 対象 fold 数
        n_folds_valid: 有効（test 指標を読めた）fold 数
        failed_jobs: 失敗した fold 番号の列
        aggregates: split 別 CV 集計 ``{"val": {...}, "test": {...}}``
        per_fold: split 別 fold ごと指標 ``{"val": [...], "test": [...]}``
        rank_metric: ランキングに用いた指標名（無ければ ``None``）
        rank_value_val: val ランキングに用いた指標の mean（無ければ ``None``）
        rank_value_test: test の同指標の mean（無ければ ``None``）
    """

    index: int
    name: str
    axis_values: Dict[str, Any]
    out_dir: str
    n_folds_total: int
    n_folds_valid: int = 0
    failed_jobs: List[int] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    per_fold: Dict[str, Any] = field(default_factory=dict)
    rank_metric: Optional[str] = None
    rank_value_val: Optional[float] = None
    rank_value_test: Optional[float] = None


def _run_fold_job(
    combo_config: str,
    fold_dir: str,
    weights_dir: str,
    split_csv: str,
    gpu_id: int,
) -> int:
    """1 つの (combo, fold) を ``foveamil-train --split`` のサブプロセスで学習する

    ``test_metrics.json`` が既にあればスキップ（resume）``CUDA_VISIBLE_DEVICES`` に
    ``gpu_id`` を割り当て当該 GPU へ固定するログ・結果は ``fold_dir``，重みは
    ``weights_dir`` へ保存する

    Returns:
        サブプロセス終了コード（スキップ時は 0）
    """
    result_path = os.path.join(fold_dir, FOLD_RESULT_JSON)
    if os.path.exists(result_path):
        return 0

    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "foveamil.training.train_cli",
        "--config",
        combo_config,
        "--split",
        split_csv,
        "--out",
        fold_dir,
        "--weights-out",
        weights_dir,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = os.path.join(fold_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT
        )
    return completed.returncode


def _pick_rank_metric(aggregate: Dict[str, Any]) -> tuple:
    """集計から優先順位順に最初に存在する指標名と mean を返す"""
    for metric in RANK_METRICS:
        if metric in aggregate:
            return metric, float(aggregate[metric]["mean"])
    return None, None


def _agg_mean(aggregate: Dict[str, Any], metric: Optional[str]) -> Optional[float]:
    """集計から ``metric`` の mean を取り出す無ければ ``None``"""
    if metric and metric in aggregate:
        return float(aggregate[metric]["mean"])
    return None


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """JSON を読む存在しない/壊れていれば ``None``"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - 壊れた結果は除外する
        logger.warning("could not read %s: %s", path, exc)
        return None


def _write_json(path: str, payload: Any) -> None:
    """辞書を JSON で保存する"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class SweepRunner:
    """combo 群を (combo, fold) ジョブにフラット化し GPU 並列で実行・集計する"""

    def __init__(
        self,
        combos: Sequence[Combo],
        split_files: Sequence[str],
        out_root: str,
        weights_root: str,
        gpu_ids: Optional[Sequence[int]] = None,
        jobs_per_gpu: Optional[int] = None,
    ) -> None:
        """実行器を初期化する

        Args:
            combos: 実行する combo 列
            split_files: fold 番号順の分割 CSV パス列
            out_root: ログ・結果の出力ルート（home）
            weights_root: 重みの出力ルート（Dataset）
            gpu_ids: 使用 GPU 一覧
            jobs_per_gpu: GPU あたり並列ジョブ数
        """
        self.combos = list(combos)
        self.split_files = list(split_files)
        self.out_root = out_root
        self.weights_root = weights_root
        self.gpu_ids = list(gpu_ids if gpu_ids else DEFAULT_GPU_IDS)
        self.jobs_per_gpu = int(jobs_per_gpu or DEFAULT_JOBS_PER_GPU)

    def _combo_dir(self, combo: Combo) -> str:
        return os.path.join(self.out_root, combo.name)

    def _write_combo_config(self, combo: Combo) -> str:
        """combo の解決済み設定を ``{out}/{name}/config.yaml`` に書き出す"""
        combo_dir = self._combo_dir(combo)
        os.makedirs(combo_dir, exist_ok=True)
        path = os.path.join(combo_dir, COMBO_CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(combo.config, handle, allow_unicode=True, sort_keys=True)
        return path

    def _build_jobs(self) -> List[Dict[str, Any]]:
        """全 (combo, fold) ジョブを GPU 割当付きで組み立てる"""
        jobs: List[Dict[str, Any]] = []
        i = 0
        for combo in self.combos:
            config_path = self._write_combo_config(combo)
            combo_dir = self._combo_dir(combo)
            weights_combo = os.path.join(self.weights_root, combo.name)
            for split_csv in self.split_files:
                fold = _fold_number(split_csv)
                fold_name = f"{FOLD_DIR_PREFIX}{fold}"
                jobs.append(
                    {
                        "combo_index": combo.index,
                        "fold": fold,
                        "combo_config": config_path,
                        "fold_dir": os.path.join(combo_dir, fold_name),
                        "weights_dir": os.path.join(weights_combo, fold_name),
                        "split_csv": split_csv,
                        "gpu_id": self.gpu_ids[i % len(self.gpu_ids)],
                    }
                )
                i += 1
        return jobs

    def run(self) -> Dict[str, Any]:
        """全 job を並列実行し combo ごとに集計してランキング要約を返す"""
        os.makedirs(self.out_root, exist_ok=True)
        jobs = self._build_jobs()
        max_workers = max(1, len(self.gpu_ids) * self.jobs_per_gpu)
        logger.info(
            "sweep: %d combos x %d folds = %d jobs, gpu_ids=%s, jobs_per_gpu=%d",
            len(self.combos),
            len(self.split_files),
            len(jobs),
            self.gpu_ids,
            self.jobs_per_gpu,
        )

        failures: Dict[int, List[int]] = {}
        executor = ProcessPoolExecutor(max_workers=max_workers)
        try:
            futures = {}
            for job in jobs:
                future = executor.submit(
                    _run_fold_job,
                    job["combo_config"],
                    job["fold_dir"],
                    job["weights_dir"],
                    job["split_csv"],
                    job["gpu_id"],
                )
                futures[future] = job
            for future in as_completed(futures):
                job = futures[future]
                try:
                    returncode = future.result()
                except Exception as exc:  # noqa: BLE001 - job 失敗で止めない
                    logger.error(
                        "combo %03d fold %d raised: %s",
                        job["combo_index"], job["fold"], exc,
                    )
                    returncode = 1
                if returncode != 0:
                    logger.error(
                        "combo %03d fold %d failed (returncode=%d)",
                        job["combo_index"], job["fold"], returncode,
                    )
                    failures.setdefault(job["combo_index"], []).append(job["fold"])
        except KeyboardInterrupt:
            logger.warning("interrupted; shutting down workers")
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            executor.shutdown(wait=True)

        results = [self._collect_combo(combo, failures.get(combo.index, []))
                   for combo in self.combos]
        summary = self._summarize(results)
        self._save_summary(summary, results)
        return summary

    def _read_fold_results(self, fold_dir: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """fold の test/val 指標 JSON を読む（無ければ ``None``）"""
        return {
            "test": _read_json(os.path.join(fold_dir, FOLD_RESULT_JSON)),
            "val": _read_json(os.path.join(fold_dir, METRICS_VAL_JSON)),
        }

    def _collect_combo(self, combo: Combo, failed_jobs: List[int]) -> ComboResult:
        """combo の val/test fold 結果を集めて CV 集計し ``cv_summary.json`` を書く"""
        combo_dir = self._combo_dir(combo)
        per_fold: Dict[str, List[Dict[str, Any]]] = {"test": [], "val": []}
        for split_csv in self.split_files:
            fold = _fold_number(split_csv)
            fold_dir = os.path.join(combo_dir, f"{FOLD_DIR_PREFIX}{fold}")
            res = self._read_fold_results(fold_dir)
            for split in ("test", "val"):
                if res[split] is not None:
                    res[split]["fold"] = fold
                    per_fold[split].append(res[split])

        result = ComboResult(
            index=combo.index,
            name=combo.name,
            axis_values=combo.axis_values,
            out_dir=combo_dir,
            n_folds_total=len(self.split_files),
            n_folds_valid=len(per_fold["test"]),
            failed_jobs=sorted(failed_jobs),
            per_fold=per_fold,
        )
        if per_fold["test"]:
            aggregates = {
                split: aggregate_folds_ci(per_fold[split]) for split in ("test", "val")
            }
            result.aggregates = aggregates
            # model selection は val 指標で（val 不在時のみ test にフォールバック）
            sel_agg = aggregates["val"] or aggregates["test"]
            metric, _ = _pick_rank_metric(sel_agg)
            result.rank_metric = metric
            result.rank_value_val = _agg_mean(aggregates["val"], metric)
            result.rank_value_test = _agg_mean(aggregates["test"], metric)
            cv_summary = {
                "schema_version": CV_SCHEMA_VERSION,
                "n_folds_total": len(self.split_files),
                "n_folds_valid": len(per_fold["test"]),
                "selection": {"split": SELECTION_SPLIT, "metric": metric},
                "test": {"per_fold": per_fold["test"], "aggregate": aggregates["test"]},
                "val": {"per_fold": per_fold["val"], "aggregate": aggregates["val"]},
            }
            _write_json(os.path.join(combo_dir, CV_SUMMARY_JSON), cv_summary)
        return result

    def _entry(
        self, r: ComboResult, rank_by_val: Dict[int, int], rank_by_test: Dict[int, int]
    ) -> Dict[str, Any]:
        """combo 1 件の要約エントリ（val/test 集計と両ランクを含む）を作る"""
        return {
            "index": r.index,
            "name": r.name,
            "axis_values": r.axis_values,
            "out_dir": r.out_dir,
            "n_folds_valid": r.n_folds_valid,
            "n_folds_total": r.n_folds_total,
            "failed_jobs": r.failed_jobs,
            "rank_metric": r.rank_metric,
            "val": r.aggregates.get("val"),
            "test": r.aggregates.get("test"),
            "rank_value_val": r.rank_value_val,
            "rank_value_test": r.rank_value_test,
            "rank_by_val": rank_by_val.get(r.index),
            "rank_by_test": rank_by_test.get(r.index),
        }

    def _summarize(self, results: Sequence[ComboResult]) -> Dict[str, Any]:
        """combo を val 指標でランキングし test を併記，test oracle も別途出す"""
        ranked_val = sorted(
            (r for r in results if r.rank_value_val is not None),
            key=lambda r: r.rank_value_val, reverse=True,
        )
        ranked_test = sorted(
            (r for r in results if r.rank_value_test is not None),
            key=lambda r: r.rank_value_test, reverse=True,
        )
        rank_by_val = {r.index: pos + 1 for pos, r in enumerate(ranked_val)}
        rank_by_test = {r.index: pos + 1 for pos, r in enumerate(ranked_test)}

        entries = [self._entry(r, rank_by_val, rank_by_test) for r in results]
        by_index = {e["index"]: e for e in entries}

        best_by_val = by_index.get(ranked_val[0].index) if ranked_val else None
        oracle = by_index.get(ranked_test[0].index) if ranked_test else None
        if oracle is not None:
            oracle = dict(oracle)
            oracle["note"] = "upper bound, not for reporting"

        return {
            "selection_metric": ranked_val[0].rank_metric if ranked_val else None,
            "selection_split": SELECTION_SPLIT,
            "report_split": REPORT_SPLIT,
            "n_combos": len(results),
            "n_folds": len(self.split_files),
            "gpu_ids": self.gpu_ids,
            "jobs_per_gpu": self.jobs_per_gpu,
            "axis_keys": varying_axis_keys(self.combos),
            "best_by_val": best_by_val,
            "oracle_by_test": oracle,
            "combos": entries,
            "failed": [r.index for r in results if r.failed_jobs],
        }

    def _write_detailed_csv(self, results: Sequence[ComboResult]) -> None:
        """全 combo×fold×split の指標を long format で ``sweep_detailed.csv`` に書く"""
        axis_keys = varying_axis_keys(self.combos)
        rows: List[Dict[str, Any]] = []
        for r in results:
            for split in ("val", "test"):
                for fold_metrics in r.per_fold.get(split, []):
                    row: Dict[str, Any] = {
                        "combo_index": r.index, "combo_name": r.name,
                    }
                    for key in axis_keys:
                        row[key] = r.axis_values.get(key)
                    row["fold"] = fold_metrics.get("fold")
                    row["split"] = split
                    for metric in SUMMARY_METRICS:
                        if metric in fold_metrics:
                            row[metric] = fold_metrics[metric]
                    rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(self.out_root, SWEEP_DETAILED_CSV), index=False
            )

    def _save_summary(self, summary: Dict[str, Any], results: Sequence[ComboResult]) -> None:
        """要約を JSON・人間可読 md・詳細 CSV で保存する"""
        _write_json(os.path.join(self.out_root, SWEEP_SUMMARY_JSON), summary)
        md_path = os.path.join(self.out_root, SWEEP_SUMMARY_MD)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(_summary_markdown(summary))
        self._write_detailed_csv(results)
        logger.info("saved sweep summary to %s", self.out_root)


def _fmt_agg(aggregate: Optional[Dict[str, Any]], metric: Optional[str]) -> str:
    """集計から ``metric`` の ``mean±std`` 表記を作る無ければ ``-``"""
    if not metric or not aggregate or metric not in aggregate:
        return "-"
    stats = aggregate[metric]
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def _best_block(summary: Dict[str, Any], metric: Optional[str]) -> List[str]:
    """val 選定 best とその test を示すブロックを作る"""
    best = summary.get("best_by_val")
    if not best:
        return ["(有効な結果がありませんでした)", ""]
    val = _fmt_agg(best.get("val"), metric)
    test = _fmt_agg(best.get("test"), metric)
    oracle = summary.get("oracle_by_test")
    lines = [
        f"## 選定 best（val 選定・test 報告 指標 {metric}）",
        "",
        f"- combo: `{best.get('name')}`",
        f"- val {metric}: {val}（選定基準）",
        f"- **test {metric}: {test}（報告値）**",
        "",
    ]
    if oracle:
        lines += [
            f"## test oracle（上限・報告には使わない）",
            "",
            f"- combo: `{oracle.get('name')}` / test {metric}: "
            f"{_fmt_agg(oracle.get('test'), metric)}",
            "",
        ]
    return lines


def _summary_markdown(summary: Dict[str, Any]) -> str:
    """要約辞書を val rank 昇順の md テーブルへ整形する（内部パスを載せない）"""
    axis_keys = summary.get("axis_keys") or []
    metric = summary.get("selection_metric")
    header = [
        "rank(val)", "combo", *axis_keys,
        f"val {metric}", f"test {metric}", "folds",
    ]

    rows = []
    for entry in summary.get("combos", []):
        axis_values = entry.get("axis_values", {})
        rank = entry.get("rank_by_val")
        cells = [
            str(rank if rank is not None else "-"),
            entry.get("name", ""),
            *[str(axis_values.get(k, "")) for k in axis_keys],
            _fmt_agg(entry.get("val"), metric),
            _fmt_agg(entry.get("test"), metric),
            f"{entry.get('n_folds_valid')}/{entry.get('n_folds_total')}",
        ]
        rows.append(cells)

    rows.sort(key=lambda c: (c[0] == "-", c[0]))

    lines = [
        f"# sweep summary（{summary.get('n_combos')} combos x "
        f"{summary.get('n_folds')} folds）",
        "",
        *_best_block(summary, metric),
        "## 全 combo（val rank 昇順）",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
