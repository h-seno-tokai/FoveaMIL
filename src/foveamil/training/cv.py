"""分割 CSV に基づく単一 fold 実行と交差検証の集計

``run_fold`` は ``train`` / ``val`` / ``test`` 列を持つ分割 CSV から 3 分割の
データセットを作り，``Trainer`` で学習・評価して test 指標を返す特徴は
``config.feature_root`` をそのまま読む（事前にステージ済みである前提）
``run_cross_validation`` は複数 fold の test 指標を集め，主要指標の fold 間
mean±std を計算して保存するログ・結果は ``save_root``，重みは ``weights_root`` へ
fold ごとに分けて保存する
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from foveamil.evaluation.stats import mean_ci_bootstrap, mean_ci_t
from foveamil.training.config import TrainConfig
from foveamil.training.dataset import FeatureBagDataset, build_label_dict
from foveamil.training.trainer import Trainer
from foveamil.training.yaml_config import train_config_to_dict
from foveamil.utils.provenance import collect_run_meta

logger = logging.getLogger(__name__)

# 分割 CSV の列名
SPLIT_COLUMNS = ("train", "val", "test")
# fold 間集計の対象とする主要指標
SUMMARY_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "weighted_f1",
    "macro_f1",
    "kappa",
    "mcc",
    "macro_auc",
    "weighted_auc",
    "ovo_macro_auc",
    "macro_auprc",
)
# CV 集計のスキーマ版
CV_SCHEMA_VERSION = 1
# CV 集計の保存ファイル名
CV_SUMMARY_JSON = "cv_summary.json"
# 後方互換の test 指標ファイル名（skip-done が参照）
TEST_METRICS_JSON = "test_metrics.json"
# 再現情報ファイル名
RUN_META_JSON = "run_meta.json"
# fold ディレクトリ名の接頭辞
FOLD_DIR_PREFIX = "fold"
# ブートストラップ反復数
BOOTSTRAP_N = 10000
# 信頼区間の有意水準
CI_ALPHA = 0.05
# 集計対象の split
AGG_SPLITS = ("test", "val")
# タイムスタンプの書式
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _foveamil_version() -> Optional[str]:
    """インストール済み foveamil のバージョンを返す取得不能なら ``None``"""
    try:
        from importlib.metadata import version

        return version("foveamil")
    except Exception:  # noqa: BLE001
        return None


def _write_json(path: str, payload: Any) -> None:
    """辞書を JSON で保存する"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_split(split_csv: str) -> Dict[str, List[str]]:
    """分割 CSV から ``train`` / ``val`` / ``test`` の slide_id 列を読む"""
    df = pd.read_csv(split_csv)
    splits: Dict[str, List[str]] = {}
    for col in SPLIT_COLUMNS:
        splits[col] = [
            str(s) for s in df[col].dropna().tolist()
        ]
    return splits


def run_fold(
    config: TrainConfig,
    split_csv: str,
    save_path: str,
    weights_dir: Optional[str] = None,
    save_train: bool = False,
) -> Dict[str, Any]:
    """単一 fold を学習・評価し val/test の指標と再現情報を保存して返す

    val/test の予測 CSV・指標 JSON・混同行列・学習履歴は ``Trainer`` が ``save_path`` へ，
    後方互換の ``test_metrics.json`` と再現情報 ``run_meta.json`` を本関数が書く

    Args:
        config: 学習設定
        split_csv: ``train`` / ``val`` / ``test`` 列を持つ分割 CSV
        save_path: この fold のログ・結果の出力先ディレクトリ
        weights_dir: 重み（``.pt``）の保存先ディレクトリ``None`` なら ``save_path``
        save_train: ``True`` なら train split も評価・保存する

    Returns:
        ``{"test": {...,"fold"}, "val": {...,"fold"}, "selection": {...},
        "class_breakdown": {...}}``
    """
    splits = _read_split(split_csv)
    label_dict = build_label_dict(config.labels_csv, classes=config.classes)
    config.n_cls = len(set(label_dict.values()))

    def _make_dataset(slide_ids: List[str]) -> FeatureBagDataset:
        return FeatureBagDataset(
            feature_root=config.feature_root,
            encoder=config.encoder,
            magnifications=config.magnifications,
            slide_ids=slide_ids,
            labels_csv=config.labels_csv,
            label_dict=label_dict,
            feature_type=config.feature_type,
        )

    train_ds = _make_dataset(splits["train"])
    val_ds = _make_dataset(splits["val"])
    test_ds = _make_dataset(splits["test"])

    trainer = Trainer(
        config, train_ds, val_ds, test_ds, save_path, weights_dir=weights_dir
    )
    start = time.time()
    trainer.train()
    results = trainer.evaluate_best(save_train=save_train)
    end = time.time()

    fold_name = os.path.basename(save_path)
    test_metrics = dict(results["test"])
    test_metrics["fold"] = fold_name
    val_metrics = dict(results["val"])
    val_metrics["fold"] = fold_name

    _write_json(os.path.join(save_path, TEST_METRICS_JSON), test_metrics)
    _write_run_meta(
        save_path, config, split_csv, results["selection"],
        results["class_breakdown"], start, end,
    )

    return {
        "test": test_metrics,
        "val": val_metrics,
        "selection": results["selection"],
        "class_breakdown": results["class_breakdown"],
    }


def _write_run_meta(
    save_path: str,
    config: TrainConfig,
    split_csv: str,
    selection: Dict[str, Any],
    class_breakdown: Dict[str, Dict[str, int]],
    start: float,
    end: float,
) -> None:
    """fold の再現情報を ``run_meta.json`` に書く"""
    timing = {
        "start": time.strftime(_TS_FORMAT, time.localtime(start)),
        "end": time.strftime(_TS_FORMAT, time.localtime(end)),
        "duration_sec": float(end - start),
    }
    meta = collect_run_meta(
        config=train_config_to_dict(config),
        selection=selection,
        timing=timing,
        labels_csv=config.labels_csv,
        split_csv=split_csv,
        class_breakdown=class_breakdown,
        version=_foveamil_version(),
    )
    _write_json(os.path.join(save_path, RUN_META_JSON), meta)


def aggregate_folds(
    per_fold: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """fold ごとの test 指標から主要指標の fold 間 mean±std を集計する

    指標が 1 つも無い fold は寄与しない指標ごとに値を持つ fold だけで集計する

    Args:
        per_fold: fold ごとの test 指標辞書の列

    Returns:
        ``{metric: {mean, std}}`` の集計辞書
    """
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [m[metric] for m in per_fold if metric in m]
        if not values:
            continue
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        logger.info(
            "%s: %.4f +/- %.4f", metric, aggregate[metric]["mean"],
            aggregate[metric]["std"],
        )
    return aggregate


def aggregate_folds_ci(
    per_fold: List[Dict[str, float]],
    n_boot: int = BOOTSTRAP_N,
    alpha: float = CI_ALPHA,
) -> Dict[str, Dict[str, float]]:
    """fold ごとの指標から mean/std と信頼区間（t・ブートストラップ）を集計する

    指標ごとに値を持つ fold だけで集計する``aggregate_folds`` と mean/std は一致し，
    ``n`` と t/ブートストラップ信頼区間の上下限を追加する

    Args:
        per_fold: fold ごとの指標辞書の列
        n_boot: ブートストラップ反復数
        alpha: 有意水準

    Returns:
        ``{metric: {mean, std, n, ci_t_low, ci_t_high, ci_boot_low, ci_boot_high}}``
    """
    aggregate: Dict[str, Dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [m[metric] for m in per_fold if metric in m]
        if not values:
            continue
        mean_t, ci_t_low, ci_t_high = mean_ci_t(values, alpha=alpha)
        _, ci_boot_low, ci_boot_high = mean_ci_bootstrap(
            values, alpha=alpha, n_boot=n_boot
        )
        aggregate[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
            "ci_t_low": ci_t_low,
            "ci_t_high": ci_t_high,
            "ci_boot_low": ci_boot_low,
            "ci_boot_high": ci_boot_high,
        }
    return aggregate


def run_cross_validation(
    config: TrainConfig,
    split_paths: List[str],
    save_root: str,
    weights_root: Optional[str] = None,
) -> Dict:
    """複数 fold を実行し主要指標の fold 間 mean±std を集計する

    Args:
        config: 学習設定
        split_paths: 各 fold の分割 CSV パス列
        save_root: ログ・結果の出力ルート（fold ごとに ``{save_root}/fold{i}``）
        weights_root: 重みの出力ルート（fold ごとに ``{weights_root}/fold{i}``）
            ``None`` なら ``save_root``

    Returns:
        ``test``/``val`` ごとに ``per_fold`` と CI 付き ``aggregate`` を含む辞書
    """
    os.makedirs(save_root, exist_ok=True)
    per_fold = {"test": [], "val": []}
    for i, split_csv in enumerate(split_paths):
        fold_path = os.path.join(save_root, f"{FOLD_DIR_PREFIX}{i + 1}")
        fold_weights = (
            os.path.join(weights_root, f"{FOLD_DIR_PREFIX}{i + 1}")
            if weights_root is not None
            else None
        )
        logger.info("=== fold %d/%d: %s ===", i + 1, len(split_paths), split_csv)
        result = run_fold(config, split_csv, fold_path, weights_dir=fold_weights)
        per_fold["test"].append(result["test"])
        per_fold["val"].append(result["val"])

    summary: Dict[str, Any] = {
        "schema_version": CV_SCHEMA_VERSION,
        "n_folds_total": len(split_paths),
        "n_folds_valid": len(per_fold["test"]),
        "selection": {"save_metric": config.save_metric},
    }
    for split in AGG_SPLITS:
        summary[split] = {
            "per_fold": per_fold[split],
            "aggregate": aggregate_folds_ci(per_fold[split]),
        }
    out_path = os.path.join(save_root, CV_SUMMARY_JSON)
    _write_json(out_path, summary)
    logger.info("saved CV summary to %s", out_path)
    return summary
