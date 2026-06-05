"""保存済み予測・集計から再学習なしで評価成果物を生成する

sweep の出力（``sweep_summary.json`` / 各 combo の ``cv_summary.json`` /
``fold*/predictions_{split}.csv`` / ``run_meta.json``）を直接読み，ROC/PR/
キャリブレーション図・combo 間有意差検定・人間可読レポートを作る学習は
一切しない（予測を二次利用するだけ）matplotlib が無ければ図は省く
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from foveamil.evaluation.stats import (
    nadeau_bengio_corrected_t,
    wilcoxon_signed_rank,
)

logger = logging.getLogger(__name__)

# sweep 出力のファイル名
SWEEP_SUMMARY_JSON = "sweep_summary.json"
CV_SUMMARY_JSON = "cv_summary.json"
RUN_META_JSON = "run_meta.json"
PREDICTIONS_CSV_TEMPLATE = "predictions_{split}.csv"
# 予測 CSV の確率列の接頭辞
PROB_PREFIX = "prob_"
# 既定のキャリブレーション bin 数
DEFAULT_N_BINS = 10
# fold ディレクトリ名の接頭辞
FOLD_DIR_PREFIX = "fold"


def _prob_columns(df: pd.DataFrame) -> List[str]:
    """``prob_*`` 列を class 添字順に返す"""
    cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(PROB_PREFIX):]))


def load_predictions(fold_dir: str, split: str) -> Optional[pd.DataFrame]:
    """fold の予測 CSV を読む無ければ ``None``"""
    path = os.path.join(fold_dir, PREDICTIONS_CSV_TEMPLATE.format(split=split))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def pool_predictions(
    combo_dir: str, split: str, fold_names: List[str]
) -> Optional[pd.DataFrame]:
    """combo の全 fold の予測を縦結合する読めなければ ``None``"""
    frames = []
    for name in fold_names:
        df = load_predictions(os.path.join(combo_dir, name), split)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _y_true_prob(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """予測 DataFrame から ``(y_true, y_prob[N,C])`` を取り出す"""
    cols = _prob_columns(df)
    return df["y_true"].to_numpy(), df[cols].to_numpy(dtype=float)


def compute_ece(df: pd.DataFrame, n_bins: int = DEFAULT_N_BINS) -> float:
    """予測信頼度のキャリブレーション誤差（ECE）を返す"""
    y_true, y_prob = _y_true_prob(df)
    if len(y_true) == 0:
        return float("nan")
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        count = int(mask.sum())
        if count:
            ece += count / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _matplotlib():
    """Agg バックエンドの pyplot を返す無ければ ``None``"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001
        logger.info("matplotlib unavailable, skipping figure: %s", exc)
        return None


def plot_roc(df: pd.DataFrame, classes: List[str], out_png: str) -> bool:
    """per-class OvR の ROC 曲線（と AUC）を描く成功で ``True``"""
    plt = _matplotlib()
    if plt is None:
        return False
    y_true, y_prob = _y_true_prob(df)
    fig, ax = plt.subplots()
    for i, name in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def plot_pr(df: pd.DataFrame, classes: List[str], out_png: str) -> bool:
    """per-class OvR の PR 曲線（と AUPRC）を描く成功で ``True``"""
    plt = _matplotlib()
    if plt is None:
        return False
    y_true, y_prob = _y_true_prob(df)
    fig, ax = plt.subplots()
    for i, name in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, i])
        ax.plot(recall, precision, label=f"{name} (AUPRC={auc(recall, precision):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (one-vs-rest)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def plot_calibration(
    df: pd.DataFrame, out_png: str, n_bins: int = DEFAULT_N_BINS
) -> bool:
    """信頼度ビンごとの精度（reliability diagram）を描く成功で ``True``"""
    plt = _matplotlib()
    if plt is None:
        return False
    y_true, y_prob = _y_true_prob(df)
    conf = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum():
            centers.append((bins[i] + bins[i + 1]) / 2.0)
            accs.append(correct[mask].mean())
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect")
    ax.plot(centers, accs, "o-", label=f"ECE={compute_ece(df, n_bins):.3f}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def _per_fold_values(cv: Dict[str, Any], split: str, metric: str) -> List[float]:
    """cv_summary の per_fold から ``metric`` の値列を取り出す"""
    folds = cv.get(split, {}).get("per_fold", [])
    return [m[metric] for m in folds if metric in m]


def compare_combos(
    cv_a: Dict[str, Any],
    cv_b: Dict[str, Any],
    split: str,
    metric: str,
    n_train: int,
    n_test: int,
) -> Dict[str, Any]:
    """2 combo の同指標を Wilcoxon と Nadeau-Bengio 補正 t で比較する"""
    a = _per_fold_values(cv_a, split, metric)
    b = _per_fold_values(cv_b, split, metric)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diffs = [a[i] - b[i] for i in range(n)]
    return {
        "metric": metric,
        "split": split,
        "mean_a": float(np.mean(a)) if a else float("nan"),
        "mean_b": float(np.mean(b)) if b else float("nan"),
        "wilcoxon": wilcoxon_signed_rank(a, b),
        "nadeau_bengio": nadeau_bengio_corrected_t(diffs, n_train, n_test),
    }
