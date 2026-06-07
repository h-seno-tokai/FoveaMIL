"""保存済み学習履歴と CV 集計から学習曲線・指標要約図を生成する

各 combo の ``fold*/history.csv`` を読み，per-epoch 検証指標の fold 平均±帯
（min-max または std）と ``val_loss`` 最小の best epoch 標示を描く複数 combo の
重ね描き比較もできる``cv_summary.json`` の per-fold 指標を二次利用して combo 横断の
mean±CI 棒図とクラス部分集合の per-class F1 棒図を描く学習や再推論はせず保存済み値を
読むだけで matplotlib が無ければ図を省き例外を投げない
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from foveamil.evaluation.group_metrics import class_f1_key, group_f1_summary
from foveamil.evaluation.stability import (
    EPOCH_COL,
    _by_epoch,
    _metric_column,
    fold_history_paths,
    load_history,
)

logger = logging.getLogger(__name__)

# 帯の種類
BAND_MINMAX = "minmax"
BAND_STD = "std"
BAND_KINDS = (BAND_MINMAX, BAND_STD)
# best epoch を測る検証損失の列名
VAL_LOSS_COL = "val_loss"
# 帯の不透明度
_BAND_ALPHA = 0.2
# best epoch 標示の縦線の不透明度
_BEST_LINE_ALPHA = 0.6
# 図の既定 dpi
_DEFAULT_DPI = 200

_NAN = float("nan")


def _matplotlib():
    """Agg バックエンドの pyplot を返す無ければ ``None``"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # noqa: BLE001 - 図は任意機能
        logger.info("matplotlib unavailable, skipping figure: %s", exc)
        return None


def _aligned_metric_matrix(
    histories: Sequence[pd.DataFrame], metric: str
) -> Tuple[np.ndarray, np.ndarray]:
    """fold 群の per-epoch 検証指標を共通エポックで揃えた行列を返す

    各 fold を ``epoch`` 昇順にし，全 fold に共通するエポック値の交差集合で揃える
    ``epoch`` 列が無い fold は行 index をエポックとみなす指標列を持つ fold のみ使う

    Args:
        histories: combo の各 fold の学習履歴
        metric: per-epoch で追う検証指標名（``val_`` 接頭辞は省略可）

    Returns:
        ``(epochs[E], values[F,E])``共通エポックが無ければ空配列の対
    """
    series: List[Tuple[np.ndarray, np.ndarray]] = []
    for history in histories:
        col = _metric_column(history, metric)
        if col is None:
            continue
        ordered = _by_epoch(history)
        if EPOCH_COL in ordered.columns:
            epochs = ordered[EPOCH_COL].to_numpy(dtype=float)
        else:
            epochs = np.arange(len(ordered), dtype=float)
        values = ordered[col].to_numpy(dtype=float)
        series.append((epochs, values))

    if not series:
        return np.empty(0), np.empty((0, 0))

    common = series[0][0]
    for epochs, _ in series[1:]:
        common = np.intersect1d(common, epochs)
    if common.size == 0:
        return np.empty(0), np.empty((0, 0))

    rows: List[np.ndarray] = []
    for epochs, values in series:
        index = {float(e): i for i, e in enumerate(epochs)}
        rows.append(np.array([values[index[float(e)]] for e in common], dtype=float))
    return common, np.vstack(rows)


def epoch_curve(
    combo_dir: str, metric: str, band: str = BAND_MINMAX
) -> Dict[str, Any]:
    """combo の per-epoch 検証指標を fold 平均±帯と best epoch でまとめる

    各 fold の ``history.csv`` を共通エポックで揃え，エポックごとに fold 平均と帯
    （``minmax`` なら最小・最大，``std`` なら平均±標準偏差）を計算する best epoch は
    fold 平均 ``val_loss`` が最小となるエポックとする履歴が読めなければ空にする

    Args:
        combo_dir: ``fold*/history.csv`` を含む combo ディレクトリ
        metric: per-epoch で追う検証指標名（``val_`` 接頭辞は省略可）
        band: 帯の種類（``"minmax"`` / ``"std"``）

    Returns:
        ``{"metric","band","n_folds","epochs","mean","low","high",
        "best_epoch","best_value"}``（配列は list）
    """
    if band not in BAND_KINDS:
        raise ValueError(f"band must be one of {BAND_KINDS}, got '{band}'")

    histories = [
        h for h in (load_history(p) for p in fold_history_paths(combo_dir))
        if h is not None and not h.empty
    ]
    epochs, matrix = _aligned_metric_matrix(histories, metric)
    empty = {
        "metric": metric,
        "band": band,
        "n_folds": 0,
        "epochs": [],
        "mean": [],
        "low": [],
        "high": [],
        "best_epoch": _NAN,
        "best_value": _NAN,
    }
    if matrix.size == 0:
        return empty

    mean = matrix.mean(axis=0)
    if band == BAND_MINMAX:
        low = matrix.min(axis=0)
        high = matrix.max(axis=0)
    else:
        std = matrix.std(axis=0)
        low = mean - std
        high = mean + std

    best_epoch_value, best_value = _best_epoch_from_loss(histories, epochs, mean)
    return {
        "metric": metric,
        "band": band,
        "n_folds": int(matrix.shape[0]),
        "epochs": epochs.tolist(),
        "mean": mean.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "best_epoch": best_epoch_value,
        "best_value": best_value,
    }


def _best_epoch_from_loss(
    histories: Sequence[pd.DataFrame], epochs: np.ndarray, metric_mean: np.ndarray
) -> Tuple[float, float]:
    """fold 平均 ``val_loss`` が最小のエポックと，そこでの指標平均値を返す

    各 fold の ``val_loss`` を共通エポックで揃えて平均し最小エポックを選ぶ
    ``val_loss`` 列が無い等で揃わなければ ``(nan, nan)``

    Args:
        histories: combo の各 fold の学習履歴
        epochs: 指標の共通エポック列
        metric_mean: 各共通エポックでの指標 fold 平均

    Returns:
        ``(best_epoch, best_value)``
    """
    loss_epochs, loss_matrix = _aligned_metric_matrix(histories, VAL_LOSS_COL)
    if loss_matrix.size == 0:
        return _NAN, _NAN
    loss_mean = loss_matrix.mean(axis=0)
    best_idx = int(np.argmin(loss_mean))
    best_epoch_value = float(loss_epochs[best_idx])
    pos = np.where(epochs == best_epoch_value)[0]
    best_value = float(metric_mean[pos[0]]) if pos.size else _NAN
    return best_epoch_value, best_value


def plot_curves(
    curves: Sequence[Tuple[str, Dict[str, Any]]],
    metric: str,
    out_png: str,
    band: str = BAND_MINMAX,
    show_best: bool = True,
) -> bool:
    """複数 combo の per-epoch 曲線を fold 平均±帯で重ね描きする

    各 combo を別色で描き，帯は半透明で塗る``show_best`` のとき各曲線の best epoch を
    縦線と点で標示する有効な曲線が無ければ図を作らず ``False``matplotlib 不在でも
    ``False``

    Args:
        curves: ``(label, epoch_curve の戻り値)`` の列
        metric: 軸ラベルに使う指標名
        out_png: 保存先 PNG パス
        band: 帯ラベルに使う種類名
        show_best: best epoch 標示の有無

    Returns:
        保存できたら ``True``
    """
    plt = _matplotlib()
    if plt is None:
        return False
    valid = [(label, c) for label, c in curves if c.get("epochs")]
    if not valid:
        logger.warning("no curve data to plot for metric %s", metric)
        return False

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    for i, (label, c) in enumerate(valid):
        color = cmap(i % cmap.N)
        epochs = np.asarray(c["epochs"], dtype=float)
        mean = np.asarray(c["mean"], dtype=float)
        low = np.asarray(c["low"], dtype=float)
        high = np.asarray(c["high"], dtype=float)
        ax.plot(epochs, mean, color=color, label=f"{label} (n={c['n_folds']})")
        ax.fill_between(epochs, low, high, color=color, alpha=_BAND_ALPHA)
        if show_best and not np.isnan(c["best_epoch"]):
            ax.axvline(
                c["best_epoch"], color=color, linestyle="--",
                linewidth=0.8, alpha=_BEST_LINE_ALPHA,
            )
            if not np.isnan(c["best_value"]):
                ax.plot([c["best_epoch"]], [c["best_value"]], "o", color=color)

    band_label = "min-max" if band == BAND_MINMAX else "±1 std"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"Validation {metric} (fold mean, band={band_label})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DEFAULT_DPI)
    plt.close(fig)
    return True


def summary_bars(
    combos: Sequence[Tuple[str, List[Dict[str, float]]]], metric: str
) -> List[Dict[str, Any]]:
    """combo 横断で指定指標の mean±CI を集計し棒図用のレコード列を返す

    各 combo の per-fold 指標から ``aggregate_folds_ci`` で mean/std/t-CI を求める
    指標を持たない combo は ``mean`` を ``nan`` にする

    Args:
        combos: ``(label, per_fold 指標辞書列)`` の列
        metric: 比較する指標名

    Returns:
        ``{"label","mean","std","ci_low","ci_high","n"}`` の列
    """
    from foveamil.training.cv import aggregate_folds_ci

    records: List[Dict[str, Any]] = []
    for label, per_fold in combos:
        agg = aggregate_folds_ci(per_fold).get(metric)
        if agg is None:
            records.append({
                "label": label, "mean": _NAN, "std": _NAN,
                "ci_low": _NAN, "ci_high": _NAN, "n": 0,
            })
            continue
        records.append({
            "label": label,
            "mean": agg["mean"],
            "std": agg["std"],
            "ci_low": agg["ci_t_low"],
            "ci_high": agg["ci_t_high"],
            "n": agg["n"],
        })
    return records


def plot_summary_bars(
    records: Sequence[Dict[str, Any]], metric: str, out_png: str
) -> bool:
    """combo 横断の mean±CI を棒図で描く

    CI 上下限が有効なら誤差棒は CI幅，無効なら std を使う有効な棒が無ければ ``False``
    matplotlib 不在でも ``False``

    Args:
        records: ``summary_bars`` の戻り値
        metric: 軸ラベルに使う指標名
        out_png: 保存先 PNG パス

    Returns:
        保存できたら ``True``
    """
    plt = _matplotlib()
    if plt is None:
        return False
    valid = [r for r in records if not np.isnan(r["mean"])]
    if not valid:
        logger.warning("no summary data to plot for metric %s", metric)
        return False

    labels = [r["label"] for r in valid]
    means = np.array([r["mean"] for r in valid], dtype=float)
    err = _error_bars(valid, means)
    x = np.arange(len(valid))

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=err, capsize=4, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across combos (mean ± 95% CI)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DEFAULT_DPI)
    plt.close(fig)
    return True


def _error_bars(
    records: Sequence[Dict[str, Any]], means: np.ndarray
) -> np.ndarray:
    """棒図の非対称誤差棒 ``[2,N]`` を作る（CI 有効なら CI幅，無効なら std）"""
    lower = np.empty(len(records), dtype=float)
    upper = np.empty(len(records), dtype=float)
    for i, r in enumerate(records):
        ci_low, ci_high = r["ci_low"], r["ci_high"]
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            lower[i] = max(means[i] - ci_low, 0.0)
            upper[i] = max(ci_high - means[i], 0.0)
        else:
            std = 0.0 if np.isnan(r["std"]) else r["std"]
            lower[i] = upper[i] = std
    return np.vstack([lower, upper])


def per_class_f1_bars(
    per_fold: List[Dict[str, float]], class_indices: Sequence[int]
) -> Dict[str, Any]:
    """指定クラス集合の per-class F1（fold 平均）と group-F1 を返す

    ``group_f1_summary`` を流用し，集合内各クラスの fold 平均 F1 と集合全体の
    非加重平均 group-F1 をまとめる

    Args:
        per_fold: fold ごとの指標辞書の列（``class_i_f1`` を含む）
        class_indices: 対象のクラス index 集合

    Returns:
        ``{"class_indices","per_class","group_mean","group_std","n"}``
    """
    summary = group_f1_summary(per_fold, class_indices)
    return {
        "class_indices": list(class_indices),
        "per_class": summary["per_class"],
        "group_mean": summary["mean"],
        "group_std": summary["std"],
        "n": summary["n"],
    }


def plot_per_class_f1(
    bars: Dict[str, Any], out_png: str, label: Optional[str] = None
) -> bool:
    """クラス部分集合の per-class F1 を棒図で描き group-F1 を水平線で示す

    有効な per-class 値が無ければ ``False``matplotlib 不在でも ``False``

    Args:
        bars: ``per_class_f1_bars`` の戻り値
        out_png: 保存先 PNG パス
        label: タイトルに添える combo 名

    Returns:
        保存できたら ``True``
    """
    plt = _matplotlib()
    if plt is None:
        return False
    items = [
        (i, v) for i, v in bars["per_class"].items() if not np.isnan(v)
    ]
    if not items:
        logger.warning("no per-class F1 to plot")
        return False

    indices = [i for i, _ in items]
    values = np.array([v for _, v in items], dtype=float)
    x = np.arange(len(items))

    fig, ax = plt.subplots()
    ax.bar(x, values, color="tab:green")
    if not np.isnan(bars["group_mean"]):
        ax.axhline(
            bars["group_mean"], color="black", linestyle="--", linewidth=0.8,
            label=f"group-F1={bars['group_mean']:.3f}",
        )
        ax.legend(fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([class_f1_key(i) for i in indices], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    title = "Per-class F1 (subset)"
    if label:
        title = f"{title} - {label}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=_DEFAULT_DPI)
    plt.close(fig)
    return True
