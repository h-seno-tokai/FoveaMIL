"""``foveamil-curves`` ― 保存済み学習履歴・CV 集計から学習曲線と指標要約図を作る CLI

sweep 出力ルート（``--in``）から combo を解決し，指定 combo の ``fold*/history.csv`` を
読んで per-epoch 検証指標の fold 平均±帯と best epoch 標示を描く複数 combo を
``--combo`` 複数指定すると重ね描き比較になる``cv_summary.json`` の per-fold 指標を
二次利用し，combo 横断の mean±CI 棒図とクラス部分集合の per-class F1 棒図も作る
学習はせず matplotlib 不在なら図を省き JSON のみ書く
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from foveamil.evaluation.curves import (
    BAND_KINDS,
    BAND_MINMAX,
    epoch_curve,
    per_class_f1_bars,
    plot_curves,
    plot_per_class_f1,
    plot_summary_bars,
    summary_bars,
)

logger = logging.getLogger(__name__)

# sweep 出力のファイル名
SWEEP_SUMMARY_JSON = "sweep_summary.json"
CV_SUMMARY_JSON = "cv_summary.json"
# 出力サブディレクトリ既定名
DEFAULT_REPORT_DIR = "curves"
# 既定の per-epoch 検証指標
DEFAULT_METRIC = "macro_f1"
# 集計対象の既定 split
DEFAULT_SPLIT = "test"
# 出力ファイル名テンプレート
CURVES_PNG = "curves_{metric}.png"
SUMMARY_PNG = "summary_{metric}.png"
PER_CLASS_PNG = "per_class_f1_{combo}.png"
SUMMARY_JSON = "curves.json"


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _combo_entries(in_root: str) -> List[Dict[str, Any]]:
    """sweep_summary から combo エントリ列を得る無ければ単一 combo 扱い"""
    summary = _load_json(os.path.join(in_root, SWEEP_SUMMARY_JSON))
    if summary and summary.get("combos"):
        return summary["combos"]
    name = os.path.basename(os.path.normpath(in_root))
    return [{"name": name, "out_dir": in_root}]


def _combo_dir(in_root: str, entry: Dict[str, Any]) -> str:
    """combo エントリから出力ディレクトリを解決する"""
    out_dir = entry.get("out_dir")
    if out_dir and os.path.isdir(out_dir):
        return out_dir
    candidate = os.path.join(in_root, entry["name"])
    if os.path.isdir(candidate):
        return candidate
    return out_dir or candidate


def _resolve_combos(
    in_root: str, names: Optional[Sequence[str]]
) -> List[Tuple[str, str]]:
    """指定 combo 名（無ければ全 combo）を ``(label, dir)`` の列に解決する"""
    entries = _combo_entries(in_root)
    by_name = {e["name"]: e for e in entries}
    if not names:
        return [(e["name"], _combo_dir(in_root, e)) for e in entries]
    resolved: List[Tuple[str, str]] = []
    for name in names:
        entry = by_name.get(name)
        if entry is None:
            # combo 名がサブディレクトリとして直接存在する場合も許す
            direct = os.path.join(in_root, name)
            if os.path.isdir(direct):
                resolved.append((name, direct))
            else:
                logger.warning("combo not found: %s", name)
            continue
        resolved.append((name, _combo_dir(in_root, entry)))
    return resolved


def _per_fold_metrics(combo_dir: str, split: str) -> List[Dict[str, float]]:
    """combo の ``cv_summary.json`` から指定 split の per-fold 指標列を得る"""
    cv = _load_json(os.path.join(combo_dir, CV_SUMMARY_JSON))
    if not cv:
        return []
    return cv.get(split, {}).get("per_fold", [])


def _parse_classes(spec: Optional[str]) -> List[int]:
    """``"4,5,6"`` 形式のクラス集合指定を整数列に変換する"""
    if not spec:
        return []
    return [int(tok) for tok in spec.split(",") if tok.strip()]


def run(args: argparse.Namespace) -> int:
    """学習曲線・指標要約図の生成本体"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    out_dir = args.out or os.path.join(args.input, DEFAULT_REPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    combos = _resolve_combos(args.input, args.combo)
    if not combos:
        logger.warning("no combos resolved from %s", args.input)

    make_plots = not args.no_plots
    payload: Dict[str, Any] = {
        "input": os.path.basename(os.path.normpath(args.input)),
        "metric": args.metric,
        "split": args.split,
        "band": args.band,
        "combos": [label for label, _ in combos],
        "figures": [],
    }

    curves = [
        (label, epoch_curve(combo_dir, args.metric, band=args.band))
        for label, combo_dir in combos
    ]
    payload["epoch_curve"] = {label: c for label, c in curves}
    if make_plots:
        curves_png = os.path.join(out_dir, CURVES_PNG.format(metric=args.metric))
        if plot_curves(
            curves, args.metric, curves_png, band=args.band,
            show_best=not args.no_best,
        ):
            payload["figures"].append(os.path.basename(curves_png))

    per_fold_by_combo = [
        (label, _per_fold_metrics(combo_dir, args.split))
        for label, combo_dir in combos
    ]
    records = summary_bars(per_fold_by_combo, args.summary_metric)
    payload["summary"] = records
    if make_plots:
        summary_png = os.path.join(
            out_dir, SUMMARY_PNG.format(metric=args.summary_metric)
        )
        if plot_summary_bars(records, args.summary_metric, summary_png):
            payload["figures"].append(os.path.basename(summary_png))

    classes = _parse_classes(args.classes)
    if classes:
        payload["per_class_f1"] = {}
        for label, per_fold in per_fold_by_combo:
            bars = per_class_f1_bars(per_fold, classes)
            payload["per_class_f1"][label] = bars
            if make_plots:
                pc_png = os.path.join(out_dir, PER_CLASS_PNG.format(combo=label))
                if plot_per_class_f1(bars, pc_png, label=label):
                    payload["figures"].append(os.path.basename(pc_png))

    summary_path = os.path.join(out_dir, SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("wrote curves summary to %s", summary_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-curves`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-curves",
        description="Plot training curves (per-epoch validation metric, fold "
        "mean ± band, best epoch) and metric summary bars from saved history "
        "and cv_summary (no retraining).",
    )
    parser.add_argument("--in", dest="input", required=True, help="Sweep --out root.")
    parser.add_argument(
        "--out", default=None, help="Output dir (default: {in}/curves)."
    )
    parser.add_argument(
        "--combo", action="append", default=None, metavar="NAME",
        help="Combo name to include (may be repeated; default: all combos).",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        help="Per-epoch validation metric for curves (val_ prefix optional).",
    )
    parser.add_argument(
        "--band", default=BAND_MINMAX, choices=list(BAND_KINDS),
        help="Band kind around the fold mean (minmax or std).",
    )
    parser.add_argument(
        "--no-best", action="store_true",
        help="Do not mark the best epoch (min fold-mean val_loss).",
    )
    parser.add_argument(
        "--summary-metric", default=DEFAULT_METRIC,
        help="Metric for the cross-combo mean±CI summary bars.",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, choices=["val", "test", "train"],
        help="Split for summary/per-class bars (default: test).",
    )
    parser.add_argument(
        "--classes", default=None, metavar="i,j,k",
        help="Comma-separated class indices for the per-class F1 subset bars.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figures (JSON only)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-curves`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
