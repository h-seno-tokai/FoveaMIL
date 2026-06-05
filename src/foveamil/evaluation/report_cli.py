"""``foveamil-eval`` コマンド

sweep の出力（``--in``）を読み，再学習なしで ROC/PR/キャリブレーション図・combo 間
有意差検定・人間可読レポートを生成する``--compare a:b`` で combo 名の対を検定する
selection は val で選び test を報告する方針を踏襲し，test@best-test は oracle 上限と
して扱う出力は ``--out``（既定 ``{in}/report/``）へ書く
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from foveamil.evaluation.report import (
    CV_SUMMARY_JSON,
    RUN_META_JSON,
    SWEEP_SUMMARY_JSON,
    compare_combos,
    compute_ece,
    plot_calibration,
    plot_pr,
    plot_roc,
    pool_predictions,
)

logger = logging.getLogger(__name__)

# 出力サブディレクトリ既定名
DEFAULT_REPORT_DIR = "report"
# 既定の selection/報告指標
DEFAULT_METRIC = "macro_auc"
# 図のファイル名テンプレート
ROC_PNG = "roc_{combo}.png"
PR_PNG = "pr_{combo}.png"
CALIBRATION_PNG = "calibration_{combo}.png"
# 検定結果のファイル名
SIGNIFICANCE_JSON_TEMPLATE = "significance_{split}.json"
SIGNIFICANCE_MD_TEMPLATE = "significance_{split}.md"
# レポート本文のファイル名
REPORT_MD = "report.md"


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fold_names(combo_dir: str) -> List[str]:
    """combo 直下の ``fold*`` ディレクトリ名を番号順に返す"""
    dirs = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(combo_dir, "fold*"))
        if os.path.isdir(p)
    ]
    return sorted(dirs, key=lambda d: int("".join(ch for ch in d if ch.isdigit()) or 0))


def _classes_and_counts(
    combo_dir: str, fold_names: List[str]
) -> Tuple[List[str], int, int]:
    """先頭 fold の run_meta から class 名・train/test サンプル数を得る"""
    classes: List[str] = []
    n_train = n_test = 0
    for name in fold_names:
        meta = _load_json(os.path.join(combo_dir, name, RUN_META_JSON))
        if not meta:
            continue
        data = meta.get("data", {})
        classes = data.get("classes") or classes
        breakdown = data.get("class_breakdown", {})
        n_train = sum(breakdown.get("train", {}).values())
        n_test = sum(breakdown.get("test", {}).values())
        break
    return classes, int(n_train), int(n_test)


def _combo_entries(in_root: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """sweep_summary から (best_by_val, combos) を得る無ければ単一 combo 扱い"""
    summary = _load_json(os.path.join(in_root, SWEEP_SUMMARY_JSON))
    if summary:
        return summary.get("best_by_val"), summary.get("combos", [])
    name = os.path.basename(os.path.normpath(in_root))
    single = {"name": name, "out_dir": in_root}
    return single, [single]


def _combo_dir(in_root: str, entry: Dict[str, Any]) -> str:
    """combo エントリから出力ディレクトリを解決する"""
    out_dir = entry.get("out_dir")
    if out_dir and os.path.isdir(out_dir):
        return out_dir
    return os.path.join(in_root, entry["name"])


def _figures_for(
    in_root: str, entry: Dict[str, Any], split: str, out_dir: str, n_bins: int,
    make_plots: bool,
) -> Dict[str, Any]:
    """1 combo の予測を pool し図・ECE を作る"""
    combo_dir = _combo_dir(in_root, entry)
    fold_names = _fold_names(combo_dir)
    df = pool_predictions(combo_dir, split, fold_names)
    if df is None or not len(df):
        logger.warning("no %s predictions for combo %s", split, entry["name"])
        return {"ece": None, "figures": []}
    classes, _, _ = _classes_and_counts(combo_dir, fold_names)
    if not classes:
        n_cls = len([c for c in df.columns if c.startswith("prob_")])
        classes = [str(i) for i in range(n_cls)]

    ece = compute_ece(df, n_bins)
    figures: List[str] = []
    if make_plots:
        roc_png = os.path.join(out_dir, ROC_PNG.format(combo=entry["name"]))
        pr_png = os.path.join(out_dir, PR_PNG.format(combo=entry["name"]))
        cal_png = os.path.join(out_dir, CALIBRATION_PNG.format(combo=entry["name"]))
        if plot_roc(df, classes, roc_png):
            figures.append(os.path.basename(roc_png))
        if plot_pr(df, classes, pr_png):
            figures.append(os.path.basename(pr_png))
        if plot_calibration(df, cal_png, n_bins):
            figures.append(os.path.basename(cal_png))
    return {"ece": ece, "figures": figures, "n_samples": int(len(df))}


def _run_comparisons(
    in_root: str, entries: List[Dict[str, Any]], compares: List[str],
    split: str, metric: str,
) -> List[Dict[str, Any]]:
    """``--compare a:b`` の各対を検定する"""
    by_name = {e["name"]: e for e in entries}
    results: List[Dict[str, Any]] = []
    for pair in compares:
        if ":" not in pair:
            logger.warning("ignoring malformed --compare %r (need a:b)", pair)
            continue
        name_a, name_b = pair.split(":", 1)
        if name_a not in by_name or name_b not in by_name:
            logger.warning("compare combo not found: %s", pair)
            continue
        dir_a = _combo_dir(in_root, by_name[name_a])
        dir_b = _combo_dir(in_root, by_name[name_b])
        cv_a = _load_json(os.path.join(dir_a, CV_SUMMARY_JSON))
        cv_b = _load_json(os.path.join(dir_b, CV_SUMMARY_JSON))
        if not cv_a or not cv_b:
            logger.warning("missing cv_summary for %s", pair)
            continue
        _, n_train, n_test = _classes_and_counts(dir_a, _fold_names(dir_a))
        result = compare_combos(cv_a, cv_b, split, metric, n_train, n_test)
        result["a"] = name_a
        result["b"] = name_b
        results.append(result)
    return results


def _agg_line(aggregate: Optional[Dict[str, Any]], metric: str) -> str:
    """集計から ``mean±std (95% CI)`` 表記を作る"""
    if not aggregate or metric not in aggregate:
        return "-"
    s = aggregate[metric]
    ci = ""
    if s.get("ci_t_low") is not None and s.get("ci_t_low") == s.get("ci_t_low"):
        ci = f"（95%CI {s['ci_t_low']:.4f}–{s['ci_t_high']:.4f}）"
    return f"{s['mean']:.4f}±{s['std']:.4f}{ci}"


def _build_markdown(
    best: Optional[Dict[str, Any]], metric: str, split: str,
    figures: Dict[str, Any], comparisons: List[Dict[str, Any]],
) -> str:
    """レポート md を組み立てる（内部パスを載せない）"""
    lines = [
        "# 評価レポート",
        "",
        "model selection は **validation で選び test を報告**（test@best-test は "
        "oracle 上限であり報告には用いない）",
        "",
    ]
    if best:
        lines += [
            "## 最良 combo（val 選定）",
            "",
            f"- combo: `{best.get('name')}`",
            f"- val {metric}: {_agg_line(best.get('val'), metric)}",
            f"- **test {metric}: {_agg_line(best.get('test'), metric)}（報告値）**",
            f"- test ECE: {figures.get('ece'):.4f}" if figures.get("ece") is not None
            else "- test ECE: -",
            "",
        ]
        if figures.get("figures"):
            lines.append("## 図")
            lines.append("")
            for png in figures["figures"]:
                lines.append(f"![{png}]({png})")
            lines.append("")
    if comparisons:
        lines += [
            f"## combo 間有意差（{split}・{metric}）",
            "",
            "| A | B | mean A | mean B | Wilcoxon p | Nadeau-Bengio p |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in comparisons:
            wp = c["wilcoxon"]["pvalue"]
            nbp = c["nadeau_bengio"]["pvalue"]
            lines.append(
                f"| {c['a']} | {c['b']} | {c['mean_a']:.4f} | {c['mean_b']:.4f} | "
                f"{wp:.4g} | {nbp:.4g} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """レポート生成本体"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    out_dir = args.out or os.path.join(args.input, DEFAULT_REPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    best, entries = _combo_entries(args.input)
    targets = entries if args.all_combos else ([best] if best else [])

    best_figures: Dict[str, Any] = {"ece": None, "figures": []}
    for entry in targets:
        if not entry:
            continue
        figures = _figures_for(
            args.input, entry, args.split, out_dir, args.bins, not args.no_plots
        )
        if best and entry.get("name") == best.get("name"):
            best_figures = figures

    comparisons = _run_comparisons(
        args.input, entries, args.compare or [], args.split, args.metric
    )
    if comparisons:
        sig_json = os.path.join(out_dir, SIGNIFICANCE_JSON_TEMPLATE.format(split=args.split))
        with open(sig_json, "w", encoding="utf-8") as handle:
            json.dump(comparisons, handle, ensure_ascii=False, indent=2)

    md = _build_markdown(best, args.metric, args.split, best_figures, comparisons)
    report_path = os.path.join(out_dir, REPORT_MD)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    logger.info("wrote report to %s", report_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-eval`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-eval",
        description="Generate ROC/PR/calibration figures, combo significance tests "
        "and a markdown report from a sweep output (no retraining).",
    )
    parser.add_argument("--in", dest="input", required=True, help="Sweep --out root.")
    parser.add_argument(
        "--out", default=None, help="Report output dir (default: {in}/report)."
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test", "train"],
        help="Split to report (default: test).",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC, help="Selection/report metric."
    )
    parser.add_argument(
        "--compare", action="append", default=None, metavar="A:B",
        help="Compare two combos by name (may be repeated).",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Calibration bins (default: 10)."
    )
    parser.add_argument(
        "--all-combos", action="store_true",
        help="Make figures for every combo (default: best_by_val only).",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figures (metrics/report only)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-eval`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
