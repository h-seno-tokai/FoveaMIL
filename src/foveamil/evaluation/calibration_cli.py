"""``foveamil-calibrate`` コマンド

sweep の出力（``--in``）から combo の予測を pool し，pooled-val で temperature scaling と
クラス別ロジット補正 δ_c を当てて test に適用し，較正前後（baseline→T→T+δ_c）の指標を出す
学習・モデル再推論はしない（保存済み ``predictions_{split}.csv`` を読むだけ）

selection は val で選び test を報告する方針を踏襲し，較正パラメタも val のみで当てる
（過適合回避のため全 fold の val をプールして 1 標本で当てる）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from foveamil.evaluation.calibration import calibrate_val_to_test
from foveamil.evaluation.report import pool_predictions
from foveamil.evaluation.report_cli import (
    _classes_and_counts,
    _combo_dir,
    _combo_entries,
    _fold_names,
)

logger = logging.getLogger(__name__)

# 出力サブディレクトリ・ファイル名の既定
DEFAULT_CALIBRATION_DIR = "calibration"
CALIBRATION_JSON = "calibration.json"
CALIBRATION_MD = "calibration.md"
# val 固定（較正は必ず val で当てる）
VAL_SPLIT = "val"


def _parse_group_classes(value: Optional[str]) -> Optional[List[int]]:
    """``--group-classes 4,5,6`` を int 列に解す未指定で ``None``"""
    if not value:
        return None
    return [int(x) for x in value.split(",") if x.strip() != ""]


def _calibrate_combo(
    in_root: str,
    entry: Dict[str, Any],
    test_split: str,
    group_classes: Optional[List[int]],
    l2: float,
    top_confusions: int,
) -> Optional[Dict[str, Any]]:
    """1 combo の pooled-val→test 較正結果を返す予測が無ければ ``None``"""
    combo_dir = _combo_dir(in_root, entry)
    fold_names = _fold_names(combo_dir)
    val_df = pool_predictions(combo_dir, VAL_SPLIT, fold_names)
    test_df = pool_predictions(combo_dir, test_split, fold_names)
    if val_df is None or test_df is None or not len(val_df) or not len(test_df):
        logger.warning("no val/test predictions for combo %s", entry["name"])
        return None
    classes, _, _ = _classes_and_counts(combo_dir, fold_names)
    result = calibrate_val_to_test(
        val_df, test_df, group_classes=group_classes, l2=l2,
        top_confusions=top_confusions,
    )
    result["combo"] = entry["name"]
    if classes:
        result["classes"] = classes
    return result


def _format_metric(value: Optional[float]) -> str:
    """指標値を 4 桁で表記する None/nan は ``-``"""
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.4f}"


def _build_markdown(
    results: List[Dict[str, Any]], group_classes: Optional[List[int]]
) -> str:
    """較正結果の md を組み立てる（内部パスを載せない）"""
    objective = "group-F1" if group_classes else "macro-F1"
    lines = [
        "# 事後較正レポート",
        "",
        "pooled-val で temperature scaling と クラス別ロジット補正 δ_c を当て test に適用した"
        "（学習なし・保存済み予測のみ）",
        f"δ_c の最適化目的: **{objective}**較正パラメタは過適合回避のため全 fold の val を"
        "プールして 1 標本で当てる",
        "",
    ]
    for r in results:
        stages = r["stages"]
        base = stages.get("baseline", {})
        temp = stages.get("temperature", {})
        full = stages.get("temperature_delta", {})
        marginal = r.get("marginal", {})
        lines += [
            f"## combo `{r['combo']}`",
            "",
            f"- temperature T: {r['temperature']:.4f}（logit 源: {r['logit_source']}）",
            f"- val/test 標本数: {r['n_val']} / {r['n_test']}",
            "",
            "| 段階 | macro-F1 | group-F1 |",
            "| --- | --- | --- |",
            f"| baseline | {_format_metric(base.get('macro_f1'))} | "
            f"{_format_metric(base.get('group_f1'))} |",
            f"| +temperature | {_format_metric(temp.get('macro_f1'))} | "
            f"{_format_metric(temp.get('group_f1'))} |",
            f"| +temperature+δ | {_format_metric(full.get('macro_f1'))} | "
            f"{_format_metric(full.get('group_f1'))} |",
            "",
        ]
        mm = marginal.get("macro_f1", {})
        if mm:
            lines += [
                "段階寄与（macro-F1 限界効用）: "
                f"T={_format_metric(mm.get('temperature'))} / "
                f"δ={_format_metric(mm.get('delta'))} / "
                f"合計={_format_metric(mm.get('total'))}",
                "",
            ]
        mg = marginal.get("group_f1", {})
        if mg:
            lines += [
                "段階寄与（group-F1 限界効用）: "
                f"T={_format_metric(mg.get('temperature'))} / "
                f"δ={_format_metric(mg.get('delta'))} / "
                f"合計={_format_metric(mg.get('total'))}",
                "",
            ]
        # 少数クラス recall の before/after
        base_rec = base.get("minority_recall", {})
        full_rec = full.get("minority_recall", {})
        if base_rec:
            lines += [
                "少数クラス recall（baseline → +δ）:",
                "",
                "| class | baseline | +δ |",
                "| --- | --- | --- |",
            ]
            for c in sorted(base_rec):
                lines.append(
                    f"| {c} | {_format_metric(base_rec.get(c))} | "
                    f"{_format_metric(full_rec.get(c))} |"
                )
            lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """較正レポート生成本体"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    out_dir = args.out or os.path.join(args.input, DEFAULT_CALIBRATION_DIR)
    os.makedirs(out_dir, exist_ok=True)

    group_classes = _parse_group_classes(args.group_classes)
    best, entries = _combo_entries(args.input)
    targets = entries if args.all_combos else ([best] if best else [])

    results: List[Dict[str, Any]] = []
    for entry in targets:
        if not entry:
            continue
        result = _calibrate_combo(
            args.input, entry, args.split, group_classes, args.l2,
            args.top_confusions,
        )
        if result is not None:
            results.append(result)

    json_path = os.path.join(out_dir, CALIBRATION_JSON)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    md = _build_markdown(results, group_classes)
    md_path = os.path.join(out_dir, CALIBRATION_MD)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(md)
    logger.info("wrote calibration report to %s", md_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-calibrate`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-calibrate",
        description="Fit temperature scaling and per-class logit deltas on pooled "
        "validation predictions and apply to test (no retraining); report "
        "before/after metrics.",
    )
    parser.add_argument("--in", dest="input", required=True, help="Sweep --out root.")
    parser.add_argument(
        "--out", default=None,
        help="Report output dir (default: {in}/calibration).",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "train"],
        help="Split to apply calibration to and report (default: test).",
    )
    parser.add_argument(
        "--group-classes", default=None, metavar="C1,C2,...",
        help="Optimize/report group-F1 over these class indices (default: macro-F1).",
    )
    parser.add_argument(
        "--l2", type=float, default=None,
        help="L2 strength for per-class delta (default: module default).",
    )
    parser.add_argument(
        "--top-confusions", type=int, default=None,
        help="Top confusion outflows per minority class (default: module default).",
    )
    parser.add_argument(
        "--all-combos", action="store_true",
        help="Calibrate every combo (default: best_by_val only).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-calibrate`` コンソールスクリプトのエントリポイント"""
    from foveamil.evaluation.calibration import (
        DEFAULT_DELTA_L2,
        DEFAULT_TOP_CONFUSIONS,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.l2 is None:
        args.l2 = DEFAULT_DELTA_L2
    if args.top_confusions is None:
        args.top_confusions = DEFAULT_TOP_CONFUSIONS
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
