"""``foveamil-ablation`` ― sweep 出力をアブレーション表に集計する CLI

1 つ以上の sweep ``--out`` ルートを受け，combo ごとの手法タグと指標集計を読み，倍率
レジームごとに多倍率ベースラインとの差分 Δ を付けた markdown 表を標準出力（と任意の
ファイル）へ書く``--baseline`` 指定時は対応 fold 差から NB 補正 t の p と多重比較
補正後 p を付け，``--metric group_f1 --group-classes ...`` で group-F1 を集計できる
学習はしない（保存済みの ``cv_summary.json`` を読むだけ）
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from foveamil.evaluation.ablation import (
    BASELINE_LABEL,
    GROUP_F1_METRIC,
    collect_ablation,
    collect_ablation_rows,
    compare_to_baseline,
    format_markdown,
    format_markdown_compare,
)
from foveamil.evaluation.stats import ADJUST_HOLM, ADJUST_METHODS

# 既定の集計指標
DEFAULT_METRIC = "weighted_f1"
# 既定の集計 split
DEFAULT_SPLIT = "test"


def _parse_group_classes(values: Optional[Sequence[str]]) -> List[int]:
    """``--group-classes`` のクラス index 列を int に変換する"""
    return [int(v) for v in (values or [])]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarise sweep outputs into an ablation table."
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="One or more sweep --out roots.",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        help=f"Metric to tabulate ('{GROUP_F1_METRIC}' for group-F1).",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help="Split to read (test / val)."
    )
    parser.add_argument(
        "--group-classes", nargs="+", default=None,
        help=f"Class indices for '{GROUP_F1_METRIC}' (unweighted mean F1).",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Baseline label for Δ / corrected-t p / adjusted p columns "
        f"(e.g. '{BASELINE_LABEL}').",
    )
    parser.add_argument(
        "--n-train", type=int, default=None,
        help="Per-fold train size (required with --baseline).",
    )
    parser.add_argument(
        "--n-test", type=int, default=None,
        help="Per-fold test size (required with --baseline).",
    )
    parser.add_argument(
        "--adjust", default=ADJUST_HOLM, choices=list(ADJUST_METHODS),
        help="Multiple-comparison correction for the p column.",
    )
    parser.add_argument(
        "--out", default=None, help="Optional path to write the markdown table."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    group_classes = _parse_group_classes(args.group_classes)

    if args.baseline:
        if args.n_train is None or args.n_test is None:
            raise SystemExit("--baseline requires --n-train and --n-test")
        rows = collect_ablation_rows(
            args.inputs, args.metric, args.split, group_classes=group_classes
        )
        enriched = compare_to_baseline(
            rows, args.n_train, args.n_test,
            baseline_label=args.baseline, adjust_method=args.adjust,
        )
        table = format_markdown_compare(
            enriched, args.metric, args.split, adjust_method=args.adjust
        )
    elif args.metric == GROUP_F1_METRIC:
        # group-F1 は aggregate に無いため per_fold から集計する（Δ/p は付けない）
        rows = collect_ablation_rows(
            args.inputs, args.metric, args.split, group_classes=group_classes
        )
        for row in rows:
            row.setdefault("delta", None)
            row.setdefault("pvalue", float("nan"))
            row.setdefault("pvalue_adj", float("nan"))
        table = format_markdown_compare(
            rows, args.metric, args.split, adjust_method=args.adjust
        )
    else:
        rows = collect_ablation(args.inputs, args.metric, args.split)
        table = format_markdown(rows, args.metric, args.split)

    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
