"""``foveamil-ablation`` ― sweep 出力をアブレーション表に集計する CLI

1 つ以上の sweep ``--out`` ルートを受け，combo ごとの手法タグと指標集計を読み，倍率
レジームごとに多倍率ベースラインとの差分 Δ を付けた markdown 表を標準出力（と任意の
ファイル）へ書く学習はしない（保存済みの ``cv_summary.json`` を読むだけ）
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from foveamil.evaluation.ablation import collect_ablation, format_markdown

# 既定の集計指標
DEFAULT_METRIC = "weighted_f1"
# 既定の集計 split
DEFAULT_SPLIT = "test"


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
        "--metric", default=DEFAULT_METRIC, help="Metric to tabulate."
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help="Split to read (test / val)."
    )
    parser.add_argument(
        "--out", default=None, help="Optional path to write the markdown table."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = collect_ablation(args.inputs, args.metric, args.split)
    table = format_markdown(rows, args.metric, args.split)
    print(table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
