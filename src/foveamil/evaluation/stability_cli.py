"""``foveamil-stability`` ― 学習履歴から学習の安定性を診断する CLI

combo ディレクトリ（``--in``）配下の各 fold の ``history.csv`` を読み，終盤の検証指標
標準偏差（振動）・``val_loss`` 最小後の上昇量（過学習）・best epoch を fold 平均で
まとめて JSON へ書く``--compare`` に別 combo を渡すと，指定指標の per-fold std の
分散比をブートストラップ（決定的シード）で区間推定する学習はしない
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Optional, Sequence

from foveamil.evaluation.stability import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOT,
    DEFAULT_TAIL,
    combo_stability,
    per_fold_tail_std,
    variance_ratio_bootstrap,
)

logger = logging.getLogger(__name__)

# 既定の振動指標
DEFAULT_METRIC = "weighted_f1"
# 要約 JSON のファイル名
SUMMARY_JSON = "stability.json"
# 出力サブディレクトリ既定名
DEFAULT_REPORT_DIR = "stability"
# ブートストラップ既定シード
DEFAULT_SEED = 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-stability`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-stability",
        description="Diagnose training stability (tail oscillation, post-min "
        "val_loss rise, best epoch) from saved history.csv (no retraining).",
    )
    parser.add_argument(
        "--in", dest="input", required=True,
        help="Combo dir containing fold*/history.csv.",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        help="Validation metric for oscillation (val_ prefix optional).",
    )
    parser.add_argument(
        "--tail", type=int, default=DEFAULT_TAIL,
        help="Number of final epochs for oscillation std.",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Second combo dir to compare per-fold std variance ratio against.",
    )
    parser.add_argument(
        "--n-boot", type=int, default=DEFAULT_N_BOOT,
        help="Bootstrap resamples for the variance ratio.",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Significance level for the variance ratio CI.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Bootstrap seed (deterministic).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output dir (default: {in}/stability).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """安定性診断の本体"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    out_dir = args.out or os.path.join(args.input, DEFAULT_REPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    summary = combo_stability(args.input, args.metric, tail=args.tail)
    summary["combo_dir"] = os.path.basename(os.path.normpath(args.input))

    if args.compare:
        a = per_fold_tail_std(args.input, args.metric, tail=args.tail)
        b = per_fold_tail_std(args.compare, args.metric, tail=args.tail)
        summary["compare"] = {
            "combo_dir": os.path.basename(os.path.normpath(args.compare)),
            "variance_ratio": variance_ratio_bootstrap(
                a, b, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed
            ),
        }

    summary_path = os.path.join(out_dir, SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("wrote stability summary to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-stability`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
