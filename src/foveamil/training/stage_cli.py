"""``foveamil-stage`` コマンド

学習前に一度だけ手動実行し，対象スライドの特徴量セットを NAS からローカル SSD へ
一括コピーする``FeatureStager`` を用い，正準レイアウト
``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` の対象ファイルをキャッシュへ複製する
対象スライドは ``--splits-dir``（配下 ``split_fold*.csv`` の train/val/test 全 slide の
和集合）または ``--slides``（``slide_id`` 列の CSV / 1 行 1 個のテキスト）で指定する
既存ファイルは再利用するため再実行は冪等staged root をログに表示して返す
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from typing import List, Optional, Sequence

import pandas as pd

from foveamil.cohort.labels import load_slide_ids
from foveamil.training.accessor import FEATURE_TYPES
from foveamil.training.staging import (
    STAGE_DIR_ENV,
    STAGE_WORKERS_ENV,
    FeatureStager,
)

logger = logging.getLogger(__name__)

# 分割 CSV ファイル名の glob パターン
SPLIT_GLOB = "split_fold*.csv"
# 分割 CSV の対象列
SPLIT_COLUMNS = ("train", "val", "test")
# 環境変数も未設定のときの既定キャッシュ先
DEFAULT_CACHE_DIR = "/tmp/foveamil_feat_stage"


def _slides_from_splits(splits_dir: str) -> List[str]:
    """``--splits-dir`` 配下の全 ``split_fold*.csv`` の train/val/test の和集合を返す"""
    paths = sorted(glob.glob(os.path.join(splits_dir, SPLIT_GLOB)))
    if not paths:
        raise ValueError(f"no {SPLIT_GLOB} found under {splits_dir}")
    slides: set = set()
    for path in paths:
        df = pd.read_csv(path)
        for col in SPLIT_COLUMNS:
            if col in df.columns:
                slides.update(str(s) for s in df[col].dropna().tolist())
    return sorted(slides)


def run(args: argparse.Namespace) -> int:
    """対象特徴セットをステージし staged root を表示する"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.splits_dir:
        slide_ids = _slides_from_splits(args.splits_dir)
    else:
        slide_ids = sorted(load_slide_ids(args.slides))
    logger.info("target slides: %d", len(slide_ids))

    cache_dir = (
        args.cache_dir or os.environ.get(STAGE_DIR_ENV) or DEFAULT_CACHE_DIR
    )
    stager = FeatureStager(
        cache_dir=cache_dir, copy_workers=args.workers, store_fp16=args.fp16
    )
    staged_root = stager.stage_set(
        args.feature_root,
        args.encoder,
        args.magnifications,
        slide_ids,
        feature_type=args.feature_type,
    )
    logger.info("staged root: %s", staged_root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-stage`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-stage",
        description="Stage a feature set from NAS to local SSD before training "
        "(run once manually).",
    )
    parser.add_argument(
        "--feature-root",
        required=True,
        help="Feature root ({encoder}/{mag}x/{slide_id}.h5) on the NAS.",
    )
    parser.add_argument("--encoder", required=True, help="Encoder name.")
    parser.add_argument(
        "--magnifications",
        required=True,
        nargs="+",
        type=float,
        help="Magnifications to stage (e.g. 1.25 2.5 5.0).",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--splits-dir",
        help="Directory of split_fold*.csv; stages the union of "
        "train/val/test slides.",
    )
    src.add_argument(
        "--slides",
        help="CSV with a slide_id column, or a text file with one id per line.",
    )

    parser.add_argument(
        "--feature-type",
        choices=FEATURE_TYPES,
        default=None,
        help="Stage only this feature's datasets (cls/mean keep that feature "
        "+ coords, halving size); default copies the full h5.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Local SSD destination; falls back to env {STAGE_DIR_ENV} "
        f"then {DEFAULT_CACHE_DIR}.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel copy workers (processes); falls back to env "
        f"{STAGE_WORKERS_ENV} then the built-in default.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Store float feature datasets as float16 (coords keep their "
        "dtype), halving size; the loader upcasts to float32 on read. "
        "Effective only with --feature-type cls/mean (subset staging); "
        "no effect on a full/concat copy.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-stage`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
