"""``foveamil-cohort`` コマンド

サブコマンド ``labels``（master ラベル表を対象クラス・対象症例に絞り込む）と
``splits``（ラベル表から層化 K-fold CV の分割 CSV を生成する）を提供する
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Optional, Sequence

from foveamil.cohort.labels import filter_labels, load_slide_ids, write_labels
from foveamil.cohort.splits import (
    DEFAULT_K,
    DEFAULT_SEED,
    make_cv_splits,
    write_split_csv,
)


def _class_ratio_str(labels: Sequence[str]) -> str:
    """ラベル列を ``クラス名 件数 (割合%)`` の要約文字列に整形する"""
    total = len(labels)
    counts = Counter(labels)
    parts = []
    for cls in sorted(counts):
        n = counts[cls]
        pct = 100.0 * n / total if total else 0.0
        parts.append(f"{cls} {n} ({pct:.1f}%)")
    return ", ".join(parts)


def _run_labels(args: argparse.Namespace) -> None:
    restrict_to = load_slide_ids(args.restrict_to) if args.restrict_to else None
    exclude = set(args.exclude) if args.exclude else None

    df = filter_labels(
        master_csv=args.input,
        classes=args.classes,
        restrict_to=restrict_to,
        exclude=exclude,
    )
    write_labels(df, args.output)

    print(f"Wrote {len(df)} rows to {args.output}")
    print(f"  classes: {_class_ratio_str(df['label'].tolist())}")


def _run_splits(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    k = args.k if args.k is not None else DEFAULT_K
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    splits = make_cv_splits(
        labels_csv=args.labels,
        k=k,
        val_frac=args.val_frac,
        seed=seed,
    )

    # 部分集合ごとのクラス比を表示するため slide_id -> label を一度だけ作る
    import pandas as pd

    df = pd.read_csv(args.labels)[["slide_id", "label"]].astype(str)
    label_of = dict(zip(df["slide_id"], df["label"]))

    print(f"Overall: {_class_ratio_str(df['label'].tolist())} (n={len(df)})")
    print(f"Generating {len(splits)} folds (k={k}, "
          f"val_frac={args.val_frac if args.val_frac is not None else '1/(k-1)'}, "
          f"seed={seed}) into {args.output_dir}")

    for split in splits:
        out = os.path.join(args.output_dir, f"split_fold{split['fold']}.csv")
        write_split_csv(split, out)
        print(f"  fold {split['fold']}: "
              f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
        for subset in ("train", "val", "test"):
            subset_labels = [label_of[s] for s in split[subset]]
            print(f"    {subset:>5}: {_class_ratio_str(subset_labels)}")


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-cohort`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-cohort",
        description="Cohort construction: label filtering and stratified CV splits.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_labels = sub.add_parser(
        "labels", help="Filter a master label table into a task-specific label CSV."
    )
    p_labels.add_argument("--input", required=True, help="Master CSV (slide_id,label).")
    p_labels.add_argument("--output", required=True, help="Output filtered CSV.")
    p_labels.add_argument(
        "--classes", required=True, nargs="+", help="Labels to keep (space separated)."
    )
    p_labels.add_argument(
        "--restrict-to",
        default=None,
        help="Text/CSV of slide_ids; keep only the intersection.",
    )
    p_labels.add_argument(
        "--exclude", nargs="*", default=None, help="slide_ids to drop."
    )
    p_labels.set_defaults(func=_run_labels)

    p_splits = sub.add_parser(
        "splits", help="Generate stratified K-fold CV split CSVs."
    )
    p_splits.add_argument("--labels", required=True, help="Label CSV (slide_id,label).")
    p_splits.add_argument(
        "--output-dir", required=True, help="Directory for split_fold*.csv outputs."
    )
    p_splits.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"Number of folds (fallback: {DEFAULT_K}).",
    )
    p_splits.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="Val fraction of the non-test pool (fallback: 1/(k-1)).",
    )
    p_splits.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (fallback: {DEFAULT_SEED}).",
    )
    p_splits.set_defaults(func=_run_splits)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """``foveamil-cohort`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
