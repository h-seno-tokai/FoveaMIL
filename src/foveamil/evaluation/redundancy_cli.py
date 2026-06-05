"""``foveamil-redundancy`` コマンド

sweep の出力（``--in``）から val 選定の best combo と fold 重みを解決し，test split の
各スライドで融合へ入る各倍率のプーリング表現を集めて倍率間の冗長性を診断する余弦
類似度（生・中心化）・Pearson 相関・線形 CKA・実効ランクを計算し，JSON 要約と CKA・
Pearson のヒートマップ PNG を出力する学習は一切しない（保存済み重みで前向き推論のみ）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from foveamil.evaluation.redundancy import (
    aggregate_redundancy,
    collect_magnification_vectors,
    save_heatmap,
)
from foveamil.visualization import cases as cases_mod
from foveamil.visualization.loader import (
    fold_dirs,
    load_fold,
    resolve_best_combo,
)

logger = logging.getLogger(__name__)

# 出力サブディレクトリ既定名
DEFAULT_REPORT_DIR = "redundancy"
# 既定の対象 split
DEFAULT_SPLIT = "test"
# 既定の推論デバイス
DEFAULT_DEVICE = "cpu"
# 要約 JSON のファイル名
SUMMARY_JSON = "redundancy.json"
# ヒートマップ PNG のファイル名
CKA_PNG = "cka_heatmap.png"
PEARSON_PNG = "pearson_heatmap.png"
# combo 選定の種別
SELECT_BEST_VAL = "best_by_val"
SELECT_ORACLE = "oracle_by_test"
SELECT_INDEX = "index"
# 全 fold を対象にする指定
FOLD_ALL = "all"
# 倍率ラベルの接尾辞
MAG_LABEL_SUFFIX = "x"


def _resolve_feature_root(args: argparse.Namespace) -> str:
    """特徴ルートを引数か環境変数から解決する"""
    feature_root = args.feature_root or os.environ.get("FOVEAMIL_FEATURE_ROOT")
    if not feature_root:
        raise SystemExit(
            "feature_root is required (pass --feature-root or set "
            "FOVEAMIL_FEATURE_ROOT)"
        )
    return feature_root


def _slide_ids(combo_dir: str, fold_names: List[str], split: str) -> List[str]:
    """combo の予測 CSV から split のスライド識別子を集める"""
    df = cases_mod.load_cases_frame(combo_dir, split, fold_names)
    if df is None:
        raise SystemExit(f"predictions_{split}.csv not found under {combo_dir}")
    return [str(s) for s in df["slide_id"].tolist()]


def _collect_split_vectors(
    combo_dir: str,
    fold: str,
    weights_root: Optional[str],
    feature_root: str,
    split: str,
    device: torch.device,
) -> tuple:
    """各 fold の重みで担当 split スライドの倍率ベクトルを集める

    fold ごとに best 重みをロードし，その fold の予測 CSV のスライドのみを処理する
    （各スライドはちょうど 1 fold の test に属するため重複しない）

    Returns:
        ``(倍率ベクトル列, magnifications)``
    """
    dirs = fold_dirs(combo_dir, fold)
    slide_vectors: List[np.ndarray] = []
    magnifications: Optional[List[float]] = None
    for fold_dir in dirs:
        weights_dir = (
            os.path.join(weights_root, os.path.basename(combo_dir),
                         os.path.basename(fold_dir))
            if weights_root else None
        )
        loaded = load_fold(fold_dir, weights_dir=weights_dir, device=str(device))
        magnifications = loaded.magnifications
        ids = _slide_ids(combo_dir, [os.path.basename(fold_dir)], split)
        for slide_id in ids:
            vectors = collect_magnification_vectors(
                loaded.model, feature_root, loaded.encoder, slide_id,
                loaded.magnifications, loaded.feature_type, device=device,
            )
            slide_vectors.append(vectors)
    return slide_vectors, (magnifications or [])


def _mag_labels(magnifications: Sequence[float]) -> List[str]:
    """倍率列を軸ラベル（例 ``1.25x``）に整形する"""
    return [f"{m}{MAG_LABEL_SUFFIX}" for m in magnifications]


def _write_heatmaps(
    summary: Dict[str, Any], magnifications: Sequence[float], out_dir: str
) -> List[str]:
    """CKA・Pearson 行列をヒートマップ PNG に保存し作れたファイル名を返す"""
    labels = _mag_labels(magnifications)
    written: List[str] = []
    pairs = (
        ("cka_matrix", CKA_PNG, "Inter-magnification linear CKA"),
        ("pearson_matrix", PEARSON_PNG, "Inter-magnification Pearson correlation"),
    )
    for key, png_name, title in pairs:
        if key not in summary:
            continue
        matrix = np.asarray(summary[key], dtype=float)
        out_png = os.path.join(out_dir, png_name)
        if save_heatmap(matrix, labels, title, out_png):
            written.append(png_name)
    return written


def run(args: argparse.Namespace) -> int:
    """冗長性診断の本体"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    feature_root = _resolve_feature_root(args)
    out_dir = args.out or os.path.join(args.input, DEFAULT_REPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    combo_dir = resolve_best_combo(
        args.input, select=args.select, combo_index=args.combo_index
    )
    logger.info("resolved combo: %s", combo_dir)

    slide_vectors, magnifications = _collect_split_vectors(
        combo_dir, args.fold, args.weights_root, feature_root, args.split, device
    )
    logger.info(
        "collected %d slides, %d magnifications",
        len(slide_vectors), len(magnifications),
    )

    summary = aggregate_redundancy(slide_vectors)
    summary["combo_dir"] = os.path.basename(os.path.normpath(combo_dir))
    summary["split"] = args.split
    summary["magnifications"] = list(magnifications)
    summary["figures"] = _write_heatmaps(summary, magnifications, out_dir)

    summary_path = os.path.join(out_dir, SUMMARY_JSON)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info("wrote redundancy summary to %s", summary_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-redundancy`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-redundancy",
        description="Diagnose inter-magnification representation redundancy "
        "(cosine, Pearson, linear CKA, effective rank) on the test split of a "
        "sweep's best-by-val combo (no retraining).",
    )
    parser.add_argument("--in", dest="input", required=True, help="Sweep --out root.")
    parser.add_argument(
        "--out", default=None,
        help="Output dir (default: {in}/redundancy).",
    )
    parser.add_argument(
        "--feature-root", default=None,
        help="Feature root (or set FOVEAMIL_FEATURE_ROOT).",
    )
    parser.add_argument(
        "--weights-root", default=None,
        help="Weights root mirroring combo/fold layout (default: alongside folds).",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, choices=["val", "test", "train"],
        help="Split to diagnose (default: test).",
    )
    parser.add_argument(
        "--select", default=SELECT_BEST_VAL,
        choices=[SELECT_BEST_VAL, SELECT_ORACLE, SELECT_INDEX],
        help="Combo selection (default: best_by_val).",
    )
    parser.add_argument(
        "--combo-index", type=int, default=None,
        help="Combo index when --select index.",
    )
    parser.add_argument(
        "--fold", default=FOLD_ALL,
        help="Fold number or 'all' (default: all).",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, help="Inference device (default: cpu)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-redundancy`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
