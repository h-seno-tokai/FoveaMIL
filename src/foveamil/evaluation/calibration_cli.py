"""保存済み予測（logits）を読み込んでキャリブレーション・閾値最適化を遡及実行する CLI
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score

from foveamil.evaluation.calibration import (
    optimize_temperature,
    optimize_offsets,
    apply_calibration,
    pool_predictions_by_fold
)
from foveamil.evaluation.report import PROB_PREFIX, load_predictions

logger = logging.getLogger(__name__)

# ロジット列の接頭辞
LOGIT_PREFIX = "logit_"


def _logit_columns(df: pd.DataFrame) -> List[str]:
    """logit_* 列を class 添字順に返す"""
    cols = [c for c in df.columns if c.startswith(LOGIT_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(LOGIT_PREFIX):]))


def main():
    parser = argparse.ArgumentParser(description="Retrospective calibration and threshold optimization")
    parser.add_argument("combo_dir", help="Combo directory containing fold*/predictions_{split}.csv")
    parser.add_argument("--save-root", help="Output directory (default: combo_dir/calibration)")
    parser.add_argument("--target-metric", default="macro_f1", choices=["macro_f1", "minority_recall"])
    parser.add_argument("--minority-indices", type=int, nargs="+", help="Indices of minority classes")
    parser.add_argument("--folds", type=int, nargs="+", help="Specific folds to use (default: all fold* dirs)")
    parser.add_argument("--val-split", default="val", help="Split name to use for optimization")
    parser.add_argument("--test-split", default="test", help="Split name to apply calibrated parameters")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    save_root = args.save_root or os.path.join(args.combo_dir, "calibration")
    os.makedirs(save_root, exist_ok=True)

    # Fold ディレクトリの探索
    fold_dirs = [d for d in os.listdir(args.combo_dir) if d.startswith("fold")]
    if args.folds:
        fold_dirs = [f"fold{i}" for i in args.folds if f"fold{i}" in fold_dirs]
    fold_dirs = sorted(fold_dirs)

    if not fold_dirs:
        print(f"No fold directories found in {args.combo_dir}")
        return

    # 1. バリデーションデータの収集 (pooled-val)
    val_data: Dict[int, Dict[str, np.ndarray]] = {}
    for i, fold_name in enumerate(fold_dirs):
        df = load_predictions(os.path.join(args.combo_dir, fold_name), args.val_split)
        if df is not None:
            logit_cols = _logit_columns(df)
            val_data[i] = {
                "y_true": df["y_true"].to_numpy(),
                "logits": df[logit_cols].to_numpy()
            }
    
    if not val_data:
        print(f"No validation predictions found for split {args.val_split}")
        return

    pooled_y_true, pooled_logits = pool_predictions_by_fold(val_data)
    n_cls = pooled_logits.shape[1]

    # 2. 最適化
    print(f"Optimizing on pooled validation data (N={len(pooled_y_true)}, n_cls={n_cls})...")
    temperature = optimize_temperature(pooled_y_true, pooled_logits)
    offsets = optimize_offsets(
        pooled_y_true, 
        pooled_logits / temperature, 
        target_metric=args.target_metric,
        minority_indices=args.minority_indices
    )

    print(f"--- Calibration Results ---")
    print(f"Temperature: {temperature:.4f}")
    print(f"Offsets: {offsets}")

    # 3. テストデータへの適用と評価
    print(f"\nApplying to {args.test_split} split per fold...")
    all_test_y_true = []
    all_test_y_pred_orig = []
    all_test_y_pred_calib = []

    for fold_name in fold_dirs:
        df = load_predictions(os.path.join(args.combo_dir, fold_name), args.test_split)
        if df is None:
            continue
        
        logit_cols = _logit_columns(df)
        y_true = df["y_true"].to_numpy()
        logits = df[logit_cols].to_numpy()
        
        # オリジナルの予測（温度 1.0, オフセットなし）
        y_pred_orig = logits.argmax(axis=1)
        
        # キャリブレーション後の予測
        y_pred_calib, _ = apply_calibration(logits, temperature, offsets)
        
        all_test_y_true.append(y_true)
        all_test_y_pred_orig.append(y_pred_orig)
        all_test_y_pred_calib.append(y_pred_calib)

    if not all_test_y_true:
        print(f"No test predictions found for split {args.test_split}")
        return

    y_true_test = np.concatenate(all_test_y_true)
    y_pred_orig_test = np.concatenate(all_test_y_pred_orig)
    y_pred_calib_test = np.concatenate(all_test_y_pred_calib)

    # 4. レポート表示
    print("\n[Baseline (Original)]")
    print(classification_report(y_true_test, y_pred_orig_test, digits=4))
    
    print("\n[After Calibration & Threshold Optimization]")
    print(classification_report(y_true_test, y_pred_calib_test, digits=4))

    # 指標の比較
    orig_f1 = f1_score(y_true_test, y_pred_orig_test, average="macro")
    calib_f1 = f1_score(y_true_test, y_pred_calib_test, average="macro")
    
    print(f"Macro-F1 Improvement: {orig_f1:.4f} -> {calib_f1:.4f} ({calib_f1 - orig_f1:+.4f})")
    
    if args.minority_indices:
        orig_recalls = recall_score(y_true_test, y_pred_orig_test, average=None)
        calib_recalls = recall_score(y_true_test, y_pred_calib_test, average=None)
        
        orig_min_recall = np.mean([orig_recalls[i] for i in args.minority_indices])
        calib_min_recall = np.mean([calib_recalls[i] for i in args.minority_indices])
        
        print(f"Minority Recall Improvement: {orig_min_recall:.4f} -> {calib_min_recall:.4f} ({calib_min_recall - orig_min_recall:+.4f})")

    # 結果の保存
    summary = {
        "temperature": temperature,
        "offsets": offsets.tolist(),
        "baseline": {
            "macro_f1": orig_f1,
        },
        "calibrated": {
            "macro_f1": calib_f1,
        }
    }
    
    with open(os.path.join(save_root, "calibration_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {save_root}")


if __name__ == "__main__":
    main()
