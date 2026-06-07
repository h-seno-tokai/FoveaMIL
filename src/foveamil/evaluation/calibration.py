"""推論後の後処理によるキャリブレーションと閾値最適化
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import log_loss, f1_score, recall_score

logger = logging.getLogger(__name__)


def optimize_temperature(y_true: np.ndarray, logits: np.ndarray) -> float:
    """Temperature scaling の温度 T をバリデーションデータで最適化する

    T > 0 を動かし，負の対数尤度（Cross Entropy）を最小化する

    Args:
        y_true: 正解クラス [N]
        logits: 未スケーリングのロジット [N, n_cls]

    Returns:
        最適化された温度 T
    """
    def objective(t: float) -> float:
        # np.clip で 0 除算や極端な値を防ぐ
        t = max(t, 1e-6)
        probs = softmax(logits / t, axis=1)
        return log_loss(y_true, probs, labels=np.arange(logits.shape[1]))

    res = minimize(
        lambda x: objective(x[0]),
        x0=[1.0],
        bounds=[(0.01, 10.0)],
        method="L-BFGS-B"
    )
    t_opt = float(res.x[0])
    logger.info("Optimized temperature: %.4f (initial log_loss: %.4f -> optimized: %.4f)",
                t_opt, objective(1.0), objective(t_opt))
    return t_opt


def optimize_offsets(
    y_true: np.ndarray,
    logits: np.ndarray,
    target_metric: str = "macro_f1",
    minority_indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """クラス別ロジット補正（offsets）をバリデーションデータで最適化する

    logits_adj = logits + offsets としたときの指定指標を最大化する
    L2 正則化を加えて過適合を抑制する

    Args:
        y_true: 正解クラス [N]
        logits: ロジット [N, n_cls]
        target_metric: 最適化対象の指標 ("macro_f1", "minority_recall", "arithmetic_mean")
        minority_indices: 少数クラスの index 集合（target_metric="minority_recall" 等で利用）

    Returns:
        最適化されたオフセット [n_cls]
    """
    n_cls = logits.shape[1]
    
    def get_score(offsets: np.ndarray) -> float:
        adj_logits = logits + offsets
        y_pred = adj_logits.argmax(axis=1)
        
        if target_metric == "macro_f1":
            return f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        if target_metric == "minority_recall":
            if minority_indices is None:
                return recall_score(y_true, y_pred, average="macro", zero_division=0)
            recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
            return np.mean([recalls[i] for i in minority_indices])
        
        # 既定は macro F1
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    # 目的関数: 1 - score + regularization
    def objective(offsets: np.ndarray) -> float:
        score = get_score(offsets)
        # 過適合回避のため L2 正則化（微小）
        reg = 1e-4 * np.sum(offsets**2)
        return (1.0 - score) + reg

    # クラス初期オフセットは 0
    res = minimize(
        objective,
        x0=np.zeros(n_cls),
        method="Nelder-Mead", # 離散的な argmax を含むため微分不要な手法を選択
        options={"maxiter": 1000}
    )
    
    offsets_opt = res.x
    logger.info("Optimized offsets for %s: %s (initial score: %.4f -> optimized: %.4f)",
                target_metric, offsets_opt, 1.0 - objective(np.zeros(n_cls)), 1.0 - objective(offsets_opt))
    return offsets_opt


def apply_calibration(
    logits: np.ndarray,
    temperature: float = 1.0,
    offsets: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """最適化されたパラメータを適用し，予測クラスと確率を返す

    Args:
        logits: 元のロジット
        temperature: 温度 T
        offsets: ロジットオフセット [n_cls]

    Returns:
        (y_hat, y_prob)
    """
    scaled_logits = logits / temperature
    if offsets is not None:
        scaled_logits = scaled_logits + offsets
    
    y_hat = scaled_logits.argmax(axis=1)
    y_prob = softmax(scaled_logits, axis=1)
    return y_hat, y_prob


def pool_predictions_by_fold(
    fold_predictions: Dict[int, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """複数 fold の予測を統合（pooled-val）する

    Args:
        fold_predictions: {fold_idx: {"y_true": np.ndarray, "logits": np.ndarray}}

    Returns:
        (pooled_y_true, pooled_logits)
    """
    all_y_true = []
    all_logits = []
    for f_idx in sorted(fold_predictions.keys()):
        all_y_true.append(fold_predictions[f_idx]["y_true"])
        all_logits.append(fold_predictions[f_idx]["logits"])
    
    return np.concatenate(all_y_true), np.concatenate(all_logits)
