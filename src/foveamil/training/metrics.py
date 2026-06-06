"""予測の蓄積から分類指標を集計するロガー

``log`` で 1 サンプルずつ予測クラス・正解クラス（任意で確率）を蓄積し，``get_summary``
で accuracy / F1（weighted・macro・per-class）/ precision・recall（weighted・macro・
per-class）/ kappa（重みなし・名義クラス）/ AUC（確率があれば OvR per-class，多クラスでは
macro・weighted）をまとめた辞書を返す``get_confusion_matrix`` で混同行列を返す
数値が安定して出ない場合は当該指標を安全に省きログに残す
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# 既定のクラス数
DEFAULT_N_CLS = 3
# Cohen's kappa の weights 指定（名義クラスは順序を持たないため重みなし）
KAPPA_WEIGHTS = None
# precision/recall/F1 のゼロ割時の値
ZERO_DIVISION = 0
# 多クラス AUC を計算する最小クラス数
MULTICLASS_AUC_MIN = 2
# 二値判定に必要なユニーククラス数
BINARY_DISTINCT = 2
# macro 平均指定
AVERAGE_MACRO = "macro"
# weighted 平均指定
AVERAGE_WEIGHTED = "weighted"
# OvR（one-vs-rest）多クラス指定
MULTICLASS_OVR = "ovr"
# OvO（one-vs-one）多クラス指定
MULTICLASS_OVO = "ovo"
# 混同行列の行正規化のゼロ割回避用の最小値
_NORM_EPS = 1e-12


class MetricLogger:
    """予測を蓄積して分類指標を集計するロガー

    Args:
        n_cls: クラス数
    """

    def __init__(self, n_cls: int = DEFAULT_N_CLS) -> None:
        self.n_cls = n_cls
        self.y_pred: List[int] = []
        self.y_true: List[int] = []
        self.y_prob: List[np.ndarray] = []
        self.y_logit: List[np.ndarray] = []

    def log(
        self,
        Y_hat: torch.Tensor,
        Y: torch.Tensor,
        Y_prob: Optional[torch.Tensor] = None,
        Y_logit: Optional[torch.Tensor] = None,
    ) -> None:
        """1 サンプル分の予測クラス・正解クラス（任意で確率・logit）を蓄積する

        Args:
            Y_hat: 予測クラス（スカラ相当）
            Y: 正解クラス（スカラ相当）
            Y_prob: クラス確率``None`` 以外なら AUC 用に保持する
            Y_logit: クラス logit``None`` 以外なら予測保存用に保持する
        """
        self.y_pred.append(int(Y_hat))
        self.y_true.append(int(Y))
        if Y_prob is not None:
            self.y_prob.append(Y_prob.detach().cpu().numpy().flatten())
        if Y_logit is not None:
            self.y_logit.append(Y_logit.detach().cpu().numpy().flatten())

    def get_arrays(self) -> Dict[str, Optional[np.ndarray]]:
        """蓄積した生配列を返す（予測 CSV 保存用）

        Returns:
            ``{"y_true": (N,), "y_pred": (N,), "y_prob": (N, C) or None,
            "y_logit": (N, C) or None}``slide_id は保持しない（呼び出し側が
            loader 順に集める）
        """
        return {
            "y_true": np.asarray(self.y_true, dtype=int),
            "y_pred": np.asarray(self.y_pred, dtype=int),
            "y_prob": np.asarray(self.y_prob) if self.y_prob else None,
            "y_logit": np.asarray(self.y_logit) if self.y_logit else None,
        }

    def _add_auc(self, summary: Dict[str, float]) -> None:
        """確率が蓄積されていれば AUC 指標を ``summary`` に追加する

        OvR per-class AUC（両クラスが揃う場合のみ）と，多クラスでは macro・weighted を
        加える計算できない指標は省く
        """
        if not self.y_prob:
            return
        try:
            y_prob = np.array(self.y_prob)
            y_true = np.array(self.y_true)
            n_classes = y_prob.shape[1]
            for i in range(n_classes):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) >= BINARY_DISTINCT:
                    summary[f"class_{i}_auc"] = float(
                        roc_auc_score(y_true_binary, y_prob[:, i])
                    )
            if n_classes > MULTICLASS_AUC_MIN:
                try:
                    summary["macro_auc"] = float(
                        roc_auc_score(
                            y_true, y_prob,
                            multi_class=MULTICLASS_OVR, average=AVERAGE_MACRO,
                        )
                    )
                    summary["weighted_auc"] = float(
                        roc_auc_score(
                            y_true, y_prob,
                            multi_class=MULTICLASS_OVR, average=AVERAGE_WEIGHTED,
                        )
                    )
                    summary["ovo_macro_auc"] = float(
                        roc_auc_score(
                            y_true, y_prob,
                            multi_class=MULTICLASS_OVO, average=AVERAGE_MACRO,
                        )
                    )
                except ValueError as exc:
                    logger.warning("skipped multiclass AUC: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not compute AUC: %s", exc)

    def get_summary(self) -> Dict[str, float]:
        """蓄積した予測から指標辞書を作る

        Returns:
            accuracy / weighted・macro F1 / weighted・macro precision・recall /
            kappa / per-class f1・precision・recall / （確率があれば）AUC を含む辞書
        """
        acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)

        f1 = f1_score(
            y_true=self.y_true, y_pred=self.y_pred, average=None,
            zero_division=ZERO_DIVISION,
        )
        weighted_f1 = f1_score(
            y_true=self.y_true, y_pred=self.y_pred, average="weighted",
            zero_division=ZERO_DIVISION,
        )
        macro_f1 = f1_score(
            y_true=self.y_true, y_pred=self.y_pred, average="macro",
            zero_division=ZERO_DIVISION,
        )

        precision = precision_score(
            y_true=self.y_true, y_pred=self.y_pred, average=None,
            zero_division=ZERO_DIVISION,
        )
        weighted_precision = precision_score(
            y_true=self.y_true, y_pred=self.y_pred, average="weighted",
            zero_division=ZERO_DIVISION,
        )
        macro_precision = precision_score(
            y_true=self.y_true, y_pred=self.y_pred, average="macro",
            zero_division=ZERO_DIVISION,
        )

        recall = recall_score(
            y_true=self.y_true, y_pred=self.y_pred, average=None,
            zero_division=ZERO_DIVISION,
        )
        weighted_recall = recall_score(
            y_true=self.y_true, y_pred=self.y_pred, average="weighted",
            zero_division=ZERO_DIVISION,
        )
        macro_recall = recall_score(
            y_true=self.y_true, y_pred=self.y_pred, average="macro",
            zero_division=ZERO_DIVISION,
        )

        try:
            kappa = float(
                cohen_kappa_score(
                    self.y_true, self.y_pred, weights=KAPPA_WEIGHTS
                )
            )
            if np.isnan(kappa):
                kappa = float(ZERO_DIVISION)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not compute kappa: %s", exc)
            kappa = float(ZERO_DIVISION)

        summary: Dict[str, float] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(
                balanced_accuracy_score(self.y_true, self.y_pred)
            ),
            "mcc": self._safe_mcc(),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "macro_precision": float(macro_precision),
            "weighted_recall": float(weighted_recall),
            "macro_recall": float(macro_recall),
            "kappa": kappa,
        }
        specificity = self._specificity_per_class()
        for i in range(len(f1)):
            summary[f"class_{i}_f1"] = float(f1[i])
            summary[f"class_{i}_precision"] = float(precision[i])
            summary[f"class_{i}_recall"] = float(recall[i])
            summary[f"class_{i}_sensitivity"] = float(recall[i])
            summary[f"class_{i}_specificity"] = float(specificity[i])

        self._add_auc(summary)
        self._add_auprc(summary)

        logger.info(
            "metrics: acc=%.4f wF1=%.4f mF1=%.4f kappa=%.4f",
            summary["accuracy"],
            summary["weighted_f1"],
            summary["macro_f1"],
            summary["kappa"],
        )
        return summary

    def _safe_mcc(self) -> float:
        """Matthews 相関係数を返す計算不能/NaN は ``ZERO_DIVISION`` 値"""
        try:
            value = float(matthews_corrcoef(self.y_true, self.y_pred))
            if np.isnan(value):
                return float(ZERO_DIVISION)
            return value
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not compute MCC: %s", exc)
            return float(ZERO_DIVISION)

    def _specificity_per_class(self) -> np.ndarray:
        """各クラスの OvR 特異度 ``TN / (TN + FP)`` を返す"""
        cm = confusion_matrix(
            self.y_true, self.y_pred, labels=list(range(self.n_cls))
        )
        total = cm.sum()
        specificity = np.zeros(self.n_cls, dtype=float)
        for i in range(self.n_cls):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = total - tp - fp - fn
            denom = tn + fp
            specificity[i] = tn / denom if denom > 0 else 0.0
        return specificity

    def _add_auprc(self, summary: Dict[str, float]) -> None:
        """確率があれば AUPRC（per-class OvR と macro）を ``summary`` に追加する

        両クラスが揃う class のみ計算し，存在分の平均を macro とする
        """
        if not self.y_prob:
            return
        try:
            y_prob = np.array(self.y_prob)
            y_true = np.array(self.y_true)
            per_class: List[float] = []
            for i in range(y_prob.shape[1]):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) >= BINARY_DISTINCT:
                    value = float(
                        average_precision_score(y_true_binary, y_prob[:, i])
                    )
                    summary[f"class_{i}_auprc"] = value
                    per_class.append(value)
            if per_class:
                summary["macro_auprc"] = float(np.mean(per_class))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not compute AUPRC: %s", exc)

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """蓄積した予測から混同行列を返す

        Args:
            normalize: ``True`` なら行（正解クラス）で正規化した割合を返す
        """
        cm = confusion_matrix(
            np.array(self.y_true),
            np.array(self.y_pred),
            labels=list(range(self.n_cls)),
        )
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            return cm / np.maximum(row_sums, _NORM_EPS)
        return cm
