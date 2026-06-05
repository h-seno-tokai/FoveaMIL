"""training.metrics の拡張指標・生配列のユニット"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)

from foveamil.training.metrics import MetricLogger


def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _fill(logger, y_true, y_pred, probs=None, logits=None):
    for i in range(len(y_true)):
        prob = torch.tensor(probs[i]) if probs is not None else None
        logit = torch.tensor(logits[i]) if logits is not None else None
        logger.log(torch.tensor(y_pred[i]), torch.tensor(y_true[i]), prob, logit)


def test_extended_scalar_metrics_match_sklearn():
    y_true = [0, 1, 2, 0, 1, 2, 0, 1]
    y_pred = [0, 1, 2, 0, 2, 2, 1, 1]
    logger = MetricLogger(n_cls=3)
    _fill(logger, y_true, y_pred)
    s = logger.get_summary()
    assert s["balanced_accuracy"] == balanced_accuracy_score(y_true, y_pred)
    assert s["mcc"] == matthews_corrcoef(y_true, y_pred)
    # 既存キーが残っている（後方互換）
    for key in ("accuracy", "weighted_f1", "macro_f1", "kappa",
                "class_0_f1", "class_0_precision", "class_0_recall"):
        assert key in s


def test_specificity_sensitivity():
    # 2クラスで手計算: cls1 を positive とみなす
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    logger = MetricLogger(n_cls=2)
    _fill(logger, y_true, y_pred)
    s = logger.get_summary()
    # cls1: TP=1, FN=1, FP=1, TN=2 -> specificity=2/3, sensitivity(recall)=1/2
    assert s["class_1_specificity"] == 2 / 3
    assert s["class_1_sensitivity"] == 0.5


def test_auprc_and_ovo_with_probabilities():
    rng = np.random.default_rng(0)
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    logits = rng.normal(size=(9, 3))
    probs = np.array([_softmax(l) for l in logits])
    logger = MetricLogger(n_cls=3)
    _fill(logger, y_true, [int(p.argmax()) for p in probs], probs=probs, logits=logits)
    s = logger.get_summary()
    assert "macro_auprc" in s and "class_0_auprc" in s
    assert "ovo_macro_auc" in s and "macro_auc" in s
    # macro_auprc が per-class の平均と一致
    per = [s[f"class_{i}_auprc"] for i in range(3)]
    assert s["macro_auprc"] == np.mean(per)
    assert s["ovo_macro_auc"] == roc_auc_score(
        y_true, probs, multi_class="ovo", average="macro"
    )


def test_missing_class_fold_omits_auc_keys():
    # test fold に class 2 が居ない -> per-class auc/auprc は欠損キーで省略される
    y_true = [0, 0, 1, 1]
    probs = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1],
                      [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]])
    logger = MetricLogger(n_cls=3)
    _fill(logger, y_true, [int(p.argmax()) for p in probs], probs=probs)
    s = logger.get_summary()
    assert "class_2_auc" not in s   # 片クラスのみ -> 省略
    assert "class_2_auprc" not in s


def test_get_arrays_shapes():
    y_true = [0, 1, 2]
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    logits = np.log(probs)
    logger = MetricLogger(n_cls=3)
    _fill(logger, y_true, [0, 1, 2], probs=probs, logits=logits)
    arr = logger.get_arrays()
    assert arr["y_true"].shape == (3,)
    assert arr["y_pred"].shape == (3,)
    assert arr["y_prob"].shape == (3, 3)
    assert arr["y_logit"].shape == (3, 3)


def test_get_arrays_none_when_no_prob():
    logger = MetricLogger(n_cls=2)
    _fill(logger, [0, 1], [0, 1])
    arr = logger.get_arrays()
    assert arr["y_prob"] is None and arr["y_logit"] is None


def test_confusion_matrix_normalize():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    logger = MetricLogger(n_cls=2)
    _fill(logger, y_true, y_pred)
    cm = logger.get_confusion_matrix()
    cmn = logger.get_confusion_matrix(normalize=True)
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert cmn[0].tolist() == [0.5, 0.5]
    assert cmn[1].tolist() == [0.0, 1.0]
