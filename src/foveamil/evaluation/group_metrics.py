"""保存済み per-fold 指標から指定クラス集合の group-F1 を算出する

``cv_summary.json`` の per_fold に格納された per-class F1（``class_i_f1``）を読み，
指定したクラス index 集合の非加重平均（macro 相当）を fold ごとに求める集合内の
per-class 値も併記できる空集合・欠損クラスでは ``nan`` を返し例外を投げない学習や
再推論はせず保存済み値を二次利用するだけ
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

# per-fold 指標辞書での per-class F1 キーの書式
CLASS_F1_KEY_TEMPLATE = "class_{i}_f1"

_NAN = float("nan")


def class_f1_key(class_index: int) -> str:
    """クラス index から per-class F1 のキー名を作る"""
    return CLASS_F1_KEY_TEMPLATE.format(i=class_index)


def group_f1_from_fold(
    fold_metrics: Dict[str, float], class_indices: Sequence[int]
) -> float:
    """1 fold の指標辞書から指定クラス集合の非加重平均 F1 を返す

    集合内で辞書に存在するクラスの F1 のみを平均する有効な per-class F1 が 1 つも
    無い（空集合・全クラス欠損）場合は ``nan``

    Args:
        fold_metrics: 1 fold の指標辞書（``class_i_f1`` を含む）
        class_indices: 平均対象のクラス index 集合

    Returns:
        group-F1（非加重平均）有効値が無ければ ``nan``
    """
    values = [
        float(fold_metrics[class_f1_key(i)])
        for i in class_indices
        if class_f1_key(i) in fold_metrics
    ]
    if not values:
        return _NAN
    return float(np.mean(values))


def group_f1_per_fold(
    per_fold: List[Dict[str, float]], class_indices: Sequence[int]
) -> List[float]:
    """各 fold の group-F1 を fold 順に返す

    Args:
        per_fold: fold ごとの指標辞書の列
        class_indices: 平均対象のクラス index 集合

    Returns:
        fold ごとの group-F1 値列
    """
    return [group_f1_from_fold(m, class_indices) for m in per_fold]


def group_f1_summary(
    per_fold: List[Dict[str, float]], class_indices: Sequence[int]
) -> Dict[str, Any]:
    """指定クラス集合の group-F1 を per-fold で集計し集合内 per-class も併記する

    fold 間の mean/std は ``nan`` を除いた有効 fold のみで取る（有効が無ければ
    mean/std とも ``nan``）``per_class`` は集合内の各クラスの per-fold 平均 F1 を返し，
    その class が全 fold で欠損なら ``nan``

    Args:
        per_fold: fold ごとの指標辞書の列
        class_indices: group-F1 を構成するクラス index 集合

    Returns:
        ``{"class_indices", "per_fold", "mean", "std", "n", "per_class"}``
        （``per_class`` は ``{class_index: 平均 F1}``）
    """
    fold_values = group_f1_per_fold(per_fold, class_indices)
    valid = [v for v in fold_values if not np.isnan(v)]
    mean = float(np.mean(valid)) if valid else _NAN
    std = float(np.std(valid)) if valid else _NAN

    per_class: Dict[int, float] = {}
    for i in class_indices:
        key = class_f1_key(i)
        vals = [float(m[key]) for m in per_fold if key in m]
        per_class[i] = float(np.mean(vals)) if vals else _NAN

    return {
        "class_indices": list(class_indices),
        "per_fold": fold_values,
        "mean": mean,
        "std": std,
        "n": len(valid),
        "per_class": per_class,
    }
