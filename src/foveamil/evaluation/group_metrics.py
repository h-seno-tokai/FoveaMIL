"""保存済み per-fold 指標と保存済み予測から指定クラス集合の group-F1 を算出する

``cv_summary.json`` の per_fold に格納された per-class F1（``class_i_f1``）を読み，
指定したクラス index 集合の非加重平均（macro 相当）を fold ごとに求める集合内の
per-class 値も併記できる空集合・欠損クラスでは ``nan`` を返し例外を投げない
保存済み予測（``predictions_{split}.csv`` の ``y_true`` / ``y_pred``）を全 fold で
プールし，同じクラス集合の非加重平均 F1 を 1 値として算出することもできる学習や
再推論はせず保存済み値を二次利用するだけ
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from foveamil.evaluation.report import FOLD_DIR_PREFIX, pool_predictions

# per-fold 指標辞書での per-class F1 キーの書式
CLASS_F1_KEY_TEMPLATE = "class_{i}_f1"
# 予測 CSV の正解/予測クラス列名
Y_TRUE_COL = "y_true"
Y_PRED_COL = "y_pred"
# プール時に各行へ付与する出所キー列（多 seed / 多 out_root のどの run 由来かを表す）
SOURCE_COL = "source"
# f1_score のゼロ割時の値
ZERO_DIVISION = 0

_NAN = float("nan")


def class_f1_key(class_index: int) -> str:
    """クラス index から per-class F1 のキー名を作る"""
    return CLASS_F1_KEY_TEMPLATE.format(i=class_index)


def group_f1_from_fold(
    fold_metrics: Dict[str, float], class_indices: Sequence[int]
) -> float:
    """1 fold の指標辞書から指定クラス集合の非加重平均 F1 を返す

    集合内で辞書に存在するクラスの F1 のみを平均する（欠損クラスは除外）有効な
    per-class F1 が 1 つも無い（空集合・全クラス欠損）場合は ``nan``この
    「存在クラスのみ平均」の規約は :func:`pooled_group_f1` の欠損クラス扱い
    （support0 クラスを除外）と一致する

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


def pooled_group_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_indices: Sequence[int]
) -> float:
    """プールした正解/予測クラスから指定クラス集合の非加重平均 F1 を返す

    fold をまたいでプールした全症例の予測に対し，``class_indices`` の各クラスの
    F1（OvR・ゼロ割は 0）を求めその非加重平均を取るfold 平均ではなく全症例を
    1 つに束ねた上での単一値である

    欠損クラス（``y_true`` に出現しない＝support0 のクラス）は :func:`group_f1_from_fold`
    と扱いを揃えて平均から除外する（存在クラスのみの非加重平均）これにより同じクラス
    集合で per-fold（存在クラスのみ平均）とプールの欠損クラス扱いが一致する有効クラスが
    1 つも無い・空集合・空標本では ``nan``

    Args:
        y_true: プールした正解クラス ``[N]``
        y_pred: プールした予測クラス ``[N]``
        class_indices: 平均対象のクラス index 集合

    Returns:
        プール group-F1（存在クラスのみの非加重平均）有効クラスなし・空集合・
        標本なしなら ``nan``
    """
    labels = list(class_indices)
    if not labels or len(y_true) == 0:
        return _NAN
    # support0 クラスは per-fold の欠損扱いに揃えて除外する
    present = set(np.unique(np.asarray(y_true)).tolist())
    eval_labels = [c for c in labels if c in present]
    if not eval_labels:
        return _NAN
    per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=eval_labels,
        average=None, zero_division=ZERO_DIVISION,
    )
    return float(np.mean(per_class))


def _fold_names(combo_dir: str) -> List[str]:
    """combo 直下の ``fold*`` ディレクトリ名を昇順で返す"""
    if not os.path.isdir(combo_dir):
        return []
    names = [
        name
        for name in os.listdir(combo_dir)
        if name.startswith(FOLD_DIR_PREFIX)
        and os.path.isdir(os.path.join(combo_dir, name))
    ]
    return sorted(names)


def pool_combo_predictions(
    combo_dirs: Sequence[str],
    split: str,
    sources: Optional[Sequence[Any]] = None,
) -> Optional[pd.DataFrame]:
    """複数 combo ディレクトリの全 fold 予測を縦結合する

    各 combo 直下の ``fold*`` を自動検出し ``predictions_{split}.csv`` をプールする
    （複数 out_root / seed の同一手法を 1 つに束ねる用途）読める予測が無ければ ``None``

    ``sources`` を渡すと各 combo_dir 由来の行に出所キー（``source`` 列）を付与する
    多 seed / 多 out_root では同一 slide_id が run ごとに重複するため，対応付け
    （paired 比較の merge）は ``[slide_id, source]`` を単位にしないと直積化して
    対応が壊れるその単位を成立させるための列である``sources`` 省略時は combo_dir
    パスを出所キーに使う

    Args:
        combo_dirs: combo ディレクトリのパス列
        split: ``test`` / ``val`` / ``train``
        sources: 各 combo_dir に対応する出所キー列（``combo_dirs`` と同長）省略時は
            combo_dir パスを使う

    Returns:
        ``source`` 列を含むプール予測 DataFrame読めなければ ``None``
    """
    if sources is not None and len(sources) != len(combo_dirs):
        raise ValueError("sources は combo_dirs と同長である必要がある")
    frames = []
    for i, combo_dir in enumerate(combo_dirs):
        df = pool_predictions(combo_dir, split, _fold_names(combo_dir))
        if df is not None and len(df):
            df = df.copy()
            df[SOURCE_COL] = sources[i] if sources is not None else combo_dir
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def pooled_group_f1_from_predictions(
    df: pd.DataFrame, class_indices: Sequence[int]
) -> float:
    """予測 DataFrame の ``y_true`` / ``y_pred`` からプール group-F1 を返す

    Args:
        df: ``y_true`` / ``y_pred`` 列を持つプール済み予測
        class_indices: 平均対象のクラス index 集合

    Returns:
        プール group-F1空・列欠損なら ``nan``
    """
    if df is None or len(df) == 0:
        return _NAN
    if Y_TRUE_COL not in df.columns or Y_PRED_COL not in df.columns:
        return _NAN
    y_true = df[Y_TRUE_COL].to_numpy()
    y_pred = df[Y_PRED_COL].to_numpy()
    return pooled_group_f1(y_true, y_pred, class_indices)
