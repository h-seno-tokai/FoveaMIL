"""保存済み予測から成功/失敗症例を選ぶ（純データ処理）

sweep が保存した ``predictions_{split}.csv``（slide_id,y_true,y_pred,prob_*）を読み，
``y_true == y_pred`` で成功/失敗に二分して代表症例を返す予測の読み込みは
``evaluation.report`` と同契約（fold 横断プール対応）WSI もモデルも知らない
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from foveamil.evaluation.report import pool_predictions

# 予測 CSV の確率列の接頭辞
PROB_PREFIX = "prob_"
# 代表症例の選定基準
BY_CONFIDENCE = "confidence"
BY_MARGIN = "margin"


@dataclass
class CaseRef:
    """1 症例の選定結果

    Attributes:
        slide_id: スライド識別子
        y_true: 正解クラス
        y_pred: 予測クラス
        correct: 正解か（``y_true == y_pred``）
        confidence: 予測確率の最大値
        margin: 最大確率と次点の差
    """

    slide_id: str
    y_true: int
    y_pred: int
    correct: bool
    confidence: float
    margin: float


def _prob_columns(df: pd.DataFrame) -> List[str]:
    """``prob_*`` 列を class 添字順に返す"""
    cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(PROB_PREFIX):]))


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    """予測 DataFrame に correct / confidence / margin 列を足す"""
    out = df.copy()
    cols = _prob_columns(df)
    probs = out[cols].to_numpy(dtype=float) if cols else np.zeros((len(out), 1))
    out["correct"] = out["y_true"] == out["y_pred"]
    out["confidence"] = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    out["margin"] = (
        sorted_p[:, -1] - sorted_p[:, -2] if probs.shape[1] >= 2 else sorted_p[:, -1]
    )
    return out


def load_cases_frame(
    combo_dir: str, split: str, fold_names: List[str]
) -> Optional[pd.DataFrame]:
    """combo の全 fold 予測をプールし補助列付きで返す読めなければ ``None``"""
    df = pool_predictions(combo_dir, split, fold_names)
    if df is None or not len(df):
        return None
    return _augment(df)


def split_success_failure(df: pd.DataFrame) -> tuple:
    """``(成功 df, 失敗 df)`` に二分する（``y_true == y_pred``）"""
    augmented = df if "correct" in df.columns else _augment(df)
    return augmented[augmented["correct"]], augmented[~augmented["correct"]]


def _to_caseref(row) -> CaseRef:
    return CaseRef(
        slide_id=str(row["slide_id"]),
        y_true=int(row["y_true"]),
        y_pred=int(row["y_pred"]),
        correct=bool(row["correct"]),
        confidence=float(row["confidence"]),
        margin=float(row["margin"]),
    )


def pick_cases(
    df: pd.DataFrame,
    n: Optional[int] = None,
    per_class: Optional[int] = None,
    target_class: Optional[int] = None,
    by: str = BY_CONFIDENCE,
) -> List[CaseRef]:
    """代表症例を選んで :class:`CaseRef` の列で返す

    ``target_class`` 指定時は正解クラスで絞り，``by`` 降順に並べる``per_class`` 指定時は
    正解クラスごとに上位 ``per_class`` 件，そうでなければ上位 ``n`` 件を返す

    Args:
        df: 補助列付き予測 DataFrame（無ければ内部で付与）
        n: 返す総件数（``per_class`` 未指定時）
        per_class: 正解クラスごとの件数
        target_class: 正解クラスで絞る
        by: 並べ替え基準（``confidence`` / ``margin``）

    Returns:
        :class:`CaseRef` の列
    """
    work = df if "confidence" in df.columns else _augment(df)
    if target_class is not None:
        work = work[work["y_true"] == target_class]
    work = work.sort_values(by, ascending=False)

    if per_class is not None:
        picked = work.groupby("y_true", sort=True).head(per_class)
        picked = picked.sort_values(["y_true", by], ascending=[True, False])
        return [_to_caseref(r) for _, r in picked.iterrows()]
    if n is not None:
        work = work.head(n)
    return [_to_caseref(r) for _, r in work.iterrows()]
