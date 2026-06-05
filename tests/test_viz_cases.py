"""visualization.cases のユニット"""

import pandas as pd

from foveamil.visualization.cases import (
    pick_cases,
    split_success_failure,
)


def _df():
    return pd.DataFrame({
        "slide_id": ["A", "B", "C", "D", "E", "F"],
        "y_true": [0, 0, 1, 1, 2, 2],
        "y_pred": [0, 1, 1, 0, 2, 2],   # A,C,E,F 正解 / B,D 失敗
        "prob_0": [0.9, 0.2, 0.1, 0.6, 0.1, 0.2],
        "prob_1": [0.05, 0.7, 0.8, 0.3, 0.2, 0.1],
        "prob_2": [0.05, 0.1, 0.1, 0.1, 0.7, 0.7],
    })


def test_split_success_failure():
    success, failure = split_success_failure(_df())
    assert set(success["slide_id"]) == {"A", "C", "E", "F"}
    assert set(failure["slide_id"]) == {"B", "D"}


def test_pick_cases_per_class_confidence():
    success, _ = split_success_failure(_df())
    picked = pick_cases(success, per_class=1, by="confidence")
    # 正解クラス 0,1,2 から各 1 件
    classes = sorted(c.y_true for c in picked)
    assert classes == [0, 1, 2]
    # クラス2 は E(0.7) と F(0.7) のうち confidence 最大が選ばれる
    assert all(c.correct for c in picked)


def test_pick_cases_target_class_and_n():
    df = _df()
    picked = pick_cases(df, n=2, target_class=2, by="confidence")
    assert len(picked) == 2
    assert all(c.y_true == 2 for c in picked)
    # confidence 降順
    assert picked[0].confidence >= picked[1].confidence
