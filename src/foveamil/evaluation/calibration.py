"""保存済み予測に対する事後較正とクラス別ロジット補正（再学習なし）

sweep が保存した ``predictions_{split}.csv`` の logit/prob を二次利用し，validation で
較正パラメタを当てて test に適用する学習やモデル再推論は一切しない（保存済み予測を
読むだけ）

提供する較正は 2 段:
  1. temperature scaling: val のロジットを温度 T で割って NLL（多クラス交差エントロピー）を
     最小化する 1 次元最適化 確率の鋭さだけを直す（argmax は不変なので分類指標は変えない）
     よって T の効用は分類指標でなく確率較正にある＝段階表では test の NLL・ECE で観測する
     （F1 が T で不変なのは仕様）
  2. クラス別ロジット補正 δ_c: T 適用後のロジットへクラスごとの加算項 δ_c を入れ，pooled-val の
     macro-F1（または指定クラス集合の group-F1）を最大化する 座標降下＋δ_c への L2 正則で
     過適合を抑える argmax を動かすので少数クラスの recall を引き上げ得る

F1 の規約: macro-F1・group-F1 とも present-only 平均（y_true 不在クラスを除外）で出す
per_class_recall の nan 規約・per-fold 側 group_f1_from_fold の『キー不在クラスを除外』と一致し
不在クラスを F1=0 算入する誤減点（δ_c 目的の希釈含む）を避ける

過適合回避: 較正は必ず全 fold の val をプールした 1 標本で当てる（fold ごとに当てると
標本が薄く過適合する）δ_c は L2 で 0 方向へ縮め，T 単独 → δ_c 追加 の段階で指標の限界効用を
分解して出す（δ_c が effが無ければ T 止まりで足りると判る）

縮退（空クラス・logit 欠損・標本不足）では nan を返すか恒等変換へ落とし例外を投げない
最適化は決定的（同入力で同結果）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, recall_score

from foveamil.evaluation.report import PROB_PREFIX

logger = logging.getLogger(__name__)

# 予測 CSV のロジット列の接頭辞
LOGIT_PREFIX = "logit_"
# 温度の探索範囲（NLL 最小化の 1 次元ブラケット）
TEMPERATURE_BOUNDS = (0.05, 100.0)
# 恒等温度（較正なし）
IDENTITY_TEMPERATURE = 1.0
# δ_c の L2 正則係数（既定）0 方向へ縮めて過適合を抑える
DEFAULT_DELTA_L2 = 1e-2
# 座標降下の最大ラウンド数と 1 軸あたりの探索範囲・点数
DEFAULT_DELTA_ROUNDS = 5
DELTA_GRID_HALF_WIDTH = 3.0
DELTA_GRID_POINTS = 25
# 確率からロジットを復元する際の下限クリップ（log(0) 回避）
PROB_FLOOR = 1e-12
# 主要流出先混同の既定報告本数
DEFAULT_TOP_CONFUSIONS = 3
# ECE の既定 bin 数（report.py の DEFAULT_N_BINS と一致）
DEFAULT_ECE_BINS = 10

_NAN = float("nan")


def _logit_columns(df: pd.DataFrame) -> List[str]:
    """``logit_*`` 列を class 添字順に返す無ければ空"""
    cols = [c for c in df.columns if c.startswith(LOGIT_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(LOGIT_PREFIX):]))


def _prob_columns(df: pd.DataFrame) -> List[str]:
    """``prob_*`` 列を class 添字順に返す"""
    cols = [c for c in df.columns if c.startswith(PROB_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(PROB_PREFIX):]))


def extract_logits(df: pd.DataFrame) -> Optional[np.ndarray]:
    """予測 DataFrame からロジット行列 ``[N, C]`` を取り出す

    ``logit_*`` 列があればそれを使う無ければ ``prob_*`` を log して復元する
    （温度較正には差が定数倍を除いて同値）両方無ければ ``None``

    Args:
        df: 予測 DataFrame

    Returns:
        ロジット行列 ``[N, C]``取得不能なら ``None``
    """
    logit_cols = _logit_columns(df)
    if logit_cols:
        return df[logit_cols].to_numpy(dtype=float)
    prob_cols = _prob_columns(df)
    if prob_cols:
        prob = df[prob_cols].to_numpy(dtype=float)
        return np.log(np.clip(prob, PROB_FLOOR, None))
    return None


def _y_true(df: pd.DataFrame) -> np.ndarray:
    """予測 DataFrame の正解ラベル列を返す"""
    return df["y_true"].to_numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    """行ごとの softmax を数値安定に計算する"""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y_true: np.ndarray) -> float:
    """ロジットと正解から平均負対数尤度を返す"""
    return _nll_from_prob(_softmax(logits), y_true)


def _nll_from_prob(prob: np.ndarray, y_true: np.ndarray) -> float:
    """較正後確率と正解から平均負対数尤度（NLL）を返す標本無しで nan"""
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0:
        return _NAN
    picked = prob[np.arange(n), y_true]
    return float(-np.mean(np.log(np.clip(picked, PROB_FLOOR, None))))


def _ece_from_prob(
    prob: np.ndarray, y_true: np.ndarray, n_bins: int = DEFAULT_ECE_BINS
) -> float:
    """較正後確率と正解から期待較正誤差（ECE）を返す

    各 bin で最大確率（信頼度）の平均と正解率の差の絶対値を標本数で重み付けて足す
    report.py の compute_ece と同じ定義（あちらは DataFrame 入力 こちらは確率配列入力）
    標本無しで nan
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    if n == 0:
        return _NAN
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        count = int(mask.sum())
        if count:
            ece += count / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(
    logits: np.ndarray, y_true: np.ndarray
) -> float:
    """val のロジットで NLL を最小化する温度 T を返す

    T で割ったロジットの softmax NLL を 1 次元有界最適化で最小化する標本不足・
    クラス 1 種のみ等の縮退では恒等温度 ``1.0`` を返す（例外を投げない）

    Args:
        logits: val のロジット ``[N, C]``
        y_true: val の正解ラベル ``[N]``（0..C-1）

    Returns:
        最適温度 T（>0）縮退時は ``1.0``
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)
    if logits.ndim != 2 or len(logits) < 1 or len(np.unique(y_true)) < 2:
        return IDENTITY_TEMPERATURE

    def objective(log_t: float) -> float:
        # 正値制約のため log スケールで探索する
        temperature = float(np.exp(log_t))
        return _nll(logits / temperature, y_true)

    lo, hi = TEMPERATURE_BOUNDS
    result = minimize_scalar(
        objective, bounds=(float(np.log(lo)), float(np.log(hi))), method="bounded"
    )
    if not result.success:
        return IDENTITY_TEMPERATURE
    return float(np.exp(result.x))


def _present_classes(y_true: np.ndarray, n_classes: int) -> List[int]:
    """y_true に支持標本が 1 つ以上あるクラスの index 列（昇順）"""
    return [c for c in range(n_classes) if int(np.sum(y_true == c)) > 0]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """present-only macro-F1（y_true に不在のクラスは平均から除外する）

    不在クラスを F1=0 で算入せず除外する規約は per_class_recall の nan 規約と一致させる
    （不在クラスを 0 減点すると T/δ_c の効用が希釈され誤誘導するため）present クラスが
    1 つも無ければ nan
    """
    present = _present_classes(y_true, n_classes)
    if len(y_true) == 0 or not present:
        return _NAN
    per_class = f1_score(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )
    return float(np.mean(per_class))


def _group_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_indices: Sequence[int]
) -> float:
    """指定クラス集合の present-only 非加重平均 F1

    集合のうち y_true に支持標本があるクラスのみで平均する（不在クラスは除外）
    per-fold 側 group_f1_from_fold の『キー不在クラスを除外』と定義を一致させる
    present な対象クラスが無ければ nan
    """
    indices = list(class_indices)
    if len(y_true) == 0 or not indices:
        return _NAN
    present = [c for c in indices if int(np.sum(y_true == c)) > 0]
    if not present:
        return _NAN
    per_class = f1_score(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )
    return float(np.mean(per_class))


def _objective_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    group_classes: Optional[Sequence[int]],
) -> float:
    """δ_c 最適化の目的（group 指定なら group-F1，無ければ macro-F1）"""
    if group_classes:
        return _group_f1(y_true, y_pred, group_classes)
    return _macro_f1(y_true, y_pred, n_classes)


def fit_class_deltas(
    logits: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    group_classes: Optional[Sequence[int]] = None,
    l2: float = DEFAULT_DELTA_L2,
    rounds: int = DEFAULT_DELTA_ROUNDS,
) -> np.ndarray:
    """val でクラス別加算ロジット δ_c を座標降下で当てる

    各 δ_c をグリッド探索で順に動かし，pooled-val の macro-F1（``group_classes``
    指定時はその group-F1）から L2 罰則 ``l2 * sum(δ^2)`` を引いた値を最大化する
    L2 は δ を 0 方向へ縮めて過適合を抑える縮退（標本無し・クラス 1 種）では
    ゼロベクトル（恒等）を返す

    探索は決定的（固定グリッド・固定巡回順）で同入力なら同結果

    Args:
        logits: T 適用後の val ロジット ``[N, C]``
        y_true: val の正解ラベル ``[N]``
        n_classes: クラス数 C
        group_classes: 目的を group-F1 にする際の対象クラス集合（None で macro-F1）
        l2: δ への L2 正則係数
        rounds: 座標降下のラウンド数

    Returns:
        δ ベクトル ``[C]``縮退時はゼロベクトル
    """
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)
    deltas = np.zeros(n_classes, dtype=float)
    if logits.ndim != 2 or len(logits) < 1 or len(np.unique(y_true)) < 2:
        return deltas

    grid = np.linspace(
        -DELTA_GRID_HALF_WIDTH, DELTA_GRID_HALF_WIDTH, DELTA_GRID_POINTS
    )

    def penalized(current: np.ndarray) -> float:
        pred = (logits + current).argmax(axis=1)
        score = _objective_score(y_true, pred, n_classes, group_classes)
        if np.isnan(score):
            return -np.inf
        return score - l2 * float(np.sum(current ** 2))

    best_obj = penalized(deltas)
    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            trial = deltas.copy()
            local_best_obj = best_obj
            local_best_val = deltas[c]
            for value in grid:
                trial[c] = value
                obj = penalized(trial)
                # tie は 0 に近い δ を優先（過適合回避）
                if obj > local_best_obj or (
                    obj == local_best_obj and abs(value) < abs(local_best_val)
                ):
                    local_best_obj = obj
                    local_best_val = value
            if local_best_val != deltas[c]:
                deltas[c] = local_best_val
                best_obj = local_best_obj
                improved = True
        if not improved:
            break
    return deltas


def apply_calibration(
    logits: np.ndarray, temperature: float, deltas: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """ロジットに T と δ_c を適用し ``(prob[N,C], pred[N])`` を返す

    Args:
        logits: 入力ロジット ``[N, C]``
        temperature: 温度 T（>0）
        deltas: クラス別加算項 ``[C]``None で δ なし

    Returns:
        ``(較正後確率, argmax 予測)``
    """
    logits = np.asarray(logits, dtype=float)
    scaled = logits / float(temperature)
    if deltas is not None:
        scaled = scaled + np.asarray(deltas, dtype=float)
    prob = _softmax(scaled)
    return prob, prob.argmax(axis=1)


def _per_class_recall(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> Dict[int, float]:
    """クラスごとの recall（当該クラスの正解標本が無ければ nan）"""
    out: Dict[int, float] = {}
    if len(y_true) == 0:
        return {c: _NAN for c in range(n_classes)}
    recalls = recall_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None,
        zero_division=0,
    )
    for c in range(n_classes):
        support = int(np.sum(y_true == c))
        out[c] = float(recalls[c]) if support > 0 else _NAN
    return out


def _top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    minority_classes: Sequence[int],
    top_k: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """少数クラスごとに主要な誤分類先（流出先）を件数降順で返す"""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for c in minority_classes:
        mask = y_true == c
        support = int(np.sum(mask))
        if support == 0:
            out[c] = []
            continue
        preds = y_pred[mask]
        flows: List[Dict[str, Any]] = []
        for target in range(n_classes):
            if target == c:
                continue
            count = int(np.sum(preds == target))
            if count > 0:
                flows.append(
                    {"to": target, "count": count, "rate": count / support}
                )
        flows.sort(key=lambda d: (-d["count"], d["to"]))
        out[c] = flows[:top_k]
    return out


def _minority_classes(y_true: np.ndarray, n_classes: int) -> List[int]:
    """支持標本数が中央値未満のクラスを少数クラスとみなす"""
    supports = np.array([int(np.sum(y_true == c)) for c in range(n_classes)])
    present = supports[supports > 0]
    if present.size == 0:
        return []
    threshold = float(np.median(present))
    return [c for c in range(n_classes) if 0 < supports[c] < threshold]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    group_classes: Optional[Sequence[int]] = None,
    top_confusions: int = DEFAULT_TOP_CONFUSIONS,
    prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """予測の指標一式（macro-F1・group-F1・per-class recall・主要流出先・NLL/ECE）を返す

    macro-F1/group-F1 は present-only 平均（y_true 不在クラスを除外＝per-class recall の
    nan 規約と一致）temperature scaling は argmax 不変なので分類指標（F1/recall）は
    構造的に変えない T の効用は確率較正にあり ``prob`` を渡すと NLL・ECE を算出して返す
    （F1 が T で不変なのは仕様であり T の価値は NLL↓/ECE↓ で観測する）

    Args:
        y_true: 正解ラベル ``[N]``
        y_pred: 予測ラベル ``[N]``
        n_classes: クラス数 C
        group_classes: group-F1 の対象クラス集合（None で算出しない）
        top_confusions: 少数クラスごとに報告する流出先の本数
        prob: 較正後確率 ``[N, C]``渡すと NLL/ECE を算出（None で両者 nan）

    Returns:
        ``{"macro_f1", "group_f1", "minority_recall", "top_confusions",
           "nll", "ece"}``
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    minority = _minority_classes(y_true, n_classes)
    recalls = _per_class_recall(y_true, y_pred, n_classes)
    nll = _nll_from_prob(prob, y_true) if prob is not None else _NAN
    ece = _ece_from_prob(prob, y_true) if prob is not None else _NAN
    return {
        "macro_f1": _macro_f1(y_true, y_pred, n_classes),
        "group_f1": (
            _group_f1(y_true, y_pred, group_classes)
            if group_classes else None
        ),
        "nll": nll,
        "ece": ece,
        "minority_classes": minority,
        "minority_recall": {c: recalls[c] for c in minority},
        "per_class_recall": recalls,
        "top_confusions": _top_confusions(
            y_true, y_pred, n_classes, minority, top_confusions
        ),
    }


def calibrate_val_to_test(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_classes: Optional[Sequence[int]] = None,
    l2: float = DEFAULT_DELTA_L2,
    top_confusions: int = DEFAULT_TOP_CONFUSIONS,
) -> Dict[str, Any]:
    """pooled-val で T と δ_c を当て test に適用し before/after を返す

    段階寄与分解: ``baseline``（較正なし）→ ``temperature``（T 単独）→ ``temperature_delta``
    （T+δ_c）の 3 段で test 指標を出し，各段の限界効用（前段との差）を併記する
    各段は分類指標（present-only macro/group-F1・recall）に加え確率較正指標 NLL/ECE を出す
    temperature scaling は argmax 不変なので F1/recall を構造的に変えない（T 段の F1 寄与が
    0 なのは仕様）T の本来効用は NLL↓/ECE↓ で観測できるよう段間差を marginal に併記する
    val/test の logit が取れない・標本不足では恒等変換へ落とし指標は計算できる範囲で返す
    （例外を投げない）

    Args:
        val_df: pooled-val の予測（``y_true`` と ``logit_*`` または ``prob_*``）
        test_df: test の予測（同上）
        group_classes: 目的・報告を group-F1 にする対象クラス集合（None で macro-F1 のみ）
        l2: δ_c の L2 正則係数
        top_confusions: 少数クラスごとに報告する流出先本数

    Returns:
        ``{"temperature", "deltas", "n_classes", "n_val", "n_test",
           "stages": {段名: 指標}, "marginal": {...}, "logit_source": str}``
    """
    val_logits = extract_logits(val_df)
    test_logits = extract_logits(test_df)
    val_y = _y_true(val_df)
    test_y = _y_true(test_df)

    # クラス数は prob/logit 列数から決める（欠損クラスがあっても固定 C を保つ）
    n_classes = max(
        len(_prob_columns(val_df)), len(_logit_columns(val_df)),
        len(_prob_columns(test_df)), len(_logit_columns(test_df)),
        int(np.max(test_y)) + 1 if len(test_y) else 0,
        int(np.max(val_y)) + 1 if len(val_y) else 0,
    )

    logit_source = "logit" if _logit_columns(val_df) else "prob"

    if val_logits is None or test_logits is None or n_classes < 2:
        # 較正不能恒等で before のみ返す
        base_pred = (
            test_logits.argmax(axis=1) if test_logits is not None
            else _fallback_pred(test_df)
        )
        baseline = evaluate_predictions(
            test_y, base_pred, n_classes, group_classes, top_confusions
        )
        return {
            "temperature": IDENTITY_TEMPERATURE,
            "deltas": [0.0] * n_classes,
            "n_classes": n_classes,
            "n_val": int(len(val_y)),
            "n_test": int(len(test_y)),
            "stages": {"baseline": baseline},
            "marginal": {},
            "logit_source": logit_source,
        }

    temperature = fit_temperature(val_logits, val_y)
    val_scaled = val_logits / temperature
    deltas = fit_class_deltas(
        val_scaled, val_y, n_classes, group_classes=group_classes, l2=l2
    )

    # test に各段を適用確率も渡し NLL/ECE を段ごとに出す
    base_prob, base_pred = apply_calibration(test_logits, IDENTITY_TEMPERATURE)
    temp_prob, temp_pred = apply_calibration(test_logits, temperature)
    full_prob, full_pred = apply_calibration(test_logits, temperature, deltas)

    stages = {
        "baseline": evaluate_predictions(
            test_y, base_pred, n_classes, group_classes, top_confusions,
            prob=base_prob,
        ),
        "temperature": evaluate_predictions(
            test_y, temp_pred, n_classes, group_classes, top_confusions,
            prob=temp_prob,
        ),
        "temperature_delta": evaluate_predictions(
            test_y, full_pred, n_classes, group_classes, top_confusions,
            prob=full_prob,
        ),
    }
    marginal = _marginal_utility(stages, group_classes)

    return {
        "temperature": temperature,
        "deltas": deltas.tolist(),
        "n_classes": n_classes,
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
        "stages": stages,
        "marginal": marginal,
        "logit_source": logit_source,
    }


def _fallback_pred(df: pd.DataFrame) -> np.ndarray:
    """logit 不能時の予測保存済み ``y_pred`` があれば使い無ければ prob argmax"""
    if "y_pred" in df.columns:
        return df["y_pred"].to_numpy()
    prob_cols = _prob_columns(df)
    if prob_cols:
        return df[prob_cols].to_numpy(dtype=float).argmax(axis=1)
    return np.zeros(len(df), dtype=int)


def _diff(after: float, before: float) -> float:
    """段間差（nan を伝播）"""
    if np.isnan(after) or np.isnan(before):
        return _NAN
    return float(after - before)


def _marginal_utility(
    stages: Dict[str, Dict[str, Any]], group_classes: Optional[Sequence[int]]
) -> Dict[str, Any]:
    """段階寄与分解 baseline→T→T+δ の各段の限界効用を返す

    macro/group-F1 は分類指標で T 段では構造的に不変（argmax 不変）になる T 本来の
    効用は確率較正なので NLL・ECE の段間差も併記する（負＝改善）これにより F1 が
    T で 0 寄与に見えても T の価値（NLL↓/ECE↓）が観測できる
    """
    base = stages["baseline"]
    temp = stages["temperature"]
    full = stages["temperature_delta"]
    out: Dict[str, Any] = {
        "macro_f1": {
            "temperature": _diff(temp["macro_f1"], base["macro_f1"]),
            "delta": _diff(full["macro_f1"], temp["macro_f1"]),
            "total": _diff(full["macro_f1"], base["macro_f1"]),
        },
        "nll": {
            "temperature": _diff(temp["nll"], base["nll"]),
            "delta": _diff(full["nll"], temp["nll"]),
            "total": _diff(full["nll"], base["nll"]),
        },
        "ece": {
            "temperature": _diff(temp["ece"], base["ece"]),
            "delta": _diff(full["ece"], temp["ece"]),
            "total": _diff(full["ece"], base["ece"]),
        },
    }
    if group_classes:
        out["group_f1"] = {
            "temperature": _diff(temp["group_f1"], base["group_f1"]),
            "delta": _diff(full["group_f1"], temp["group_f1"]),
            "total": _diff(full["group_f1"], base["group_f1"]),
        }
    return out
