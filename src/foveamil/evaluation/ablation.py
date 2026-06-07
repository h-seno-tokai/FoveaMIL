"""sweep 出力をアブレーション表に集計する

各 combo の ``config.yaml`` から手法タグ（ABMIL / CLAM / 多倍率ベースライン
``FoveaMIL(no-A/B/C/D)`` / A・B・D の組合せ / MCTS）と倍率レジームを判定し，
``cv_summary.json`` の集計から指標の mean±std と信頼区間を読む同一倍率レジーム内で
多倍率ベースライン（成分すべて off）との差分 Δ・対応 fold 差からの NB 補正 t の p・
多重比較補正後 p を付け，group-F1（指定クラス集合の非加重平均）も指標にできる
複数の sweep 出力ルートをまたいで集計できる（成分群と MCTS を別ルートで回した場合に
1 表へまとめる）素集計（:func:`collect_ablation`）は後方互換で残すプール group-F1 比較
（:func:`pooled_group_f1_compare`）は予測 CSV を出所キー（seed 由来）付きでプールし，
``[slide_id, source]`` で対応付けて多 seed の直積化を防ぐ集める行は予測 CSV ベースの
:func:`collect_pooled_rows`（per-fold class F1 欠如での無言脱落を避ける）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from foveamil.evaluation.group_metrics import (
    SOURCE_COL,
    Y_PRED_COL,
    Y_TRUE_COL,
    group_f1_summary,
    pool_combo_predictions,
    pooled_group_f1,
)

logger = logging.getLogger(__name__)
from foveamil.evaluation.stats import (
    adjust_pvalues,
    nadeau_bengio_corrected_t,
    paired_group_f1_permutation_test,
    stratified_bootstrap_group_f1_ci,
)

# combo ディレクトリ内のファイル名
COMBO_CONFIG_NAME = "config.yaml"
CV_SUMMARY_JSON = "cv_summary.json"
# 多倍率ベースライン（A/B/C/D すべて off の差分可能駆動）のラベル
BASELINE_LABEL = "FoveaMIL(no-A/B/C/D)"
# 成分タグの接頭辞（自前アーキの構成を表す）
METHOD_PREFIX = "FoveaMIL+"
# 探索駆動（C）のラベル
MCTS_LABEL = "FoveaMIL+MCTS(C)"
# 単一倍率のラベル
ABMIL_LABEL = "ABMIL"
CLAM_LABEL = "CLAM"
# 探索駆動を表す zoom_driver の値
ZOOM_DRIVER_MCTS = "mcts"
# 設定キー
MAGNIFICATIONS_KEY = "magnifications"
INSTANCE_LOSS_KEY = "instance_loss"
ZOOM_DRIVER_KEY = "zoom_driver"
DECORRELATION_WEIGHT_KEY = "decorrelation_weight"
AUX_NORM_KEY = "aux_norm"
SELECTOR_KEY = "selector"
# 出所キーの素にする seed 設定キー（多 seed プールで run を識別する）
SEED_KEY = "seed"
# B(スパース) を表す aux_norm 値
SPARSE_AUX_NORM = "entmax"
# D(多様性) を表す selector 値
DPP_SELECTOR = "dpp"


def tag_combo(config: Dict[str, Any]) -> Tuple[str, str]:
    """combo 設定から ``(倍率レジーム, 手法ラベル)`` を判定する

    単一倍率は ``instance_loss`` で ABMIL/CLAM，多倍率は ``zoom_driver`` と
    A(``decorrelation_weight>0``)/B(``aux_norm==entmax``)/D(``selector==dpp``) の
    組合せでラベル付けする

    Args:
        config: combo の解決済み設定辞書

    Returns:
        ``(regime, label)``
    """
    mags = list(config[MAGNIFICATIONS_KEY])
    if len(mags) == 1:
        regime = f"single-{_fmt_mag(mags[0])}x"
        label = CLAM_LABEL if config.get(INSTANCE_LOSS_KEY) else ABMIL_LABEL
        return regime, label

    regime = "multi-" + "/".join(_fmt_mag(m) for m in mags) + "x"
    if config.get(ZOOM_DRIVER_KEY) == ZOOM_DRIVER_MCTS:
        return regime, MCTS_LABEL

    methods = []
    if float(config.get(DECORRELATION_WEIGHT_KEY, 0.0)) > 0.0:
        methods.append("A")
    if config.get(AUX_NORM_KEY) == SPARSE_AUX_NORM:
        methods.append("B")
    if config.get(SELECTOR_KEY) == DPP_SELECTOR:
        methods.append("D")
    if not methods:
        return regime, BASELINE_LABEL
    return regime, METHOD_PREFIX + "".join(methods)


def _fmt_mag(mag: float) -> str:
    """倍率を表示用文字列にする（整数なら小数点を落とす）"""
    return str(int(mag)) if float(mag).is_integer() else str(mag)


def _read_yaml(path: str) -> Optional[Dict[str, Any]]:
    """YAML を読む存在しなければ ``None``"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """JSON を読む存在しなければ ``None``"""
    import json

    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_ablation(
    out_roots: List[str], metric: str, split: str = "test"
) -> List[Dict[str, Any]]:
    """sweep 出力ルート群から combo ごとのアブレーション行を集める

    各ルート直下の combo ディレクトリ（``config.yaml`` と ``cv_summary.json`` を持つ）
    を走査し，手法タグと指標の集計（mean/std/CI）を読む``cv_summary`` の当該 split
    集計に ``metric`` が無い combo は飛ばす

    Args:
        out_roots: sweep の ``--out`` ルートの列
        metric: 集計指標名（例 ``weighted_f1``）
        split: 集計する split（``test`` / ``val``）

    Returns:
        行辞書の列（``regime`` / ``label`` / ``mean`` / ``std`` / ``ci_low`` /
        ``ci_high`` / ``n`` / ``combo`` / ``root``）
    """
    rows: List[Dict[str, Any]] = []
    for root in out_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            combo_dir = os.path.join(root, name)
            config = _read_yaml(os.path.join(combo_dir, COMBO_CONFIG_NAME))
            summary = _read_json(os.path.join(combo_dir, CV_SUMMARY_JSON))
            if config is None or summary is None:
                continue
            aggregate = summary.get(split, {}).get("aggregate", {})
            if metric not in aggregate:
                continue
            regime, label = tag_combo(config)
            agg = aggregate[metric]
            rows.append(
                {
                    "regime": regime,
                    "label": label,
                    "mean": float(agg["mean"]),
                    "std": float(agg["std"]),
                    "ci_low": agg.get("ci_t_low"),
                    "ci_high": agg.get("ci_t_high"),
                    "n": agg.get("n"),
                    "combo": name,
                    "root": root,
                }
            )
    return rows


# group-F1 を指す擬似メトリクス名（クラス集合を併せて指定する）
GROUP_F1_METRIC = "group_f1"


def _per_fold(summary: Dict[str, Any], split: str) -> List[Dict[str, float]]:
    """cv_summary から当該 split の per_fold 指標列を取り出す"""
    return summary.get(split, {}).get("per_fold", [])


def _metric_per_fold(
    summary: Dict[str, Any],
    split: str,
    metric: str,
    group_classes: Optional[Sequence[int]] = None,
) -> List[float]:
    """combo の per_fold から ``metric`` の fold ごとの値列を作る

    ``metric == GROUP_F1_METRIC`` のときは ``group_classes`` の非加重平均（group-F1）を
    fold ごとに算出する通常指標は per_fold 辞書のキーをそのまま読む（欠損 fold は除外）

    Args:
        summary: combo の ``cv_summary.json`` 辞書
        split: ``test`` / ``val``
        metric: 指標名または ``GROUP_F1_METRIC``
        group_classes: group-F1 のクラス index 集合

    Returns:
        fold ごとの値列（group-F1 では nan を含み得る）
    """
    folds = _per_fold(summary, split)
    if metric == GROUP_F1_METRIC:
        return group_f1_summary(folds, group_classes or [])["per_fold"]
    return [float(m[metric]) for m in folds if metric in m]


def _row_mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    """nan を除いた値列の ``(mean, std, n)`` を返す有効値なしは nan"""
    import numpy as np

    valid = [v for v in values if not np.isnan(v)]
    if not valid:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


def collect_ablation_rows(
    out_roots: List[str],
    metric: str,
    split: str = "test",
    group_classes: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """per_fold を保持した combo 行を集める（差分・検定の素材）

    各 combo の per_fold から ``metric``（または group-F1）の fold 値列を取り出し，
    mean/std と共に保持する``cv_summary`` の per_fold に当該指標が無い combo は飛ばす
    （group-F1 は全 fold で構成クラスが欠損なら飛ばす）

    Args:
        out_roots: sweep の ``--out`` ルートの列
        metric: 指標名または ``GROUP_F1_METRIC``
        split: 集計する split（``test`` / ``val``）
        group_classes: group-F1 のクラス index 集合

    Returns:
        行辞書の列（``regime`` / ``label`` / ``per_fold`` / ``mean`` / ``std`` /
        ``n`` / ``combo`` / ``root``）
    """
    import numpy as np

    rows: List[Dict[str, Any]] = []
    for root in out_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            combo_dir = os.path.join(root, name)
            config = _read_yaml(os.path.join(combo_dir, COMBO_CONFIG_NAME))
            summary = _read_json(os.path.join(combo_dir, CV_SUMMARY_JSON))
            if config is None or summary is None:
                continue
            values = _metric_per_fold(summary, split, metric, group_classes)
            if not values or all(np.isnan(v) for v in values):
                continue
            mean, std, n = _row_mean_std(values)
            regime, label = tag_combo(config)
            rows.append(
                {
                    "regime": regime,
                    "label": label,
                    "per_fold": list(values),
                    "mean": mean,
                    "std": std,
                    "n": n,
                    "combo": name,
                    "root": root,
                }
            )
    return rows


def compare_to_baseline(
    rows: List[Dict[str, Any]],
    n_train: int,
    n_test: int,
    baseline_label: str = BASELINE_LABEL,
    adjust_method: str = "holm",
) -> List[Dict[str, Any]]:
    """各レジーム内で baseline 行に対する Δ・NB 補正 t の p・補正後 p を付与する

    レジームごとに ``baseline_label`` の行を基準にし，各 method の対応 fold 差
    （method - baseline・共通 fold 数まで）から平均差 Δ と Nadeau-Bengio 補正 t の
    p 値を求める同一レジームの method 群の p に :func:`adjust_pvalues` で多重比較補正を
    かけ補正後 p を付ける baseline が無いレジーム・baseline 自身・fold 数不一致の縮退では
    Δ/p を ``None``/``nan`` にする入力 ``rows`` は :func:`collect_ablation_rows` の行

    Args:
        rows: per_fold を持つ combo 行
        n_train: 1 fold の訓練サンプル数（NB 補正用）
        n_test: 1 fold の test サンプル数（NB 補正用）
        baseline_label: 基準とする手法ラベル
        adjust_method: 多重比較補正法（``holm`` / ``fdr_bh``）

    Returns:
        ``rows`` の各行に ``delta`` / ``pvalue`` / ``pvalue_adj`` を加えた新しい行列
    """
    by_regime: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], []).append(row)

    enriched: List[Dict[str, Any]] = []
    for regime in sorted(by_regime):
        group = by_regime[regime]
        baseline = next(
            (r for r in group if r["label"] == baseline_label), None
        )
        base_fold = baseline["per_fold"] if baseline is not None else None

        pvalues: List[float] = []
        targets: List[Dict[str, Any]] = []
        out_group: List[Dict[str, Any]] = []
        for row in group:
            new_row = dict(row)
            new_row["delta"] = None
            new_row["pvalue"] = float("nan")
            new_row["pvalue_adj"] = float("nan")
            if base_fold is not None and row["label"] != baseline_label:
                diffs = _paired_diffs(row["per_fold"], base_fold)
                if diffs:
                    new_row["delta"] = float(sum(diffs) / len(diffs))
                    res = nadeau_bengio_corrected_t(diffs, n_train, n_test)
                    new_row["pvalue"] = res["pvalue"]
                    pvalues.append(res["pvalue"])
                    targets.append(new_row)
            out_group.append(new_row)

        if pvalues:
            adjusted = adjust_pvalues(pvalues, method=adjust_method)["adjusted"]
            for target, adj in zip(targets, adjusted):
                target["pvalue_adj"] = adj
        enriched.extend(out_group)
    return enriched


def _paired_diffs(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """共通 fold まで対応差 ``a - b`` を取り両方有効な fold のみ残す"""
    import numpy as np

    n = min(len(a), len(b))
    diffs: List[float] = []
    for i in range(n):
        if not (np.isnan(a[i]) or np.isnan(b[i])):
            diffs.append(float(a[i] - b[i]))
    return diffs


# 予測 CSV のスライド識別列
SLIDE_ID_COL = "slide_id"


def collect_pooled_rows(
    out_roots: List[str], split: str = "test"
) -> List[Dict[str, Any]]:
    """プール group-F1 比較用に予測 CSV ベースで combo 行を集める

    プール経路は per-fold の per-class F1（``cv_summary`` 依存）ではなく，保存済み
    予測 CSV から症例を集める``cv_summary`` に per-class F1 が無くても予測さえ
    あれば対象に含める（per-fold 経路の :func:`collect_ablation_rows` で無言脱落
    していた combo を拾う）``config.yaml`` を持つが当該 split の予測 CSV が読めない
    combo は警告して飛ばす（無言 skip はしない）

    Args:
        out_roots: sweep の ``--out`` ルートの列
        split: 読む予測 split（``test`` / ``val``）

    Returns:
        行辞書の列（``regime`` / ``label`` / ``combo`` / ``root``）
    """
    rows: List[Dict[str, Any]] = []
    for root in out_roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            combo_dir = os.path.join(root, name)
            config = _read_yaml(os.path.join(combo_dir, COMBO_CONFIG_NAME))
            if config is None:
                continue
            preds = pool_combo_predictions([combo_dir], split)
            if preds is None or not len(preds):
                logger.warning(
                    "pooled: combo %s に split=%s の予測 CSV が無く脱落",
                    combo_dir, split,
                )
                continue
            regime, label = tag_combo(config)
            rows.append(
                {"regime": regime, "label": label, "combo": name, "root": root}
            )
    return rows


def _combo_dirs_for_label(
    rows: List[Dict[str, Any]], regime: str, label: str
) -> List[str]:
    """同一レジーム・同一手法ラベルの combo ディレクトリ群を集める

    同手法が複数 out_root / seed に跨る場合に全 combo を束ねる用途
    """
    return [
        os.path.join(row["root"], row["combo"])
        for row in rows
        if row["regime"] == regime and row["label"] == label
    ]


def _combo_seed(combo_dir: str) -> Optional[Any]:
    """combo の ``config.yaml`` から seed を読む無ければ ``None``"""
    config = _read_yaml(os.path.join(combo_dir, COMBO_CONFIG_NAME))
    if config is not None and SEED_KEY in config and config[SEED_KEY] is not None:
        return config[SEED_KEY]
    return None


def _combo_sources(combo_dirs: Sequence[str]) -> List[Any]:
    """combo ディレクトリ群の出所キー列を作る

    出所キーは method と baseline で **同一 run（同一 seed）同士**が一致する値で
    なければ ``[slide_id, source]`` の対応付けが成立しない各 combo の seed を読み，
    全 combo で seed が読めるならそれを使う（多 seed プールで run を一意に識別）
    seed を読めない combo があるときは label の combo 列の **位置 index** に揃える
    （単一 CV や seed 未記録の出力で method/baseline を同位置同士で対応付ける）

    Args:
        combo_dirs: combo ディレクトリのパス列（label ごとの combo 群）

    Returns:
        各 combo の出所キー列（``combo_dirs`` と同長）
    """
    seeds = [_combo_seed(d) for d in combo_dirs]
    if seeds and all(s is not None for s in seeds):
        return seeds
    # seed 未記録の出力では位置 index で揃える（method/baseline を同位置同士で対応）
    return list(range(len(combo_dirs)))


def pooled_group_f1_compare(
    rows: List[Dict[str, Any]],
    group_classes: Sequence[int],
    split: str = "test",
    baseline_label: str = BASELINE_LABEL,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """各レジーム内で baseline に対するプール group-F1 の Δ・並べ替え p・bootstrap CI を付与する

    fold（必要なら複数 out_root / seed）の予測をプールし，baseline と各手法を
    対応付けて同一テスト症例集合に揃える各 combo 行に出所キー（seed 由来の
    ``source`` 列）を付け，対応付けは ``[slide_id, source]`` を単位に行う単一 CV では
    各 slide が test に 1 度だけ現れ対応は 1 対 1多 seed プールでは同一 slide_id が
    seed ごとに現れるため，``slide_id`` だけで merge すると直積化して N が水増しされ
    method-seed と別 baseline-seed が誤対応する``source``（seed）を merge キーに
    含めて同一 seed 同士で 1 対 1 に揃えることでこれを防ぐ対応した予測に対し，プール
    group-F1 の差 Δ・対応あり並べ替え検定の p・クラス層化 bootstrap の差の CI を求める

    baseline が無いレジーム・baseline 自身・対応症例が無い縮退では Δ/p/CI を
    ``None``/``nan`` にする入力 ``rows`` は :func:`collect_ablation_rows` の行
    （``regime`` / ``label`` / ``root`` / ``combo`` を用いる）

    Args:
        rows: combo 行（``regime`` / ``label`` / ``root`` / ``combo`` を持つ）
        group_classes: group-F1 を構成するクラス index 集合
        split: 読む予測 split（``test`` / ``val``）
        baseline_label: 基準とする手法ラベル
        n_perm: 並べ替え反復数
        n_boot: bootstrap 反復数
        seed: 乱数シード（再現性のため固定）

    Returns:
        ``rows`` の各行に ``pooled_gf1`` / ``pooled_delta`` / ``perm_pvalue`` /
        ``boot_ci_low`` / ``boot_ci_high`` を加えた新しい行列
    """
    import numpy as np

    labels = list(group_classes)
    by_regime: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], []).append(row)

    enriched: List[Dict[str, Any]] = []
    for regime in sorted(by_regime):
        group = by_regime[regime]
        base_dirs = _combo_dirs_for_label(group, regime, baseline_label)
        base_df = (
            pool_combo_predictions(base_dirs, split, _combo_sources(base_dirs))
            if base_dirs
            else None
        )

        seen_labels: set = set()
        for row in group:
            new_row = dict(row)
            new_row["pooled_gf1"] = float("nan")
            new_row["pooled_delta"] = None
            new_row["perm_pvalue"] = float("nan")
            new_row["boot_ci_low"] = float("nan")
            new_row["boot_ci_high"] = float("nan")

            label = row["label"]
            # 同手法が複数 combo に跨っても各ラベルにつき 1 度だけ集計する
            if label in seen_labels:
                enriched.append(new_row)
                continue
            seen_labels.add(label)

            method_dirs = _combo_dirs_for_label(group, regime, label)
            method_df = pool_combo_predictions(
                method_dirs, split, _combo_sources(method_dirs)
            )
            if method_df is not None and len(method_df):
                new_row["pooled_gf1"] = pooled_group_f1(
                    method_df[Y_TRUE_COL].to_numpy(),
                    method_df[Y_PRED_COL].to_numpy(),
                    labels,
                )

            if (
                base_df is None
                or method_df is None
                or label == baseline_label
                or not labels
            ):
                enriched.append(new_row)
                continue

            # [slide_id, source] を単位に 1 対 1 対応させる（多 seed の直積を防ぐ）
            merged = method_df.merge(
                base_df, on=[SLIDE_ID_COL, SOURCE_COL], suffixes=("_m", "_b")
            )
            if not len(merged):
                enriched.append(new_row)
                continue
            y_true = merged[f"{Y_TRUE_COL}_m"].to_numpy()
            y_pred_m = merged[f"{Y_PRED_COL}_m"].to_numpy()
            y_pred_b = merged[f"{Y_PRED_COL}_b"].to_numpy()

            gf1_m = pooled_group_f1(y_true, y_pred_m, labels)
            gf1_b = pooled_group_f1(y_true, y_pred_b, labels)
            if not (np.isnan(gf1_m) or np.isnan(gf1_b)):
                new_row["pooled_delta"] = float(gf1_m - gf1_b)
            perm = paired_group_f1_permutation_test(
                y_true, y_pred_m, y_pred_b, labels, n_perm=n_perm, seed=seed
            )
            new_row["perm_pvalue"] = perm["pvalue"]
            ci = stratified_bootstrap_group_f1_ci(
                y_true, y_pred_m, labels, y_pred_b=y_pred_b,
                n_boot=n_boot, seed=seed,
            )
            new_row["boot_ci_low"] = ci["ci_low"]
            new_row["boot_ci_high"] = ci["ci_high"]
            enriched.append(new_row)
    return enriched


def format_markdown_pooled(
    rows: List[Dict[str, Any]], split: str = "test"
) -> str:
    """プール group-F1 の Δ・並べ替え p・bootstrap CI を markdown 表に整形する

    入力は :func:`pooled_group_f1_compare` の行レジームごとにプール group-F1・
    Δ・並べ替え p・bootstrap CI を並べる

    Args:
        rows: プール比較を付与した combo 行
        split: 集計 split（見出し用）

    Returns:
        markdown 文字列
    """
    import numpy as np

    if not rows:
        return f"# Pooled group-F1 ({split})\n\n(no combos found)\n"

    regimes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        regimes.setdefault(row["regime"], []).append(row)

    lines = [f"# Pooled group-F1 ({split})", ""]
    for regime in sorted(regimes):
        group = sorted(
            regimes[regime],
            key=lambda r: (-1.0 if np.isnan(r["pooled_gf1"]) else r["pooled_gf1"]),
            reverse=True,
        )
        lines.append(f"## {regime}")
        lines.append("")
        lines.append(
            "| method | pooled group-F1 | Δ vs baseline | p (perm) | "
            "95% CI (Δ) | combo |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in group:
            gf1 = (
                "-" if np.isnan(row["pooled_gf1"]) else f"{row['pooled_gf1']:.4f}"
            )
            delta = (
                f"{row['pooled_delta']:+.4f}"
                if row.get("pooled_delta") is not None
                else "-"
            )
            p_perm = _fmt_p(row.get("perm_pvalue"))
            lo, hi = row.get("boot_ci_low"), row.get("boot_ci_high")
            ci = (
                f"[{lo:.4f}, {hi:.4f}]"
                if not (np.isnan(lo) or np.isnan(hi))
                else "-"
            )
            lines.append(
                f"| {row['label']} | {gf1} | {delta} | {p_perm} | {ci} | "
                f"{row['combo']} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_markdown(rows: List[Dict[str, Any]], metric: str, split: str = "test") -> str:
    """アブレーション行を倍率レジームごとの markdown 表に整形する

    各レジーム内で多倍率ベースライン（``FoveaMIL(no-A/B/C/D)``）との差分 Δ を付けるベース
    ラインが無いレジーム（単一倍率など）では Δ 欄を空にする

    Args:
        rows: :func:`collect_ablation` の行
        metric: 表に出す指標名
        split: 集計 split（見出し用）

    Returns:
        markdown 文字列
    """
    if not rows:
        return f"# Ablation ({metric}, {split})\n\n(no combos found)\n"

    regimes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        regimes.setdefault(row["regime"], []).append(row)

    lines = [f"# Ablation ({metric}, {split})", ""]
    for regime in sorted(regimes):
        group = sorted(regimes[regime], key=lambda r: r["mean"], reverse=True)
        baseline = next(
            (r["mean"] for r in group if r["label"] == BASELINE_LABEL), None
        )
        lines.append(f"## {regime}")
        lines.append("")
        lines.append(f"| method | {metric} (mean ± std) | 95% CI | Δ vs baseline | combo |")
        lines.append("|---|---|---|---|---|")
        for row in group:
            mean_std = f"{row['mean']:.4f} ± {row['std']:.4f}"
            if row["ci_low"] is not None and row["ci_high"] is not None:
                ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
            else:
                ci = "-"
            if baseline is not None and row["label"] != BASELINE_LABEL:
                delta = f"{row['mean'] - baseline:+.4f}"
            else:
                delta = "-"
            lines.append(
                f"| {row['label']} | {mean_std} | {ci} | {delta} | {row['combo']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt_p(value: Optional[float]) -> str:
    """p 値（nan/None 可）を表示文字列にする"""
    import numpy as np

    if value is None or np.isnan(value):
        return "-"
    return f"{value:.4g}"


def format_markdown_compare(
    rows: List[Dict[str, Any]],
    metric: str,
    split: str = "test",
    adjust_method: str = "holm",
) -> str:
    """Δ・補正 t の p・補正後 p を含むアブレーション表に整形する

    入力は :func:`compare_to_baseline` の行（``delta`` / ``pvalue`` / ``pvalue_adj``
    を持つ）レジームごとに mean±std・Δ・p・補正後 p を並べる

    Args:
        rows: 差分・検定を付与した combo 行
        metric: 表に出す指標名
        split: 集計 split（見出し用）
        adjust_method: 補正後 p 列の見出しに出す補正法名

    Returns:
        markdown 文字列
    """
    if not rows:
        return f"# Ablation ({metric}, {split})\n\n(no combos found)\n"

    regimes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        regimes.setdefault(row["regime"], []).append(row)

    lines = [f"# Ablation ({metric}, {split})", ""]
    for regime in sorted(regimes):
        group = sorted(regimes[regime], key=lambda r: r["mean"], reverse=True)
        lines.append(f"## {regime}")
        lines.append("")
        lines.append(
            f"| method | {metric} (mean ± std) | Δ vs baseline | p (NB) | "
            f"p ({adjust_method}) | combo |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in group:
            mean_std = f"{row['mean']:.4f} ± {row['std']:.4f}"
            delta = f"{row['delta']:+.4f}" if row.get("delta") is not None else "-"
            p_raw = _fmt_p(row.get("pvalue"))
            p_adj = _fmt_p(row.get("pvalue_adj"))
            lines.append(
                f"| {row['label']} | {mean_std} | {delta} | {p_raw} | {p_adj} | "
                f"{row['combo']} |"
            )
        lines.append("")
    return "\n".join(lines)
