"""sweep 出力をアブレーション表に集計する

各 combo の ``config.yaml`` から手法タグ（ABMIL / CLAM / ZoomMIL ベースライン / A・B・D
の組合せ / MCTS）と倍率レジームを判定し，``cv_summary.json`` の test 集計から指標の
mean±std と信頼区間を読む同一倍率レジーム内で多倍率ベースライン（手法すべて off）との
差分 Δ を付けた markdown 表を作る複数の sweep 出力ルートをまたいで集計できる
（A/B/D と MCTS を別ルートで回した場合に 1 表へまとめる）
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

# combo ディレクトリ内のファイル名
COMBO_CONFIG_NAME = "config.yaml"
CV_SUMMARY_JSON = "cv_summary.json"
# 多倍率ベースライン（A/B/D すべて off の差分可能駆動）のラベル
BASELINE_LABEL = "ZoomMIL(baseline)"
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
        return regime, "ZoomMIL+MCTS(C)"

    methods = []
    if float(config.get(DECORRELATION_WEIGHT_KEY, 0.0)) > 0.0:
        methods.append("A")
    if config.get(AUX_NORM_KEY) == SPARSE_AUX_NORM:
        methods.append("B")
    if config.get(SELECTOR_KEY) == DPP_SELECTOR:
        methods.append("D")
    if not methods:
        return regime, BASELINE_LABEL
    return regime, "ZoomMIL+" + "".join(methods)


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


def format_markdown(rows: List[Dict[str, Any]], metric: str, split: str = "test") -> str:
    """アブレーション行を倍率レジームごとの markdown 表に整形する

    各レジーム内で多倍率ベースライン（``ZoomMIL(baseline)``）との差分 Δ を付けるベース
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
