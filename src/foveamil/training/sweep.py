"""sweep 設定を combo へ展開し fold 並列で実行してランキングする

設定は ``sweep``（リスト→展開する軸）/ ``fixed``（全 combo 共通スカラ）/ ``resolve``
（解決の起点）/ ``parallel``（制御）の 4 ブロックからなる``encoder`` と ``feature_type``
は直積でなく妥当な組合せのみ残す（cls/concat は ``has_cls=True`` のエンコーダのみ）
他の軸は直積展開する各 combo に解決済みの ``in_feat_dim`` / ``feature_root`` /
``labels_csv`` / ``n_cls`` を載せ，``job=(combo, fold)`` を ``foveamil-train --split`` の
サブプロセスとして GPU へ割り当て並列実行する``test_metrics.json`` 既存の job は
スキップ（resume），失敗は記録して継続し，combo ごとに CV 集計してランキングする
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from foveamil.encoders import ENCODERS
from foveamil.training.accessor import (
    FEATURE_TYPE_MEAN,
    FEATURE_TYPES,
    FeatureAccessor,
    _format_mag,
)
from foveamil.training.config import TrainConfig
from foveamil.training.cv import (
    CV_SCHEMA_VERSION,
    CV_SUMMARY_JSON,
    SUMMARY_METRICS,
    aggregate_folds_ci,
)
from foveamil.training.resolve import (
    ResolvedPaths,
    _fold_number,
    normalize_mags,
    resolve_in_feat_dim,
)

logger = logging.getLogger(__name__)

# sweep 軸のうち特別扱いするキー
ENCODER_KEY = "encoder"
FEATURE_TYPE_KEY = "feature_type"
MAGNIFICATIONS_KEY = "magnifications"
# インスタンス補助損失の有効化キー（単一倍率のみ有効）
INSTANCE_LOSS_KEY = "instance_loss"
# 補助アテンション正規化器名のキー（多倍率のみ有効）
AUX_NORM_KEY = "aux_norm"
# 多倍率（ズーム）でのみ意味を持つキー（単一倍率では学習に無関係）
ZOOM_PARAM_KEYS = ("k_sample", "k_sigma", "topk_method", "aux_norm")
# aux_norm の値ごとにのみ意味を持つキー（他の値や単一倍率では無関係）
AUX_NORM_TEMPERATURE = "temperature"
AUX_NORM_ENTMAX = "entmax"
AUX_NORM_PARAM_KEYS = {
    "aux_norm_temperature": AUX_NORM_TEMPERATURE,
    "aux_norm_alpha": AUX_NORM_ENTMAX,
}
# ズーム駆動の選択キー（単一倍率ではズーム自体が無いため無関係）
ZOOM_DRIVER_KEY = "zoom_driver"
# 探索駆動を表す zoom_driver の値
ZOOM_DRIVER_MCTS = "mcts"
# zoom_driver="mcts" 時のみ意味を持つキー（それ以外は無関係）
MCTS_PARAM_KEYS = (
    "mcts_planner",
    "mcts_simulations",
    "mcts_max_considered",
    "policy_loss_weight",
    "value_loss_weight",
    "policy_entropy_weight",
    "mcts_hidden_dim",
)
# instance_loss 有効時のみ意味を持つキー（無効時は無関係）
INSTANCE_PARAM_KEYS = ("bag_weight", "inst_k", "inst_subtyping")
# 倍率間冗長性罰則の重みキー（多倍率のみ有効）
DECORRELATION_WEIGHT_KEY = "decorrelation_weight"
# decorrelation_weight 有効時のみ意味を持つキー（無効時は無関係）
DECORRELATION_PARAM_KEYS = ("decorrelation_method",)
# 選択コントローラ名キー
SELECTOR_KEY = "selector"
# DPP 選択コントローラ名
SELECTOR_DPP = "dpp"
# selector=="dpp" かつ多倍率でのみ意味を持つキー（他では無関係）
DPP_PARAM_KEYS = (
    "dpp_similarity",
    "dpp_temperature",
    "dpp_quality_beta",
    "dpp_rbf_gamma",
    "dpp_use_gumbel",
    "dpp_diversity_weight",
)
# 単一倍率を表す倍率数
SINGLE_MAG = 1
# 自動解決されるため設定に手書きを許さないキー
AUTO_RESOLVED_KEYS = ("in_feat_dim", "feature_root", "labels_csv", "n_cls", "classes")
# combo ディレクトリ名の接頭辞
COMBO_DIR_PREFIX = "combo_"
# combo 名でパス非対応文字を置換する文字
NAME_SANITIZE_SUB = "_"
# combo ごとの解決済み設定ファイル名
COMBO_CONFIG_NAME = "config.yaml"
# 単一 fold の test 指標ファイル名（foveamil-train の出力，skip-done 判定にも使う）
FOLD_RESULT_JSON = "test_metrics.json"
# 単一 fold の val 指標ファイル名（foveamil-train の出力）
METRICS_VAL_JSON = "metrics_val.json"
# fold ディレクトリ名の接頭辞
FOLD_DIR_PREFIX = "fold"
# combo ランキングに使う split（model selection は val）
SELECTION_SPLIT = "val"
# 報告に使う split
REPORT_SPLIT = "test"
# sweep 要約の保存ファイル名（機械可読）
SWEEP_SUMMARY_JSON = "sweep_summary.json"
# sweep 要約の保存ファイル名（人間可読）
SWEEP_SUMMARY_MD = "sweep_summary.md"
# sweep の詳細フラット CSV ファイル名（全 combo×fold×split）
SWEEP_DETAILED_CSV = "sweep_detailed.csv"
# 既定の使用 GPU 一覧
DEFAULT_GPU_IDS = (0,)
# 既定の GPU あたり並列ジョブ数
DEFAULT_JOBS_PER_GPU = 1
# ランキングの主要指標（優先順位順に最初に見つかったものを使う）
RANK_METRICS = ("macro_auc", "weighted_f1", "macro_f1", "accuracy", "kappa")


@dataclass
class Combo:
    """1 combo の展開結果

    Attributes:
        index: combo 連番
        name: パス安全な combo 名（出力ディレクトリ名）
        config: ``TrainConfig`` 互換の設定辞書（解決済み値を含む）
        axis_values: この combo の各 sweep 軸の値（表示用パスを含めない）
    """

    index: int
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    axis_values: Dict[str, Any] = field(default_factory=dict)


def _train_config_fields() -> frozenset:
    """``TrainConfig`` のフィールド名集合を返す"""
    return frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _train_config_defaults() -> Dict[str, Any]:
    """``TrainConfig`` の各フィールドの既定値を返す"""
    return {f.name: f.default for f in dataclasses.fields(TrainConfig)}


def _reject_auto_resolved(block_name: str, block: Dict[str, Any]) -> None:
    """自動解決キーが設定に手書きされていればエラーにする"""
    for key in AUTO_RESOLVED_KEYS:
        if key in block:
            raise ValueError(
                f"'{key}' must not be set in '{block_name}'; it is resolved "
                "automatically from resolve.n_cls / encoder / feature_type"
            )


def _valid_pair(encoder: str, feature_type: str) -> bool:
    """``(encoder, feature_type)`` が有効か返す

    ``mean`` は全エンコーダで有効``cls`` / ``concat`` は ``has_cls=True`` の
    エンコーダのみ有効（``has_cls=False`` の h5 には cls 特徴が無い）
    """
    if feature_type == FEATURE_TYPE_MEAN:
        return True
    return ENCODERS[encoder].has_cls


def _encoder_feature_pairs(
    encoders: Sequence[str], feature_types: Sequence[str]
) -> List[tuple]:
    """有効な ``(encoder, feature_type)`` ペアのみを列挙する（直積にしない）"""
    for enc in encoders:
        if enc not in ENCODERS:
            raise KeyError(
                f"unknown encoder '{enc}'; available: {sorted(ENCODERS)}"
            )
    for ft in feature_types:
        if ft not in FEATURE_TYPES:
            raise ValueError(
                f"unknown feature_type '{ft}'; available: {sorted(FEATURE_TYPES)}"
            )
    return [
        (enc, ft)
        for enc in encoders
        for ft in feature_types
        if _valid_pair(enc, ft)
    ]


def _as_mag_sets(value: Any) -> List[List[float]]:
    """``magnifications`` の値を倍率セットの列に正規化する

    要素がすべて list なら倍率セットの列（軸），そうでなければ単一セットとみなす

    Args:
        value: 単一セット（``[1.25, 2.5]``）または複数セット（``[[...], [...]]``）

    Returns:
        正規化済みの倍率セット列（``[[float, ...], ...]``）
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"magnifications must be a non-empty list, got {value!r}")
    if all(isinstance(item, list) for item in value):
        return [normalize_mags(item) for item in value]
    return [normalize_mags(value)]


def _listify(value: Any) -> List[Any]:
    """スカラを 1 要素 list に包む（既に list ならそのまま）"""
    return value if isinstance(value, list) else [value]


def _sanitize(text: str) -> str:
    """combo 名でパス非対応文字を安全な文字へ置換する"""
    return re.sub(r"[^0-9A-Za-z._-]", NAME_SANITIZE_SUB, str(text))


def _combo_name(index: int, encoder: str, feature_type: str, n_mags: int) -> str:
    """combo の決定的でパス安全な名前を作る"""
    enc = _sanitize(encoder)
    return f"{COMBO_DIR_PREFIX}{index:03d}__{enc}_{feature_type}_m{n_mags}"


def _disable_param(
    config: Dict[str, Any],
    axis_values: Dict[str, Any],
    defaults: Dict[str, Any],
    key: str,
) -> bool:
    """無関係なパラメータを既定値へ畳み axis_values から落とす

    設定に現れたキーのみ畳む（未指定キーは ``TrainConfig`` の既定がそのまま効く）
    明示値（既定と異なる値）を捨てた場合に ``True`` を返す（警告用）
    """
    discarded = False
    if key in config:
        discarded = config[key] != defaults[key]
        config[key] = defaults[key]
    axis_values.pop(key, None)
    return discarded


def _canonicalize_conditional(
    config: Dict[str, Any], axis_values: Dict[str, Any], defaults: Dict[str, Any]
) -> set:
    """構成に無関係な条件付きパラメータを既定値へ畳む

    正規化する）ズーム系（``k_sample`` / ``k_sigma`` / ``topk_method`` / ``aux_norm`` /
    ``selector`` / ``zoom_driver``）は多倍率のみ有効（単一倍率では畳む）``aux_norm_temperature`` /
    ``aux_norm_alpha`` は対応する ``aux_norm`` 値（``temperature`` / ``entmax``）かつ多倍率の
    ときのみ意味を持つ（他では畳む）DPP 系（``dpp_similarity`` / ``dpp_temperature`` /
    ``dpp_quality_beta`` / ``dpp_rbf_gamma`` / ``dpp_use_gumbel`` / ``dpp_diversity_weight``）は
    ``selector=="dpp"`` かつ多倍率のときのみ意味を持つ（他では畳む）MCTS 系（探索プランナ・
    模擬予算・損失重み等）は ``zoom_driver=="mcts"`` の多倍率のみ意味を持つ（それ以外は畳む）
    instance 系（``bag_weight`` / ``inst_k`` / ``inst_subtyping``）は ``instance_loss`` 有効時の
    み意味を持つ（無効時は畳む）``decorrelation_weight`` は多倍率のみ有効（単一倍率では畳む）
    ``decorrelation_method`` は ``decorrelation_weight`` が正のときのみ意味を持つ（0 では畳む）
    畳んだキーは ``axis_values`` から落とし集計・表に載せない明示値を捨てたキー集合を返す（警告用）
    """
    discarded: set = set()
    single_mag = len(config[MAGNIFICATIONS_KEY]) == SINGLE_MAG
    if not single_mag:
        # 多倍率での無効化は dropped_instance_multi が警告するため discarded には積まない
        _disable_param(config, axis_values, defaults, INSTANCE_LOSS_KEY)
    instance_on = bool(config.get(INSTANCE_LOSS_KEY, defaults[INSTANCE_LOSS_KEY]))
    if single_mag and INSTANCE_LOSS_KEY in config:
        # 保存・署名を安定させるため真偽値へ正規化する（YAML の 1/'true' 等を吸収）
        config[INSTANCE_LOSS_KEY] = instance_on
        if INSTANCE_LOSS_KEY in axis_values:
            axis_values[INSTANCE_LOSS_KEY] = instance_on

    aux_norm = config.get(AUX_NORM_KEY, defaults[AUX_NORM_KEY])
    if single_mag:
        for key in (*ZOOM_PARAM_KEYS, ZOOM_DRIVER_KEY):
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
        if _disable_param(config, axis_values, defaults, DECORRELATION_WEIGHT_KEY):
            discarded.add(DECORRELATION_WEIGHT_KEY)
        if _disable_param(config, axis_values, defaults, SELECTOR_KEY):
            discarded.add(SELECTOR_KEY)
    mcts_on = (
        not single_mag
        and config.get(ZOOM_DRIVER_KEY, defaults[ZOOM_DRIVER_KEY]) == ZOOM_DRIVER_MCTS
    )
    if not mcts_on:
        for key in MCTS_PARAM_KEYS:
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
    for key, required_aux_norm in AUX_NORM_PARAM_KEYS.items():
        if single_mag or aux_norm != required_aux_norm:
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
    dpp_on = config.get(SELECTOR_KEY, defaults[SELECTOR_KEY]) == SELECTOR_DPP
    if single_mag or not dpp_on:
        for key in DPP_PARAM_KEYS:
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
    if not instance_on:
        for key in INSTANCE_PARAM_KEYS:
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
    decorrelation_on = (
        config.get(DECORRELATION_WEIGHT_KEY, defaults[DECORRELATION_WEIGHT_KEY]) > 0.0
    )
    if not decorrelation_on:
        for key in DECORRELATION_PARAM_KEYS:
            if _disable_param(config, axis_values, defaults, key):
                discarded.add(key)
    return discarded


def _normalize_for_signature(value: Any) -> Any:
    """署名用に値を正規化する（型違いの同値を同一視する）

    ``bool`` / ``int`` / ``float`` は ``float`` へ寄せ ``1`` / ``1.0`` / ``True`` を同一視する
    list は要素ごとに再帰し，その他は文字列化する
    """
    if isinstance(value, (bool, int, float)):
        return float(value)
    if isinstance(value, list):
        return [_normalize_for_signature(item) for item in value]
    return str(value)


def _combo_signature(config: Dict[str, Any]) -> str:
    """combo 設定の決定的な署名（重複判定キー）を返す"""
    normalized = {key: _normalize_for_signature(val) for key, val in config.items()}
    return json.dumps(normalized, sort_keys=True)


def _warn_collapse(
    n_raw: int,
    n_kept: int,
    dropped_instance_multi: bool,
    discarded_keys: set,
) -> None:
    """無関係パラメータの統合・除外を警告で知らせる"""
    merged = n_raw - n_kept
    if merged > 0:
        logger.warning(
            "sweep 健全化: 構成に無関係なパラメータのみ異なる combo を %d 件統合しました"
            "（展開 %d -> %d）",
            merged, n_raw, n_kept,
        )
    if dropped_instance_multi:
        logger.warning(
            "instance_loss=True は単一倍率でのみ有効です 多倍率の combo では無効化しました"
        )
    if discarded_keys:
        logger.warning(
            "構成に無関係なため既定値へ畳んだ軸（記録しません）: %s",
            sorted(discarded_keys),
        )


def expand_combos(
    sweep: Dict[str, Any], fixed: Dict[str, Any], resolved: ResolvedPaths
) -> List[Combo]:
    """sweep / fixed / 解決済みパスから combo 一覧を生成する

    ``(encoder, feature_type)`` は制約付き join，他の軸は ``ParameterGrid`` で直積展開し，
    各 combo に解決済み ``in_feat_dim`` / ``feature_root`` / ``labels_csv`` / ``n_cls`` を
    載せる``config`` のキーは ``TrainConfig`` フィールドに限る展開後，構成に無関係な
    条件付きパラメータ（単一倍率でのズーム系，無効時の instance 系，多倍率での
    ``instance_loss``）を既定値へ畳んで重複 combo を統合する（直積の無駄を断つ）

    Args:
        sweep: 展開対象の軸辞書（``encoder`` / ``feature_type`` / ``magnifications`` を含む）
        fixed: 全 combo 共通スカラ辞書（``TrainConfig`` フィールド）
        resolved: 解決済みパス群

    Returns:
        combo の列（連番・名前・設定・軸値を持つ）

    Raises:
        ValueError: 必須軸の欠落，自動解決キーの手書き，未知の設定キー
        KeyError: 未知のエンコーダ名
    """
    _reject_auto_resolved("sweep", sweep)
    _reject_auto_resolved("fixed", fixed)

    for required in (ENCODER_KEY, FEATURE_TYPE_KEY, MAGNIFICATIONS_KEY):
        if required not in sweep:
            raise ValueError(f"sweep is missing required axis '{required}'")

    pairs = _encoder_feature_pairs(
        _listify(sweep[ENCODER_KEY]), _listify(sweep[FEATURE_TYPE_KEY])
    )
    mag_sets = _as_mag_sets(sweep[MAGNIFICATIONS_KEY])

    other_axes = {
        key: _listify(value)
        for key, value in sweep.items()
        if key not in (ENCODER_KEY, FEATURE_TYPE_KEY, MAGNIFICATIONS_KEY)
    }
    other_combos = list(ParameterGrid(other_axes)) if other_axes else [{}]

    known = _train_config_fields()
    defaults = _train_config_defaults()

    raw: List[Combo] = []
    for encoder, feature_type in pairs:
        for mags in mag_sets:
            for other in other_combos:
                config: Dict[str, Any] = dict(fixed)
                config.update(other)
                config[ENCODER_KEY] = encoder
                config[FEATURE_TYPE_KEY] = feature_type
                config[MAGNIFICATIONS_KEY] = mags
                config["in_feat_dim"] = resolve_in_feat_dim(encoder, feature_type)
                config["feature_root"] = resolved.feature_root_base
                config["labels_csv"] = resolved.labels_csv
                config["n_cls"] = resolved.n_cls

                unknown = set(config) - known
                if unknown:
                    raise ValueError(
                        f"unknown config keys (not TrainConfig fields): "
                        f"{sorted(unknown)}"
                    )

                axis_values: Dict[str, Any] = {
                    ENCODER_KEY: encoder,
                    FEATURE_TYPE_KEY: feature_type,
                    MAGNIFICATIONS_KEY: mags,
                    **other,
                }
                raw.append(
                    Combo(index=-1, name="", config=config, axis_values=axis_values)
                )

    return _collapse_combos(raw, defaults)


def _collapse_combos(
    raw: Sequence[Combo], defaults: Dict[str, Any]
) -> List[Combo]:
    """無関係パラメータを畳んで重複 combo を統合し連番・名前を振り直す"""
    dropped_instance_multi = False
    discarded_keys: set = set()
    seen: set = set()
    kept: List[Combo] = []
    for combo in raw:
        single_mag = len(combo.config[MAGNIFICATIONS_KEY]) == SINGLE_MAG
        if not single_mag and bool(
            combo.config.get(INSTANCE_LOSS_KEY, defaults[INSTANCE_LOSS_KEY])
        ):
            dropped_instance_multi = True
        discarded_keys |= _canonicalize_conditional(
            combo.config, combo.axis_values, defaults
        )
        signature = _combo_signature(combo.config)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(combo)

    _warn_collapse(len(raw), len(kept), dropped_instance_multi, discarded_keys)

    combos: List[Combo] = []
    for index, combo in enumerate(kept):
        combos.append(
            Combo(
                index=index,
                name=_combo_name(
                    index,
                    combo.config[ENCODER_KEY],
                    combo.config[FEATURE_TYPE_KEY],
                    len(combo.config[MAGNIFICATIONS_KEY]),
                ),
                config=combo.config,
                axis_values=combo.axis_values,
            )
        )
    return combos


def varying_axis_keys(combos: Sequence[Combo]) -> List[str]:
    """combo 間で値が複数に分かれた軸キーを返す（結果表の列・通知用）

    パス系の固定値を含まないため通知・表に載せても安全
    """
    seen: Dict[str, set] = {}
    for combo in combos:
        for key, value in combo.axis_values.items():
            seen.setdefault(key, set()).add(repr(value))
    return sorted(key for key, values in seen.items() if len(values) > 1)


def verify_feature_dims(combos: Sequence[Combo]) -> None:
    """各 combo の解決 ``in_feat_dim`` を実 h5 の特徴次元と突き合わせる

    ``(feature_root, encoder, feature_type, base_mag)`` ごとに 1 スライドだけ開き，
    :meth:`FeatureAccessor.feature_dim` と一致するか確認する特徴の不在や cls 欠落も
    ここで検出する

    Raises:
        ValueError: 特徴が見つからない，または次元が解決値と一致しない場合
    """
    import glob

    checked: set = set()
    for combo in combos:
        encoder = combo.config[ENCODER_KEY]
        feature_type = combo.config[FEATURE_TYPE_KEY]
        feature_root = combo.config["feature_root"]
        base_mag = combo.config[MAGNIFICATIONS_KEY][0]
        key = (feature_root, encoder, feature_type, base_mag)
        if key in checked:
            continue
        checked.add(key)

        mag_dir = os.path.join(feature_root, encoder, _format_mag(base_mag))
        slides = sorted(glob.glob(os.path.join(mag_dir, "*.h5")))
        if not slides:
            raise ValueError(
                f"no feature h5 found for encoder={encoder} mag={base_mag} "
                f"under {mag_dir}"
            )
        slide_id = os.path.splitext(os.path.basename(slides[0]))[0]
        accessor = FeatureAccessor(feature_root, encoder, slide_id, feature_type)
        try:
            actual = accessor.feature_dim(base_mag)
        finally:
            accessor.close()
        expected = combo.config["in_feat_dim"]
        if actual != expected:
            raise ValueError(
                f"in_feat_dim mismatch for encoder={encoder} "
                f"feature_type={feature_type}: resolved {expected} but h5 has "
                f"{actual} (slide {slide_id})"
            )


@dataclass
class ComboResult:
    """1 combo の実行・集計結果

    Attributes:
        index: combo 連番
        name: combo 名
        axis_values: sweep 軸の値（表示用）
        out_dir: この combo の出力先
        n_folds_total: 対象 fold 数
        n_folds_valid: 有効（test 指標を読めた）fold 数
        failed_jobs: 失敗した fold 番号の列
        aggregates: split 別 CV 集計 ``{"val": {...}, "test": {...}}``
        per_fold: split 別 fold ごと指標 ``{"val": [...], "test": [...]}``
        rank_metric: ランキングに用いた指標名（無ければ ``None``）
        rank_value_val: val ランキングに用いた指標の mean（無ければ ``None``）
        rank_value_test: test の同指標の mean（無ければ ``None``）
    """

    index: int
    name: str
    axis_values: Dict[str, Any]
    out_dir: str
    n_folds_total: int
    n_folds_valid: int = 0
    failed_jobs: List[int] = field(default_factory=list)
    aggregates: Dict[str, Any] = field(default_factory=dict)
    per_fold: Dict[str, Any] = field(default_factory=dict)
    rank_metric: Optional[str] = None
    rank_value_val: Optional[float] = None
    rank_value_test: Optional[float] = None


def _run_fold_job(
    combo_config: str,
    fold_dir: str,
    weights_dir: str,
    split_csv: str,
    gpu_id: int,
) -> int:
    """1 つの (combo, fold) を ``foveamil-train --split`` のサブプロセスで学習する

    ``test_metrics.json`` が既にあればスキップ（resume）``CUDA_VISIBLE_DEVICES`` に
    ``gpu_id`` を割り当て当該 GPU へ固定するログ・結果は ``fold_dir``，重みは
    ``weights_dir`` へ保存する

    Returns:
        サブプロセス終了コード（スキップ時は 0）
    """
    result_path = os.path.join(fold_dir, FOLD_RESULT_JSON)
    if os.path.exists(result_path):
        return 0

    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "foveamil.training.train_cli",
        "--config",
        combo_config,
        "--split",
        split_csv,
        "--out",
        fold_dir,
        "--weights-out",
        weights_dir,
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_path = os.path.join(fold_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT
        )
    return completed.returncode


def run_jobs_on_gpu_pool(
    jobs: Sequence[Dict[str, Any]],
    gpu_ids: Sequence[int],
    jobs_per_gpu: int,
    run_fn,
) -> Dict[int, int]:
    """各 GPU に ``jobs_per_gpu`` スロットを持つ動的キューで jobs を実行する

    GPU スロットトークンを ``jobs_per_gpu`` 個ずつキューへ入れ，worker は次の job ごと
    に空きトークンを取得して ``run_fn(job, gpu_id)`` を実行し終了後にトークンを戻す
    job が終わるたびに空いた GPU へ次の pending job が割り当てられるため GPU が遊ばず，
    GPU あたり同時実行数は ``jobs_per_gpu`` を超えない``fold_dir`` に
    ``test_metrics.json`` が既にある job は GPU を取らずスキップする実行は
    ``CUDA_VISIBLE_DEVICES`` を介す ``foveamil-train`` サブプロセスで GIL を解放するため
    スレッドプールで足りる

    Args:
        jobs: ジョブ辞書の列（各々 ``fold_dir`` を持つ）
        gpu_ids: 使用 GPU 一覧
        jobs_per_gpu: GPU あたり同時実行数
        run_fn: ``(job, gpu_id) -> returncode`` の実行関数

    Returns:
        ジョブ添字 → 終了コード（例外は終了コード 1 に正規化）
    """
    gpu_list = list(gpu_ids) if gpu_ids else list(DEFAULT_GPU_IDS)
    per_gpu = max(1, int(jobs_per_gpu))
    slots: "queue.Queue[int]" = queue.Queue()
    for gpu in gpu_list:
        for _ in range(per_gpu):
            slots.put(gpu)
    max_workers = len(gpu_list) * per_gpu

    def _execute(job: Dict[str, Any]) -> int:
        result_path = os.path.join(job["fold_dir"], FOLD_RESULT_JSON)
        if os.path.exists(result_path):
            return 0
        gpu = slots.get()
        try:
            return run_fn(job, gpu)
        finally:
            slots.put(gpu)

    results: Dict[int, int] = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(_execute, job): index for index, job in enumerate(jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                returncode = future.result()
            except Exception as exc:  # noqa: BLE001 - job 失敗で止めない
                logger.error("job %d (%s) raised: %s", index, jobs[index].get("fold_dir"), exc)
                returncode = 1
            results[index] = returncode
    except KeyboardInterrupt:
        logger.warning("interrupted; cancelling pending jobs")
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False)
        raise
    finally:
        executor.shutdown(wait=True)
    return results


def _pick_rank_metric(aggregate: Dict[str, Any]) -> tuple:
    """集計から優先順位順に最初に存在する指標名と mean を返す"""
    for metric in RANK_METRICS:
        if metric in aggregate:
            return metric, float(aggregate[metric]["mean"])
    return None, None


def _agg_mean(aggregate: Dict[str, Any], metric: Optional[str]) -> Optional[float]:
    """集計から ``metric`` の mean を取り出す無ければ ``None``"""
    if metric and metric in aggregate:
        return float(aggregate[metric]["mean"])
    return None


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    """JSON を読む存在しない/壊れていれば ``None``"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - 壊れた結果は除外する
        logger.warning("could not read %s: %s", path, exc)
        return None


def _write_json(path: str, payload: Any) -> None:
    """辞書を JSON で保存する"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class SweepRunner:
    """combo 群を (combo, fold) ジョブにフラット化し GPU 並列で実行・集計する"""

    def __init__(
        self,
        combos: Sequence[Combo],
        split_files: Sequence[str],
        out_root: str,
        weights_root: str,
        gpu_ids: Optional[Sequence[int]] = None,
        jobs_per_gpu: Optional[int] = None,
    ) -> None:
        """実行器を初期化する

        Args:
            combos: 実行する combo 列
            split_files: fold 番号順の分割 CSV パス列
            out_root: ログ・結果の出力ルート（home）
            weights_root: 重みの出力ルート（Dataset）
            gpu_ids: 使用 GPU 一覧
            jobs_per_gpu: GPU あたり並列ジョブ数
        """
        self.combos = list(combos)
        self.split_files = list(split_files)
        self.out_root = out_root
        self.weights_root = weights_root
        self.gpu_ids = list(gpu_ids if gpu_ids else DEFAULT_GPU_IDS)
        self.jobs_per_gpu = int(jobs_per_gpu or DEFAULT_JOBS_PER_GPU)

    def _combo_dir(self, combo: Combo) -> str:
        return os.path.join(self.out_root, combo.name)

    def _write_combo_config(self, combo: Combo) -> str:
        """combo の解決済み設定を ``{out}/{name}/config.yaml`` に書き出す"""
        combo_dir = self._combo_dir(combo)
        os.makedirs(combo_dir, exist_ok=True)
        path = os.path.join(combo_dir, COMBO_CONFIG_NAME)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(combo.config, handle, allow_unicode=True, sort_keys=True)
        return path

    def _build_jobs(self) -> List[Dict[str, Any]]:
        """全 (combo, fold) ジョブを組み立てる（GPU は動的キューが実行時に割り当てる）"""
        jobs: List[Dict[str, Any]] = []
        for combo in self.combos:
            config_path = self._write_combo_config(combo)
            combo_dir = self._combo_dir(combo)
            weights_combo = os.path.join(self.weights_root, combo.name)
            for split_csv in self.split_files:
                fold = _fold_number(split_csv)
                fold_name = f"{FOLD_DIR_PREFIX}{fold}"
                jobs.append(
                    {
                        "combo_index": combo.index,
                        "fold": fold,
                        "combo_config": config_path,
                        "fold_dir": os.path.join(combo_dir, fold_name),
                        "weights_dir": os.path.join(weights_combo, fold_name),
                        "split_csv": split_csv,
                    }
                )
        return jobs

    def run(self) -> Dict[str, Any]:
        """全 job を動的 GPU キューで並列実行し combo ごとに集計してランキング要約を返す"""
        os.makedirs(self.out_root, exist_ok=True)
        jobs = self._build_jobs()
        logger.info(
            "sweep: %d combos x %d folds = %d jobs, gpu_ids=%s, jobs_per_gpu=%d (dynamic queue)",
            len(self.combos),
            len(self.split_files),
            len(jobs),
            self.gpu_ids,
            self.jobs_per_gpu,
        )

        def run_fn(job: Dict[str, Any], gpu_id: int) -> int:
            return _run_fold_job(
                job["combo_config"],
                job["fold_dir"],
                job["weights_dir"],
                job["split_csv"],
                gpu_id,
            )

        returncodes = run_jobs_on_gpu_pool(
            jobs, self.gpu_ids, self.jobs_per_gpu, run_fn
        )

        failures: Dict[int, List[int]] = {}
        for index, job in enumerate(jobs):
            if returncodes.get(index, 1) != 0:
                logger.error(
                    "combo %03d fold %d failed (returncode=%s)",
                    job["combo_index"], job["fold"], returncodes.get(index),
                )
                failures.setdefault(job["combo_index"], []).append(job["fold"])

        results = [self._collect_combo(combo, failures.get(combo.index, []))
                   for combo in self.combos]
        summary = self._summarize(results)
        self._save_summary(summary, results)
        return summary

    def _read_fold_results(self, fold_dir: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """fold の test/val 指標 JSON を読む（無ければ ``None``）"""
        return {
            "test": _read_json(os.path.join(fold_dir, FOLD_RESULT_JSON)),
            "val": _read_json(os.path.join(fold_dir, METRICS_VAL_JSON)),
        }

    def _collect_combo(self, combo: Combo, failed_jobs: List[int]) -> ComboResult:
        """combo の val/test fold 結果を集めて CV 集計し ``cv_summary.json`` を書く"""
        combo_dir = self._combo_dir(combo)
        per_fold: Dict[str, List[Dict[str, Any]]] = {"test": [], "val": []}
        for split_csv in self.split_files:
            fold = _fold_number(split_csv)
            fold_dir = os.path.join(combo_dir, f"{FOLD_DIR_PREFIX}{fold}")
            res = self._read_fold_results(fold_dir)
            for split in ("test", "val"):
                if res[split] is not None:
                    res[split]["fold"] = fold
                    per_fold[split].append(res[split])

        result = ComboResult(
            index=combo.index,
            name=combo.name,
            axis_values=combo.axis_values,
            out_dir=combo_dir,
            n_folds_total=len(self.split_files),
            n_folds_valid=len(per_fold["test"]),
            failed_jobs=sorted(failed_jobs),
            per_fold=per_fold,
        )
        if per_fold["test"]:
            aggregates = {
                split: aggregate_folds_ci(per_fold[split]) for split in ("test", "val")
            }
            result.aggregates = aggregates
            # model selection は val 指標で（val 不在時のみ test にフォールバック）
            sel_agg = aggregates["val"] or aggregates["test"]
            metric, _ = _pick_rank_metric(sel_agg)
            result.rank_metric = metric
            result.rank_value_val = _agg_mean(aggregates["val"], metric)
            result.rank_value_test = _agg_mean(aggregates["test"], metric)
            cv_summary = {
                "schema_version": CV_SCHEMA_VERSION,
                "n_folds_total": len(self.split_files),
                "n_folds_valid": len(per_fold["test"]),
                "selection": {"split": SELECTION_SPLIT, "metric": metric},
                "test": {"per_fold": per_fold["test"], "aggregate": aggregates["test"]},
                "val": {"per_fold": per_fold["val"], "aggregate": aggregates["val"]},
            }
            _write_json(os.path.join(combo_dir, CV_SUMMARY_JSON), cv_summary)
        return result

    def _entry(
        self, r: ComboResult, rank_by_val: Dict[int, int], rank_by_test: Dict[int, int]
    ) -> Dict[str, Any]:
        """combo 1 件の要約エントリ（val/test 集計と両ランクを含む）を作る"""
        return {
            "index": r.index,
            "name": r.name,
            "axis_values": r.axis_values,
            "out_dir": r.out_dir,
            "n_folds_valid": r.n_folds_valid,
            "n_folds_total": r.n_folds_total,
            "failed_jobs": r.failed_jobs,
            "rank_metric": r.rank_metric,
            "val": r.aggregates.get("val"),
            "test": r.aggregates.get("test"),
            "rank_value_val": r.rank_value_val,
            "rank_value_test": r.rank_value_test,
            "rank_by_val": rank_by_val.get(r.index),
            "rank_by_test": rank_by_test.get(r.index),
        }

    def _summarize(self, results: Sequence[ComboResult]) -> Dict[str, Any]:
        """combo を val 指標でランキングし test を併記，test oracle も別途出す"""
        ranked_val = sorted(
            (r for r in results if r.rank_value_val is not None),
            key=lambda r: r.rank_value_val, reverse=True,
        )
        ranked_test = sorted(
            (r for r in results if r.rank_value_test is not None),
            key=lambda r: r.rank_value_test, reverse=True,
        )
        rank_by_val = {r.index: pos + 1 for pos, r in enumerate(ranked_val)}
        rank_by_test = {r.index: pos + 1 for pos, r in enumerate(ranked_test)}

        entries = [self._entry(r, rank_by_val, rank_by_test) for r in results]
        by_index = {e["index"]: e for e in entries}

        best_by_val = by_index.get(ranked_val[0].index) if ranked_val else None
        oracle = by_index.get(ranked_test[0].index) if ranked_test else None
        if oracle is not None:
            oracle = dict(oracle)
            oracle["note"] = "upper bound, not for reporting"

        return {
            "selection_metric": ranked_val[0].rank_metric if ranked_val else None,
            "selection_split": SELECTION_SPLIT,
            "report_split": REPORT_SPLIT,
            "n_combos": len(results),
            "n_folds": len(self.split_files),
            "gpu_ids": self.gpu_ids,
            "jobs_per_gpu": self.jobs_per_gpu,
            "axis_keys": varying_axis_keys(self.combos),
            "best_by_val": best_by_val,
            "oracle_by_test": oracle,
            "combos": entries,
            "failed": [r.index for r in results if r.failed_jobs],
        }

    def _write_detailed_csv(self, results: Sequence[ComboResult]) -> None:
        """全 combo×fold×split の指標を long format で ``sweep_detailed.csv`` に書く"""
        axis_keys = varying_axis_keys(self.combos)
        rows: List[Dict[str, Any]] = []
        for r in results:
            for split in ("val", "test"):
                for fold_metrics in r.per_fold.get(split, []):
                    row: Dict[str, Any] = {
                        "combo_index": r.index, "combo_name": r.name,
                    }
                    for key in axis_keys:
                        row[key] = r.axis_values.get(key)
                    row["fold"] = fold_metrics.get("fold")
                    row["split"] = split
                    for metric in SUMMARY_METRICS:
                        if metric in fold_metrics:
                            row[metric] = fold_metrics[metric]
                    rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(self.out_root, SWEEP_DETAILED_CSV), index=False
            )

    def _save_summary(self, summary: Dict[str, Any], results: Sequence[ComboResult]) -> None:
        """要約を JSON・人間可読 md・詳細 CSV で保存する"""
        _write_json(os.path.join(self.out_root, SWEEP_SUMMARY_JSON), summary)
        md_path = os.path.join(self.out_root, SWEEP_SUMMARY_MD)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(_summary_markdown(summary))
        self._write_detailed_csv(results)
        logger.info("saved sweep summary to %s", self.out_root)


def _fmt_agg(aggregate: Optional[Dict[str, Any]], metric: Optional[str]) -> str:
    """集計から ``metric`` の ``mean±std`` 表記を作る無ければ ``-``"""
    if not metric or not aggregate or metric not in aggregate:
        return "-"
    stats = aggregate[metric]
    return f"{stats['mean']:.4f}±{stats['std']:.4f}"


def _best_block(summary: Dict[str, Any], metric: Optional[str]) -> List[str]:
    """val 選定 best とその test を示すブロックを作る"""
    best = summary.get("best_by_val")
    if not best:
        return ["(有効な結果がありませんでした)", ""]
    val = _fmt_agg(best.get("val"), metric)
    test = _fmt_agg(best.get("test"), metric)
    oracle = summary.get("oracle_by_test")
    lines = [
        f"## 選定 best（val 選定・test 報告 指標 {metric}）",
        "",
        f"- combo: `{best.get('name')}`",
        f"- val {metric}: {val}（選定基準）",
        f"- **test {metric}: {test}（報告値）**",
        "",
    ]
    if oracle:
        lines += [
            f"## test oracle（上限・報告には使わない）",
            "",
            f"- combo: `{oracle.get('name')}` / test {metric}: "
            f"{_fmt_agg(oracle.get('test'), metric)}",
            "",
        ]
    return lines


def _summary_markdown(summary: Dict[str, Any]) -> str:
    """要約辞書を val rank 昇順の md テーブルへ整形する（内部パスを載せない）"""
    axis_keys = summary.get("axis_keys") or []
    metric = summary.get("selection_metric")
    header = [
        "rank(val)", "combo", *axis_keys,
        f"val {metric}", f"test {metric}", "folds",
    ]

    rows = []
    for entry in summary.get("combos", []):
        axis_values = entry.get("axis_values", {})
        rank = entry.get("rank_by_val")
        cells = [
            str(rank if rank is not None else "-"),
            entry.get("name", ""),
            *[str(axis_values.get(k, "")) for k in axis_keys],
            _fmt_agg(entry.get("val"), metric),
            _fmt_agg(entry.get("test"), metric),
            f"{entry.get('n_folds_valid')}/{entry.get('n_folds_total')}",
        ]
        rows.append(cells)

    rows.sort(key=lambda c: (c[0] == "-", c[0]))

    lines = [
        f"# sweep summary（{summary.get('n_combos')} combos x "
        f"{summary.get('n_folds')} folds）",
        "",
        *_best_block(summary, metric),
        "## 全 combo（val rank 昇順）",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
