"""sweep 出力と可視化を繋ぐ唯一のグルー（モデル構築を知る部品）

``sweep_summary.json`` の best_by_val（または oracle/index）から combo を解決し，fold の
``run_meta.json`` の config で FoveaMIL を再構築して ``model_best_{save_metric}.pt`` を
ロードするモデル構築は :class:`Trainer` の構築と同一にするため，topk セレクタの引数は
``trainer._topk_kwargs``（topk_method→k_sigma 写像）を共有する（再実装しない）
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from foveamil.models import FoveaMIL
from foveamil.training.config import TrainConfig
from foveamil.training.trainer import build_foveamil_from_config

# sweep / fold の出力ファイル名
SWEEP_SUMMARY_JSON = "sweep_summary.json"
RUN_META_JSON = "run_meta.json"
FOLD_DIR_PREFIX = "fold"
# best 重みのファイル名テンプレート
WEIGHTS_TEMPLATE = "model_best_{metric}.pt"
# combo 選択の種別
SELECT_BEST_VAL = "best_by_val"
SELECT_ORACLE = "oracle_by_test"
SELECT_INDEX = "index"


@dataclass
class LoadedModel:
    """再構築・ロード済みのモデルと推論に要る設定

    Attributes:
        model: eval 化された FoveaMIL
        magnifications: 倍率列（低→高）
        encoder: エンコーダ名
        feature_type: ``mean`` / ``cls`` / ``concat``
        n_cls: クラス数
        classes: クラス名の並び（無ければ ``None``）
        save_metric: best 選択基準（重みファイル名に対応）
    """

    model: FoveaMIL
    magnifications: List[float]
    encoder: str
    feature_type: str
    n_cls: int
    classes: Optional[List[str]]
    save_metric: str


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _train_config_from_dict(config: Dict[str, Any]) -> TrainConfig:
    """run_meta の config 辞書から ``TrainConfig`` を復元する（既知フィールドのみ）"""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    return TrainConfig(**{k: v for k, v in config.items() if k in fields})


def build_model(config: Dict[str, Any]) -> FoveaMIL:
    """run_meta の config から FoveaMIL を ``Trainer.__init__`` と同一引数で構築する

    モデル組立は ``build_foveamil_from_config`` を共有し，正規化器・選択コントローラ・
    top-k 引数の写像のズレを防ぐ
    """
    cfg = _train_config_from_dict(config)
    return build_foveamil_from_config(cfg, len(cfg.magnifications))


def resolve_best_combo(
    sweep_root: str, select: str = SELECT_BEST_VAL, combo_index: Optional[int] = None
) -> str:
    """sweep_summary から combo の出力ディレクトリを解決する

    Args:
        sweep_root: ``sweep_summary.json`` のあるルート
        select: ``best_by_val`` / ``oracle_by_test`` / ``index``
        combo_index: ``select=index`` 時の combo 連番

    Returns:
        combo の出力ディレクトリ
    """
    summary = _load_json(os.path.join(sweep_root, SWEEP_SUMMARY_JSON))
    if select == SELECT_INDEX:
        entry = next(
            (c for c in summary["combos"] if c["index"] == combo_index), None
        )
    else:
        entry = summary.get(select)
    if not entry:
        raise ValueError(f"could not resolve combo for select={select} in {sweep_root}")
    out_dir = entry.get("out_dir")
    if out_dir and os.path.isdir(out_dir):
        return out_dir
    return os.path.join(sweep_root, entry["name"])


def fold_dirs(combo_dir: str, fold: str) -> List[str]:
    """combo の fold ディレクトリ群を返す（``fold="all"`` で全て）"""
    import glob

    if fold != "all":
        return [os.path.join(combo_dir, f"{FOLD_DIR_PREFIX}{fold}")]
    dirs = [
        p for p in glob.glob(os.path.join(combo_dir, f"{FOLD_DIR_PREFIX}*"))
        if os.path.isdir(p)
    ]
    return sorted(dirs, key=lambda d: int("".join(ch for ch in os.path.basename(d) if ch.isdigit()) or 0))


def load_fold(
    fold_dir: str, weights_dir: Optional[str] = None, device: str = "cpu"
) -> LoadedModel:
    """fold の run_meta から FoveaMIL を再構築し best 重みをロードする

    Args:
        fold_dir: ``run_meta.json`` のある fold ディレクトリ（home 側）
        weights_dir: 重み（``.pt``）のあるディレクトリ``None`` なら ``fold_dir``
        device: 推論デバイス

    Returns:
        :class:`LoadedModel`
    """
    meta = _load_json(os.path.join(fold_dir, RUN_META_JSON))
    config = meta["config"]
    save_metric = meta["selection"]["save_metric"]
    weights_root = weights_dir if weights_dir is not None else fold_dir
    weights_path = os.path.join(weights_root, WEIGHTS_TEMPLATE.format(metric=save_metric))

    model = build_model(config)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        magnifications=list(config["magnifications"]),
        encoder=config["encoder"],
        feature_type=config["feature_type"],
        n_cls=int(config["n_cls"]),
        classes=meta.get("data", {}).get("classes"),
        save_metric=save_metric,
    )
