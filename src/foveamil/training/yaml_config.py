"""YAML から ``TrainConfig`` を読み込む補助

``load_train_config`` は YAML の辞書を ``TrainConfig`` の既知フィールドへ写し，
任意の ``overrides`` で上書きする未知キーは警告ログを出して無視する
``train_config_to_dict`` は ``TrainConfig`` を素の辞書へ戻す（保存・sweep 展開用）
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Optional

import yaml

from foveamil.training.config import TrainConfig

logger = logging.getLogger(__name__)


def _known_fields() -> frozenset:
    """``TrainConfig`` のフィールド名集合を返す"""
    return frozenset(f.name for f in dataclasses.fields(TrainConfig))


def _apply_mapping(values: Dict[str, Any], target: Dict[str, Any]) -> None:
    """``values`` の各項目を既知フィールドのみ ``target`` に写す

    未知キーは警告ログを出して無視する
    """
    known = _known_fields()
    for key, value in values.items():
        if key in known:
            target[key] = value
        else:
            logger.warning("unknown config key ignored: %s", key)


def load_train_config(
    yaml_path: str, overrides: Optional[Dict[str, Any]] = None
) -> TrainConfig:
    """YAML を読み ``TrainConfig`` を構築する

    Args:
        yaml_path: 設定 YAML のパス（トップレベルは辞書）
        overrides: ``{field: value}`` の上書き辞書（任意）

    Returns:
        構築済みの ``TrainConfig``
    """
    with open(yaml_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config YAML must be a mapping: {yaml_path}")

    fields: Dict[str, Any] = {}
    _apply_mapping(raw, fields)
    if overrides:
        _apply_mapping(overrides, fields)

    return TrainConfig(**fields)


def train_config_to_dict(config: TrainConfig) -> Dict[str, Any]:
    """``TrainConfig`` を素の辞書へ変換する"""
    return dataclasses.asdict(config)
