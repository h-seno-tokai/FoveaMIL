"""Virchow2-mini（ViT-Small 自家蒸留モデル）による特徴抽出エンコーダ

ローカルの ``.pt`` チェックポイントをロードするチェックポイントパスは環境変数
``VIRCHOW_MINI_CHECKPOINT`` から取得する
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional, Tuple

import timm
import torch
from torch import Tensor, nn

from foveamil.encoders.base import PatchEncoder

# Virchow2-mini の特徴次元
_FEATURE_DIM = 384
# チェックポイントパスを与える環境変数名
CHECKPOINT_ENV = "VIRCHOW_MINI_CHECKPOINT"
# timm のベースモデル名
_TIMM_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
# チェックポイント内で state_dict を格納するキー
_STATE_DICT_KEY = "model_state_dict"
# state_dict キーから除去する prefix
_STRIP_PREFIXES = ("module.", "student.", "backbone.")
# CLS 1 + register 4 を除いた patch トークン開始位置
_PATCH_TOKEN_START = 5


def _strip_state_dict_prefixes(state_dict: "OrderedDict[str, Tensor]") -> "OrderedDict[str, Tensor]":
    """state_dict の各キー先頭から既知の prefix を除去する"""
    stripped: "OrderedDict[str, Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        name = key
        for prefix in _STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
        stripped[name] = value
    return stripped


class Virchow2MiniEncoder(PatchEncoder):
    """ローカル ``.pt`` から ViT-Small を構築する特徴抽出器

    出力 ``[B, 261, 384]`` から cls=``[:, 0, :]``，patch ``[:, 5:, :]`` を
    空間平均して pooled を返すチェックポイントは環境変数
    ``VIRCHOW_MINI_CHECKPOINT`` が指すパスからロードする
    """

    name = "Virchow2-mini-dinov2"
    feature_dim = _FEATURE_DIM
    has_cls = True

    def _build_model(self) -> nn.Module:
        checkpoint_path = os.environ.get(CHECKPOINT_ENV)
        if not checkpoint_path:
            raise RuntimeError(
                f"environment variable {CHECKPOINT_ENV} is not set; "
                "point it to the Virchow2-mini checkpoint .pt"
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        model = timm.create_model(
            _TIMM_MODEL_NAME,
            pretrained=False,
            num_classes=0,
            img_size=224,
            patch_size=14,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            reg_tokens=4,
            dynamic_img_size=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if _STATE_DICT_KEY not in checkpoint:
            raise ValueError(
                f"checkpoint missing key '{_STATE_DICT_KEY}': {list(checkpoint.keys())}"
            )
        state_dict = _strip_state_dict_prefixes(checkpoint[_STATE_DICT_KEY])
        model.load_state_dict(state_dict, strict=False)
        return model

    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        tokens = self._model.forward_features(patches)
        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, _PATCH_TOKEN_START:, :]
        pooled = self._pool_patch_tokens(patch_tokens)
        return pooled, cls
