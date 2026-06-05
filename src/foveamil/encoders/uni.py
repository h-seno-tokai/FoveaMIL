"""UNI2-h（ViT 系病理基盤モデル）による特徴抽出エンコーダ"""

from __future__ import annotations

from typing import Optional, Tuple

import timm
import torch
from torch import Tensor, nn

from foveamil.encoders.auth import ensure_hf_auth
from foveamil.encoders.base import PatchEncoder

# UNI2-h の特徴次元
_FEATURE_DIM = 1536
# HuggingFace Hub のモデル ID
_HF_MODEL_ID = "hf-hub:MahmoodLab/UNI2-h"
# CLS 1 + register 8 トークンを除いた patch トークン開始位置
_PATCH_TOKEN_START = 9


class UNI2hEncoder(PatchEncoder):
    """UNI2-h を ``forward_features`` で実行する特徴抽出器

    出力トークン列 ``[B, 265, 1536]`` から cls=``[:, 0, :]``，
    patch トークン ``[:, 9:, :]`` を空間平均して pooled を返す（gated モデル）
    """

    name = "UNI2-h"
    feature_dim = _FEATURE_DIM
    has_cls = True

    def _build_model(self) -> nn.Module:
        ensure_hf_auth()
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        return timm.create_model(_HF_MODEL_ID, pretrained=True, **timm_kwargs)

    def _forward_tokens(self, patches: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        tokens = self._model.forward_features(patches)
        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, _PATCH_TOKEN_START:, :]
        pooled = self._pool_patch_tokens(patch_tokens)
        return pooled, cls
