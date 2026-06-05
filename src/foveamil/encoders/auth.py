"""HuggingFace Hub 認証ヘルパ

gated モデルのロード前に呼ぶ環境変数 ``HF_TOKEN`` があればそれでログインし，
無ければ HuggingFace 標準キャッシュに保存済みのログインに委ねる
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# HF トークンを与える環境変数名
HF_TOKEN_ENV = "HF_TOKEN"


def ensure_hf_auth() -> None:
    """HuggingFace Hub のログインを確保する（失敗しても例外にしない）

    環境変数 ``HF_TOKEN`` があればそのトークンでログインする無ければ
    ``huggingface-cli login`` 等で保存済みのキャッシュ済みログインに委ねる
    """
    token = os.environ.get(HF_TOKEN_ENV)
    if not token:
        logger.info("no %s set; relying on cached HuggingFace login", HF_TOKEN_ENV)
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    logger.info("logged in to HuggingFace Hub via %s", HF_TOKEN_ENV)
