"""メモリ／GC ユーティリティ

``gc.collect()`` の実行と，プロセス RSS の ``DEBUG`` ログ出力を提供する
RSS の取得は ``psutil`` があるときのみ行う（任意依存）
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)

try:  # psutil は任意無くてもメモリログを諦めるだけで GC は機能する
    import psutil

    _PROCESS = psutil.Process()
except Exception:  # pragma: no cover - psutil 不在時のフォールバック
    psutil = None
    _PROCESS = None

# RSS をバイトから MiB に直す除数
_BYTES_PER_MIB = 1024 * 1024


def log_memory(label: str) -> None:
    """現在のプロセス RSS を ``DEBUG`` レベルで出力する（``psutil`` があるとき）

    Args:
        label: ログ行に添える文脈ラベル
    """
    if _PROCESS is None:
        logger.debug("%s", label)
        return
    rss_mib = _PROCESS.memory_info().rss / _BYTES_PER_MIB
    logger.debug("%s | RSS=%.0f MiB", label, rss_mib)


def force_gc() -> None:
    """``gc.collect()`` を実行し，回収したオブジェクト数を ``DEBUG`` で記録する"""
    collected = gc.collect()
    logger.debug("gc.collect() reclaimed %d objects", collected)
