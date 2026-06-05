"""WSI ファイルをローカル一時領域へコピーして読み出すステージング部品

ネットワーク越しの WSI を実行前にローカル SSD へ複製し，ローカルパスを返す
処理後の個別削除・全削除，後続ファイルのバックグラウンド先読み，キャッシュ総量の
LRU 退避に対応する
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Dict, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# キャッシュ総量の既定上限（GB）
DEFAULT_MAX_CACHE_GB = 100.0
# 既定の先読み件数（0 なら先読みしない）
DEFAULT_PREFETCH_AHEAD = 2
# バイト ⇔ GB 換算係数
BYTES_PER_GB = 1024 ** 3
# キャッシュ先ルートを与える環境変数名
STAGE_DIR_ENV = "FOVEAMIL_STAGE_DIR"
# 環境変数も未設定のときに /tmp 配下へ作る既定ディレクトリ名の接頭辞
_DEFAULT_DIR_PREFIX = "foveamil_wsi_stage_"


def _default_cache_dir() -> str:
    """環境変数が未設定のときの既定キャッシュ先を返す（プロセス毎に衝突しない名前）"""
    return os.path.join("/tmp", f"{_DEFAULT_DIR_PREFIX}{os.getpid()}")


class WSIStager:
    """WSI ファイルをローカルキャッシュへコピーしてローカルパスを返す管理器

    Args:
        cache_dir: ローカルキャッシュ先``None`` のとき環境変数
            ``FOVEAMIL_STAGE_DIR`` を読み，それも無ければ ``/tmp`` 配下の
            プロセス固有ディレクトリを使う
        max_cache_size_gb: キャッシュ総量の上限（GB）超過時は古いものから退避する
        prefetch_ahead: ``stage`` 時に先読みする後続ファイル数``0`` で先読みしない
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size_gb: float = DEFAULT_MAX_CACHE_GB,
        prefetch_ahead: int = DEFAULT_PREFETCH_AHEAD,
    ) -> None:
        if cache_dir is None:
            cache_dir = os.environ.get(STAGE_DIR_ENV) or _default_cache_dir()
        self.cache_dir = cache_dir
        self.max_cache_size = int(max_cache_size_gb * BYTES_PER_GB)
        self.prefetch_ahead = prefetch_ahead

        os.makedirs(self.cache_dir, exist_ok=True)

        # src_path -> (local_path, size_bytes, last_access_time)
        self._entries: Dict[str, Tuple[str, int, float]] = {}
        self._lock = threading.RLock()
        self._prefetch_threads: list[threading.Thread] = []

        logger.info(
            "WSIStager ready: dir=%s max=%.1fGB prefetch=%d",
            self.cache_dir,
            max_cache_size_gb,
            self.prefetch_ahead,
        )

    def stage(self, src_path: str, upcoming: Optional[Sequence[str]] = None) -> str:
        """``src_path`` をキャッシュへコピーしてローカルパスを返す

        既にキャッシュ済みなら再利用しアクセス時刻を更新する``upcoming`` が
        与えられれば先頭から ``prefetch_ahead`` 件をバックグラウンドで先読みする

        Args:
            src_path: コピー元 WSI の絶対パス
            upcoming: 続けて処理予定の WSI パス列（先読み候補）

        Returns:
            ローカルキャッシュ上のパス
        """
        self._join_prefetch_threads()
        local_path = self._copy(src_path)

        if upcoming and self.prefetch_ahead > 0:
            for next_src in list(upcoming)[: self.prefetch_ahead]:
                thread = threading.Thread(
                    target=self._copy, args=(next_src,), daemon=True
                )
                thread.start()
                self._prefetch_threads.append(thread)

        return local_path

    def release(self, src_path: str) -> None:
        """``src_path`` に対応するローカルコピーを削除し管理から除く"""
        local_path = self._local_path_for(src_path)
        with self._lock:
            entry = self._entries.pop(src_path, None)
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.debug("released %s", os.path.basename(local_path))
        if entry is None and not os.path.exists(local_path):
            logger.debug("release skipped (not staged): %s", os.path.basename(local_path))

    def cleanup_all(self) -> None:
        """キャッシュディレクトリごと削除する（存在しなくてもエラーにしない）"""
        self._join_prefetch_threads()
        with self._lock:
            self._entries.clear()
        try:
            shutil.rmtree(self.cache_dir)
            logger.info("cleaned up cache dir: %s", self.cache_dir)
        except FileNotFoundError:
            logger.debug("cache dir already absent: %s", self.cache_dir)

    def staged(
        self, paths: Sequence[str]
    ) -> Iterator[Tuple[str, str]]:
        """``paths`` を順にステージし ``(src_path, local_path)`` を yield する

        次へ進む前に直前の src を ``release`` し，残りを先読み候補として渡す

        Args:
            paths: 処理する WSI パスの列

        Yields:
            ``(src_path, local_path)`` の組
        """
        paths = list(paths)
        previous: Optional[str] = None
        for index, src_path in enumerate(paths):
            if previous is not None:
                self.release(previous)
            local_path = self.stage(src_path, upcoming=paths[index + 1 :])
            yield src_path, local_path
            previous = src_path
        if previous is not None:
            self.release(previous)

    def __enter__(self) -> "WSIStager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup_all()

    def _local_path_for(self, src_path: str) -> str:
        """``src_path`` に対応するローカルキャッシュ上のパスを返す"""
        return os.path.join(self.cache_dir, os.path.basename(src_path))

    def _current_cache_size(self) -> int:
        """現在のキャッシュ総量（バイト）を返す"""
        with self._lock:
            return sum(size for _, size, _ in self._entries.values())

    def _evict_oldest(self) -> None:
        """アクセス時刻が最も古いエントリを 1 件退避する"""
        with self._lock:
            if not self._entries:
                return
            oldest_src = min(
                self._entries, key=lambda key: self._entries[key][2]
            )
            local_path, size, _ = self._entries.pop(oldest_src)
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.debug(
                "evicted %s (%.2fGB)", os.path.basename(local_path), size / BYTES_PER_GB
            )

    def _copy(self, src_path: str) -> str:
        """``src_path`` をローカルキャッシュへコピーしてローカルパスを返す

        既に管理下にあれば再コピーせずアクセス時刻だけ更新する
        コピー前に必要なら LRU 退避でサイズ上限を保つ
        """
        local_path = self._local_path_for(src_path)

        with self._lock:
            if src_path in self._entries:
                _, size, _ = self._entries[src_path]
                self._entries[src_path] = (local_path, size, time.time())
                logger.debug("reuse staged %s", os.path.basename(local_path))
                return local_path

        if not os.path.exists(local_path):
            logger.debug("copying %s", os.path.basename(src_path))
            shutil.copy2(src_path, local_path)

        size = os.path.getsize(local_path)

        # コピー済み分を含めて上限を超える間，古いものから退避する
        while self._entries and self._current_cache_size() + size > self.max_cache_size:
            self._evict_oldest()

        with self._lock:
            self._entries[src_path] = (local_path, size, time.time())

        logger.debug("staged %s (%.2fGB)", os.path.basename(local_path), size / BYTES_PER_GB)
        return local_path

    def _join_prefetch_threads(self) -> None:
        """進行中の先読みスレッドの完了を待つ"""
        for thread in self._prefetch_threads:
            if thread.is_alive():
                thread.join()
        self._prefetch_threads = []
