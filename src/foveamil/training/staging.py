"""特徴量セットをローカル SSD へ一括コピーしてから読むステージング部品

正準レイアウト ``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` の対象ファイルを
ローカルキャッシュへ複製し，新しいルート（キャッシュ先）を返す
コピー前に必要容量と SSD 空き容量を確認し，収まればコピー，収まらなければ警告して
元のルートを返す（NAS 直読フォールバック）
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# キャッシュ先ルートを与える環境変数名
STAGE_DIR_ENV = "FOVEAMIL_STAGE_DIR"
# 既定の空き容量安全マージン（空きの ``1 - margin`` までを使う）
DEFAULT_FREE_MARGIN = 0.1
# バイト ⇔ GB 換算係数
BYTES_PER_GB = 1024 ** 3
# 環境変数も未設定のときに /tmp 配下へ作る既定ディレクトリ名の接頭辞
_DEFAULT_DIR_PREFIX = "foveamil_feat_stage_"


def _format_mag(magnification: float) -> str:
    """倍率を正準レイアウトのディレクトリ名（例 ``1.25`` → ``"1.25x"``）にする"""
    return f"{magnification}x"


def _default_cache_dir() -> str:
    """環境変数が未設定のときの既定キャッシュ先を返す（プロセス毎に衝突しない名前）"""
    return os.path.join("/tmp", f"{_DEFAULT_DIR_PREFIX}{os.getpid()}")


class FeatureStager:
    """特徴量セットをローカルキャッシュへ一括コピーする管理器

    Args:
        cache_dir: ローカルキャッシュ先``None`` のとき環境変数
            ``FOVEAMIL_STAGE_DIR`` を読み，それも無ければ ``/tmp`` 配下の
            プロセス固有ディレクトリを使う
        free_space_margin: 空き容量の安全マージン（例 ``0.1`` で空きの 90% まで使う）
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        free_space_margin: float = DEFAULT_FREE_MARGIN,
    ) -> None:
        if cache_dir is None:
            cache_dir = os.environ.get(STAGE_DIR_ENV) or _default_cache_dir()
        self.cache_dir = cache_dir
        self.free_space_margin = free_space_margin
        self._staged = False

    def _target_files(
        self,
        feature_root: str,
        encoder: str,
        magnifications: Sequence[float],
        slide_ids: Sequence[str],
    ) -> List[str]:
        """対象 slide_id × 倍率の特徴 H5 のうち実在するものの相対パス列を返す

        相対パスは ``feature_root`` 起点（``{encoder}/{mag}x/{slide_id}.h5``）

        Args:
            feature_root: 特徴ルートディレクトリ
            encoder: エンコーダ名
            magnifications: 倍率の列
            slide_ids: 対象 slide_id の集合/列

        Returns:
            ``feature_root`` からの相対パス列（実在ファイルのみ）
        """
        rels: List[str] = []
        for slide_id in slide_ids:
            for mag in magnifications:
                rel = os.path.join(encoder, _format_mag(mag), f"{slide_id}.h5")
                if os.path.exists(os.path.join(feature_root, rel)):
                    rels.append(rel)
        return rels

    def _copy_atomic(self, src: str, dst: str) -> None:
        """``src`` を ``dst`` へアトミックにコピーする（tmp 名→rename）

        ``dst`` が既にあれば再利用しコピーしない

        Args:
            src: コピー元の絶対パス
            dst: コピー先の絶対パス
        """
        if os.path.exists(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def stage_set(
        self,
        feature_root: str,
        encoder: str,
        magnifications: Sequence[float],
        slide_ids: Sequence[str],
    ) -> str:
        """対象特徴セットをキャッシュへ一括コピーし新しいルートを返す

        必要容量が SSD 空きの ``1 - free_space_margin`` 以内なら ``cache_dir`` 構造へ
        コピーし ``cache_dir`` を返す（既存ファイルは再利用）収まらなければ警告して
        元の ``feature_root`` を返す（NAS 直読フォールバック）

        Args:
            feature_root: 特徴ルートディレクトリ
            encoder: エンコーダ名
            magnifications: 倍率の列
            slide_ids: 対象 slide_id の集合/列

        Returns:
            キャッシュ先ルート（コピー時）または元の ``feature_root``（フォールバック時）
        """
        rels = self._target_files(feature_root, encoder, magnifications, slide_ids)
        required = sum(
            os.path.getsize(os.path.join(feature_root, rel)) for rel in rels
        )

        os.makedirs(self.cache_dir, exist_ok=True)
        free = shutil.disk_usage(self.cache_dir).free
        usable = free * (1.0 - self.free_space_margin)

        if required > usable:
            logger.warning(
                "feature set does not fit on local SSD "
                "(need %.2fGB, free %.2fGB), falling back to direct NAS read "
                "(training will be slower)",
                required / BYTES_PER_GB,
                free / BYTES_PER_GB,
            )
            return feature_root

        for rel in rels:
            self._copy_atomic(
                os.path.join(feature_root, rel), os.path.join(self.cache_dir, rel)
            )
        self._staged = True

        logger.info(
            "staged feature set: need %.2fGB / free %.2fGB / %d files -> %s",
            required / BYTES_PER_GB,
            free / BYTES_PER_GB,
            len(rels),
            self.cache_dir,
        )
        return self.cache_dir

    def localize(self, path: str) -> str:
        """単一ファイルをキャッシュへコピーしキャッシュ上のパスを返す

        既にステージング済みのレイアウトに対応するパスがあればそれを返す
        ``path`` がキャッシュ配下を指す場合はそのまま返す

        Args:
            path: コピー元ファイルの絶対パス

        Returns:
            キャッシュ上のパス
        """
        if os.path.commonpath([os.path.abspath(path), os.path.abspath(self.cache_dir)]) == os.path.abspath(self.cache_dir):
            return path
        dst = os.path.join(self.cache_dir, os.path.basename(path))
        self._copy_atomic(path, dst)
        self._staged = True
        return dst

    def cleanup(self) -> None:
        """ステージングディレクトリを削除する（存在しなくてもエラーにしない）"""
        try:
            shutil.rmtree(self.cache_dir)
            logger.info("cleaned up stage dir: %s", self.cache_dir)
        except FileNotFoundError:
            logger.debug("stage dir already absent: %s", self.cache_dir)
        self._staged = False

    def __enter__(self) -> "FeatureStager":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.cleanup()
