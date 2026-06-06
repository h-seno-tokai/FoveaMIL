"""特徴量セットをローカル SSD へ一括コピーしてから読むステージング部品

正準レイアウト ``{feature_root}/{encoder}/{mag}x/{slide_id}.h5`` の対象ファイルを
ローカルキャッシュへ複製し，新しいルート（キャッシュ先）を返す
コピー前に必要容量と SSD 空き容量を確認し，収まればコピー，収まらなければ警告して
元のルートを返す（NAS 直読フォールバック）
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Sequence, Set

from foveamil.training.accessor import (
    CLS_DATASET,
    COORDS_DATASET,
    FEATURE_TYPE_CLS,
    FEATURE_TYPE_MEAN,
    POOLED_DATASET,
)

logger = logging.getLogger(__name__)

# キャッシュ先ルートを与える環境変数名
STAGE_DIR_ENV = "FOVEAMIL_STAGE_DIR"
# 既定の空き容量安全マージン（空きの ``1 - margin`` までを使う）
DEFAULT_FREE_MARGIN = 0.1
# バイト ⇔ GB 換算係数
BYTES_PER_GB = 1024 ** 3
# 環境変数も未設定のときに /tmp 配下へ作る既定ディレクトリ名の接頭辞
_DEFAULT_DIR_PREFIX = "foveamil_feat_stage_"
# コピーの既定並列ワーカ数（h5 再シリアライズ/コピーをプロセス並列にする）
DEFAULT_COPY_WORKERS = 8
# 並列ワーカ数を上書きする環境変数名
STAGE_WORKERS_ENV = "FOVEAMIL_STAGE_WORKERS"
# keep 指定時に必要容量を見積るためのサンプル h5 数（全件メタ走査を避ける）
_SIZE_SAMPLE_FILES = 64
# feature_type ごとにコピーするデータセット（``None`` は h5 全体をそのまま複製）
_KEEP_DATASETS = {
    FEATURE_TYPE_CLS: (CLS_DATASET, COORDS_DATASET),
    FEATURE_TYPE_MEAN: (POOLED_DATASET, COORDS_DATASET),
}


def _needed_datasets(feature_type: Optional[str]) -> Optional[Set[str]]:
    """``feature_type`` で必要なデータセット名集合を返す（全体複製なら ``None``）

    ``cls`` / ``mean`` は対応する 1 特徴 ``+ coords`` のみで足り，``concat`` や未指定は
    h5 全体を複製する
    """
    keep = _KEEP_DATASETS.get(feature_type)
    return set(keep) if keep is not None else None


def _format_mag(magnification: float) -> str:
    """倍率を正準レイアウトのディレクトリ名（例 ``1.25`` → ``"1.25x"``）にする"""
    return f"{magnification}x"


def _default_cache_dir() -> str:
    """環境変数が未設定のときの既定キャッシュ先を返す（プロセス毎に衝突しない名前）"""
    return os.path.join("/tmp", f"{_DEFAULT_DIR_PREFIX}{os.getpid()}")


def _resolve_copy_workers(copy_workers: Optional[int]) -> int:
    """並列ワーカ数を解決する（引数 > 環境変数 > 既定の順・下限 1）"""
    if copy_workers is None:
        env = os.environ.get(STAGE_WORKERS_ENV)
        copy_workers = int(env) if env else DEFAULT_COPY_WORKERS
    return max(1, int(copy_workers))


def _write_subset_h5(src: str, dst: str, keep: Set[str]) -> None:
    """``src`` の ``keep`` データセットと属性のみを ``dst`` の h5 へ書く"""
    import h5py

    with h5py.File(src, "r") as fsrc, h5py.File(dst, "w") as fdst:
        for key, value in fsrc.attrs.items():
            fdst.attrs[key] = value
        for name in keep:
            if name not in fsrc:
                continue
            source = fsrc[name]
            out = fdst.create_dataset(name, data=source[()])
            for key, value in source.attrs.items():
                out.attrs[key] = value


def _copy_atomic(src: str, dst: str, keep: Optional[Set[str]] = None) -> None:
    """``src`` を ``dst`` へアトミックにコピーする（tmp 名→rename・冪等）

    ``dst`` が既にあれば再利用しコピーしない``keep`` 指定時は h5 全体でなく
    指定データセット（+ root/データセット属性）のみを書いた縮小 h5 を作る``None``
    なら h5 全体を複製するプロセス並列ワーカから呼べるようモジュール関数にしている
    """
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp"
    if keep is None:
        shutil.copy2(src, tmp)
    else:
        _write_subset_h5(src, tmp, keep)
    os.replace(tmp, dst)


class FeatureStager:
    """特徴量セットをローカルキャッシュへ一括コピーする管理器

    Args:
        cache_dir: ローカルキャッシュ先``None`` のとき環境変数
            ``FOVEAMIL_STAGE_DIR`` を読み，それも無ければ ``/tmp`` 配下の
            プロセス固有ディレクトリを使う
        free_space_margin: 空き容量の安全マージン（例 ``0.1`` で空きの 90% まで使う）
        copy_workers: コピーの並列ワーカ数``None`` のとき環境変数
            ``FOVEAMIL_STAGE_WORKERS`` を読み，それも無ければ ``DEFAULT_COPY_WORKERS``
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        free_space_margin: float = DEFAULT_FREE_MARGIN,
        copy_workers: Optional[int] = None,
    ) -> None:
        if cache_dir is None:
            cache_dir = os.environ.get(STAGE_DIR_ENV) or _default_cache_dir()
        self.cache_dir = cache_dir
        self.free_space_margin = free_space_margin
        self.copy_workers = _resolve_copy_workers(copy_workers)
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

    def _copy_atomic(
        self, src: str, dst: str, keep: Optional[Set[str]] = None
    ) -> None:
        """単一ファイルをアトミックにコピーする（モジュール関数へ委譲）"""
        _copy_atomic(src, dst, keep)

    def _copy_many(
        self,
        feature_root: str,
        rels: Sequence[str],
        keep: Optional[Set[str]],
    ) -> None:
        """対象ファイル列をワーカ数に応じて直列/プロセス並列でコピーする

        ``copy_workers <= 1`` か対象が 1 件以下なら直列h5 の再シリアライズは
        h5py のグローバルロックでスレッド並列が効かないためプロセス並列にする
        spawn コンテキストで子に HDF5 状態を継承させず安全に並列化する各
        ``_copy_atomic`` は冪等なので既存ファイルはワーカ側で即スキップされる
        """
        tasks = [
            (
                os.path.join(feature_root, rel),
                os.path.join(self.cache_dir, rel),
                keep,
            )
            for rel in rels
        ]
        if self.copy_workers <= 1 or len(tasks) <= 1:
            for src, dst, sub in tasks:
                _copy_atomic(src, dst, sub)
            return
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=self.copy_workers, mp_context=ctx
        ) as pool:
            futures = [
                pool.submit(_copy_atomic, src, dst, sub)
                for src, dst, sub in tasks
            ]
            for future in as_completed(futures):
                future.result()

    def _required_bytes(
        self, feature_root: str, rels: Sequence[str], keep: Optional[Set[str]]
    ) -> int:
        """コピーに必要なバイト数を返す

        ``keep`` 指定時は h5 を等間隔に最大 ``_SIZE_SAMPLE_FILES`` 件サンプルし
        該当データセットの論理サイズ（``shape × itemsize``）の平均×件数で推定する
        （全件メタ走査だと数万ファイルで数十分かかるため）``None`` なら h5 全体の
        ファイルサイズを合計する
        """
        if keep is None:
            return sum(
                os.path.getsize(os.path.join(feature_root, rel)) for rel in rels
            )
        count = len(rels)
        if count == 0:
            return 0
        import h5py

        sample_n = min(_SIZE_SAMPLE_FILES, count)
        step = count / sample_n
        indices = sorted({int(i * step) for i in range(sample_n)})
        sampled = 0
        for idx in indices:
            with h5py.File(os.path.join(feature_root, rels[idx]), "r") as handle:
                for name in keep:
                    if name in handle:
                        dataset = handle[name]
                        sampled += int(dataset.dtype.itemsize) * int(
                            dataset.size
                        )
        return int(sampled / len(indices) * count)

    def stage_set(
        self,
        feature_root: str,
        encoder: str,
        magnifications: Sequence[float],
        slide_ids: Sequence[str],
        feature_type: Optional[str] = None,
    ) -> str:
        """対象特徴セットをキャッシュへ一括コピーし新しいルートを返す

        必要容量が SSD 空きの ``1 - free_space_margin`` 以内なら ``cache_dir`` 構造へ
        コピーし ``cache_dir`` を返す（既存ファイルは再利用）収まらなければ警告して
        元の ``feature_root`` を返す（NAS 直読フォールバック）``feature_type`` に
        ``cls`` / ``mean`` を与えると h5 全体でなく該当特徴 ``+ coords`` のみを書いた
        縮小 h5 を作り容量を約半分にする（``concat`` や未指定は h5 全体を複製する）

        Args:
            feature_root: 特徴ルートディレクトリ
            encoder: エンコーダ名
            magnifications: 倍率の列
            slide_ids: 対象 slide_id の集合/列
            feature_type: ``cls`` / ``mean`` で縮小コピー``None`` は h5 全体を複製

        Returns:
            キャッシュ先ルート（コピー時）または元の ``feature_root``（フォールバック時）
        """
        rels = self._target_files(feature_root, encoder, magnifications, slide_ids)
        keep = _needed_datasets(feature_type)
        required = self._required_bytes(feature_root, rels, keep)

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

        self._copy_many(feature_root, rels, keep)
        self._staged = True

        logger.info(
            "staged feature set (%s): need %.2fGB / free %.2fGB / %d files "
            "/ %d workers -> %s",
            feature_type or "full",
            required / BYTES_PER_GB,
            free / BYTES_PER_GB,
            len(rels),
            self.copy_workers,
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
