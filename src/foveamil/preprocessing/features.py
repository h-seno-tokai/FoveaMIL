"""座標 H5 と WSI からパッチ特徴を抽出し，倍率ごとに 1 ファイルへ保存する

WSI×エンコーダ×倍率ごとに次のレイアウトで H5 を書く:
  ``{out_root}/{encoder.name}/{mag}x/{slide_id}.h5``
  dataset ``coords``: 座標 H5 の ``coords`` をそのまま ``(N, 2) int32``
  dataset ``patches``: pooled 特徴 ``(N, feature_dim) float32``
  dataset ``patches_cls``: cls 特徴 ``(N, feature_dim) float32``（``has_cls=True`` のみ）
  attrs: ``case_id``, ``encoder``, ``feature_dim``, ``has_cls``, ``magnification``, ``n_patches``

座標は level-0 ``(x, y)`` であり，倍率ごとに最寄りピラミッドレベルから読んで
``patch_size`` へ縮小する``skip_background`` 指定時は階層背景検出で背景パッチの
順伝播を省き，ダミー背景特徴で埋める
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import queue
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import h5py
import numpy as np
import openslide
import torch

from foveamil.encoders.base import PatchEncoder
from foveamil.utils.memory import force_gc, log_memory
from foveamil.wsi.slide import get_actual_max_magnification

logger = logging.getLogger(__name__)

# 既定のパッチサイズ（画素）
DEFAULT_PATCH_SIZE = 224
# 既定のベース（最低）倍率
DEFAULT_BASE_MAG = 1.25
# 既定の最高倍率（背景検出の判定基準）
DEFAULT_HIGHEST_MAG = 40.0
# 背景判定の彩度しきい値（HSV 彩度の最大値/255 がこれ未満なら背景）
DEFAULT_SATURATION_THRESHOLD = 0.05
# 既定のパッチ I/O 並列スレッド数
DEFAULT_NUM_WORKERS = 4
# ベストレベル探索で要求する読み出しサイズの patch_size 倍率（過剰拡大を避ける）
MIN_QUALITY_FACTOR = 1.5
# 背景彩度マップを作るピラミッドレベルの上限（``min(これ, level_count-1)``）
LOW_RES_LEVEL_CAP = 2
# coords dataset の dtype
COORDS_DTYPE = np.int32
# 特徴 dataset の dtype
FEATURE_DTYPE = np.float32
# RGB 画素値の最大（テンソル化時の正規化除数）
PIXEL_MAX = 255.0
# 解凍済みタイルの共有キャッシュ上限（バイト）SVS は 256x256 の JPEG タイルで，
# 224 パッチは複数タイルにまたがるため隣接パッチが同じタイルを再解凍する
# 全デコードスレッドで 1 つのキャッシュを共有し各タイルの解凍を 1 回に抑える
# 画素は read_region と同一（キャッシュは解凍結果の再利用のみ）
TILE_CACHE_BYTES = 1024 ** 3
# 一時出力ファイル名の接尾辞（同一 dir・pid 一意名でアトミック書き込みに使う）
TMP_SUFFIX = ".tmp"
# CPU 専用ワーカを表す物理デバイス ID の番兵
CPU_DEVICE_SENTINEL = -1
# ワーカ結果待ちのポーリング間隔（秒）
RESULT_POLL_SECONDS = 1.0
# 各ワーカ join の待機上限（秒）超過分は terminate する
WORKER_JOIN_TIMEOUT = 30.0
# ワーカ別ステージ先ディレクトリ名の接頭辞
STAGE_SUBDIR_PREFIX = "foveamil_features_stage"


def _format_mag(magnification: float) -> str:
    """倍率を座標ファイル名と同じ表記（例 ``1.25`` → ``"1.25x"``）にする"""
    return f"{magnification}x"


def _coords_h5_path(coords_dir: str, slide_id: str, magnification: float) -> str:
    """指定倍率の座標 H5 パスを返す（``coordinates`` モジュールの命名と一致）"""
    return os.path.join(coords_dir, _format_mag(magnification), f"{slide_id}.h5")


def _output_h5_path(
    out_root: str, encoder_name: str, magnification: float, slide_id: str
) -> str:
    """出力 H5 パス ``{out_root}/{encoder}/{mag}x/{slide_id}.h5`` を返す"""
    return os.path.join(
        out_root, encoder_name, _format_mag(magnification), f"{slide_id}.h5"
    )


def _read_coords(coords_path: str) -> np.ndarray:
    """座標 H5 の ``coords`` を ``(N, 2) int32`` で読み出す"""
    with h5py.File(coords_path, "r") as f:
        return np.asarray(f["coords"][:], dtype=COORDS_DTYPE).reshape(-1, 2)


def _best_level_and_read_size(
    wsi: openslide.OpenSlide, magnification: float, actual_max_mag: int, patch_size: int
) -> Tuple[int, int]:
    """倍率に対するベストレベルと読み出しサイズを返す

    ``level0_size = ceil(patch_size * actual_max_mag / magnification)`` を基準に，
    最上位レベルから下って ``level0_size / level_downsample >= patch_size *
    MIN_QUALITY_FACTOR`` を満たす最初のレベルを採用する

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        magnification: 目標倍率
        actual_max_mag: WSI の最大倍率
        patch_size: パッチの一辺（画素）

    Returns:
        ``(best_level, read_size)``
    """
    downsample_factor = actual_max_mag / magnification
    level0_size = int(np.ceil(patch_size * downsample_factor))
    min_acceptable = patch_size * MIN_QUALITY_FACTOR

    best_level = 0
    read_size = level0_size
    for level in range(wsi.level_count - 1, -1, -1):
        level_size = level0_size / wsi.level_downsamples[level]
        if level_size >= min_acceptable:
            best_level = level
            read_size = int(np.ceil(level_size))
            break
    return best_level, read_size


def _stream_forward_indices(
    wsi_path: str,
    coords: np.ndarray,
    indices: List[int],
    best_level: int,
    read_size: int,
    patch_size: int,
    encoder: PatchEncoder,
    num_workers: int,
    prefetch: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``indices`` のパッチをデコードしながら順伝播し特徴を返す

    デコード（CPU, スレッド）と順伝播（GPU）を重ねるため，バックグラウンドの
    プロデューサがバッチ単位でパッチをデコードしてキューに積み，本スレッドは
    届いたバッチを順に ``encoder`` へ流す全パッチのデコードを待たずに GPU を
    回し続けるバッチ構成・デコード・正規化・順伝播は逐次版と同一で出力は一致する

    各パッチは ``read_region`` で読み，``read_size != patch_size`` なら BILINEAR で
    ``patch_size`` に縮小し ``[0, 1]`` スケールにする

    Args:
        wsi_path: 読み出す WSI のパス（スレッドローカルに開く）
        coords: 全パッチの level-0 座標 ``(N, 2)``
        indices: 順伝播対象の ``coords`` 上インデックス列（出力はこの順）
        best_level: 読み出すピラミッドレベル
        read_size: そのレベルで読み出す一辺
        patch_size: 出力パッチの一辺
        encoder: ロード済みエンコーダ
        num_workers: デコードの並列スレッド数
        prefetch: 先読みするバッチ数（デコード済みバッチのキュー上限）

    Returns:
        ``(pooled[len(indices), dim] float32, cls[...] float32 または None)``
    """
    tile_cache = openslide.OpenSlideCache(TILE_CACHE_BYTES)
    thread_local = threading.local()

    def _get_wsi() -> openslide.OpenSlide:
        if not hasattr(thread_local, "wsi"):
            wsi = openslide.OpenSlide(wsi_path)
            wsi.set_cache(tile_cache)  # 全スレッドで共有しタイル再解凍を防ぐ
            thread_local.wsi = wsi
        return thread_local.wsi

    def _decode(idx: int) -> torch.Tensor:
        x, y = coords[idx]
        region = _get_wsi().read_region(
            location=(int(x), int(y)), level=best_level, size=(read_size, read_size)
        )
        img = np.asarray(region.convert("RGB"))
        if read_size != patch_size:
            img = cv2.resize(
                img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
            )
        return torch.from_numpy(img).permute(2, 0, 1).float() / PIXEL_MAX

    batch_size = encoder.batch_size
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    batch_queue: "queue.Queue" = queue.Queue(maxsize=max(1, prefetch))

    def _produce() -> None:
        try:
            with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
                for batch_indices in batches:
                    tensors = list(executor.map(_decode, batch_indices))
                    batch_queue.put(torch.stack(tensors, dim=0))
            batch_queue.put(None)
        except Exception as exc:  # プロデューサ側の失敗を本スレッドへ伝える
            batch_queue.put(exc)

    producer = threading.Thread(target=_produce, daemon=True)
    producer.start()

    pooled_chunks: List[np.ndarray] = []
    cls_chunks: List[np.ndarray] = []
    while True:
        item = batch_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        pooled, cls = encoder.forward(_normalize(item, encoder))
        pooled_chunks.append(pooled.float().cpu().numpy())
        if cls is not None:
            cls_chunks.append(cls.float().cpu().numpy())
    producer.join()

    pooled_array = np.concatenate(pooled_chunks, axis=0).astype(FEATURE_DTYPE)
    cls_array = (
        np.concatenate(cls_chunks, axis=0).astype(FEATURE_DTYPE) if cls_chunks else None
    )
    return pooled_array, cls_array


def _normalize(patches: torch.Tensor, encoder: PatchEncoder) -> torch.Tensor:
    """``[B, 3, H, W]`` を ``encoder`` の平均・標準偏差で正規化する"""
    mean = torch.tensor(encoder.normalizer_mean).view(1, 3, 1, 1)
    std = torch.tensor(encoder.normalizer_std).view(1, 3, 1, 1)
    return (patches - mean) / std


def _is_background(img_rgb: np.ndarray, saturation_threshold: float) -> bool:
    """RGB パッチが背景か判定する（HSV 彩度の最大値/255 < しきい値）"""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 1].max() / PIXEL_MAX) < saturation_threshold


def extract_dummy_feature(
    wsi_path: str,
    encoder: PatchEncoder,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """WSI の背景領域からダミー背景特徴 ``(pooled, cls)`` を 1 度だけ抽出する

    Args:
        wsi_path: 読み出す WSI のパス
        encoder: ロード済み（または自動ロードされる）エンコーダ
        patch_size: パッチの一辺（画素）
        saturation_threshold: 背景判定の彩度しきい値

    Returns:
        ``(pooled[feature_dim], cls[feature_dim] または None)``
    """
    encoder.load()
    wsi = openslide.OpenSlide(wsi_path)
    try:
        return _extract_dummy_feature(wsi, encoder, patch_size, saturation_threshold)
    finally:
        wsi.close()


def _extract_dummy_feature(
    wsi: openslide.OpenSlide,
    encoder: PatchEncoder,
    patch_size: int,
    saturation_threshold: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """背景領域から 1 回だけダミー特徴（pooled, cls）を抽出する

    コーナー・辺の候補から背景パッチ（彩度の最大値がしきい値未満）を探し，
    見つからなければ左上を使う

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        encoder: ロード済みエンコーダ
        patch_size: パッチの一辺（画素）
        saturation_threshold: 背景判定の彩度しきい値

    Returns:
        ``(pooled[feature_dim], cls[feature_dim] または None)``
    """
    width, height = wsi.level_dimensions[0]
    candidates = [
        (0, 0),
        (width - patch_size, 0),
        (0, height - patch_size),
        (width - patch_size, height - patch_size),
        (width // 2, 0),
        (0, height // 2),
    ]

    chosen: Optional[np.ndarray] = None
    for x, y in candidates:
        x = max(0, min(x, width - patch_size))
        y = max(0, min(y, height - patch_size))
        region = wsi.read_region(location=(int(x), int(y)), level=0, size=(patch_size, patch_size))
        img = np.asarray(region.convert("RGB"))
        if _is_background(img, saturation_threshold):
            chosen = img
            break

    if chosen is None:
        region = wsi.read_region(location=(0, 0), level=0, size=(patch_size, patch_size))
        chosen = np.asarray(region.convert("RGB"))

    tensor = torch.from_numpy(chosen).permute(2, 0, 1).float().unsqueeze(0) / PIXEL_MAX
    pooled, cls = encoder.forward(_normalize(tensor, encoder))
    dummy_pooled = pooled.float().cpu().numpy().reshape(-1)
    dummy_cls = cls.float().cpu().numpy().reshape(-1) if cls is not None else None
    return dummy_pooled, dummy_cls


def _build_tissue_set(
    wsi: openslide.OpenSlide,
    coords_dir: str,
    slide_id: str,
    highest_magnification: float,
    actual_max_mag: int,
    patch_size: int,
    saturation_threshold: float,
) -> Set[Tuple[int, int]]:
    """最高倍率の座標を彩度マップで判定し，組織パッチの ``(x, y)`` 集合を作る

    低解像度（``min(LOW_RES_LEVEL_CAP, level_count-1)`` レベル）の彩度マップ上で，
    各最高倍率パッチの対応領域の最大彩度が ``saturation_threshold`` 以上なら組織とする

    Args:
        wsi: 開いた :class:`openslide.OpenSlide`
        coords_dir: 座標 H5 のディレクトリ
        slide_id: 対象 slide_id
        highest_magnification: 最高倍率
        actual_max_mag: WSI の最大倍率
        patch_size: パッチの一辺（画素）
        saturation_threshold: 背景判定の彩度しきい値

    Returns:
        組織と判定した level-0 座標 ``(x, y)`` の集合（座標が無ければ空集合）
    """
    coords_path = _coords_h5_path(coords_dir, slide_id, highest_magnification)
    if not os.path.exists(coords_path):
        logger.warning("highest-mag coords not found, background skip disabled: %s", coords_path)
        return set()

    level = min(LOW_RES_LEVEL_CAP, wsi.level_count - 1)
    level_dims = wsi.level_dimensions[level]
    level_downsample = wsi.level_downsamples[level]
    region = wsi.read_region((0, 0), level, level_dims)
    full = np.asarray(region.convert("RGB"))
    saturation = cv2.cvtColor(full, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / PIXEL_MAX
    del full
    force_gc()

    highest_coords = _read_coords(coords_path)
    highest_patch_size_level0 = int(patch_size * (actual_max_mag / highest_magnification))
    sat_patch_size = max(1, int(highest_patch_size_level0 / level_downsample))

    tissue: Set[Tuple[int, int]] = set()
    for x, y in highest_coords:
        sat_x = min(int(x / level_downsample), saturation.shape[1] - 1)
        sat_y = min(int(y / level_downsample), saturation.shape[0] - 1)
        sat_x_end = min(sat_x + sat_patch_size, saturation.shape[1])
        sat_y_end = min(sat_y + sat_patch_size, saturation.shape[0])
        if saturation[sat_y:sat_y_end, sat_x:sat_x_end].max() >= saturation_threshold:
            tissue.add((int(x), int(y)))

    del saturation
    force_gc()
    logger.info(
        "%s tissue patches at %s: %d / %d",
        slide_id,
        _format_mag(highest_magnification),
        len(tissue),
        len(highest_coords),
    )
    return tissue


def _tissue_indices(
    coords: np.ndarray,
    magnification: float,
    base_magnification: float,
    highest_magnification: float,
    tissue_set: Set[Tuple[int, int]],
    actual_max_mag: int,
    patch_size: int,
) -> List[int]:
    """背景スキップ時に順伝播すべきパッチのインデックス列を返す

    ``magnification == base_magnification`` または ``tissue_set`` が空のときは全件を返す
    最高倍率は ``tissue_set`` に直接含まれるか，低倍率は子の最高倍率座標が自分の
    level-0 ボックス内に 1 つでもあれば組織とする

    Args:
        coords: 対象倍率の level-0 座標 ``(N, 2)``
        magnification: 対象倍率
        base_magnification: ベース倍率（背景スキップ無効）
        highest_magnification: 最高倍率
        tissue_set: 最高倍率の組織座標集合
        actual_max_mag: WSI の最大倍率
        patch_size: パッチの一辺（画素）

    Returns:
        順伝播対象の ``coords`` 上インデックス列
    """
    if magnification == base_magnification or not tissue_set:
        return list(range(len(coords)))

    if magnification == highest_magnification:
        return [
            idx
            for idx, (x, y) in enumerate(coords)
            if (int(x), int(y)) in tissue_set
        ]

    # 低倍率: 各パッチの level-0 ボックス [x, x+P) x [y, y+P) に最高倍率の組織座標が
    # 1 つでも入れば組織とする低倍率パッチは間隔 P の非重複格子なので，各組織座標は
    # ちょうど 1 つのボックスに属する組織座標を所属ボックス原点へ写像し，実在する
    # 低倍率パッチと突き合わせる（元の二重ループ述語と同値で O(N^2) を解消）
    patch_size_level0 = int(patch_size * (actual_max_mag / magnification))
    coords_int = np.asarray(coords, dtype=np.int64)
    xs = np.unique(coords_int[:, 0])
    ys = np.unique(coords_int[:, 1])
    tissue = np.asarray(sorted(tissue_set), dtype=np.int64)
    xi = np.searchsorted(xs, tissue[:, 0], side="right") - 1
    yi = np.searchsorted(ys, tissue[:, 1], side="right") - 1
    valid = (xi >= 0) & (yi >= 0)
    box_x = np.where(valid, xs[np.clip(xi, 0, len(xs) - 1)], -1)
    box_y = np.where(valid, ys[np.clip(yi, 0, len(ys) - 1)], -1)
    inside = (
        valid
        & (tissue[:, 0] < box_x + patch_size_level0)
        & (tissue[:, 1] < box_y + patch_size_level0)
    )
    tissue_boxes = set(zip(box_x[inside].tolist(), box_y[inside].tolist()))
    return [
        idx
        for idx, (x, y) in enumerate(coords)
        if (int(x), int(y)) in tissue_boxes
    ]


def _write_features_h5(
    out_path: str,
    coords: np.ndarray,
    pooled: np.ndarray,
    cls: Optional[np.ndarray],
    *,
    slide_id: str,
    encoder: PatchEncoder,
    magnification: float,
) -> None:
    """特徴・座標・属性を出力 H5 にアトミックに書き出す（正準レイアウト）

    同一ディレクトリの一時ファイルへ書いてから ``os.replace`` で確定する
    中断・例外時は一時ファイルを削除し，最終パスには完成済みの H5 だけが現れる
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}{TMP_SUFFIX}.{os.getpid()}"
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("coords", data=coords.astype(COORDS_DTYPE))
            f.create_dataset("patches", data=pooled.astype(FEATURE_DTYPE))
            if encoder.has_cls and cls is not None:
                f.create_dataset("patches_cls", data=cls.astype(FEATURE_DTYPE))
            f.attrs["case_id"] = slide_id
            f.attrs["encoder"] = encoder.name
            f.attrs["feature_dim"] = int(encoder.feature_dim)
            f.attrs["has_cls"] = bool(encoder.has_cls)
            f.attrs["magnification"] = _format_mag(magnification)
            f.attrs["n_patches"] = int(len(coords))
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def extract_features_for_slide(
    slide_id: str,
    wsi_path: str,
    coords_dir: str,
    encoder: PatchEncoder,
    magnifications: List[float],
    out_root: str,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    skip_background: bool = False,
    base_magnification: float = DEFAULT_BASE_MAG,
    highest_magnification: float = DEFAULT_HIGHEST_MAG,
    saturation_threshold: float = DEFAULT_SATURATION_THRESHOLD,
    num_workers: int = DEFAULT_NUM_WORKERS,
    dummy_features: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None,
) -> List[str]:
    """1 スライドの全倍率について特徴を抽出し出力 H5 を書く

    各倍率の出力が既存ならその倍率はスキップする``skip_background`` 指定時は
    最高倍率の彩度判定で組織集合を作り，背景パッチは ``dummy_features`` で埋める
    （``dummy_features`` 未指定時はこのスライドの背景から 1 度だけ抽出する）

    Args:
        slide_id: 対象 slide_id（出力ファイル名・``case_id`` に使う）
        wsi_path: 読み出す WSI のパス
        coords_dir: 座標 H5 のディレクトリ
        encoder: ロード済み（または ``forward`` 時に自動ロードされる）エンコーダ
        magnifications: 抽出する倍率の列
        out_root: 出力ルートディレクトリ
        patch_size: パッチの一辺（画素）
        skip_background: 背景パッチの順伝播を省くか
        base_magnification: 背景スキップを無効にするベース倍率
        highest_magnification: 背景検出の判定基準となる最高倍率
        saturation_threshold: 背景判定の彩度しきい値
        num_workers: パッチ I/O の並列スレッド数
        dummy_features: 流用するダミー背景特徴 ``(pooled, cls)``

    Returns:
        書き出した（既存スキップを除く）出力 H5 パスの一覧
    """
    encoder.load()

    pending = [
        mag
        for mag in magnifications
        if not os.path.exists(
            _output_h5_path(out_root, encoder.name, mag, slide_id)
        )
    ]
    if not pending:
        logger.info("%s: all magnifications already present, skipping", slide_id)
        return []

    wsi = openslide.OpenSlide(wsi_path)
    written: List[str] = []
    try:
        actual_max_mag = get_actual_max_magnification(wsi)

        tissue_set: Set[Tuple[int, int]] = set()
        if skip_background:
            if dummy_features is None:
                dummy_features = _extract_dummy_feature(
                    wsi, encoder, patch_size, saturation_threshold
                )
            tissue_set = _build_tissue_set(
                wsi,
                coords_dir,
                slide_id,
                highest_magnification,
                actual_max_mag,
                patch_size,
                saturation_threshold,
            )

        for mag in pending:
            out_path = _output_h5_path(out_root, encoder.name, mag, slide_id)
            coords_path = _coords_h5_path(coords_dir, slide_id, mag)
            if not os.path.exists(coords_path):
                raise FileNotFoundError(f"coords H5 not found: {coords_path}")

            coords = _read_coords(coords_path)
            n = len(coords)
            log_memory(f"{slide_id} {_format_mag(mag)} start: {n} patches")

            pooled = np.zeros((n, encoder.feature_dim), dtype=FEATURE_DTYPE)
            cls = (
                np.zeros((n, encoder.feature_dim), dtype=FEATURE_DTYPE)
                if encoder.has_cls
                else None
            )

            if skip_background and dummy_features is not None:
                indices = _tissue_indices(
                    coords,
                    mag,
                    base_magnification,
                    highest_magnification,
                    tissue_set,
                    actual_max_mag,
                    patch_size,
                )
                pooled[:] = dummy_features[0]
                if cls is not None and dummy_features[1] is not None:
                    cls[:] = dummy_features[1]
            else:
                indices = list(range(n))

            if indices:
                best_level, read_size = _best_level_and_read_size(
                    wsi, mag, actual_max_mag, patch_size
                )
                feat_pooled, feat_cls = _stream_forward_indices(
                    wsi_path,
                    coords,
                    indices,
                    best_level,
                    read_size,
                    patch_size,
                    encoder,
                    num_workers,
                )
                pooled[indices] = feat_pooled
                if cls is not None and feat_cls is not None:
                    cls[indices] = feat_cls

            _write_features_h5(
                out_path,
                coords,
                pooled,
                cls,
                slide_id=slide_id,
                encoder=encoder,
                magnification=mag,
            )
            written.append(out_path)
            logger.info(
                "%s %s -> %d patches (%d forwarded)",
                slide_id,
                _format_mag(mag),
                n,
                len(indices),
            )
            force_gc()
    finally:
        wsi.close()
        force_gc()

    return written


def _slides_needing_work(
    slides: Sequence[Tuple[str, str]],
    out_root: str,
    encoder_name: str,
    magnifications: Sequence[float],
) -> List[Tuple[str, str]]:
    """全倍率の出力が揃っていない ``(slide_id, src_path)`` だけを返す

    ステージ前にこの判定で済みスライドを除くと無駄なローカルコピーを避けられる
    倍率単位の細かいスキップは :func:`extract_features_for_slide` が担う
    """
    return [
        (slide_id, src_path)
        for slide_id, src_path in slides
        if not all(
            os.path.exists(_output_h5_path(out_root, encoder_name, mag, slide_id))
            for mag in magnifications
        )
    ]


def _visible_physical_devices() -> List[int]:
    """親プロセスに見えている物理 GPU ID の列を返す（CUDA を初期化しない）

    ``CUDA_VISIBLE_DEVICES`` が設定されていればその値を物理 ID 列として解釈する
    未設定なら ``torch.cuda.device_count`` で全 GPU を数える GPU が無ければ空列
    ``device_count`` / ``is_available`` は CUDA コンテキストを作らない
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [int(tok.strip()) for tok in env.split(",") if tok.strip()]
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def resolve_worker_devices(gpu_ids: Optional[Sequence[int]]) -> List[int]:
    """使用する物理 GPU ID の列を親で（CUDA を初期化せずに）解決する

    優先順は ``gpu_ids`` > ``CUDA_VISIBLE_DEVICES`` > 全可視 GPU > CPU
    ``gpu_ids`` を与える場合は親に見えている GPU の部分集合でなければならない
    GPU が無ければ ``[CPU_DEVICE_SENTINEL]`` を返す（CPU ワーカ 1 つ）

    Raises:
        ValueError: ``gpu_ids`` が可視 GPU の部分集合でない場合
    """
    visible = _visible_physical_devices()
    if gpu_ids:
        requested = list(gpu_ids)
        if visible:
            invalid = [g for g in requested if g not in visible]
            if invalid:
                raise ValueError(
                    f"gpu_ids {invalid} は可視 GPU {visible} に含まれない"
                )
        return requested
    if visible:
        return visible
    return [CPU_DEVICE_SENTINEL]


def _worker_loop(
    device_id: int,
    cache_dir: str,
    cfg: Dict,
    task_queue,
    result_queue,
) -> None:
    """1 物理 GPU に常駐し ``task_queue`` から WSI を取り出して抽出するワーカ

    起動直後に自分の GPU だけを可視化してからエンコーダを 1 度だけロードする
    番兵 ``None`` を受け取るまで「ステージ→抽出→解放」を繰り返す
    背景ダミー特徴は初回に 1 度算出してワーカ内で流用する
    各スライドの結果 ``(slide_id, 成功フラグ, トレース or None)`` を返す
    1 スライドの失敗は捕捉して継続する
    """
    if device_id == CPU_DEVICE_SENTINEL:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    from foveamil.encoders import build_encoder
    from foveamil.wsi.staging import WSIStager

    logging.basicConfig(
        level=logging.DEBUG if cfg.get("verbose") else logging.INFO,
        format=f"%(levelname)s [gpu{device_id}] %(name)s: %(message)s",
    )

    encoder = build_encoder(
        cfg["encoder"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )
    encoder.load()

    stager: Optional["WSIStager"] = None
    if cfg["stage"]:
        stager = WSIStager(cache_dir=cache_dir, prefetch_ahead=0)
        atexit.register(stager.cleanup_all)

    dummy_features: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None

    while True:
        item = task_queue.get()
        if item is None:
            break
        slide_id, src_path = item
        local_path = src_path
        try:
            if stager is not None:
                local_path = stager.stage(src_path)
            if cfg["skip_background"] and dummy_features is None:
                dummy_features = extract_dummy_feature(
                    local_path,
                    encoder,
                    patch_size=cfg["patch_size"],
                    saturation_threshold=cfg["saturation_threshold"],
                )
            extract_features_for_slide(
                slide_id=slide_id,
                wsi_path=local_path,
                coords_dir=cfg["coords_dir"],
                encoder=encoder,
                magnifications=cfg["mags"],
                out_root=cfg["out"],
                patch_size=cfg["patch_size"],
                skip_background=cfg["skip_background"],
                base_magnification=cfg["base_magnification"],
                highest_magnification=cfg["highest_magnification"],
                saturation_threshold=cfg["saturation_threshold"],
                num_workers=cfg["num_workers"],
                dummy_features=dummy_features,
            )
            result_queue.put((slide_id, True, None))
        except Exception:  # noqa: BLE001 - 1 枚の失敗で全体を止めない
            result_queue.put((slide_id, False, traceback.format_exc()))
        finally:
            if stager is not None:
                stager.release(src_path)


def _prefetch_worker(cfg: Dict) -> None:
    """重みを CPU でロードしダウンロードのみ行う子プロセス用エントリ"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from foveamil.encoders import build_encoder

    build_encoder(
        cfg["encoder"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    ).load()


def _prefetch_encoder_weights(cfg: Dict) -> None:
    """エンコーダ重みを別プロセスで 1 度取得しワーカ間の同時ダウンロード競合を防ぐ

    親は CUDA を初期化しないため CPU 強制の ``spawn`` 子プロセスでロードして捨てる
    重みが既にキャッシュ済みなら再ダウンロードは起きない
    """
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_prefetch_worker, args=(cfg,))
    proc.start()
    proc.join()


def extract_features_distributed(
    slides: Sequence[Tuple[str, str]],
    devices: Sequence[int],
    cfg: Dict,
) -> Tuple[int, List[Tuple[str, str]]]:
    """スライド群を物理 GPU へ動的割当して並列抽出する

    各デバイスに常駐ワーカを 1 つ置き，空いたワーカが ``task_queue`` から次の
    スライドを取りに行く（事前のパッチ数計算や静的分割はしない）
    全倍率出力が揃ったスライドはステージ前に除外する
    1 ワーカの異常終了は他ワーカへ波及させない（残りは生存ワーカが消化する）
    親プロセスは CUDA を初期化しない（``spawn`` で各ワーカが自分の GPU を確保する）
    ワーカ別ステージ先は ``finally`` で必ず削除する

    Args:
        slides: 処理候補の ``(slide_id, src_path)`` 一覧
        devices: 使用する物理 GPU ID の列（``CPU_DEVICE_SENTINEL`` は CPU ワーカ）
        cfg: ワーカへ渡す設定（encoder/coords_dir/out/mags/batch_size/num_workers/
            patch_size/skip_background/base_magnification/highest_magnification/
            saturation_threshold/stage/verbose）すべて pickle 可能な値

    Returns:
        ``(成功件数, [(slide_id, トレース), ...])``
    """
    pending = _slides_needing_work(slides, cfg["out"], cfg["encoder"], cfg["mags"])
    skipped = len(slides) - len(pending)
    if skipped:
        logger.info("%d slides already complete, skipping", skipped)
    if not pending:
        logger.info("no slides to process")
        return 0, []

    device_list = list(devices)
    logger.info(
        "distributing %d slides over devices=%s (encoder=%s)",
        len(pending),
        device_list,
        cfg["encoder"],
    )

    _prefetch_encoder_weights(cfg)

    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for item in pending:
        task_queue.put(item)
    for _ in device_list:
        task_queue.put(None)

    stage_root = os.path.join(
        os.environ.get("FOVEAMIL_STAGE_DIR") or "/tmp",
        f"{STAGE_SUBDIR_PREFIX}_{os.getpid()}",
    )
    cache_dirs = {dev: os.path.join(stage_root, f"gpu{dev}") for dev in device_list}

    workers: List[multiprocessing.process.BaseProcess] = []
    for dev in device_list:
        proc = ctx.Process(
            target=_worker_loop,
            args=(dev, cache_dirs[dev], cfg, task_queue, result_queue),
        )
        proc.start()
        workers.append(proc)

    n_ok = 0
    failures: List[Tuple[str, str]] = []
    received = 0
    total = len(pending)
    try:
        while received < total:
            try:
                slide_id, ok, err = result_queue.get(timeout=RESULT_POLL_SECONDS)
            except queue.Empty:
                if not any(proc.is_alive() for proc in workers):
                    break
                continue
            received += 1
            if ok:
                n_ok += 1
            else:
                failures.append((slide_id, err or ""))
                logger.error("failed %s:\n%s", slide_id, err)
        for proc in workers:
            proc.join(timeout=WORKER_JOIN_TIMEOUT)
    except KeyboardInterrupt:
        logger.warning("interrupted; terminating workers")
        raise
    finally:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=WORKER_JOIN_TIMEOUT)
        shutil.rmtree(stage_root, ignore_errors=True)

    unreported = total - received
    if unreported > 0:
        logger.error(
            "%d slides produced no result (worker death); resume will redo them",
            unreported,
        )
        failures.extend(
            ("<unreported>", "worker exited before reporting result")
            for _ in range(unreported)
        )

    return n_ok, failures
