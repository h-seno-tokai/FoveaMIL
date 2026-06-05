"""``foveamil-features`` コマンド

座標 H5 と WSI からパッチ特徴を抽出し，倍率ごとに 1 ファイルへ保存する

入力スライドは ``--slides``（``slide_id`` 列を持つ CSV か 1 行 1 個のテキスト
:class:`WSIResolver` でパス解決）または ``--wsi-dir``（ディレクトリ内の対応 WSI を全件）で
指定する（排他）``--overrides`` で ``slide_id,path`` のパス対応表を与えられる

``--gpu-ids`` で複数の物理 GPU を指定すると，各 GPU の常駐ワーカが空き次第
スライドを取りに行く（動的割当）未指定時は可視 GPU 全て，無ければ CPU を使う
全倍率の出力が揃ったスライドは再開時に処理対象から除く

1 スライドの失敗は記録して継続し，最後に成功／失敗を集計する``--stage`` 指定時は
:class:`WSIStager` でローカル SSD へ退避してから読み，処理後に解放する
``--notify`` 指定時は開始・完了・エラーに日本語のメールを送る
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from foveamil.cohort.labels import load_slide_ids
from foveamil.encoders import ENCODERS
from foveamil.preprocessing.features import (
    DEFAULT_BASE_MAG,
    DEFAULT_HIGHEST_MAG,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SATURATION_THRESHOLD,
    extract_features_distributed,
    resolve_worker_devices,
)
from foveamil.utils.notify import send_email
from foveamil.wsi.resolver import (
    SUPPORTED_WSI_EXTENSIONS,
    WSIResolutionError,
    WSIResolver,
)

logger = logging.getLogger(__name__)

# 既定の推論バッチサイズ（``--batch-size`` 未指定かつ環境変数も無いとき）
DEFAULT_BATCH_SIZE = 256
# 既定のパッチ I/O 並列ワーカ数（``--num-workers`` 未指定かつ環境変数も無いとき）
DEFAULT_NUM_WORKERS = 4
# バッチサイズ・ワーカ数を上書きする環境変数名
ENV_BATCH_SIZE = "PREPROCESS_BATCH_SIZE"
ENV_NUM_WORKERS = "PREPROCESS_NUM_WORKERS"
# 1 分あたりの秒数（処理時間の分換算用）
SECONDS_PER_MINUTE = 60
# 通知メールの件名
NOTIFY_SUBJECT_START = "🔬 FoveaMIL 特徴抽出をはじめました"
NOTIFY_SUBJECT_DONE = "✅ 特徴抽出が完了しました！"
NOTIFY_SUBJECT_ERROR = "⚠️ 特徴抽出でエラーが発生しました"


def _load_dotenv_if_available() -> None:
    """``.env`` があれば読み込む``python-dotenv`` が無ければ何もしない"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _resolve_int(cli_value: Optional[int], env_name: str, default: int) -> int:
    """整数設定値を ``CLI 値 → 環境変数 → 既定`` の優先順で解決する"""
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return int(env_value)
    return default


def _slides_from_dir(wsi_dir: str) -> List[Tuple[str, str]]:
    """ディレクトリ内の対応 WSI を ``(slide_id, path)`` の一覧として返す"""
    pairs: List[Tuple[str, str]] = []
    for ext in SUPPORTED_WSI_EXTENSIONS:
        for path in glob.glob(os.path.join(wsi_dir, f"*.{ext}")):
            pairs.append((Path(path).stem, os.path.abspath(path)))
    return sorted(set(pairs))


def _slides_from_resolver(
    slides_file: str, resolver: WSIResolver
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """slide_id ファイルを読み，解決できた組と失敗した組に分けて返す

    Returns:
        ``(resolved, failures)````resolved`` は ``(slide_id, path)`` の一覧，
        ``failures`` は ``(slide_id, エラーメッセージ)`` の一覧
    """
    slide_ids = sorted(load_slide_ids(slides_file))
    resolved: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str]] = []
    for slide_id in slide_ids:
        try:
            resolved.append((slide_id, resolver.resolve(slide_id)))
        except WSIResolutionError as exc:
            failures.append((slide_id, str(exc)))
    return resolved, failures


def _resolve_input_slides(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """入力指定（``--wsi-dir`` / ``--slides``）から処理対象を決める

    Returns:
        ``(slides, resolution_failures)````slides`` は ``(slide_id, path)`` 一覧
    """
    if args.wsi_dir:
        return _slides_from_dir(args.wsi_dir), []

    if args.overrides:
        resolver = WSIResolver.from_overrides_csv(
            args.overrides, base_path=args.wsi_base_path
        )
    else:
        resolver = WSIResolver(base_path=args.wsi_base_path)
    return _slides_from_resolver(args.slides, resolver)


def _start_body(
    args: argparse.Namespace, n_slides: int, batch_size: int, num_workers: int
) -> str:
    """開始通知メールの本文（日本語・フレンドリー）を組み立てる"""
    return (
        "特徴抽出をはじめました 🙌\n"
        "\n"
        f"エンコーダ: {args.encoder}\n"
        f"対象スライド数: {n_slides} 枚\n"
        f"倍率: {list(args.mags)}\n"
        f"バッチサイズ: {batch_size}\n"
        f"並列ワーカ数: {num_workers}\n"
        "\n"
        "完了したらまたお知らせしますね，少々お待ちください ☕"
    )


def _done_body(
    args: argparse.Namespace,
    n_ok: int,
    n_failed: int,
    elapsed_seconds: float,
) -> str:
    """完了通知メールの本文（日本語・フレンドリー）を組み立てる"""
    minutes = elapsed_seconds / SECONDS_PER_MINUTE
    return (
        "おつかれさまでした，特徴抽出が完了しました 🎉\n"
        "\n"
        f"成功スライド: {n_ok} 枚\n"
        f"失敗スライド: {n_failed} 枚\n"
        f"エンコーダ: {args.encoder}\n"
        f"倍率: {list(args.mags)}\n"
        f"出力先: {args.out}\n"
        f"処理時間: {elapsed_seconds:.0f} 秒（約 {minutes:.1f} 分）\n"
        "\n"
        "ゆっくり休んでくださいね 😊"
    )


def _parse_int_list(value: str) -> List[int]:
    """カンマ区切りの整数文字列を整数リストにする（空トークンは無視）"""
    return [int(tok.strip()) for tok in value.split(",") if tok.strip()]


def run(args: argparse.Namespace) -> int:
    """特徴抽出を実行する終了コード（失敗が 1 件でもあれば 1）を返す"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_if_available()

    start_time = time.monotonic()
    try:
        return _run_extraction(args, start_time)
    except Exception as exc:  # noqa: BLE001 - 通知後に再送出する
        if getattr(args, "notify", False):
            elapsed = time.monotonic() - start_time
            send_email(
                NOTIFY_SUBJECT_ERROR,
                f"特徴抽出の途中でエラーが発生しました 🙇\n\n"
                f"内容: {exc}\n"
                f"経過時間: {elapsed:.0f} 秒",
            )
        raise


def _build_worker_cfg(
    args: argparse.Namespace, batch_size: int, num_workers: int
) -> dict:
    """ワーカへ渡す設定辞書を組み立てる（すべて pickle 可能な値）"""
    return {
        "encoder": args.encoder,
        "coords_dir": args.coords_dir,
        "out": args.out,
        "mags": list(args.mags),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "patch_size": args.patch_size,
        "skip_background": args.skip_background,
        "base_magnification": args.base_magnification,
        "highest_magnification": args.highest_magnification,
        "saturation_threshold": args.saturation_threshold,
        "stage": args.stage,
        "verbose": args.verbose,
    }


def _run_extraction(args: argparse.Namespace, start_time: float) -> int:
    """特徴抽出本体``run`` から呼ばれ，終了コードを返す"""
    if len(args.mags) < 1:
        logger.error("at least one magnification is required")
        return 1
    os.makedirs(args.out, exist_ok=True)

    batch_size = _resolve_int(args.batch_size, ENV_BATCH_SIZE, DEFAULT_BATCH_SIZE)
    num_workers = _resolve_int(args.num_workers, ENV_NUM_WORKERS, DEFAULT_NUM_WORKERS)

    slides, resolution_failures = _resolve_input_slides(args)
    for slide_id, err in resolution_failures:
        logger.error("could not resolve %s: %s", slide_id, err)

    if not slides:
        logger.error("no slides to process")
        return 1

    devices = resolve_worker_devices(args.gpu_ids)
    logger.info(
        "extracting %s for %d slides @ mags=%s batch=%d workers=%d devices=%s "
        "skip_bg=%s stage=%s",
        args.encoder,
        len(slides),
        list(args.mags),
        batch_size,
        num_workers,
        devices,
        args.skip_background,
        args.stage,
    )

    if args.notify:
        send_email(
            NOTIFY_SUBJECT_START,
            _start_body(args, len(slides), batch_size, num_workers),
        )

    cfg = _build_worker_cfg(args, batch_size, num_workers)
    n_ok, process_failures = extract_features_distributed(slides, devices, cfg)

    total_failures = len(resolution_failures) + len(process_failures)
    logger.info(
        "completed: %d ok, %d failed (%d unresolved, %d processing) -> %s",
        n_ok,
        total_failures,
        len(resolution_failures),
        len(process_failures),
        args.out,
    )

    if args.notify:
        send_email(
            NOTIFY_SUBJECT_DONE,
            _done_body(
                args,
                n_ok=n_ok,
                n_failed=total_failures,
                elapsed_seconds=time.monotonic() - start_time,
            ),
        )

    return 0 if total_failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-features`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-features",
        description="Extract patch features from coordinate H5 files and WSIs into "
        "per-magnification H5 files ({out}/{encoder}/{mag}x/{slide_id}.h5).",
    )

    parser.add_argument(
        "--encoder",
        required=True,
        choices=sorted(ENCODERS),
        help="Encoder name (registered in ENCODERS).",
    )
    parser.add_argument(
        "--coords-dir",
        required=True,
        help="Directory holding coordinate H5 files ({mag}x/{slide_id}.h5).",
    )
    parser.add_argument(
        "--out", required=True, help="Output root directory for feature H5 files."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--slides",
        help="labels.csv (with a 'slide_id' column) or a text file of slide_ids, "
        "one per line; resolved to file paths via WSIResolver.",
    )
    src.add_argument(
        "--wsi-dir",
        help="Process every supported WSI directly inside this directory.",
    )

    parser.add_argument(
        "--mags",
        type=float,
        nargs="+",
        required=True,
        help="Magnifications to extract (e.g. 1.25 2.5 5.0 10.0 20.0 40.0).",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional CSV (slide_id,path) overriding path resolution.",
    )
    parser.add_argument(
        "--wsi-base-path",
        default=None,
        help="Root directory for slide_id resolution (default: $WSI_BASE_PATH).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size in pixels (default {DEFAULT_PATCH_SIZE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Inference batch size (default: ${ENV_BATCH_SIZE} or {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=f"Patch I/O threads per GPU worker (default: ${ENV_NUM_WORKERS} or {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=_parse_int_list,
        default=None,
        help="Comma-separated physical GPU ids to distribute slides over (e.g. 0,1,2). "
        "Default: all visible GPUs (CUDA_VISIBLE_DEVICES), or CPU if none.",
    )
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Skip forward pass for background patches and fill with a dummy feature.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=float,
        default=DEFAULT_SATURATION_THRESHOLD,
        help=f"HSV saturation threshold for background (default {DEFAULT_SATURATION_THRESHOLD}).",
    )
    parser.add_argument(
        "--base-magnification",
        type=float,
        default=DEFAULT_BASE_MAG,
        help=f"Base magnification (background skip disabled) (default {DEFAULT_BASE_MAG}).",
    )
    parser.add_argument(
        "--highest-magnification",
        type=float,
        default=DEFAULT_HIGHEST_MAG,
        help=f"Highest magnification for background detection (default {DEFAULT_HIGHEST_MAG}).",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage WSIs to local SSD via WSIStager before reading.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send friendly start/done emails via Gmail SMTP.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-features`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
