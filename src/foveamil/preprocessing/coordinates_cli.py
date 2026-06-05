"""``foveamil-coords`` コマンド

入力スライドは ``--slides``（``slide_id`` 列を持つ CSV か 1 行 1 個の slide_id テキスト
:class:`WSIResolver` でパス解決）または ``--wsi-dir``（ディレクトリ内の対応 WSI を全件）で
指定する（排他）``--overrides`` で ``slide_id,path`` のパス対応表を与えられる

1 スライドの失敗は記録して継続し，最後に成功／失敗を集計する
``--num-workers > 1`` で WSI 単位に並列処理する
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
from foveamil.preprocessing.coordinates import process_wsi, validate_magnifications
from foveamil.utils.notify import send_email
from foveamil.wsi.resolver import (
    SUPPORTED_WSI_EXTENSIONS,
    WSIResolutionError,
    WSIResolver,
)
from foveamil.wsi.tissue import SimpleTissueMask

logger = logging.getLogger(__name__)

# 既定のパッチサイズ・組織割合しきい値
DEFAULT_PATCH_SIZE = 224
DEFAULT_TISSUE_THRESHOLD = 0.1
# 1 分あたりの秒数（処理時間の分換算用）
SECONDS_PER_MINUTE = 60
# 完了通知・エラー通知のメール件名
NOTIFY_SUBJECT_DONE = "foveamil-coords completed"
NOTIFY_SUBJECT_ERROR = "foveamil-coords error"


def _load_dotenv_if_available() -> None:
    """``.env`` があれば読み込む``python-dotenv`` が無ければ何もしない"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


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


def _process_one(task: dict) -> Tuple[str, bool, Optional[str]]:
    """1 スライドを処理するワーカー（``multiprocessing.Pool`` で pickle 可能）

    Returns:
        ``(slide_id, 成功フラグ, エラートレース or None)``
    """
    import traceback

    try:
        process_wsi(
            wsi_path=task["wsi_path"],
            output_dir=task["output_dir"],
            magnifications=task["magnifications"],
            patch_size=task["patch_size"],
            stride=task["stride"],
            tissue_threshold=task["tissue_threshold"],
            mask_generator=SimpleTissueMask(sigma=task["mask_sigma"]),
            slide_id=task["slide_id"],
        )
        return task["slide_id"], True, None
    except Exception:  # noqa: BLE001 - 1 枚の失敗で全体を止めない
        return task["slide_id"], False, traceback.format_exc()


def _build_tasks(
    slides: Sequence[Tuple[str, str]], args: argparse.Namespace
) -> List[dict]:
    """``(slide_id, path)`` 一覧からワーカー用タスク辞書の一覧を作る"""
    return [
        {
            "slide_id": slide_id,
            "wsi_path": wsi_path,
            "output_dir": args.out,
            "magnifications": list(args.mags),
            "patch_size": args.patch_size,
            "stride": args.stride,
            "tissue_threshold": args.tissue_threshold,
            "mask_sigma": SimpleTissueMask().sigma,
        }
        for slide_id, wsi_path in slides
    ]


def _resolve_input_slides(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """入力指定（``--wsi-dir`` / ``--slides``）から処理対象を決める

    Returns:
        ``(slides, resolution_failures)````slides`` は ``(slide_id, path)`` 一覧
    """
    if args.wsi_dir:
        return _slides_from_dir(args.wsi_dir), []

    overrides_csv = args.overrides
    if overrides_csv:
        resolver = WSIResolver.from_overrides_csv(
            overrides_csv, base_path=args.wsi_base_path
        )
    else:
        resolver = WSIResolver(base_path=args.wsi_base_path)
    return _slides_from_resolver(args.slides, resolver)


def _summary_body(
    args: argparse.Namespace,
    n_ok: int,
    n_unresolved: int,
    n_processing_failed: int,
    elapsed_seconds: float,
) -> str:
    """完了通知メールの本文（プレーンテキスト）を組み立てる"""
    minutes = elapsed_seconds / SECONDS_PER_MINUTE
    lines = [
        f"ok: {n_ok}",
        f"failed: {n_unresolved + n_processing_failed}",
        f"unresolved: {n_unresolved}",
        f"processing_failed: {n_processing_failed}",
        f"output_dir: {args.out}",
        f"magnifications: {list(args.mags)}",
        f"patch_size: {args.patch_size}",
        f"stride: {args.stride}",
        f"elapsed: {elapsed_seconds:.0f}s ({minutes:.1f}min)",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    """座標抽出を実行する終了コード（失敗が 1 件でもあれば 1）を返す"""
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
                f"error: {exc}\nelapsed: {elapsed:.0f}s",
            )
        raise


def _run_extraction(args: argparse.Namespace, start_time: float) -> int:
    """座標抽出本体``run`` から呼ばれ，終了コードを返す"""
    try:
        validate_magnifications(args.mags)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    os.makedirs(args.out, exist_ok=True)

    slides, resolution_failures = _resolve_input_slides(args)
    for slide_id, err in resolution_failures:
        logger.error("could not resolve %s: %s", slide_id, err)

    if not slides:
        logger.error("no slides to process")
        return 1

    logger.info(
        "processing %d slides @ mags=%s patch=%d stride=%d threshold=%.3g workers=%d",
        len(slides),
        list(args.mags),
        args.patch_size,
        args.stride,
        args.tissue_threshold,
        args.num_workers,
    )

    tasks = _build_tasks(slides, args)
    process_failures: List[Tuple[str, str]] = []
    n_ok = 0

    if args.num_workers > 1:
        from multiprocessing import Pool

        with Pool(processes=args.num_workers) as pool:
            for slide_id, ok, err in pool.imap_unordered(_process_one, tasks):
                if ok:
                    n_ok += 1
                else:
                    process_failures.append((slide_id, err or ""))
                    logger.error("failed %s:\n%s", slide_id, err)
    else:
        for task in tasks:
            slide_id, ok, err = _process_one(task)
            if ok:
                n_ok += 1
            else:
                process_failures.append((slide_id, err or ""))
                logger.error("failed %s:\n%s", slide_id, err)

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
        body = _summary_body(
            args,
            n_ok=n_ok,
            n_unresolved=len(resolution_failures),
            n_processing_failed=len(process_failures),
            elapsed_seconds=time.monotonic() - start_time,
        )
        send_email(NOTIFY_SUBJECT_DONE, body)

    return 0 if total_failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-coords`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-coords",
        description="Extract hierarchical multi-resolution tissue patch coordinates "
        "from WSIs into per-magnification H5 files.",
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
        "--out", required=True, help="Output directory for coordinate H5 files."
    )
    parser.add_argument(
        "--mags",
        type=float,
        nargs="+",
        required=True,
        help="Magnifications low->high, adjacent ratio 2.0 (e.g. 1.25 2.5 5.0 10.0).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Patch size in pixels (default {DEFAULT_PATCH_SIZE}).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride in pixels (default: same as --patch-size).",
    )
    parser.add_argument(
        "--tissue-threshold",
        type=float,
        default=DEFAULT_TISSUE_THRESHOLD,
        help=f"Min tissue fraction per patch (default {DEFAULT_TISSUE_THRESHOLD}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel WSI workers (default 1).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send a summary email on completion (and on error) via Gmail SMTP.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``foveamil-coords`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride is None:
        args.stride = args.patch_size
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
