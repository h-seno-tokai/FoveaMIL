"""``foveamil-sweep`` コマンド

``resolve`` / ``sweep`` / ``fixed`` / ``parallel`` の 4 ブロックからなる YAML を読み，
``(encoder, feature_type)`` を制約付き，他の軸を直積で展開して combo を作る各 combo を
fold へ展開し ``foveamil-train --split`` のサブプロセスとして GPU へ割り当て並列実行する
パス・特徴次元・split は ``resolve`` から自動解決する``--dry-run`` は展開結果と job 数・
解決値を表示して実行しないログ・結果は ``--out``（home），重み（``.pt``）は
``--weights-out``（Dataset，未指定なら ``--out``）へ分けて保存する``--notify`` 指定時は
開始・完了に日本語のメールを送る
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

import yaml

from foveamil.training.resolve import resolve_paths, resolve_split_files, verify_n_cls
from foveamil.training.sweep import (
    DEFAULT_GPU_HEADROOM_MB,
    SWEEP_SUMMARY_MD,
    SweepRunner,
    expand_combos,
    varying_axis_keys,
    verify_feature_dims,
)
from foveamil.utils.notify import send_email

logger = logging.getLogger(__name__)

# YAML トップレベルで許すブロック名
CONFIG_BLOCKS = ("resolve", "sweep", "fixed", "parallel")
# 入力 YAML を再現性のため複製する名前
SWEEP_CONFIG_COPY = "sweep_config.yaml"
# 通知メールの件名
NOTIFY_SUBJECT_START = "🔬 FoveaMIL sweep をはじめました"
NOTIFY_SUBJECT_DONE = "✅ sweep が完了しました！"
NOTIFY_SUBJECT_ERROR = "⚠️ sweep でエラーが発生しました"


def _load_dotenv_if_available() -> None:
    """``.env`` があれば読み込む``python-dotenv`` が無ければ何もしない"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """``"0,1"`` を ``[0, 1]`` に解釈する``None`` ならそのまま返す"""
    if value is None:
        return None
    return [int(tok) for tok in value.split(",") if tok.strip() != ""]


def _load_config(path: str) -> Dict[str, Any]:
    """sweep 設定 YAML（4 ブロックの辞書）を読む"""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"sweep YAML must be a mapping: {path}")
    unknown = set(raw) - set(CONFIG_BLOCKS)
    if unknown:
        raise ValueError(
            f"unknown top-level blocks {sorted(unknown)}; "
            f"allowed: {list(CONFIG_BLOCKS)}"
        )
    return raw


def _print_dry_run(combos, split_files, resolved) -> None:
    """展開結果と job 数・解決値を表示する（実行しない）"""
    axis_keys = varying_axis_keys(combos)
    print(f"resolved labels_csv : {resolved.labels_csv}")
    print(f"resolved splits_dir : {resolved.splits_dir}")
    print(f"feature_root base   : {resolved.feature_root_base}")
    print(f"folds               : {len(split_files)}")
    print(f"combos              : {len(combos)}")
    print(f"jobs (combo x fold) : {len(combos) * len(split_files)}")
    print(f"varying axes        : {axis_keys}")
    print("")
    for combo in combos:
        shown = {k: combo.axis_values.get(k) for k in axis_keys}
        print(
            f"  [{combo.index:03d}] {combo.name}  "
            f"in_feat_dim={combo.config['in_feat_dim']}  {shown}"
        )


def _start_body(n_combos: int, n_jobs: int, gpu_ids, jobs_per_gpu: int) -> str:
    """開始通知メールの本文（日本語・フレンドリー）を組み立てる"""
    return (
        "sweep をはじめました 🙌\n"
        "\n"
        f"combo 数: {n_combos}\n"
        f"job 数（combo×fold）: {n_jobs}\n"
        f"使用 GPU 数: {len(gpu_ids)}\n"
        f"GPU あたり並列数: {jobs_per_gpu}\n"
        "\n"
        "完了したらまたお知らせしますね，少々お待ちください ☕"
    )


def _metric_mean(aggregate: Optional[Dict[str, Any]], metric: Optional[str]) -> str:
    """集計から ``metric`` の mean 表記を作る無ければ ``-``"""
    if metric and aggregate and metric in aggregate:
        return f"{aggregate[metric]['mean']:.4f}"
    return "-"


def _best_lines(summary: Dict[str, Any]) -> str:
    """val 選定 best の探索軸と val/test 指標を通知用の行に整える（内部パス非表示）"""
    best = summary.get("best_by_val")
    if not best:
        return "(有効な結果がありませんでした)"
    metric = summary.get("selection_metric")
    axis_keys = summary.get("axis_keys") or []
    axis_values = best.get("axis_values", {})
    shown = {k: axis_values[k] for k in axis_keys if k in axis_values}
    param_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(shown.items()))
    if not param_lines:
        param_lines = "  (探索軸なし)"
    val = _metric_mean(best.get("val"), metric)
    test = _metric_mean(best.get("test"), metric)
    score = (
        f"val {metric}: {val} / test {metric}: {test}" if metric else "(指標なし)"
    )
    return f"最良 combo（val 選定）:\n{param_lines}\n\n{score}"


def _done_body(summary: Dict[str, Any], elapsed: float) -> str:
    """完了通知メールの本文（日本語・フレンドリー）を組み立てる"""
    minutes = elapsed / 60.0
    n_failed = len(summary.get("failed", []))
    return (
        "おつかれさまでした，sweep が完了しました 🎉\n"
        "\n"
        f"combo 数: {summary.get('n_combos')}（fold 失敗を含む combo {n_failed} 件）\n"
        "\n"
        f"{_best_lines(summary)}\n"
        "\n"
        f"処理時間: {elapsed:.0f} 秒（約 {minutes:.1f} 分）\n"
        "\n"
        "ゆっくり休んでくださいね 😊"
    )


def run(args: argparse.Namespace) -> int:
    """sweep を実行する失敗 fold があっても完走し 0 を返す"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_if_available()

    config = _load_config(args.config)
    resolve_block = config.get("resolve", {})
    sweep_block = config.get("sweep", {})
    fixed_block = config.get("fixed", {})
    parallel_block = config.get("parallel", {})

    resolved = resolve_paths(
        n_cls=resolve_block["n_cls"],
        folds=resolve_block["folds"],
        cohort_root=resolve_block.get("cohort_root", "cohort"),
        feature_root_base=resolve_block["feature_root"],
    )
    verify_n_cls(resolved.labels_csv, resolved.n_cls)

    combos = expand_combos(sweep_block, fixed_block, resolved)
    split_files = resolve_split_files(resolved.splits_dir, resolved.folds)

    if args.dry_run:
        _print_dry_run(combos, split_files, resolved)
        return 0

    gpu_ids = _parse_int_list(args.gpu_ids) or parallel_block.get("gpu_ids")
    jobs_per_gpu = args.jobs_per_gpu or parallel_block.get("jobs_per_gpu")
    mem_per_job_mb = args.mem_per_job_mb or parallel_block.get("mem_per_job_mb")
    mem_headroom_mb = args.mem_headroom_mb or parallel_block.get(
        "mem_headroom_mb", DEFAULT_GPU_HEADROOM_MB
    )

    verify_feature_dims(combos)

    os.makedirs(args.out, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(args.out, SWEEP_CONFIG_COPY))
    weights_out = args.weights_out if args.weights_out else args.out

    start_time = time.monotonic()
    n_jobs = len(combos) * len(split_files)
    if args.notify:
        send_email(
            NOTIFY_SUBJECT_START,
            _start_body(len(combos), n_jobs, gpu_ids or [0], jobs_per_gpu or 1),
        )

    try:
        runner = SweepRunner(
            combos=combos,
            split_files=split_files,
            out_root=args.out,
            weights_root=weights_out,
            gpu_ids=gpu_ids,
            jobs_per_gpu=jobs_per_gpu,
            mem_per_job_mb=mem_per_job_mb,
            mem_headroom_mb=mem_headroom_mb,
        )
        summary = runner.run()
    except Exception as exc:  # noqa: BLE001 - 通知後に再送出する
        if args.notify:
            elapsed = time.monotonic() - start_time
            send_email(
                NOTIFY_SUBJECT_ERROR,
                "sweep の途中でエラーが発生しました 🙇\n\n"
                f"内容: {exc}\n"
                f"経過時間: {elapsed:.0f} 秒",
            )
        raise

    logger.info("sweep summary: %s", os.path.join(args.out, SWEEP_SUMMARY_MD))
    if args.notify:
        send_email(
            NOTIFY_SUBJECT_DONE, _done_body(summary, time.monotonic() - start_time)
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-sweep`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-sweep",
        description="Expand a sweep YAML (resolve/sweep/fixed/parallel) into "
        "combos and run each (combo, fold) as a foveamil-train subprocess "
        "across GPUs.",
    )
    parser.add_argument("--config", required=True, help="Sweep config YAML path.")
    parser.add_argument(
        "--out", required=True, help="Output root for logs/config/results."
    )
    parser.add_argument(
        "--weights-out",
        default=None,
        help="Output root for model weights (.pt); falls back to --out.",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU ids overriding parallel.gpu_ids (e.g. 0,1).",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=None,
        help="Parallel jobs per GPU overriding parallel.jobs_per_gpu; with "
        "--mem-per-job-mb it caps concurrent jobs per GPU.",
    )
    parser.add_argument(
        "--mem-per-job-mb",
        type=int,
        default=None,
        help="Use the GPU-memory-aware scheduler, reserving this VRAM (MB) per "
        "job; injects jobs onto GPUs with enough free memory.",
    )
    parser.add_argument(
        "--mem-headroom-mb",
        type=int,
        default=None,
        help="GPU free-memory safety margin (MB) for the memory-aware scheduler.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded combos, job count and resolved values; do not run.",
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
    """``foveamil-sweep`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
