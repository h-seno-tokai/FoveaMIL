"""``foveamil-train`` コマンド

YAML 設定から ``TrainConfig`` を構築し，単一 fold（``--split``）または複数 fold の
交差検証（``--splits-dir``）を実行する``--override key=value`` は任意の
``TrainConfig`` フィールドを上書きし，値は YAML リテラルとして解釈するログ・config・
結果（JSON / tensorboard / 混同行列）は ``--out`` 配下に保存する重み（``.pt``）は
``--weights-out`` 配下に保存し，未指定なら ``--out`` にフォールバックする
``--notify`` 指定時は開始・完了・エラーに日本語のメールを送る
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import yaml

from foveamil.training.cv import run_cross_validation, run_fold
from foveamil.training.yaml_config import load_train_config
from foveamil.utils.notify import send_email

logger = logging.getLogger(__name__)

# 分割 CSV ファイル名の glob パターン（fold 番号を含む）
SPLIT_GLOB = "split_fold*.csv"
# 完了通知に載せる主要指標（CV 集計の優先順）
NOTIFY_METRICS = ("macro_auc", "weighted_f1", "macro_f1", "accuracy", "kappa")
# 通知メールの件名
NOTIFY_SUBJECT_START = "🔬 FoveaMIL 学習をはじめました"
NOTIFY_SUBJECT_DONE = "✅ 学習が完了しました！"
NOTIFY_SUBJECT_ERROR = "⚠️ 学習でエラーが発生しました"


def _load_dotenv_if_available() -> None:
    """``.env`` があれば読み込む``python-dotenv`` が無ければ何もしない"""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _parse_overrides(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``key=value`` の列を ``{key: value}`` に解釈する（値は YAML リテラル）"""
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value: {item}")
        key, _, raw = item.partition("=")
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def _fold_number(path: str) -> int:
    """``split_fold{n}.csv`` から fold 番号 ``n`` を取り出す"""
    stem = os.path.basename(path)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _collect_split_paths(
    splits_dir: str, folds: Optional[List[int]]
) -> List[str]:
    """``--splits-dir`` から対象 fold の分割 CSV を fold 番号順に集める

    Args:
        splits_dir: ``split_fold*.csv`` を含むディレクトリ
        folds: 対象 fold 番号の一覧（``None`` なら全件）

    Returns:
        fold 番号昇順の分割 CSV パス一覧
    """
    import glob

    paths = sorted(
        glob.glob(os.path.join(splits_dir, SPLIT_GLOB)), key=_fold_number
    )
    if folds is not None:
        wanted = set(folds)
        paths = [p for p in paths if _fold_number(p) in wanted]
    return paths


def _parse_folds(value: Optional[str]) -> Optional[List[int]]:
    """``"1,2,4"`` を ``[1, 2, 4]`` に解釈する``None`` ならそのまま返す"""
    if value is None:
        return None
    return [int(tok) for tok in value.split(",") if tok.strip() != ""]


def _summary_line(config) -> str:
    """通知本文用の設定要約（内部名・ホスト名を含めない）を組み立てる"""
    return (
        f"エンコーダ: {config.encoder}\n"
        f"倍率: {config.magnifications}\n"
        f"特徴タイプ: {config.feature_type}\n"
        f"選択数 k / 融合: {config.k_sample} / {config.fusion}\n"
        f"top-k 手法: {config.topk_method}\n"
        f"クラス数: {config.n_cls}"
    )


def _start_body(config, n_folds: int) -> str:
    """開始通知メールの本文（日本語・フレンドリー）を組み立てる"""
    return (
        "学習をはじめました 🙌\n"
        "\n"
        f"{_summary_line(config)}\n"
        f"fold 数: {n_folds}\n"
        "\n"
        "完了したらまたお知らせしますね，少々お待ちください ☕"
    )


def _metrics_lines(result: Dict[str, Any]) -> str:
    """完了通知用に主要指標の行（CV なら mean±std，単一 fold なら値）を作る"""
    aggregate = result.get("aggregate")
    if isinstance(aggregate, dict):
        lines = []
        for metric in NOTIFY_METRICS:
            if metric in aggregate:
                stats = aggregate[metric]
                lines.append(
                    f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}"
                )
        return "\n".join(lines) if lines else "(指標なし)"

    lines = []
    for metric in NOTIFY_METRICS:
        if metric in result:
            lines.append(f"{metric}: {result[metric]:.4f}")
    return "\n".join(lines) if lines else "(指標なし)"


def _done_body(config, result: Dict[str, Any], n_folds: int, elapsed: float) -> str:
    """完了通知メールの本文（日本語・フレンドリー）を組み立てる"""
    minutes = elapsed / 60.0
    return (
        "おつかれさまでした，学習が完了しました 🎉\n"
        "\n"
        f"{_summary_line(config)}\n"
        f"fold 数: {n_folds}\n"
        "\n"
        f"{_metrics_lines(result)}\n"
        "\n"
        f"処理時間: {elapsed:.0f} 秒（約 {minutes:.1f} 分）\n"
        "\n"
        "ゆっくり休んでくださいね 😊"
    )


def run(args: argparse.Namespace) -> int:
    """学習を実行する成功なら 0，例外時は再送出する前に通知する"""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_if_available()

    start_time = time.monotonic()
    try:
        return _run_training(args, start_time)
    except Exception as exc:  # noqa: BLE001 - 通知後に再送出する
        if getattr(args, "notify", False):
            elapsed = time.monotonic() - start_time
            send_email(
                NOTIFY_SUBJECT_ERROR,
                "学習の途中でエラーが発生しました 🙇\n\n"
                f"内容: {exc}\n"
                f"経過時間: {elapsed:.0f} 秒",
            )
        raise


def _run_training(args: argparse.Namespace, start_time: float) -> int:
    """学習本体``run`` から呼ばれ終了コードを返す"""
    overrides = _parse_overrides(args.override)
    config = load_train_config(args.config, overrides)
    config.save_path = args.out
    os.makedirs(args.out, exist_ok=True)
    weights_out = args.weights_out if args.weights_out else args.out
    if args.weights_out:
        os.makedirs(args.weights_out, exist_ok=True)

    if args.split:
        split_paths = [args.split]
    else:
        folds = _parse_folds(args.folds)
        split_paths = _collect_split_paths(args.splits_dir, folds)
        if not split_paths:
            logger.error("no split_fold*.csv found under %s", args.splits_dir)
            return 1

    n_folds = len(split_paths)
    logger.info(
        "training encoder=%s mags=%s topk=%s fusion=%s folds=%d -> %s",
        config.encoder,
        config.magnifications,
        config.topk_method,
        config.fusion,
        n_folds,
        args.out,
    )

    if args.notify:
        send_email(NOTIFY_SUBJECT_START, _start_body(config, n_folds))

    if args.split:
        # run_fold が test_metrics.json と run_meta.json を保存する
        fold = run_fold(config, args.split, args.out, weights_dir=weights_out)
        report: Dict[str, Any] = fold["test"]
    else:
        cv = run_cross_validation(
            config, split_paths, args.out, weights_root=weights_out
        )
        report = {"aggregate": cv["test"]["aggregate"]}

    if args.notify:
        send_email(
            NOTIFY_SUBJECT_DONE,
            _done_body(
                config, report, n_folds, time.monotonic() - start_time
            ),
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """``foveamil-train`` の引数パーサを構築する"""
    parser = argparse.ArgumentParser(
        prog="foveamil-train",
        description="Train FoveaMIL from a YAML config on a single fold "
        "(--split) or cross-validation (--splits-dir).",
    )
    parser.add_argument(
        "--config", required=True, help="Training config YAML path."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a TrainConfig field (value parsed as a YAML literal); "
        "may be repeated.",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--split", help="Single split CSV with train/val/test columns."
    )
    src.add_argument(
        "--splits-dir",
        help="Directory of split_fold*.csv files for cross-validation.",
    )

    parser.add_argument(
        "--folds",
        default=None,
        help="Comma-separated fold numbers to run from --splits-dir "
        "(default: all).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output root for logs/config/results "
        "(JSON, tensorboard, confusion matrix).",
    )
    parser.add_argument(
        "--weights-out",
        default=None,
        help="Output root for model weights (.pt); falls back to --out.",
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
    """``foveamil-train`` コンソールスクリプトのエントリポイント"""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
