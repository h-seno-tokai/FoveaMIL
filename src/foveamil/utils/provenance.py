"""実験の再現情報（provenance）を収集する

git リビジョン・実行環境・入力ファイルのハッシュを集め，``run_meta`` 辞書に組み立てる
git 未初期化や情報取得失敗時は例外を投げず ``None`` を入れる（再現情報の欠落で学習を
止めない）
"""

from __future__ import annotations

import hashlib
import platform
import socket
import subprocess
from typing import Any, Dict, List, Optional

# run_meta のスキーマ版
META_SCHEMA_VERSION = 1
# ファイルハッシュ読み込みのチャンクサイズ
_HASH_CHUNK = 1 << 20
# git コマンドのタイムアウト秒
_GIT_TIMEOUT = 5


def _run_git(args: List[str], cwd: Optional[str]) -> Optional[str]:
    """``git`` をサブプロセス実行し標準出力を返す失敗時は ``None``"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except Exception:  # noqa: BLE001 - git 不在や実行失敗は None 扱い
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """現在の git リビジョン（短縮 SHA）を返す未初期化なら ``None``"""
    return _run_git(["rev-parse", "--short", "HEAD"], cwd)


def git_dirty(cwd: Optional[str] = None) -> Optional[bool]:
    """作業ツリーに未コミット変更があるか返す取得できなければ ``None``"""
    status = _run_git(["status", "--porcelain"], cwd)
    if status is None:
        return None
    return status != ""


def collect_env() -> Dict[str, Optional[str]]:
    """実行環境（python/torch/cuda/GPU/host/platform）を集めて返す

    各項目は取得失敗時 ``None``torch は遅延 import し，無くても落ちない
    """
    env: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": None,
        "torch": None,
        "cuda": None,
        "gpu_name": None,
    }
    try:
        env["hostname"] = socket.gethostname()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch 不在/CUDA 無しでも続行
        pass
    return env


def file_sha256(path: Optional[str]) -> Optional[str]:
    """ファイルの sha256 を返す``None`` パスや読めない場合は ``None``"""
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:  # noqa: BLE001 - 読めないファイルは None
        return None


def collect_run_meta(
    *,
    config: Dict[str, Any],
    selection: Dict[str, Any],
    timing: Dict[str, Any],
    labels_csv: Optional[str],
    split_csv: Optional[str],
    class_breakdown: Dict[str, Dict[str, int]],
    version: Optional[str] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """1 fold 分の再現情報をまとめた ``run_meta`` 辞書を組み立てる

    Args:
        config: 解決済み設定（``TrainConfig`` を辞書化したもの）
        selection: モデル選択の記録（save_metric / best_epoch / best_val_value / n_epochs 等）
        timing: ``{"start": ..., "end": ..., "duration_sec": ...}``
        labels_csv: ラベル CSV のパス（ハッシュ対象）
        split_csv: 分割 CSV のパス（ハッシュ対象）
        class_breakdown: ``{"train": {label: n}, "val": {...}, "test": {...}}``
        version: パッケージ版（任意）
        cwd: git 情報を取るディレクトリ（任意）

    Returns:
        ``run_meta`` 辞書
    """
    return {
        "schema_version": META_SCHEMA_VERSION,
        "timestamp_start": timing.get("start"),
        "timestamp_end": timing.get("end"),
        "duration_sec": timing.get("duration_sec"),
        "config": config,
        "selection": selection,
        "code": {
            "git_sha": git_sha(cwd),
            "git_dirty": git_dirty(cwd),
            "foveamil_version": version,
        },
        "env": collect_env(),
        "data": {
            "labels_csv_sha256": file_sha256(labels_csv),
            "split_csv_sha256": file_sha256(split_csv),
            "encoder": config.get("encoder"),
            "feature_type": config.get("feature_type"),
            "magnifications": config.get("magnifications"),
            "n_cls": config.get("n_cls"),
            "classes": config.get("classes"),
            "class_breakdown": class_breakdown,
        },
    }
