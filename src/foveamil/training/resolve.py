"""sweep 設定から学習に必要なパス・特徴次元・倍率を解決する

``n_cls`` から labels CSV と splits ディレクトリ（``cv{folds}`` まで），エンコーダ名と
feature_type から入力特徴次元，倍率表記（``"1.25x"`` / ``1.25``）から float 列を解決する
解決できない場合（未対応 fold 数，存在しない labels/splits，本数不一致）は生成手順を含む
明確なエラーで止めるパスは存在検証まで行い，後段の展開・学習へ確定値だけを渡す
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Sequence, Union

from foveamil.encoders import ENCODERS
from foveamil.training.accessor import FEATURE_TYPE_CONCAT

# 対応する fold 数（事前生成した cv5/cv10 のみ）
SUPPORTED_FOLDS = (5, 10)
# 推奨 fold 数
RECOMMENDED_FOLDS = 10
# labels CSV のファイル名テンプレート
LABELS_TEMPLATE = "labels_{n}class.csv"
# splits 配下のクラス別ディレクトリ名テンプレート
SPLITS_CLASS_TEMPLATE = "{n}class"
# クラス別ディレクトリ配下の fold 数別サブディレクトリ名テンプレート
CV_SUBDIR_TEMPLATE = "cv{folds}"
# labels ディレクトリ名
LABELS_DIRNAME = "labels"
# splits ディレクトリ名
SPLITS_DIRNAME = "splits"
# 分割 CSV ファイル名の glob パターン
SPLIT_GLOB = "split_fold*.csv"
# 倍率表記の接尾辞
MAG_SUFFIX = "x"
# concat 時の特徴次元倍率
CONCAT_DIM_FACTOR = 2


@dataclass
class ResolvedPaths:
    """解決済みのパス群

    Attributes:
        n_cls: クラス数
        folds: fold 数
        labels_csv: ``slide_id,label`` の CSV パス
        splits_dir: ``split_fold*.csv`` を含む ``cv{folds}`` ディレクトリ
        feature_root_base: 特徴ルートの base（``{encoder}/{mag}x`` は後段が付与）
    """

    n_cls: int
    folds: int
    labels_csv: str
    splits_dir: str
    feature_root_base: str


def _normalize_one_mag(value: Union[str, float, int]) -> float:
    """単一の倍率表記を float にする（``"1.25x"`` / ``"1.25"`` / ``1.25`` を許容）"""
    if isinstance(value, str):
        text = value[:-len(MAG_SUFFIX)] if value.endswith(MAG_SUFFIX) else value
        return float(text)
    return float(value)


def normalize_mags(mags: Sequence[Union[str, float, int]]) -> List[float]:
    """倍率セット 1 つを float 列へ正規化する

    Args:
        mags: 倍率表記の列（``["1.25x", "2.5x"]`` でも ``[1.25, 2.5]`` でも可）

    Returns:
        float の倍率列（低→高の順は呼び出し側の入力を保つ）
    """
    return [_normalize_one_mag(m) for m in mags]


def resolve_in_feat_dim(encoder: str, feature_type: str) -> int:
    """エンコーダと feature_type から入力特徴次元を解決する

    ``concat`` は pooled と cls を次元連結するため素の次元の 2 倍になる素の次元は
    エンコーダクラス属性 ``feature_dim`` を真実源とする（インスタンス化しない）

    Args:
        encoder: 登録エンコーダ名
        feature_type: ``"mean"`` / ``"cls"`` / ``"concat"``

    Returns:
        入力特徴次元
    """
    if encoder not in ENCODERS:
        raise KeyError(
            f"unknown encoder '{encoder}'; available: {sorted(ENCODERS)}"
        )
    base = ENCODERS[encoder].feature_dim
    if feature_type == FEATURE_TYPE_CONCAT:
        return base * CONCAT_DIM_FACTOR
    return base


def _fold_number(path: str) -> int:
    """``split_fold{n}.csv`` から fold 番号 ``n`` を取り出す"""
    stem = os.path.basename(path)
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else -1


def _splits_dir(cohort_root: str, n_cls: int, folds: int) -> str:
    """``{cohort}/splits/{n}class/cv{folds}`` を組む"""
    return os.path.join(
        cohort_root,
        SPLITS_DIRNAME,
        SPLITS_CLASS_TEMPLATE.format(n=n_cls),
        CV_SUBDIR_TEMPLATE.format(folds=folds),
    )


def _labels_csv(cohort_root: str, n_cls: int) -> str:
    """``{cohort}/labels/labels_{n}class.csv`` を組む"""
    return os.path.join(
        cohort_root, LABELS_DIRNAME, LABELS_TEMPLATE.format(n=n_cls)
    )


def _gen_splits_hint(labels_csv: str, splits_dir: str, folds: int) -> str:
    """splits 不在時に提示する生成コマンドを返す"""
    return (
        "foveamil-cohort splits "
        f"--labels {labels_csv} --output-dir {splits_dir} --k {folds}"
    )


def resolve_paths(
    n_cls: int, folds: int, cohort_root: str, feature_root_base: str
) -> ResolvedPaths:
    """labels と splits のパスを解決し存在を検証する

    Args:
        n_cls: クラス数（labels/splits の解決キー）
        folds: fold 数（``SUPPORTED_FOLDS`` のみ）
        cohort_root: ``labels/`` と ``splits/`` を含むコホートルート
        feature_root_base: 特徴ルートの base（``${VAR}`` を展開する）

    Returns:
        解決済みパス群

    Raises:
        ValueError: 未対応 fold 数，feature_root 未展開，labels/splits の不在
    """
    if folds not in SUPPORTED_FOLDS:
        raise ValueError(
            f"folds={folds} is not supported; choose one of {SUPPORTED_FOLDS} "
            f"(recommended: {RECOMMENDED_FOLDS})"
        )

    base = os.path.expandvars(feature_root_base)
    if "$" in base:
        raise ValueError(
            f"feature_root has an unresolved variable: '{feature_root_base}' "
            "(set it in .env or the environment)"
        )

    labels_csv = _labels_csv(cohort_root, n_cls)
    splits_dir = _splits_dir(cohort_root, n_cls, folds)

    if not os.path.isfile(labels_csv):
        raise ValueError(f"labels CSV not found: {labels_csv}")
    if not os.path.isdir(splits_dir):
        raise ValueError(
            f"splits directory not found: {splits_dir}\n"
            f"generate it with: {_gen_splits_hint(labels_csv, splits_dir, folds)}"
        )

    return ResolvedPaths(
        n_cls=n_cls,
        folds=folds,
        labels_csv=labels_csv,
        splits_dir=splits_dir,
        feature_root_base=base,
    )


def verify_n_cls(labels_csv: str, n_cls: int) -> None:
    """labels CSV のユニークラベル数が ``n_cls`` と一致するか検証する

    取り違えた labels ファイルを使う事故を展開前に止める

    Raises:
        ValueError: ユニークラベル数が ``n_cls`` と一致しない場合
    """
    import pandas as pd

    actual = pd.read_csv(labels_csv)["label"].astype(str).nunique()
    if actual != n_cls:
        raise ValueError(
            f"labels {labels_csv} has {actual} classes but n_cls={n_cls}"
        )


def resolve_split_files(splits_dir: str, folds: int) -> List[str]:
    """``splits_dir`` の分割 CSV を fold 番号順に集め本数を照合する

    先頭 k 切り出しは行わず，実ファイル数が ``folds`` と一致しなければエラーにする

    Args:
        splits_dir: ``split_fold*.csv`` を含むディレクトリ
        folds: 期待する fold 数

    Returns:
        fold 番号昇順の分割 CSV パス列

    Raises:
        ValueError: 実ファイル数が ``folds`` と一致しない場合
    """
    paths = sorted(
        glob.glob(os.path.join(splits_dir, SPLIT_GLOB)), key=_fold_number
    )
    if len(paths) != folds:
        raise ValueError(
            f"expected {folds} split files in {splits_dir} but found "
            f"{len(paths)}; regenerate a true {folds}-fold set with: "
            "foveamil-cohort splits "
            f"--labels <labels.csv> --output-dir {splits_dir} --k {folds}"
        )
    return paths
