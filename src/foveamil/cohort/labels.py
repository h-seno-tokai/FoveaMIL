"""master CSV から対象クラス・対象症例に絞ったラベル表を作る

``slide_id`` は拡張子なしの WSI ベース名（``/path/SAMPLE_0001.svs`` → ``SAMPLE_0001``）を指す
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import pandas as pd


def load_slide_ids(path: str) -> set[str]:
    """テキストまたは CSV ファイルから ``slide_id`` の集合を読み込む

    入力は 1 行 1 個のテキスト，または ``slide_id`` 列を持つ CSV のいずれか
    各要素はベース名から拡張子を除いて ``slide_id`` に正規化される
    （``/path/SAMPLE_0001.svs`` → ``SAMPLE_0001``）

    Args:
        path: 1 行 1 個のテキスト，または ``slide_id`` 列を持つ CSV のパス

    Returns:
        正規化した ``slide_id`` の集合
    """
    with open(path, "r", encoding="utf-8") as fh:
        first_line = fh.readline().strip()

    # ``slide_id`` ヘッダを持つときだけ CSV として扱う
    header_fields = [c.strip() for c in first_line.split(",")]
    if "slide_id" in header_fields:
        series = pd.read_csv(path)["slide_id"].astype(str)
        raw_values: Iterable[str] = series.tolist()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            raw_values = [line.strip() for line in fh if line.strip()]

    return {_to_slide_id(value) for value in raw_values}


def _to_slide_id(value: str) -> str:
    """パスやファイル名を ``slide_id``（拡張子なしのベース名）に変換する"""
    return os.path.splitext(os.path.basename(value.strip()))[0]


def filter_labels(
    master_csv: str,
    classes: list[str],
    restrict_to: Optional[set[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """master ラベル表を対象クラス・対象症例に絞り込む

    Args:
        master_csv: ``slide_id,label`` 列を持つ master CSV のパス
        classes: 残すラベルこれに含まれない ``label`` の行は除く
        restrict_to: ``slide_id`` の集合指定時はこの集合に含まれる行だけ残す（積集合）
        exclude: 除外する ``slide_id`` の列

    Returns:
        ``slide_id,label`` 列の ``DataFrame``master の行順を保ち，残った行のみ含む
    """
    df = pd.read_csv(master_csv)[["slide_id", "label"]].copy()
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(str)

    mask = df["label"].isin(set(classes))

    if restrict_to is not None:
        mask &= df["slide_id"].isin(set(restrict_to))

    if exclude is not None:
        mask &= ~df["slide_id"].isin(set(exclude))

    return df.loc[mask, ["slide_id", "label"]].reset_index(drop=True)


def write_labels(df: pd.DataFrame, output_csv: str) -> None:
    """``slide_id,label`` の DataFrame を CSV に書き出す（ヘッダ付き・インデックスなし）

    改行コードは CRLF (``\r\n``) で統一する

    Args:
        df: ``slide_id,label`` 列を持つ DataFrame
        output_csv: 出力先パス
    """
    # pandas が出す改行を一旦正規化してから CRLF で結合する
    # （pandas のバージョンや OS に依存しないようにするため）
    csv_text = df.to_csv(index=False)
    lines = csv_text.replace("\r\n", "\n").split("\n")
    with open(output_csv, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(lines))
