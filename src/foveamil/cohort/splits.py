"""層化 K-fold 交差検証の分割を生成する

test fold は互いに重複せず，各症例はちょうど 1 つの test fold に入る各 fold では
その fold を test とし，残り ``k-1`` fold を母集団として層化した validation を抽出し，
残りを train とする同じ ``seed``/``k``/``val_frac`` なら出力は決定的に一致する
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# fold 数の既定値
DEFAULT_K = 10
# 乱数シードの既定値
DEFAULT_SEED = 42


def make_cv_splits(
    labels_csv: str,
    k: int = DEFAULT_K,
    val_frac: Optional[float] = None,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """層化 K-fold 交差検証の分割を生成する

    ``labels_csv`` の各症例を ``label`` で層化して ``k`` 個の test fold に分け，各症例が
    ちょうど 1 つの test fold に入るようにするfold ``i`` では test = fold ``i``，
    残り ``k-1`` fold の和集合から層化して validation を抽出し，残りを train とする

    Args:
        labels_csv: ``slide_id,label`` 列を持つ CSV のパス
        k: fold 数（＝返す分割数）
        val_frac: test を除いた母集団のうち validation に充てる割合``None``（既定）は
            ``1/(k-1)`` を意味し，validation は test とほぼ同サイズになる（``k=10`` で
            全体が概ね train 80% / val 10% / test 10%）
        seed: シャッフルを制御する乱数シード出力を決定的にする

    Returns:
        ``k`` 個の dict のリスト各要素は ``{"fold": i, "train": [...], "val": [...],
        "test": [...]}``（``i`` は 1 始まりの fold 番号，値は ``slide_id`` のリスト）
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    df = pd.read_csv(labels_csv)[["slide_id", "label"]].copy()
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(str)

    slide_ids = df["slide_id"].to_numpy()
    labels = df["label"].to_numpy()

    if val_frac is None:
        val_frac = 1.0 / (k - 1)
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    splits: list[dict] = []
    for fold_idx, (pool_idx, test_idx) in enumerate(skf.split(slide_ids, labels), start=1):
        pool_ids = slide_ids[pool_idx]
        pool_labels = labels[pool_idx]

        train_ids, val_ids = train_test_split(
            pool_ids,
            test_size=val_frac,
            stratify=pool_labels,
            random_state=seed,
            shuffle=True,
        )

        splits.append(
            {
                "fold": fold_idx,
                "train": sorted(train_ids.tolist()),
                "val": sorted(val_ids.tolist()),
                "test": sorted(slide_ids[test_idx].tolist()),
            }
        )

    return splits


def write_split_csv(split: dict, output_csv: str) -> None:
    """1 つの分割を ``train,val,test`` 列の CSV に書き出す

    各列はその部分集合の ``slide_id`` を縦に並べたもの短い列は末尾を空文字で
    パディングして全列の長さをそろえる

    Args:
        split: ``train``/``val``/``test`` の ``slide_id`` リストを持つ dict
        output_csv: 出力先パス
    """
    columns = {col: list(split[col]) for col in ("train", "val", "test")}
    max_len = max((len(v) for v in columns.values()), default=0)
    padded = {col: vals + [""] * (max_len - len(vals)) for col, vals in columns.items()}
    pd.DataFrame(padded, columns=["train", "val", "test"]).to_csv(output_csv, index=False)
