# `foveamil.cohort` — コホート構築（ラベル絞り込み＋層化CV split）

master ラベル表からタスク用ラベル集合を作り，再現可能な層化 K-fold 交差検証 split を
生成する小さな純粋関数群と，それらを束ねる CLI (`foveamil-cohort`) を提供する．
コードに実データはハードコードしない．以下の例はすべてダミー．

## モジュール

| ファイル | 役割 |
|---|---|
| `labels.py` | ラベル絞り込みの純粋ロジック． |
| `splits.py` | 層化 K-fold split 生成の純粋ロジック． |
| `cli.py`    | `argparse` による CLI．サブコマンド `labels` / `splits`． |

## 関数

### `labels.py`
- `load_slide_ids(path) -> set[str]`
  1行1要素のテキスト，または `slide_id` 列を持つ CSV を読み，各要素を
  **basename から拡張子を除いた** slide_id に正規化した集合を返す
  （`/path/to/wsi/SAMPLE_0001.svs` → `SAMPLE_0001`）．
- `filter_labels(master_csv, classes, restrict_to=None, exclude=None) -> DataFrame`
  master(`slide_id,label`) を読み，`label ∈ classes` の行を残す．`restrict_to`
  指定時はその slide_id 集合と積を取り，`exclude` の slide_id を除外．master の
  出現順を維持し，列は `slide_id,label`．
- `write_labels(df, output_csv)`
  ヘッダ付き・index 無し CSV を書き出す．

### `splits.py`
- `make_cv_splits(labels_csv, k=10, val_frac=None, seed=42) -> list[dict]`
  ラベルで層化して k 個の test fold に分割（各症例ちょうど1回 test）．各 fold で
  残り (k-1) fold のプールから層化して val を抽出し，残りを train とする．
  `val_frac` は test を除いた残りプールに対する val の割合（既定 `1/(k-1)`）．
  完全に決定的（同じ `seed`/`k`/`val_frac` → 同じ出力）．
  返り値は各 fold の `{"fold": k, "train": [...], "val": [...], "test": [...]}`．
- `write_split_csv(split, output_csv)`
  列ヘッダ `train,val,test` のワイド CSV．各列に slide_id を縦に並べ，短い列は
  末尾を空欄でパディング．

## CLI 使用例（ダミーのみ）

```bash
# ラベル絞り込み: master から DLBCL/FL/Reactive を抽出し，手元にある WSI 集合と積を取る
foveamil-cohort labels \
    --input  master.csv \
    --output labels_3class.csv \
    --classes DLBCL FL Reactive \
    --restrict-to available_slides.txt \
    --exclude SAMPLE_9998 SAMPLE_9999

# 層化 K-fold CV split を生成（split_fold1.csv .. split_foldK.csv）
foveamil-cohort splits \
    --labels     labels_3class.csv \
    --output-dir splits/3class/cv10 \
    --k 10 --seed 42
```

ダミー master の例:

```csv
slide_id,label
SAMPLE_0001,DLBCL
SAMPLE_0002,FL
SAMPLE_0003,Reactive
```
