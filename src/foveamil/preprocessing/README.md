# `foveamil.preprocessing` — 多重解像度 座標抽出

WSI から背景(白地)を除いた**組織パッチの座標**を，複数倍率について抽出して H5 に保存する．
中心窩(fovea)型 MIL は「低倍率で広く薄く見て，ここぞというパッチだけ高倍率で精査」する
ため，倍率間で**空間的に入れ子**になった階層座標が必要になる．これがその第一段である．

## 速度・省メモリの核（このモジュールの肝）

1. **組織マスクは最低倍率で 1 回だけ計算**（`SimpleTissueMask`）．
2. ベース（最低）倍率はグリッド走査し `tissue_fraction >= 閾値` のパッチだけ採用．
   このとき**画像本体は読まず `level_dimensions` だけ**でグリッド寸法を決める．
3. 以降の高倍率は画像もマスクも再計算せず，各親パッチを **2×2=4 子**に
   座標演算だけで細分化（倍率 2 倍ごとに面積 4 倍 → 子は親の 4 倍）．
4. 座標は全倍率で **level-0(最大倍率)ピクセル空間の (x, y)**．WSI ごとに `gc` を実行．

## モジュール

| ファイル | 役割 |
|---|---|
| `coordinates.py` | 抽出アルゴリズム本体（`process_wsi`, `extract_base_coordinates`, `subdivide_coordinates`, `validate_magnifications`）． |
| `cli.py`         | `argparse` による CLI `foveamil-coords`． |
| `features.py`    | 特徴抽出本体（`extract_features_for_slide`, `extract_dummy_feature`）． |
| `features_cli.py`| `argparse` による CLI `foveamil-features`． |

## 倍率のバリデーション

固定ホワイトリストではなく，**「倍率が昇順で，隣接比がちょうど 2.0」**という階層細分化が
成立する一般条件を直接検証する（最低 2 倍率）．例 `1.25 2.5 5.0 10.0 20.0 40.0` は通る．

## 出力仕様（WSI ごと・倍率ごとに 1 ファイル）

ファイル名 `{mag}x/{slide_id}.h5`（例 `1.25x/SAMPLE_0001.h5`）．

- dataset `coords`: shape **(N, 2)**, dtype **int32**, level-0 座標 **(x, y)**（x=列, y=行）．
- 共通 attrs: `patch_size`(int), `magnification`(float), `stride`(int),
  `downsample_factor`(int=`round(actual_max_mag/mag)`), `actual_max_mag`(int),
  `wsi_path`(str), `tissue_threshold`(float), `is_hierarchical`(bool=True)．
- base 以外のみ追加: `parent_magnification`(float)．

階層: base 座標は patch grid を level-0 に変換したもの．各親→4 子のオフセット =
`patch_size * (actual_max_mag / parent_mag) // 2`（=親パッチの level-0 サイズの半分）．
子の生成順は `for dy in (0,1): for dx in (0,1)` で `(x+dx*offset, y+dy*offset)`．

## CLI 使用例

入力指定は 3 通り（優先順位 `--wsi-dir` > `--slides`，`--overrides` は補助）:

```bash
# (A) コホートの labels.csv から（WSIResolver で slide_id をパス解決）
WSI_BASE_PATH=/path/to/wsi \
foveamil-coords \
    --slides cohort/labels/labels_3class.csv \
    --out    /path/to/coords \
    --mags   1.25 2.5 5.0 10.0 20.0 40.0 \
    --patch-size 224 --tissue-threshold 0.1 --num-workers 4

# (B) ディレクトリ内の WSI を全件処理（コホート定義なしの外部利用者向け）
foveamil-coords --wsi-dir /path/to/wsi --out /path/to/coords \
    --mags 1.25 2.5 5.0 10.0

# (C) 命名規則が崩れたスライドはオーバーライド表で救済
foveamil-coords --slides ids.txt --overrides overrides.csv \
    --out /path/to/coords --mags 1.25 2.5 5.0 10.0
```

主な引数: `--out`(必須), `--mags`(必須), `--patch-size`(既定 224),
`--stride`(既定=patch-size), `--tissue-threshold`(既定 0.1), `--num-workers`(既定 1),
`--verbose`(DEBUG ログ)．1 スライドの失敗は記録して継続し，最後に成功/失敗を集計する．

# 特徴抽出（`foveamil-features`）

座標 H5 と WSI を読み，各倍率のパッチをエンコーダに通して特徴を H5 に保存する．座標は
level-0 (x,y) として保持されているので，倍率ごとに最寄りピラミッドレベルから読んで
`patch_size` に縮小し，エンコーダの平均・標準偏差で正規化して順伝播する．

## 正準出力レイアウト（WSI×エンコーダ×倍率ごとに 1 ファイル）

`{out}/{encoder}/{mag}x/{slide_id}.h5`（例 `.../ResNet50/1.25x/SAMPLE_0001.h5`）．

- dataset `coords`: 座標 H5 の `coords` をそのまま **(N, 2) int32**．
- dataset `patches`: pooled 特徴 **(N, feature_dim) float32**．
- dataset `patches_cls`: cls 特徴 (N, feature_dim) float32．**`has_cls=True` のエンコーダのみ**
  （`ResNet50` は作らない）．
- attrs: `case_id`(=slide_id), `encoder`, `feature_dim`(int), `has_cls`(bool),
  `magnification`(文字列 "1.25x"), `n_patches`(int=N)．

全倍率の出力が揃ったスライドは処理対象から除き，倍率の出力が既存ならその倍率はスキップする
（途中中断後も同じコマンドで無駄なく再開できる）．

## 複数 GPU への動的割当

`--gpu-ids`（カンマ区切りの物理 GPU ID）で複数 GPU を指定すると，各 GPU に常駐ワーカを 1 つ置き，
**空いたワーカが次のスライドを取りに行く**（事前分割はしない）．スライドごとの計算量がばらついても
GPU が遊ばない．未指定なら可視 GPU（`CUDA_VISIBLE_DEVICES`）全て，GPU が無ければ CPU を使う．
出力 H5 は一時ファイルへ書いてから `os.replace` で確定するため，中断しても半端なファイルは残らない．

## 主な引数

`--encoder`(必須, `ENCODERS` のキー), `--coords-dir`(必須), `--out`(必須),
`--slides | --wsi-dir`(排他・どちらか必須), `--mags`(必須)．
`--patch-size`(既定 224), `--batch-size`(既定 `PREPROCESS_BATCH_SIZE`→256),
`--num-workers`(既定 `PREPROCESS_NUM_WORKERS`→4, GPU ワーカあたりのパッチ I/O スレッド数),
`--gpu-ids`(例 `0,1,2`, 未指定なら可視 GPU 全て/無ければ CPU),
`--overrides`, `--wsi-base-path`(既定 `$WSI_BASE_PATH`), `--verbose`,
`--notify`(開始・完了・エラーをメール通知)．1 スライドの失敗は記録して継続する．

## 背景スキップ（`--skip-background`）

最高倍率の座標を低解像度の彩度マップで判定し，背景パッチは順伝播せず，各ワーカが最初に処理する
WSI の背景から 1 度取ったダミー特徴で埋める．ベース倍率は背景スキップ無効（全計算）．
`--saturation-threshold`(既定 0.05), `--base-magnification`(既定 1.25),
`--highest-magnification`(既定 40.0) で挙動を調整する．無指定なら全パッチを計算する．

## ステージング（`--stage`）

`WSIStager` で WSI をローカル SSD に退避してから読み，処理後に解放する．NAS への
ネットワーク負荷を抑えたいときに使う．退避先は `FOVEAMIL_STAGE_DIR`（未設定なら `/tmp` 配下）．

## 実行例

```bash
# ResNet50 で全倍率を抽出
WSI_BASE_PATH=/path/to/wsi \
foveamil-features \
    --encoder ResNet50 \
    --coords-dir /path/to/coords \
    --out /path/to/features \
    --slides cohort/labels/labels_3class.csv \
    --mags 1.25 2.5 5.0 10.0 20.0 40.0

# Virchow2 を 3 GPU へ動的割当 + 背景スキップ + ローカル退避（中断後も同じコマンドで再開）
foveamil-features \
    --encoder Virchow2 --coords-dir /path/to/coords --out /path/to/features \
    --slides cohort/labels/labels_11class.csv --mags 1.25 2.5 5.0 10.0 20.0 40.0 \
    --gpu-ids 0,1,2 --batch-size 2048 --num-workers 8 \
    --skip-background --stage --notify
```
