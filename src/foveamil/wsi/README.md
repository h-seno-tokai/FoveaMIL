# `foveamil.wsi` — WSI アクセス層

全スライド画像(WSI)を開き，倍率/レベルを解決し，組織マスクを作る低レベル部品群．
上位の前処理(`preprocessing`)はここを土台にする．WSI 置き場のパスや患者 ID は
コードにハードコードせず，`slide_id` と環境変数 `WSI_BASE_PATH` から解決する．

## モジュール

| ファイル | 役割 |
|---|---|
| `resolver.py` | `slide_id` → WSI ファイルの絶対パス解決（`WSIResolver`）． |
| `slide.py`    | 倍率/レベル解決，画像読み出し，グリッド寸法算出． |
| `tissue.py`   | 組織マスク生成（`SimpleTissueMask`, `make_tissue_mask`）． |
| `staging.py`  | WSI をローカル SSD へ複製して読み出すステージング（`WSIStager`）． |

## `resolver.py` — `WSIResolver`

`slide_id`（例 `SAMPLE_0001`）を WSI ファイルパスへ解決する．パスリストファイルは持たず，
コホート定義(`labels.csv` の `slide_id`)＋ `WSI_BASE_PATH` から解決するのが基本思想．

解決戦略（上から順）:
1. **オーバーライド表**（`slide_id,path` の CSV）に該当があればそれを使う
   （命名規則が崩れた現実への逃げ道）．
2. なければ `WSI_BASE_PATH` 配下を `glob` して `{slide_id}.{ext}` を探す
   （対応拡張子は `SUPPORTED_WSI_EXTENSIONS`．サブディレクトリを再帰探索）．
   `WSI_BASE_PATH` は `os.pathsep`（`:`）区切りで複数ルートを与えられ，全ルートを横断探索する．
3. 0 件 / 複数ヒットは `WSIResolutionError`（どの slide_id がどう失敗したか明記）．

`base_path` 未指定時は環境変数 `WSI_BASE_PATH` を既定にする．ライブラリ層では
`.env` を強制ロードしない（`.env` 読み込みは CLI 層の責務）．

```python
from foveamil.wsi import WSIResolver

resolver = WSIResolver(base_path="/path/to/wsi")          # or $WSI_BASE_PATH
path = resolver.resolve("SAMPLE_0001")                     # -> 絶対パス
paths = resolver.resolve_many(["SAMPLE_0001", "SAMPLE_0002"])

# 命名規則が崩れたスライドはオーバーライド表で救済
resolver = WSIResolver.from_overrides_csv("overrides.csv", base_path="/path/to/wsi")
```

## `slide.py` — 倍率/レベル/グリッド

- `get_actual_max_magnification(wsi)` … level-0 の対物倍率を
  `aperio.AppMag` → `openslide.mpp-x`（`round(0.25/mpp)`）→ 既定 40x の順で推定．
- `get_level_and_size(wsi, mag, max_mag)` … **画像を読まずに** 最寄りレベルと
  その倍率での画像サイズを返す（グリッド算出に使う）．
- `read_image_at(wsi, mag, max_mag)` … 指定倍率の画像を RGB 配列で読む
  （マスク用に最低倍率を 1 回だけ読む用途）．
- `grid_shape(w, h, patch_size, stride)` … `(行数, 列数)` を返す．

## `tissue.py` — 組織マスク

H&E 染色の性質（白地=低彩度，組織=高彩度）を使い，HSV の彩度を **Otsu** で二値化，
**Gaussian 平滑**でならして組織マスク（組織=1/背景=0）を作る．`SimpleTissueMask` は
設定値だけ持つので `multiprocessing` でも安全に pickle できる．

## `staging.py` — `WSIStager`

ネットワーク越しの WSI を実行前にローカル SSD へ複製し，ローカルパスを返す．
処理後の個別削除（`release`）・全削除（`cleanup_all`）に対応する．後続ファイルを
バックグラウンドで先読みし（`prefetch_ahead`），キャッシュ総量が上限
（`max_cache_size_gb`）を超えたら古いものから LRU 退避する．

キャッシュ先は `cache_dir` で指定する．未指定なら環境変数 `FOVEAMIL_STAGE_DIR`，
それも未設定なら `/tmp` 配下のプロセス固有ディレクトリを使う．

```python
from foveamil.wsi import WSIStager

paths = ["/path/to/a.svs", "/path/to/b.svs", "/path/to/c.svs"]

# 反復ヘルパ: 各 src をステージし，次へ進む前に前を release，先読みも面倒を見る
with WSIStager() as stager:
    for src_path, local_path in stager.staged(paths):
        ...  # local_path を開いて処理する

# 個別に使う場合
stager = WSIStager(cache_dir="/tmp/wsi", max_cache_size_gb=50, prefetch_ahead=2)
local = stager.stage(paths[0], upcoming=paths[1:])
...
stager.release(paths[0])
stager.cleanup_all()
```
