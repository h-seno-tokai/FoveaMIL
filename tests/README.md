# tests — ユニットテスト

各部品のロジックを検証するユニットテスト（122 件）．純粋ロジック中心で，重い WSI・学習の
実 e2e は含めない．

## 実行

```bash
pip install -e ".[dev]"
pytest tests/
```

## 依存の二段

テストは依存の重さで 2 つに分かれる（CI [.github/workflows/tests.yml](../.github/workflows/tests.yml) もこの分けで回す）．

- **システムライブラリ不要**（pip wheel のみ）: `foveamil.models` / `foveamil.training` /
  `foveamil.evaluation` / `foveamil.utils` を読む 8 ファイル．
- **OpenSlide が必要**: `foveamil.visualization`（`region_reader` が openslide を読む）と
  `preprocessing.features` を読む 6 ファイル（`test_features` / `test_viz_*`）．`libopenslide` を
  入れてから回す．

CI の `core` ジョブは後者 6 ファイルを `--ignore` し，`full` ジョブで OpenSlide を入れて全件回す．

## 一覧

| ファイル | 検証対象 | OpenSlide |
|---|---|---|
| `test_hierarchy.py` | 倍率階層の子数・index 計算・倍率列の検証 | 不要 |
| `test_resolve.py` | sweep のパス解決・特徴次元・倍率正規化 | 不要 |
| `test_stats.py` | 信頼区間・ブートストラップ・Wilcoxon・Nadeau-Bengio 検定 | 不要 |
| `test_metrics.py` | 分類指標の sklearn 一致・混同行列 | 不要 |
| `test_cv_aggregate.py` | CV fold 集計（mean/std/信頼区間） | 不要 |
| `test_provenance.py` | 再現情報（git・環境・ファイルハッシュ）の収集 | 不要 |
| `test_sweep.py` | combo 展開・制約結合・val 選定・CV 集計 | 不要 |
| `test_models.py` | コア部品（微分可能 top-k・ゲート付きアテンション・融合・FoveaMIL forward）の形状・数値・勾配 | 不要 |
| `test_viz_geometry.py` | 可視化の座標・寸法計算（level-0 ↔ 倍率画素・子セル） | 必要 |
| `test_viz_illuminate.py` | 階層ズーム照明・正規化（percentile/minmax） | 必要 |
| `test_viz_cases.py` | 予測 CSV からの成功/失敗症例選定 | 必要 |
| `test_viz_loader.py` | sweep 出力からのモデル再構築・重み復元 | 必要 |
| `test_viz_region_reader.py` | WSI 領域読み込み（level 選択・読み出しサイズ） | 必要 |
| `test_features.py` | 特徴 H5 書き込み・デバイス解決・既存出力の探索 | 必要 |
