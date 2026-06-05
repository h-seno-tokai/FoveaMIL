# `foveamil.visualization` — アテンション可視化

学習済み FoveaMIL の階層アテンションを WSI 上に描く多重解像度の探索（低倍率で広く見て
top-k パッチだけ高倍率へズーム選択）を，研究者が読み取れる図にする保存済みの予測・重み・
設定から**再学習なし**で図を作る（`foveamil-visualize`）

## 3 つのビュー

- **overview（View A）**: 倍率 × {主アテンション(primary,pooling 寄与), 補助アテンション(aux,選択スコア)} の格子．
  H&E サムネに主を viridis・補助を cividis で半透明合成し，次倍率へ選ばれたパッチをマゼンタ枠で示す．
  多解像度探索の俯瞰．
- **zoom（View B）**: 階層ズーム照明．WSI 全体図では潰れて見えない高倍率の選択を解く．
  選択された親パッチを高解像で拡大し，内部の r×r 子セルを**子の primary アテンションの連続明度**で照らす
  （低い所は減光）．`--chain` で最低→最高倍率の選択経路を 1 行で辿る中心窩経路図．
- **compare（View C）**: 成功(`y_true==y_pred`)と失敗の症例を同一カラースケールで対比し誤りパターンを診断．

## 構造（3 層・単一責任）

- `core/extraction.py` — `extract_attention_trace`/`AttentionTrace`/`LayerTrace`．唯一の入力源（データ層）．
- `render/` — 素材層（純関数）: `geometry`(座標↔画素・子 r²)・`region_reader`(level-0 矩形読込)・
  `normalize`(percentile/minmax)・`palette`(配色定数)・`heatmap`(スカラ場→RGBA)・`blend`(alpha 合成)・
  `illuminate`(階層ズーム照明)・`panels`(matplotlib 格子/共有カラーバー/凡例)．
- `builders/` — 組立層: `overview`/`zoom`/`compare`（素材を束ねるだけ・view 追加 = builder 1 ファイル）．
- 配線層: `cases`(成功/失敗選定)・`loader`(best combo 解決＋FoveaMIL 再構築)・`io`(WSI 解決/保存)・
  `visualize`(オーケストレータ)・`visualize_cli`(CLI)．

## 入出力

入力は sweep の出力ルート（`sweep_summary.json` の best_by_val から combo→`run_meta.json` の設定で
モデルを再構築し `model_best_{metric}.pt` をロード）・`predictions_{split}.csv`（成功/失敗判定）・
`{encoder}/{mag}x/{slide}.h5` の特徴・WSI（`WSI_BASE_PATH` か `--wsi-overrides-csv`）．
`actual_max_mag` は座標 H5 の attr を優先し，無ければ WSI から取る（特徴 H5 には無い）．
出力は `--out-dir` に PNG ＋ サイドカー JSON（設定・正規化基準・忠実度の但し書き）．

## 使い方

```bash
# best combo の成功/失敗症例の overview（クラスごと 1 件）
foveamil-visualize overview --sweep-root OUT --feature-root FEAT --weights-root W \
    --coords-root COORDS --wsi-base-path WSI --per-class 1 --split test --out-dir VIZ

# 特定症例の階層ズーム経路（中心窩）
foveamil-visualize zoom --sweep-root OUT --feature-root FEAT --weights-root W \
    --slide-id SAMPLE_0001 --chain --out-dir VIZ

# 成功 vs 失敗 対比
foveamil-visualize compare --sweep-root OUT --feature-root FEAT --weights-root W \
    --per-class 1 --out-dir VIZ
```

`--dry-run` で解決した combo/fold/モデル設定/症例だけ表示し描画しない．`--fold N` で fold を選ぶ．

## 注意

生アテンションは分類寄与の代理であり厳密な特徴帰属ではない（図キャプション/サイドカーに明記）．
重い実行は WSI をローカル SSD にステージしてから（NAS 直読は負荷大）．`--n-parents`/`--n` で対象を絞る．
