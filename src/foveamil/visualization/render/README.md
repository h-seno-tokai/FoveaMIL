# `foveamil.visualization.render` — 描画素材（純関数層）

座標変換・領域読込・正規化・配色・ヒートマップ・合成・照明・図レイアウトの素材を提供する
純関数群．状態を持たず，`builders` はこれらを束ねるだけ（全体像は [visualization](../README.md)）．

| ファイル | 役割 |
|---|---|
| `geometry.py` | 座標・寸法計算（level-0 ↔ 倍率画素・子 r² セル矩形）の純関数． |
| `region_reader.py` | WSI の level-0 矩形を目標画素サイズの RGB として読む唯一の公開 API． |
| `normalize.py` | アテンションスカラを配色前に `[0, 1]` へ写す正規化（percentile / minmax）． |
| `palette.py` | 配色・ブレンド・凡例の規約を一箇所に固定する定数． |
| `heatmap.py` | 正規化スカラとパッチ矩形からヒートマップの材料（RGBA）を作る． |
| `blend.py` | 原画像と RGBA オーバーレイの alpha 合成． |
| `illuminate.py` | 階層ズーム照明（選択された親を高解像で拡大し，内部の子セルを子の主アテンション連続明度で照らす）． |
| `panels.py` | matplotlib の格子・共有カラーバー・凡例・スケールバー・dpi 等の図レイアウト素材． |
