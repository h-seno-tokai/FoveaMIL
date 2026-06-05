"""可視化の配色・ブレンド・凡例の規約を一箇所に固定する定数

配色は色覚多様性に配慮し連続・非負単調の知覚均等カラーマップを用いる主アテンション
（pooling 寄与）は viridis，補助アテンション（選択スコア）は cividis で意味の異なる二量を
描き分けるズーム選択の標示は単色マゼンタの離散枠で示すロジックは持たず定数のみ
"""

from __future__ import annotations

# 主アテンション（pooling 寄与）の連続カラーマップ
PRIMARY_CMAP = "viridis"
# 補助アテンション（選択スコア）の連続カラーマップ
AUX_CMAP = "cividis"
# ズーム選択パッチを示す離散枠の色（マゼンタ matplotlib 名）
SELECT_EDGE_COLOR = "magenta"
# ズーム選択枠の RGB（配列描画用）
SELECT_EDGE_RGB = (255, 0, 255)
# 将来の符号付き帰属用に予約する発散カラーマップ（0=白未使用）
DIVERGING_CMAP = "RdBu_r"

# オーバーレイの不透明度（原画像との alpha 合成）
OVERLAY_ALPHA = 0.4
# 階層ズームで非選択子セルを暗くする係数
DIM_FACTOR = 0.35
# 選択枠の線幅（点）
SELECT_EDGE_WIDTH = 1.5
# 図の既定 dpi
DEFAULT_DPI = 300
# 印刷用の高 dpi
PRINT_DPI = 600
# パネルタイトル等の最小フォントサイズ（点）
MIN_FONT_SIZE = 8
# 成功症例のタイトル枠色
SUCCESS_EDGE_COLOR = "tab:green"
# 失敗症例のタイトル枠色
FAILURE_EDGE_COLOR = "tab:red"
# 生アテンションは寄与の代理であり厳密な帰属ではない旨の但し書き（図に載せるため英語）
ATTRIBUTION_DISCLAIMER = (
    "raw attention is a proxy for class contribution, not exact attribution"
)
