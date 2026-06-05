#!/bin/bash
# foveamil-visualize の薄いラッパ（サブコマンド overview / zoom / compare をそのまま渡す）
# 保存済み予測・重みから WSI に attention を重ねた図を出す（学習はしない）
#
# 使用:
#   ./scripts/visualize.sh <overview|zoom|compare> [引数...]
#
# 主な引数（foveamil-visualize --help 参照）:
#   --sweep-root DIR     sweep の出力ルート（best_by_val を解決）
#   --feature-root DIR   {encoder}/{mag}x/{slide}.h5 のルート（必須）
#   --out-dir DIR        図の保存先（必須）
#   --weights-root DIR   重み（.pt）のルート（Dataset 側）
#   --coords-root DIR    actual_max_mag を取る座標ルート
#   --split / --outcome / --slide-id / --per-class / --n   症例選択
#   --fold N             可視化に使う fold
#   --dry-run            解決結果だけ表示
#
# 環境変数:
#   WSI_BASE_PATH  --wsi-base-path 未指定時の slide_id 解決ルート
#
# 例:
#   ./scripts/visualize.sh zoom --sweep-root $OUT --feature-root $F --weights-root $W \
#       --coords-root $C --slide-id SAMPLE_0001 --parent-mag 20 --chain --out-dir $VIZ

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <overview|zoom|compare> [args...]" >&2
    exit 1
fi

foveamil-visualize "$@"
