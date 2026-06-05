#!/bin/bash
# foveamil-eval の引数を組み立てて評価レポートを生成する薄いラッパ
# 保存済み予測から再学習なしで図・有意差検定・レポートを作る（学習はしない）
#
# 必須引数:
#   --in DIR             sweep の出力ルート（--out に渡したディレクトリ）
#
# 任意引数:
#   --out DIR            レポート出力先（未指定なら {in}/report）
#   --split NAME         report する split（val/test/train 既定 test）
#   --metric NAME        selection/報告指標（既定 macro_auc）
#   --compare "A:B"      combo 名の対を有意差検定（繰り返し可）
#   --bins N             キャリブレーションの bin 数（既定 10）
#   --all-combos         全 combo の図を作る（既定は best のみ）
#   --no-plots           図を作らない（指標・レポートのみ）

set -euo pipefail

IN=""
OUT=""
SPLIT=""
METRIC=""
BINS=""
ALL_COMBOS=""
NO_PLOTS=""
COMPARE=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --in)
            IN="$2"
            shift 2
            ;;
        --out)
            OUT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --compare)
            COMPARE+=(--compare "$2")
            shift 2
            ;;
        --bins)
            BINS="$2"
            shift 2
            ;;
        --all-combos)
            ALL_COMBOS="--all-combos"
            shift
            ;;
        --no-plots)
            NO_PLOTS="--no-plots"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 --in DIR [--out DIR] [--split test] [--metric macro_auc] [--compare \"A:B\"]... [--bins N] [--all-combos] [--no-plots]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$IN" ]]; then
    echo "Error: --in is required" >&2
    exit 1
fi

ARGS=(--in "$IN")
if [[ -n "$OUT" ]]; then
    ARGS+=(--out "$OUT")
fi
if [[ -n "$SPLIT" ]]; then
    ARGS+=(--split "$SPLIT")
fi
if [[ -n "$METRIC" ]]; then
    ARGS+=(--metric "$METRIC")
fi
if [[ -n "$BINS" ]]; then
    ARGS+=(--bins "$BINS")
fi
if [[ ${#COMPARE[@]} -gt 0 ]]; then
    ARGS+=("${COMPARE[@]}")
fi
if [[ -n "$ALL_COMBOS" ]]; then
    ARGS+=("$ALL_COMBOS")
fi
if [[ -n "$NO_PLOTS" ]]; then
    ARGS+=("$NO_PLOTS")
fi

foveamil-eval "${ARGS[@]}"
