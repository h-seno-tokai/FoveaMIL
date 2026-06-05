#!/bin/bash
# foveamil-sweep の引数を組み立てて sweep を実行する薄いラッパ
# メール通知は foveamil-sweep --notify が担う（このスクリプトは SMTP を持たない）
#
# 必須引数:
#   --config PATH        sweep 設定 YAML（resolve/sweep/fixed/parallel）
#   --out DIR            combo のログ・結果の出力ルート（home）
#
# 任意引数:
#   --weights-out DIR    重み（.pt）の出力ルート（Dataset 未指定なら --out）
#   --gpu-ids "0,1"      parallel.gpu_ids を上書き
#   --jobs-per-gpu N     parallel.jobs_per_gpu を上書き
#   --dry-run            展開結果と job 数・解決値を表示し実行しない
#   --notify             開始・完了・エラー時にメールを送る
#
# 環境変数:
#   FEATURE_ROOT                                    特徴ルートの base（resolve.feature_root）
#   FOVEAMIL_STAGE_DIR                              事前ステージ先（foveamil-stage 使用時）
#   GMAIL_USER / GMAIL_APP_PASSWORD / RECEIVE_USER  --notify 使用時の認証情報

set -euo pipefail

CONFIG=""
OUT=""
WEIGHTS_OUT=""
GPU_IDS=""
JOBS_PER_GPU=""
DRY_RUN=""
NOTIFY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --out)
            OUT="$2"
            shift 2
            ;;
        --weights-out)
            WEIGHTS_OUT="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --jobs-per-gpu)
            JOBS_PER_GPU="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --notify)
            NOTIFY="--notify"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 --config PATH --out DIR [--weights-out DIR] [--gpu-ids \"0,1\"] [--jobs-per-gpu N] [--dry-run] [--notify]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required" >&2
    exit 1
fi
if [[ -z "$OUT" ]]; then
    echo "Error: --out is required" >&2
    exit 1
fi

ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "$WEIGHTS_OUT" ]]; then
    ARGS+=(--weights-out "$WEIGHTS_OUT")
fi
if [[ -n "$GPU_IDS" ]]; then
    ARGS+=(--gpu-ids "$GPU_IDS")
fi
if [[ -n "$JOBS_PER_GPU" ]]; then
    ARGS+=(--jobs-per-gpu "$JOBS_PER_GPU")
fi
if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
fi
if [[ -n "$NOTIFY" ]]; then
    ARGS+=("$NOTIFY")
fi

foveamil-sweep "${ARGS[@]}"
