#!/bin/bash
# foveamil-train の引数を組み立てて学習を実行する薄いラッパ
# メール通知は foveamil-train --notify が担う（このスクリプトは SMTP を持たない）
#
# 必須引数:
#   --config PATH        学習設定 YAML
#   --out DIR            結果の出力ルート
#   --split PATH         単一 fold の分割 CSV
#   または --splits-dir DIR  split_fold*.csv のディレクトリ（交差検証）
#   （--split と --splits-dir は排他どちらか一方を指定する）
#
# 任意引数:
#   --folds "1,2,3"      --splits-dir 使用時に対象 fold 番号を絞る
#   --notify             開始・完了・エラー時にメールを送る
#
# 環境変数:
#   GMAIL_USER / GMAIL_APP_PASSWORD / RECEIVE_USER  --notify 使用時の認証情報

set -euo pipefail

CONFIG=""
OUT=""
SPLIT=""
SPLITS_DIR=""
FOLDS=""
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
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --splits-dir)
            SPLITS_DIR="$2"
            shift 2
            ;;
        --folds)
            FOLDS="$2"
            shift 2
            ;;
        --notify)
            NOTIFY="--notify"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 --config PATH --out DIR (--split PATH | --splits-dir DIR) [--folds \"1,2,3\"] [--notify]" >&2
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
if [[ -z "$SPLIT" && -z "$SPLITS_DIR" ]]; then
    echo "Error: one of --split or --splits-dir is required" >&2
    exit 1
fi
if [[ -n "$SPLIT" && -n "$SPLITS_DIR" ]]; then
    echo "Error: --split and --splits-dir are mutually exclusive" >&2
    exit 1
fi

ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "$SPLIT" ]]; then
    ARGS+=(--split "$SPLIT")
else
    ARGS+=(--splits-dir "$SPLITS_DIR")
fi
if [[ -n "$FOLDS" ]]; then
    ARGS+=(--folds "$FOLDS")
fi
if [[ -n "$NOTIFY" ]]; then
    ARGS+=("$NOTIFY")
fi

foveamil-train "${ARGS[@]}"
