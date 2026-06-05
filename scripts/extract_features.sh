#!/bin/bash
# foveamil-features の薄いラッパ（単一エンコーダ）
#   座標 H5 と WSI からパッチ特徴を抽出する 複数 GPU への動的割当・再開・再試行は
#   core(foveamil-features) が担い，本スクリプトは引数を組み立てて 1 回呼ぶだけ
#   パス・GPU は全て引数/環境変数で受け，内部固有名を持たない
#
# 必須引数:
#   --encoder NAME       エンコーダ名（ResNet50 / UNI2-h / Virchow / Virchow2 / Virchow2-mini-dinov2）
#   --coords-dir DIR     座標 H5 のディレクトリ（{mag}x/{slide_id}.h5）
#   --out DIR            特徴 H5 の出力ルート（{out}/{encoder}/{mag}x/{slide_id}.h5）
#   --slides PATH        slide_id ファイル（CSV/テキスト）WSI_BASE_PATH でパス解決
#   または --wsi-dir DIR ディレクトリ内の対応 WSI を全件処理（--slides と排他）
#
# 任意引数:
#   --mags "1.25 2.5 ..."  倍率列（未指定時は下記 DEFAULT_MAGS）
#   --batch-size N         推論バッチサイズ（未指定時は PREPROCESS_BATCH_SIZE か既定値）
#   --num-workers N        GPU ワーカあたりのパッチ I/O スレッド数
#   --gpu-ids "0,1,2"      スライドを動的割当する物理 GPU（未指定なら可視 GPU 全て，無ければ CPU）
#   --skip-background      背景パッチの順伝播を省きダミー特徴で埋める
#   --stage                WSI をローカル SSD に退避してから読む
#   --notify               開始/完了/エラーの要約メール
#   --overrides CSV        slide_id,path のパス対応表
#   --wsi-base-path DIR    slide_id 解決ルート（: 区切りで複数可）
#
# 再開: 既存出力のあるスライド/倍率は core が自動でスキップする（専用フラグ不要）
#
# 環境変数:
#   WSI_BASE_PATH                                   --slides 使用時の slide_id 解決ルート（: 区切りで複数可）
#   PREPROCESS_BATCH_SIZE / PREPROCESS_NUM_WORKERS  バッチサイズ・ワーカ数の既定上書き
#   FOVEAMIL_STAGE_DIR                              --stage 使用時のローカル退避先（未設定なら /tmp 配下）
#   VIRCHOW_MINI_CHECKPOINT                         蒸留エンコーダの重みチェックポイント
#   GMAIL_USER / GMAIL_APP_PASSWORD / RECEIVE_USER  --notify 使用時の認証情報

set -euo pipefail

DEFAULT_MAGS="1.25 2.5 5.0 10.0 20.0 40.0"

ENCODER=""
COORDS_DIR=""
OUT=""
SLIDES=""
WSI_DIR=""
MAGS="$DEFAULT_MAGS"
BATCH_SIZE=""
NUM_WORKERS=""
GPU_IDS=""
SKIP_BACKGROUND=""
STAGE=""
NOTIFY=""
OVERRIDES=""
WSI_BASE=""

USAGE="Usage: $0 --encoder NAME --coords-dir DIR --out DIR (--slides PATH | --wsi-dir DIR) \
[--mags \"1.25 2.5 ...\"] [--batch-size N] [--num-workers N] [--gpu-ids \"0,1,2\"] \
[--skip-background] [--stage] [--notify] [--overrides CSV] [--wsi-base-path DIR]"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder) ENCODER="$2"; shift 2;;
        --coords-dir) COORDS_DIR="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --slides) SLIDES="$2"; shift 2;;
        --wsi-dir) WSI_DIR="$2"; shift 2;;
        --mags) MAGS="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --num-workers) NUM_WORKERS="$2"; shift 2;;
        --gpu-ids) GPU_IDS="$2"; shift 2;;
        --skip-background) SKIP_BACKGROUND="--skip-background"; shift;;
        --stage) STAGE="--stage"; shift;;
        --notify) NOTIFY="--notify"; shift;;
        --overrides) OVERRIDES="$2"; shift 2;;
        --wsi-base-path) WSI_BASE="$2"; shift 2;;
        *) echo "Unknown option: $1" >&2; echo "$USAGE" >&2; exit 1;;
    esac
done

[[ -z "$ENCODER" ]] && { echo "Error: --encoder is required" >&2; exit 1; }
[[ -z "$COORDS_DIR" ]] && { echo "Error: --coords-dir is required" >&2; exit 1; }
[[ -z "$OUT" ]] && { echo "Error: --out is required" >&2; exit 1; }
[[ -z "$SLIDES" && -z "$WSI_DIR" ]] && { echo "Error: one of --slides or --wsi-dir is required" >&2; exit 1; }
[[ -n "$SLIDES" && -n "$WSI_DIR" ]] && { echo "Error: --slides and --wsi-dir are mutually exclusive" >&2; exit 1; }

ARGS=(--encoder "$ENCODER" --coords-dir "$COORDS_DIR" --out "$OUT" --mags $MAGS)
[[ -n "$SLIDES" ]] && ARGS+=(--slides "$SLIDES")
[[ -n "$WSI_DIR" ]] && ARGS+=(--wsi-dir "$WSI_DIR")
[[ -n "$BATCH_SIZE" ]] && ARGS+=(--batch-size "$BATCH_SIZE")
[[ -n "$NUM_WORKERS" ]] && ARGS+=(--num-workers "$NUM_WORKERS")
[[ -n "$GPU_IDS" ]] && ARGS+=(--gpu-ids "$GPU_IDS")
[[ -n "$SKIP_BACKGROUND" ]] && ARGS+=("$SKIP_BACKGROUND")
[[ -n "$STAGE" ]] && ARGS+=("$STAGE")
[[ -n "$NOTIFY" ]] && ARGS+=("$NOTIFY")
[[ -n "$OVERRIDES" ]] && ARGS+=(--overrides "$OVERRIDES")
[[ -n "$WSI_BASE" ]] && ARGS+=(--wsi-base-path "$WSI_BASE")

foveamil-features "${ARGS[@]}"
