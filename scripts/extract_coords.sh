#!/bin/bash
# foveamil-coords を本番運用するラッパ（座標抽出）
#  - 既存出力のある slide を除外して再開（--resume; 全 --mags の座標が揃う slide を飛ばす）
#  - 異常終了したら残り slide だけで再試行（--retries）
#  - 完了/失敗メール（--notify）
# 座標抽出は CPU（--num-workers で並列）のため GPU 分割は持たない
# パスは全て引数/環境変数で受け，内部固有名を持たない
#
# 必須引数:
#   --out DIR            座標 H5 の出力ディレクトリ（{out}/{mag}x/{slide_id}.h5）
#   --slides PATH        slide_id ファイル（CSV/テキスト）WSI_BASE_PATH でパス解決
#   または --wsi-dir DIR ディレクトリ内の対応 WSI を全件処理
#   （--slides と --wsi-dir は排他．--resume は --slides 指定時のみ有効）
#
# 任意引数:
#   --mags "1.25 2.5 ..."  倍率列（未指定時は下記 DEFAULT_MAGS）
#   --num-workers N        並列ワーカー数（未指定時は下記 DEFAULT_NUM_WORKERS）
#   --resume               既存出力のある slide を除外して残りだけ処理する
#   --retries N            異常終了時の再試行回数（既定 DEFAULT_RETRIES）
#   --notify               完了・エラー時に要約メールを送る
#
# 環境変数:
#   WSI_BASE_PATH          --slides 使用時の slide_id 解決ルート（: 区切りで複数可）
#   GMAIL_USER / GMAIL_APP_PASSWORD / RECEIVE_USER  --notify 使用時の認証情報

set -euo pipefail

DEFAULT_MAGS="1.25 2.5 5.0 10.0 20.0 40.0"
DEFAULT_NUM_WORKERS=12
DEFAULT_RETRIES=2

OUT=""
SLIDES=""
WSI_DIR=""
MAGS="$DEFAULT_MAGS"
NUM_WORKERS="$DEFAULT_NUM_WORKERS"
RESUME=""
RETRIES="$DEFAULT_RETRIES"
NOTIFY=""

USAGE="Usage: $0 --out DIR (--slides PATH | --wsi-dir DIR) [--mags \"1.25 2.5 ...\"] \
[--num-workers N] [--resume] [--retries N] [--notify]"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2;;
        --slides) SLIDES="$2"; shift 2;;
        --wsi-dir) WSI_DIR="$2"; shift 2;;
        --mags) MAGS="$2"; shift 2;;
        --num-workers) NUM_WORKERS="$2"; shift 2;;
        --resume) RESUME="1"; shift;;
        --retries) RETRIES="$2"; shift 2;;
        --notify) NOTIFY="1"; shift;;
        *) echo "Unknown option: $1" >&2; echo "$USAGE" >&2; exit 1;;
    esac
done

[[ -z "$OUT" ]] && { echo "Error: --out is required" >&2; exit 1; }
[[ -z "$SLIDES" && -z "$WSI_DIR" ]] && { echo "Error: one of --slides or --wsi-dir is required" >&2; exit 1; }
[[ -n "$SLIDES" && -n "$WSI_DIR" ]] && { echo "Error: --slides and --wsi-dir are mutually exclusive" >&2; exit 1; }
if [[ -z "$SLIDES" && -n "$RESUME" ]]; then
    echo "Error: --resume requires --slides (not --wsi-dir)" >&2; exit 1
fi

BASE_ARGS=(--out "$OUT" --mags $MAGS --num-workers "$NUM_WORKERS")
[[ -n "$NOTIFY" ]] && BASE_ARGS+=(--notify)

# 全 --mags の座標が揃っていない slide だけを stdout に列挙する
filter_undone() {
    python - "$1" "$OUT" "$MAGS" <<'PY'
import sys, os
slides_file, root, mags = sys.argv[1], sys.argv[2], sys.argv[3].split()
seen = set()
with open(slides_file) as f:
    for line in f:
        sid = line.split(",")[0].strip()
        if not sid or sid == "slide_id" or sid in seen:
            continue
        seen.add(sid)
        if not all(os.path.exists(os.path.join(root, f"{m}x", f"{sid}.h5")) for m in mags):
            print(sid)
PY
}

# --wsi-dir はそのまま単一実行（再開は slide リストが要るため非対応）
if [[ -n "$WSI_DIR" ]]; then
    foveamil-coords "${BASE_ARGS[@]}" --wsi-dir "$WSI_DIR"
    exit 0
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/foveamil_coords.XXXXXX")
TARGET="$WORK/targets.txt"
if [[ -n "$RESUME" ]]; then
    filter_undone "$SLIDES" > "$TARGET"
else
    awk -F, 'NR>0{s=$1; gsub(/^[ \t]+|[ \t]+$/,"",s); if(s!="" && s!="slide_id") print s}' "$SLIDES" > "$TARGET"
fi
N=$(grep -c . "$TARGET" || true)
echo "対象 slide: $N"
if [[ "${N:-0}" -eq 0 ]]; then
    echo "処理対象なし（全て出力済み）"
    exit 0
fi

tries=0
while :; do
    tries=$((tries+1))
    if [[ $tries -gt 1 ]]; then
        filter_undone "$TARGET" > "$TARGET.remain" || true
        mv "$TARGET.remain" "$TARGET"
        [[ -s "$TARGET" ]] || { echo "残りなし"; break; }
    fi
    echo "座標抽出 start try=$tries ($(grep -c . "$TARGET") slides)"
    if foveamil-coords "${BASE_ARGS[@]}" --slides "$TARGET"; then
        break
    fi
    echo "FAILED try=$tries"
    [[ $tries -gt $RETRIES ]] && { echo "give up after $tries tries" >&2; exit 1; }
    sleep 10
done
