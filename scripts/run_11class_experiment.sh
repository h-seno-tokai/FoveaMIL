#!/bin/bash
# 11 クラス本番実験を一括実行する薄いラッパ
# ベースライン(ABMIL/CLAM)＋A/B/D アブレーション(abd) と C(MCTS) 単独(mcts) を
# それぞれ別の out_root へ回し，任意で評価まで通す
#
# 前提: 特徴は事前に foveamil-stage で 1 回ステージしておく（このスクリプトはステージしない）
#       FEATURE_ROOT がステージ先（または正準特徴ルート）を指すこと
#   cls-only 実験なので cls 特徴だけステージすると容量が約半分になる（40x が巨大なため有効）:
#     foveamil-stage --feature-root <正準> --encoder Virchow2 --feature-type cls \
#       --magnifications 1.25 2.5 5.0 10.0 20.0 40.0 --splits-dir cohort/splits/11class/cv10 \
#       --cache-dir /tmp/foveamil_feat_stage
#     export FEATURE_ROOT=/tmp/foveamil_feat_stage   # ステージ先を指す
#   多並列で CPU をスラッシュさせないため OMP_NUM_THREADS=1 を必ず設定する:
#     export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
#
# 必須引数:
#   --weights-base DIR   重み(.pt)の出力ルート（Dataset 側）配下に abd/ mcts/ を作る
#
# 任意引数:
#   --out-base DIR       ログ・結果の出力ルート（home 既定 experiments/11class_virchow2）
#   --gpu-ids "0,1,2"    使用 GPU
#   --jobs-per-gpu N     abd の GPU あたり並列数（既定 config の値）
#   --mcts-jobs-per-gpu N MCTS の GPU あたり並列数（既定 config の値 MCTS は重い）
#   --eval               sweep 後に foveamil-eval と foveamil-ablation を実行する
#   --notify             各 sweep の開始・完了をメール通知する
#
# 環境変数: FEATURE_ROOT（必須）GMAIL_*（--notify 時）

set -euo pipefail

# 実験 config は頻繁に変わるため git 追跡しない configs/local/ に置く（汎用雛形は
# configs/sweep.example.yaml）CONFIG_DIR を環境変数で上書きも可
CONFIG_DIR="${CONFIG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../configs/local" && pwd)}"
ABD_CONFIG="${CONFIG_DIR}/sweep_11class_virchow2.yaml"
MCTS_CONFIG="${CONFIG_DIR}/sweep_11class_virchow2_mcts.yaml"

OUT_BASE="experiments/11class_virchow2"
WEIGHTS_BASE=""
GPU_IDS=""
JOBS_PER_GPU=""
MCTS_JOBS_PER_GPU=""
RUN_EVAL=""
NOTIFY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-base) OUT_BASE="$2"; shift 2 ;;
        --weights-base) WEIGHTS_BASE="$2"; shift 2 ;;
        --gpu-ids) GPU_IDS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --mcts-jobs-per-gpu) MCTS_JOBS_PER_GPU="$2"; shift 2 ;;
        --eval) RUN_EVAL="1"; shift ;;
        --notify) NOTIFY="--notify"; shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 --weights-base DIR [--out-base DIR] [--gpu-ids \"0,1,2\"] [--jobs-per-gpu N] [--mcts-jobs-per-gpu N] [--eval] [--notify]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${WEIGHTS_BASE}" ]]; then
    echo "Error: --weights-base is required" >&2
    exit 1
fi
if [[ -z "${FEATURE_ROOT:-}" ]]; then
    echo "Error: FEATURE_ROOT must be set (point to the staged or canonical feature root)" >&2
    exit 1
fi

run_sweep() {
    local config="$1" tag="$2" jpg="$3"
    local args=(--config "$config" --out "${OUT_BASE}/${tag}" --weights-out "${WEIGHTS_BASE}/${tag}")
    [[ -n "${GPU_IDS}" ]] && args+=(--gpu-ids "${GPU_IDS}")
    [[ -n "${jpg}" ]] && args+=(--jobs-per-gpu "${jpg}")
    [[ -n "${NOTIFY}" ]] && args+=("${NOTIFY}")
    echo ">>> sweep ${tag}: foveamil-sweep ${args[*]}"
    foveamil-sweep "${args[@]}"
}

run_sweep "${ABD_CONFIG}" "abd" "${JOBS_PER_GPU}"
run_sweep "${MCTS_CONFIG}" "mcts" "${MCTS_JOBS_PER_GPU}"

if [[ -n "${RUN_EVAL}" ]]; then
    echo ">>> eval abd"
    foveamil-eval --in "${OUT_BASE}/abd" --split test --metric weighted_f1
    echo ">>> eval mcts"
    foveamil-eval --in "${OUT_BASE}/mcts" --split test --metric weighted_f1
    echo ">>> ablation table (abd + mcts)"
    foveamil-ablation --in "${OUT_BASE}/abd" "${OUT_BASE}/mcts" \
        --metric weighted_f1 --split test --out "${OUT_BASE}/ablation.md"
fi
