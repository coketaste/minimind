#!/usr/bin/env bash
# =============================================================================
# Block D — Agent RL (tool-call rollout training)
# Prerequisites: block_b_posttrain.sh (needs full_sft weights)
# Run from repo root: bash paper/experiments/block_d_agent.sh [cuda|rocm]
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"
AGENT_DATA="${AGENT_DATA:-dataset/agent_rl.jsonl}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-../../internlm2-1_8b-reward}"
SAVE_DIR="${SAVE_DIR:-out}"
LOG_DIR="${LOG_DIR:-paper/logs/block_d}"

mkdir -p "$LOG_DIR"

run_agent() {
    local tag="$1"; local gpus="$2"; local from="$3"
    local logfile="${LOG_DIR}/${VENDOR}_agent_${tag}.log"
    echo "[block_d/agent] $tag" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    $cmd trainer/train_agent.py \
        --from_weight "$from" \
        --epochs 1 \
        --batch_size 2 \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "agent_blockd_${tag}" \
        --data_path "$AGENT_DATA" \
        2>&1 | tee -a "$logfile"
    echo "[block_d/agent] Done: $tag"
}

# Primary: 8 GPU
run_agent "dense_g8" 8 "full_sft_blockb_dense_g8"

# Single-GPU for latency / memory baseline
run_agent "dense_g1" 1 "full_sft_blockb_dense_g1"

echo "[block_d] Agent RL experiments complete. Logs in $LOG_DIR"
