#!/usr/bin/env bash
# =============================================================================
# Block C — Preference / RLHF: DPO, PPO, GRPO
# Prerequisites: block_b_posttrain.sh must have completed (needs full_sft weights)
# Requires: internlm2-1_8b-reward checkpoint at REWARD_MODEL_PATH
# Run from repo root: bash paper/experiments/block_c_rlhf.sh [cuda|rocm]
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"
DPO_DATA="${DPO_DATA:-dataset/dpo.jsonl}"
RLAIF_DATA="${RLAIF_DATA:-dataset/rlaif.jsonl}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-../../internlm2-1_8b-reward}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://localhost:8998}"
SAVE_DIR="${SAVE_DIR:-out}"
LOG_DIR="${LOG_DIR:-paper/logs/block_c}"

mkdir -p "$LOG_DIR"

# ---- C1: DPO ---------------------------------------------------------------
run_dpo() {
    local tag="$1"; local gpus="$2"; local from="$3"
    local logfile="${LOG_DIR}/${VENDOR}_dpo_${tag}.log"
    echo "[block_c/dpo] $tag" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    $cmd trainer/train_dpo.py \
        --from_weight "$from" \
        --epochs 1 \
        --batch_size 4 \
        --beta 0.15 \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "dpo_blockc_${tag}" \
        --data_path "$DPO_DATA" \
        2>&1 | tee -a "$logfile"
    echo "[block_c/dpo] Done: $tag"
}

# DPO at 1 and 8 GPU (frozen ref doubles memory; important comparison point)
run_dpo "dense_g1" 1 "full_sft_blockb_dense_g1"
run_dpo "dense_g8" 8 "full_sft_blockb_dense_g8"

# ---- C2: GRPO (rollout-dominated; two rollout engines) ---------------------
run_grpo() {
    local tag="$1"; local gpus="$2"; local from="$3"
    local engine="$4"; local num_gen="$5"
    local logfile="${LOG_DIR}/${VENDOR}_grpo_${tag}.log"
    echo "[block_c/grpo] $tag (engine=$engine, num_gen=$num_gen)" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    local extra=""
    if [ "$engine" = "sglang" ]; then
        extra="--sglang_base_url $SGLANG_BASE_URL"
    fi
    $cmd trainer/train_grpo.py \
        --from_weight "$from" \
        --epochs 1 \
        --batch_size 4 \
        --num_generations "$num_gen" \
        --max_gen_len 1024 \
        --thinking_ratio 0.9 \
        --rollout_engine "$engine" \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "grpo_blockc_${tag}" \
        --data_path "$RLAIF_DATA" \
        $extra \
        2>&1 | tee -a "$logfile"
    echo "[block_c/grpo] Done: $tag"
}

# GRPO with torch rollout engine: num_gen=4 and 8
run_grpo "dense_g8_torch_gen4" 8 "full_sft_blockb_dense_g8" torch 4
run_grpo "dense_g8_torch_gen8" 8 "full_sft_blockb_dense_g8" torch 8

# GRPO with sglang rollout engine (requires running SGLang server separately)
# Launch: python -m sglang.launch_server --model-path <qwen3-converted-path> --port 8998
run_grpo "dense_g8_sglang_gen4" 8 "full_sft_blockb_dense_g8" sglang 4

# ---- C3: PPO (actor+critic+ref+reward; most memory-intensive) --------------
run_ppo() {
    local tag="$1"; local gpus="$2"; local from="$3"; local engine="$4"
    local logfile="${LOG_DIR}/${VENDOR}_ppo_${tag}.log"
    echo "[block_c/ppo] $tag (engine=$engine)" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    local extra=""
    if [ "$engine" = "sglang" ]; then
        extra="--sglang_base_url $SGLANG_BASE_URL"
    fi
    $cmd trainer/train_ppo.py \
        --from_weight "$from" \
        --epochs 1 \
        --batch_size 2 \
        --clip_epsilon 0.2 \
        --kl_coef 0.02 \
        --ppo_update_iters 2 \
        --rollout_engine "$engine" \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "ppo_blockc_${tag}" \
        --data_path "$RLAIF_DATA" \
        $extra \
        2>&1 | tee -a "$logfile"
    echo "[block_c/ppo] Done: $tag"
}

# PPO torch rollout (baseline)
run_ppo "dense_g8_torch" 8 "full_sft_blockb_dense_g8" torch

# PPO sglang rollout (tests vendor serving stack interaction inside training loop)
run_ppo "dense_g8_sglang" 8 "full_sft_blockb_dense_g8" sglang

echo "[block_c] All RLHF experiments complete. Logs in $LOG_DIR"
