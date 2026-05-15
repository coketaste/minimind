#!/usr/bin/env bash
# =============================================================================
# Block B — Post-training: SFT, LoRA, Knowledge Distillation
# Prerequisites: block_a_pretrain.sh must have completed successfully
# Run from repo root: bash paper/experiments/block_b_posttrain.sh [cuda|rocm]
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"
SFT_DATA="${SFT_DATA:-dataset/sft_t2t_mini.jsonl}"
LORA_DATA="${LORA_DATA:-dataset/lora_medical.jsonl}"
SAVE_DIR="${SAVE_DIR:-out}"
LOG_DIR="${LOG_DIR:-paper/logs/block_b}"

mkdir -p "$LOG_DIR"

# ---- B1: Full SFT ----------------------------------------------------------
run_sft() {
    local tag="$1"; local gpus="$2"; local moe="$3"; local from="$4"
    local logfile="${LOG_DIR}/${VENDOR}_sft_${tag}.log"
    echo "[block_b/sft] $tag" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    $cmd trainer/train_full_sft.py \
        --from_weight "$from" \
        --use_moe "$moe" \
        --epochs 2 \
        --batch_size 16 \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "full_sft_blockb_${tag}" \
        --data_path "$SFT_DATA" \
        2>&1 | tee -a "$logfile"
    echo "[block_b/sft] Done: $tag"
}

# Dense SFT: 1, 4, 8 GPU
run_sft "dense_g1" 1 0 "pretrain_blocka_dense_bf16_g1_nocompile"
run_sft "dense_g4" 4 0 "pretrain_blocka_dense_bf16_g4_nocompile"
run_sft "dense_g8" 8 0 "pretrain_blocka_dense_bf16_g8_nocompile"

# MoE SFT: 8 GPU
run_sft "moe_g8"   8 1 "pretrain_blocka_moe_bf16_g8_nocompile"

# ---- B2: LoRA --------------------------------------------------------------
run_lora() {
    local tag="$1"; local gpus="$2"; local from="$3"
    local logfile="${LOG_DIR}/${VENDOR}_lora_${tag}.log"
    echo "[block_b/lora] $tag" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    $cmd trainer/train_lora.py \
        --from_weight "$from" \
        --lora_name "lora_blockb_${tag}" \
        --epochs 10 \
        --batch_size 32 \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --data_path "$LORA_DATA" \
        2>&1 | tee -a "$logfile"
    echo "[block_b/lora] Done: $tag"
}

# LoRA on dense model, 1 and 8 GPU (torch.compile auto-disabled by script)
run_lora "dense_g1" 1 "full_sft_blockb_dense_g1"
run_lora "dense_g8" 8 "full_sft_blockb_dense_g8"

# ---- B3: Knowledge Distillation --------------------------------------------
run_distill() {
    local tag="$1"; local gpus="$2"
    local student_w="$3"; local teacher_w="$4"
    local student_moe="$5"; local teacher_moe="$6"
    local logfile="${LOG_DIR}/${VENDOR}_distill_${tag}.log"
    echo "[block_b/distill] $tag" | tee -a "$logfile"
    local cmd
    [ "$gpus" -eq 1 ] && cmd="python" || cmd="torchrun --nproc_per_node=$gpus"
    $cmd trainer/train_distillation.py \
        --from_student_weight "$student_w" \
        --from_teacher_weight "$teacher_w" \
        --student_use_moe "$student_moe" \
        --teacher_use_moe "$teacher_moe" \
        --alpha 0.5 \
        --temperature 1.5 \
        --epochs 6 \
        --batch_size 32 \
        --dtype bfloat16 \
        --save_dir "$SAVE_DIR" \
        --save_weight "distill_blockb_${tag}" \
        --data_path "$SFT_DATA" \
        2>&1 | tee -a "$logfile"
    echo "[block_b/distill] Done: $tag"
}

# MoE teacher → dense student (primary configuration)
run_distill "moe2dense_g8" 8 \
    "full_sft_blockb_dense_g8" \
    "full_sft_blockb_moe_g8" \
    0 1

# Same-size dense teacher → dense student (ablation)
run_distill "dense2dense_g8" 8 \
    "full_sft_blockb_dense_g8" \
    "full_sft_blockb_dense_g8" \
    0 0

echo "[block_b] All post-training experiments complete. Logs in $LOG_DIR"
