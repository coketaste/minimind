#!/usr/bin/env bash
# =============================================================================
# Block A — Pretrain experiment runner
# Covers: dense + MoE, 1/4/8 GPU, bf16/fp16, torch.compile on/off
# Run from repo root: bash paper/experiments/block_a_pretrain.sh [cuda|rocm]
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"          # 'cuda' or 'rocm'
DATA_PATH="${DATA_PATH:-dataset/pretrain_t2t_mini.jsonl}"
SAVE_DIR="${SAVE_DIR:-out}"
LOG_DIR="${LOG_DIR:-paper/logs/block_a}"
EPOCHS=2
LOG_INTERVAL=100
SAVE_INTERVAL=99999          # save only at end

mkdir -p "$LOG_DIR"

# ---- helper ----------------------------------------------------------------
run_pretrain() {
    local tag="$1"; shift
    local gpus="$1"; shift
    local moe="$1"; shift
    local dtype="$1"; shift
    local compile="$1"; shift
    local extra_args="$*"

    local logfile="${LOG_DIR}/${VENDOR}_${tag}.log"
    echo "[block_a] Starting: $tag  (gpus=$gpus moe=$moe dtype=$dtype compile=$compile)" | tee -a "$logfile"

    local cmd
    if [ "$gpus" -eq 1 ]; then
        cmd="python trainer/train_pretrain.py"
    else
        cmd="torchrun --nproc_per_node=$gpus trainer/train_pretrain.py"
    fi

    $cmd \
        --epochs "$EPOCHS" \
        --use_moe "$moe" \
        --dtype "$dtype" \
        --use_compile "$compile" \
        --save_dir "$SAVE_DIR" \
        --save_weight "pretrain_blocka_${tag}" \
        --save_interval "$SAVE_INTERVAL" \
        --log_interval "$LOG_INTERVAL" \
        --data_path "$DATA_PATH" \
        $extra_args \
        2>&1 | tee -a "$logfile"

    echo "[block_a] Done: $tag" | tee -a "$logfile"
}

# ---- sweep -----------------------------------------------------------------
# Dense, bf16, varying GPU count, compile=0 (primary)
for gpus in 1 4 8; do
    run_pretrain "dense_bf16_g${gpus}_nocompile" "$gpus" 0 bfloat16 0
done

# Dense, bf16, 8 GPU, compile=1
run_pretrain "dense_bf16_g8_compile" 8 0 bfloat16 1

# Dense, fp16, 8 GPU (secondary dtype)
run_pretrain "dense_fp16_g8_nocompile" 8 0 float16 0

# MoE, bf16, varying GPU count
for gpus in 1 4 8; do
    run_pretrain "moe_bf16_g${gpus}_nocompile" "$gpus" 1 bfloat16 0
done

# MoE, bf16, 8 GPU, compile=1
run_pretrain "moe_bf16_g8_compile" 8 1 bfloat16 1

echo "[block_a] All pretrain experiments complete. Logs in $LOG_DIR"
