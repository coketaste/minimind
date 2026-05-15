#!/usr/bin/env bash
# =============================================================================
# Block E — Inference benchmarks
# Covers: HF native, vLLM, SGLang × dense+MoE × H100/MI350
# Prerequisites:
#   1. block_b_posttrain.sh (produces full_sft weights)
#   2. Qwen3 conversion: python scripts/convert_model.py  (run once per config)
#   3. vLLM and SGLang installed for the target vendor
# Run from repo root: bash paper/experiments/block_e_inference.sh [cuda|rocm]
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"
SAVE_DIR="${SAVE_DIR:-out}"
CONV_DIR="${CONV_DIR:-out/qwen3_converted}"   # HF-format output
LOG_DIR="${LOG_DIR:-paper/logs/block_e}"
API_HOST="127.0.0.1"
API_PORT=8998
CONCURRENCY_LEVELS="1 8 32 128"
MAX_NEW_TOKENS_LIST="128 512 2048"
BENCH_REQUESTS=200   # requests per concurrency level

mkdir -p "$LOG_DIR" "$CONV_DIR"

# ---- Step 1: Convert weights to Qwen3 format --------------------------------
convert_weights() {
    local model="$1"; local moe="$2"; local size="768"
    local out_dir="${CONV_DIR}/${model}"
    if [ -d "$out_dir" ]; then
        echo "[block_e/convert] Already exists: $out_dir, skipping."
        return
    fi
    echo "[block_e/convert] Converting $model (moe=$moe)..."
    python - <<PYEOF
import sys; sys.path.insert(0, '.')
from scripts.convert_model import convert_torch2transformers
from model.model_minimind import MiniMindConfig
import os
lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=bool($moe))
moe_suf = '_moe' if $moe else ''
torch_path = f'$SAVE_DIR/${model}_768{moe_suf}.pth'
convert_torch2transformers(torch_path, '$out_dir')
PYEOF
    echo "[block_e/convert] Done: $out_dir"
}

convert_weights "full_sft_blockb_dense_g8" 0
convert_weights "full_sft_blockb_moe_g8"   1

# ---- Step 2: HF-native single-stream ----------------------------------------
run_hf_native() {
    local model="$1"; local moe="$2"; local max_new="$3"
    local tag="${model}_mnt${max_new}"
    local logfile="${LOG_DIR}/${VENDOR}_hf_${tag}.log"
    echo "[block_e/hf] $tag" | tee -a "$logfile"
    python eval_llm.py \
        --weight "$model" \
        --use_moe "$moe" \
        --max_new_tokens "$max_new" \
        --show_speed 1 \
        --device cuda:0 \
        2>&1 | tee -a "$logfile"
}

for mnt in $MAX_NEW_TOKENS_LIST; do
    run_hf_native "full_sft_blockb_dense_g8" 0 "$mnt"
    run_hf_native "full_sft_blockb_moe_g8"   1 "$mnt"
done

# ---- Step 3: HF-native concurrent (via OpenAI-compatible API) ---------------
run_hf_api_bench() {
    local model="$1"; local moe="$2"; local concurrency="$3"
    local tag="${model}_c${concurrency}"
    local logfile="${LOG_DIR}/${VENDOR}_hfapi_${tag}.log"

    # Launch server in background, wait for it
    python scripts/serve_openai_api.py \
        --weight "$model" --use_moe "$moe" --device cuda:0 &
    SERVER_PID=$!
    sleep 8   # wait for uvicorn to be ready

    echo "[block_e/hfapi] $tag concurrency=$concurrency" | tee -a "$logfile"
    python paper/experiments/bench_client.py \
        --host "$API_HOST" --port "$API_PORT" \
        --concurrency "$concurrency" \
        --num_requests "$BENCH_REQUESTS" \
        --max_tokens 512 \
        2>&1 | tee -a "$logfile"

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}

for concurrency in $CONCURRENCY_LEVELS; do
    run_hf_api_bench "full_sft_blockb_dense_g8" 0 "$concurrency"
done

# ---- Step 4: vLLM benchmark -------------------------------------------------
run_vllm() {
    local conv_path="$1"; local concurrency="$2"; local max_new="$3"
    local label; label="$(basename "$conv_path")"
    local tag="${label}_c${concurrency}_mnt${max_new}"
    local logfile="${LOG_DIR}/${VENDOR}_vllm_${tag}.log"

    # Launch vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model "$conv_path" \
        --host "$API_HOST" --port "$API_PORT" \
        --dtype bfloat16 &
    SERVER_PID=$!
    sleep 20   # vLLM needs more startup time

    echo "[block_e/vllm] $tag" | tee -a "$logfile"
    python paper/experiments/bench_client.py \
        --host "$API_HOST" --port "$API_PORT" \
        --concurrency "$concurrency" \
        --num_requests "$BENCH_REQUESTS" \
        --max_tokens "$max_new" \
        2>&1 | tee -a "$logfile"

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}

for concurrency in $CONCURRENCY_LEVELS; do
    for mnt in 128 512 2048; do
        run_vllm "${CONV_DIR}/full_sft_blockb_dense_g8" "$concurrency" "$mnt"
        run_vllm "${CONV_DIR}/full_sft_blockb_moe_g8"   "$concurrency" "$mnt"
    done
done

# ---- Step 5: SGLang benchmark -----------------------------------------------
run_sglang() {
    local conv_path="$1"; local concurrency="$2"; local max_new="$3"
    local label; label="$(basename "$conv_path")"
    local tag="${label}_c${concurrency}_mnt${max_new}"
    local logfile="${LOG_DIR}/${VENDOR}_sglang_${tag}.log"

    python -m sglang.launch_server \
        --model-path "$conv_path" \
        --host "$API_HOST" --port "$API_PORT" &
    SERVER_PID=$!
    sleep 20

    echo "[block_e/sglang] $tag" | tee -a "$logfile"
    python paper/experiments/bench_client.py \
        --host "$API_HOST" --port "$API_PORT" \
        --concurrency "$concurrency" \
        --num_requests "$BENCH_REQUESTS" \
        --max_tokens "$max_new" \
        2>&1 | tee -a "$logfile"

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}

for concurrency in $CONCURRENCY_LEVELS; do
    for mnt in 128 512 2048; do
        run_sglang "${CONV_DIR}/full_sft_blockb_dense_g8" "$concurrency" "$mnt"
    done
done

# ---- Step 6: Tool-call quality eval -----------------------------------------
python scripts/eval_toolcall.py \
    2>&1 | tee "${LOG_DIR}/${VENDOR}_toolcall.log"

echo "[block_e] All inference experiments complete. Logs in $LOG_DIR"
