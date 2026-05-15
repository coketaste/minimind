#!/usr/bin/env bash
# =============================================================================
# Verification harness — 6 gates from the survey plan.
# Run before reporting any results.
# Usage: bash paper/experiments/verify_all.sh [cuda|rocm]
# All gates must pass before proceeding to the next experiment block.
# =============================================================================
set -euo pipefail

VENDOR="${1:-cuda}"
LOG_DIR="paper/logs/verify"
PASS=0; FAIL=0

mkdir -p "$LOG_DIR"

ok()   { echo "[PASS] $1"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

# =============================================================================
# Gate 1: Banner check
# Every run log must contain "[device] ... bf16=True dist_backend=nccl"
# =============================================================================
echo ""
echo "=== Gate 1: Device banner check ==="
missing=0
for logfile in paper/logs/block_a/"${VENDOR}"_*.log 2>/dev/null; do
    [ -f "$logfile" ] || continue
    if ! grep -q "\[device\]" "$logfile"; then
        echo "  MISSING banner in: $logfile"
        missing=$((missing+1))
    fi
    if ! grep -q "bf16=True" "$logfile"; then
        echo "  bf16=True missing in: $logfile"
        missing=$((missing+1))
    fi
    if ! grep -q "dist_backend=nccl" "$logfile"; then
        echo "  dist_backend=nccl missing in: $logfile"
        missing=$((missing+1))
    fi
done
if [ "$missing" -eq 0 ]; then
    ok "Gate 1: All banners valid"
else
    fail "Gate 1: $missing banner issues found"
fi

# =============================================================================
# Gate 2: Convergence parity (dense pretrain)
# Run both CUDA and ROCm 1-GPU dense pretrain with identical seeds, compare
# loss at step 100.  Acceptable tolerance: ±2% relative.
# =============================================================================
echo ""
echo "=== Gate 2: Convergence parity (dense pretrain 1-GPU) ==="

extract_loss_at_step() {
    local logfile="$1"; local step="$2"
    grep "($step/" "$logfile" | grep -oP 'loss:\s*\K[\d.]+' | head -1
}

cuda_log="paper/logs/block_a/cuda_dense_bf16_g1_nocompile.log"
rocm_log="paper/logs/block_a/rocm_dense_bf16_g1_nocompile.log"

if [ -f "$cuda_log" ] && [ -f "$rocm_log" ]; then
    loss_cuda=$(extract_loss_at_step "$cuda_log" 100)
    loss_rocm=$(extract_loss_at_step "$rocm_log" 100)
    if [ -n "$loss_cuda" ] && [ -n "$loss_rocm" ]; then
        python3 - "$loss_cuda" "$loss_rocm" <<'PYEOF'
import sys
a, b = float(sys.argv[1]), float(sys.argv[2])
rel = abs(a - b) / max(a, b)
print(f"  CUDA loss@100: {a:.4f}  ROCm loss@100: {b:.4f}  rel_diff: {rel*100:.2f}%")
if rel > 0.02:
    print("  PARITY FAIL: relative difference > 2%")
    sys.exit(1)
else:
    print("  PARITY PASS")
PYEOF
        ok "Gate 2: Convergence parity within 2%"
    else
        fail "Gate 2: Could not extract loss at step 100 from logs"
    fi
else
    echo "  Gate 2 SKIP: pretrain logs not yet available (run block_a first)"
fi

# =============================================================================
# Gate 3: MoE aux_loss sanity
# aux_loss must be non-zero and below 1.0 for all MoE runs
# =============================================================================
echo ""
echo "=== Gate 3: MoE aux_loss sanity ==="
moe_issues=0
for logfile in paper/logs/block_a/"${VENDOR}"_moe_*.log 2>/dev/null; do
    [ -f "$logfile" ] || continue
    # Check last reported aux_loss
    last_aux=$(grep -oP 'aux_loss:\s*\K[\d.]+' "$logfile" | tail -1)
    if [ -z "$last_aux" ]; then
        echo "  No aux_loss found in $logfile"
        moe_issues=$((moe_issues+1))
        continue
    fi
    python3 - "$last_aux" "$logfile" <<'PYEOF'
import sys
val, f = float(sys.argv[1]), sys.argv[2]
if val <= 0 or val >= 1.0:
    print(f"  aux_loss={val} out of expected range (0, 1) in {f}")
    sys.exit(1)
PYEOF
done
if [ "$moe_issues" -eq 0 ]; then
    ok "Gate 3: MoE aux_loss values are sane"
else
    fail "Gate 3: $moe_issues MoE aux_loss issues"
fi

# =============================================================================
# Gate 4: Round-trip checkpoint test
# Train 1 step on the current vendor, save .pth, load on CPU, compare loss
# =============================================================================
echo ""
echo "=== Gate 4: Round-trip checkpoint test ==="
RT_DIR="paper/logs/verify/roundtrip"
mkdir -p "$RT_DIR"

python3 - <<PYEOF 2>&1 | tee "${RT_DIR}/roundtrip.log"
import sys, os
sys.path.insert(0, '.')
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
model = MiniMindForCausalLM(cfg).to(device)

# Dummy forward
ids = torch.zeros(1, 16, dtype=torch.long, device=device)
out = model(ids, labels=ids)
loss_before = out.loss.item()

# Save
ckp_path = '${RT_DIR}/roundtrip_test.pth'
torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp_path)

# Reload on CPU
loaded = {k: v.float() for k, v in torch.load(ckp_path, map_location='cpu').items()}
model2 = MiniMindForCausalLM(cfg)
model2.load_state_dict(loaded)
model2 = model2.to(device)
out2 = model2(ids, labels=ids)
loss_after = out2.loss.item()

rel = abs(loss_before - loss_after) / max(abs(loss_before), 1e-9)
print(f"Loss before save: {loss_before:.6f}")
print(f"Loss after load : {loss_after:.6f}")
print(f"Relative diff   : {rel*100:.4f}%")
if rel > 1e-3:
    print("ROUNDTRIP FAIL: losses diverge after checkpoint round-trip")
    sys.exit(1)
else:
    print("ROUNDTRIP PASS")
PYEOF
if [ $? -eq 0 ]; then
    ok "Gate 4: Checkpoint round-trip passed"
else
    fail "Gate 4: Checkpoint round-trip FAILED"
fi

# =============================================================================
# Gate 5: Pytest gate
# =============================================================================
echo ""
echo "=== Gate 5: pytest tests/test_device_utils.py ==="
python -m pytest tests/ -v 2>&1 | tee "${LOG_DIR}/pytest_${VENDOR}.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    ok "Gate 5: All pytest tests passed"
else
    fail "Gate 5: pytest FAILED — check ${LOG_DIR}/pytest_${VENDOR}.log"
fi

# =============================================================================
# Gate 6: vLLM/SGLang greedy parity
# 20 prompts: compare greedy output of eval_llm.py vs vLLM-served Qwen3 weights
# =============================================================================
echo ""
echo "=== Gate 6: vLLM greedy parity (skipped if converted weights absent) ==="
CONV_PATH="out/qwen3_converted/full_sft_blockb_dense_g8"
if [ -d "$CONV_PATH" ]; then
    python3 paper/experiments/verify_greedy_parity.py \
        --native_weight "full_sft_blockb_dense_g8" \
        --conv_path "$CONV_PATH" \
        --num_prompts 20 \
        2>&1 | tee "${LOG_DIR}/greedy_parity_${VENDOR}.log"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        ok "Gate 6: Greedy parity passed"
    else
        fail "Gate 6: Greedy parity FAILED"
    fi
else
    echo "  Gate 6 SKIP: converted weights not found at $CONV_PATH"
    echo "  Run block_e_inference.sh first to generate converted weights."
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=================================="
echo "Verification summary for $VENDOR"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "=================================="
if [ "$FAIL" -gt 0 ]; then
    echo "STOP: Fix failures before running experiment blocks."
    exit 1
fi
echo "All gates passed. Proceed with experiments."
