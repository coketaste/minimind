#!/usr/bin/env python3
"""
Compute Model-FLOP Utilization (MFU) from MiniMind training logs.

Usage:
    python paper/experiments/compute_mfu.py \
        --log paper/logs/block_a/cuda_dense_bf16_g8_nocompile.log \
        --hidden_size 768 --num_hidden_layers 8 --use_moe 0 \
        --gpu h100

MFU = achieved_flops_per_sec / peak_flops_per_sec

Achieved FLOPs per token:
  dense : 6 * P                     (forward + 2x backward, each ~2P MACs)
  MoE   : 6 * P_active              (only active expert contributes per token)

Peak TFLOPs/s (bf16 tensor cores):
  H100 SXM5  : 989  TFLOPs/s
  MI350      : fill from spec sheet when available
"""
import argparse
import math
import re
import sys

# ---- Hardware peak FLOPs (bf16 tensor core, TFLOPs/s) ----------------------
PEAK_TFLOPS = {
    "h100":  989.0,   # H100 SXM5
    "mi350": None,    # TODO: fill from AMD spec sheet
    "mi300x": 1307.0, # MI300X (fallback)
    "a100":  312.0,   # A100 SXM4 80GB (reference)
    "rx7900xtx": 61.4, # Radeon RX 7900 XTX (consumer reference)
}

# ---- MiniMind parameter counting -------------------------------------------
def count_params_dense(hidden_size: int, num_layers: int, vocab_size: int = 6400) -> int:
    """Approximate trainable parameters for MiniMind dense model."""
    num_heads = 8
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64

    # Embedding (tied)
    embed = hidden_size * vocab_size

    per_layer = (
        # Attention
        hidden_size * num_heads * head_dim     # q_proj
        + hidden_size * num_kv_heads * head_dim  # k_proj
        + hidden_size * num_kv_heads * head_dim  # v_proj
        + num_heads * head_dim * hidden_size    # o_proj
        + 2 * head_dim                          # q_norm + k_norm weights
        # FFN (SwiGLU: gate + up + down)
        + hidden_size * intermediate_size       # gate_proj
        + hidden_size * intermediate_size       # up_proj
        + intermediate_size * hidden_size       # down_proj
        # RMSNorm (per layer, 2 norms)
        + 2 * hidden_size
    )
    # Final norm
    final_norm = hidden_size

    return embed + num_layers * per_layer + final_norm


def count_params_moe(
    hidden_size: int,
    num_layers: int,
    vocab_size: int = 6400,
    num_experts: int = 4,
    num_active: int = 1,
) -> tuple[int, int]:
    """Returns (total_params, active_params) for MiniMind MoE."""
    num_heads = 8
    num_kv_heads = 4
    head_dim = hidden_size // num_heads
    intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
    moe_intermediate_size = intermediate_size  # same by default

    embed = hidden_size * vocab_size

    attn_per_layer = (
        hidden_size * num_heads * head_dim
        + hidden_size * num_kv_heads * head_dim
        + hidden_size * num_kv_heads * head_dim
        + num_heads * head_dim * hidden_size
        + 2 * head_dim
    )
    shared_expert_params = (
        hidden_size * moe_intermediate_size   # gate
        + hidden_size * moe_intermediate_size # up
        + moe_intermediate_size * hidden_size # down
    )
    routed_expert_params = shared_expert_params  # same size per expert
    router_params = hidden_size * num_experts
    norm_params = 2 * hidden_size  # pre-attn + pre-ffn RMSNorm

    total_per_layer = (
        attn_per_layer
        + shared_expert_params               # 1 shared expert
        + num_experts * routed_expert_params  # all routed experts
        + router_params
        + norm_params
    )
    active_per_layer = (
        attn_per_layer
        + shared_expert_params               # shared always active
        + num_active * routed_expert_params   # only active routed experts
        + router_params
        + norm_params
    )

    final_norm = hidden_size
    total = embed + num_layers * total_per_layer + final_norm
    active = embed + num_layers * active_per_layer + final_norm
    return total, active


# ---- Log parsing -----------------------------------------------------------
def parse_tokens_per_sec(log_path: str) -> list[float]:
    """
    Extract per-step tokens/s from MiniMind training logs.
    Logs contain: loss: X, epoch_time: Y.Zmin (time for remaining steps)
    We parse step timing directly from elapsed wall-clock deltas.
    Falls back to epoch_time field as rough estimate.
    """
    # Pattern: Epoch:[E/N](step/total), loss: ..., epoch_time: T.Xmin
    pattern = re.compile(
        r"Epoch:\[(\d+)/\d+\]\((\d+)/(\d+)\).*?epoch_time:\s*([\d.]+)min"
    )
    records = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(2))
                total = int(m.group(3))
                eta_min = float(m.group(4))
                records.append((step, total, eta_min))

    if len(records) < 2:
        return []

    # Derive step time from successive eta_min differences
    # eta_min = remaining_steps * time_per_step (in minutes)
    # Δeta / Δstep ≈ time_per_step
    tps_list = []
    for i in range(1, len(records)):
        s0, t0, eta0 = records[i - 1]
        s1, t1, eta1 = records[i]
        ds = s1 - s0
        if ds <= 0:
            continue
        # Approximate step time (seconds) from ETA reduction
        step_time_s = (eta0 - eta1) * 60.0 / ds if eta0 > eta1 else None
        if step_time_s and step_time_s > 0:
            tps_list.append(step_time_s)

    return tps_list


def compute_mfu_from_log(
    log_path: str,
    hidden_size: int,
    num_layers: int,
    use_moe: bool,
    seq_len: int,
    batch_size: int,
    gpu: str,
) -> dict:
    peak = PEAK_TFLOPS.get(gpu.lower())
    if peak is None:
        print(f"WARNING: peak TFLOPs unknown for {gpu}; MFU will be None", file=sys.stderr)

    if use_moe:
        total_p, active_p = count_params_moe(hidden_size, num_layers)
        flops_per_token = 6 * active_p
    else:
        total_p = count_params_dense(hidden_size, num_layers)
        active_p = total_p
        flops_per_token = 6 * total_p

    tokens_per_step = seq_len * batch_size

    step_times = parse_tokens_per_sec(log_path)
    if not step_times:
        return {
            "log": log_path,
            "gpu": gpu,
            "total_params_M": total_p / 1e6,
            "active_params_M": active_p / 1e6,
            "flops_per_token": flops_per_token,
            "avg_step_time_s": None,
            "tokens_per_s": None,
            "tflops_per_s": None,
            "mfu": None,
            "warning": "Could not parse step times from log",
        }

    avg_step_s = sum(step_times) / len(step_times)
    tokens_per_s = tokens_per_step / avg_step_s
    tflops_per_s = flops_per_token * tokens_per_s / 1e12
    mfu = tflops_per_s / peak if peak else None

    return {
        "log": log_path,
        "gpu": gpu,
        "total_params_M": round(total_p / 1e6, 2),
        "active_params_M": round(active_p / 1e6, 2),
        "flops_per_token": flops_per_token,
        "avg_step_time_s": round(avg_step_s, 4),
        "tokens_per_s": round(tokens_per_s, 1),
        "tflops_per_s": round(tflops_per_s, 3),
        "mfu_pct": round(mfu * 100, 2) if mfu else None,
    }


# ---- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MFU from MiniMind training log")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--seq_len", type=int, default=340, help="max_seq_len used in training")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", default="h100", choices=list(PEAK_TFLOPS.keys()),
                        help="GPU model for peak-FLOPs lookup")
    args = parser.parse_args()

    result = compute_mfu_from_log(
        log_path=args.log,
        hidden_size=args.hidden_size,
        num_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        gpu=args.gpu,
    )
    import json
    print(json.dumps(result, indent=2))
