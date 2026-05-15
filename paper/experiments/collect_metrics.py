#!/usr/bin/env python3
"""
Aggregate training and inference metrics from all experiment logs
into a single JSON + CSV for paper table generation.

Usage:
    python paper/experiments/collect_metrics.py \
        --log_root paper/logs \
        --out paper/tables/metrics.json
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from compute_mfu import compute_mfu_from_log


# ---- Training log parsers ---------------------------------------------------

def parse_final_loss(log_path: str) -> dict:
    """Return the last reported loss / aux_loss from a training log."""
    last = {}
    pattern = re.compile(
        r"loss:\s*([\d.]+).*?logits_loss:\s*([\d.]+).*?aux_loss:\s*([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last = {
                    "loss": float(m.group(1)),
                    "logits_loss": float(m.group(2)),
                    "aux_loss": float(m.group(3)),
                }
    return last


def parse_device_banner(log_path: str) -> str:
    """Extract the [device] banner line from a log file."""
    with open(log_path) as f:
        for line in f:
            if line.startswith("[device]"):
                return line.strip()
    return "MISSING"


def parse_epoch_wall_time(log_path: str) -> float | None:
    """Return estimated total wall time in minutes from the last epoch_time field."""
    pattern = re.compile(r"epoch_time:\s*([\d.]+)min")
    last = None
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                last = float(m.group(1))
    return last


def parse_peak_memory_gb(log_path: str) -> float | None:
    """Look for torch.cuda.max_memory_allocated output in log."""
    pattern = re.compile(r"peak_mem_gb[:\s]+([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


# ---- Inference log parsers --------------------------------------------------

def parse_inference_metrics(log_path: str) -> dict:
    """Parse bench_client.py output from an inference log."""
    metrics = {}
    patterns = {
        "ttft_p50_ms":    re.compile(r"TTFT\s+p50\s*:\s*([\d.]+)\s*ms"),
        "ttft_p95_ms":    re.compile(r"TTFT\s+p95\s*:\s*([\d.]+)\s*ms"),
        "itl_p50_ms":     re.compile(r"ITL\s+p50\s*:\s*([\d.]+)\s*ms"),
        "itl_p95_ms":     re.compile(r"ITL\s+p95\s*:\s*([\d.]+)\s*ms"),
        "tokens_per_s":   re.compile(r"Throughput\s*:\s*([\d.]+)\s*tok/s"),
        "requests_per_s": re.compile(r"Requests/s\s*:\s*([\d.]+)"),
    }
    with open(log_path) as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    metrics[key] = float(m.group(1))
    return metrics


# ---- Main aggregation -------------------------------------------------------

def collect_all(log_root: str, out_path: str):
    log_root = Path(log_root)
    results = []

    for vendor in ("cuda", "rocm"):
        for block in ("block_a", "block_b", "block_c", "block_d", "block_e"):
            block_dir = log_root / block
            if not block_dir.exists():
                continue
            for log_file in sorted(block_dir.glob(f"{vendor}_*.log")):
                tag = log_file.stem[len(vendor) + 1:]  # strip vendor prefix
                rec = {
                    "vendor": vendor,
                    "block": block,
                    "tag": tag,
                    "log": str(log_file),
                    "banner": parse_device_banner(str(log_file)),
                }

                if block in ("block_a", "block_b", "block_c", "block_d"):
                    rec.update(parse_final_loss(str(log_file)))
                    rec["epoch_wall_time_min"] = parse_epoch_wall_time(str(log_file))
                    rec["peak_mem_gb"] = parse_peak_memory_gb(str(log_file))
                    # MFU: infer config from tag
                    moe = 1 if "moe" in tag else 0
                    gpus_m = re.search(r"g(\d+)", tag)
                    gpus = int(gpus_m.group(1)) if gpus_m else 1
                    # Seq len defaults from CLAUDE.md: pretrain=340, sft=768
                    seq = 768 if "sft" in tag or "lora" in tag else 340
                    bs = 16 if "sft" in tag else 32
                    gpu_model = "h100" if vendor == "cuda" else "mi350"
                    mfu_result = compute_mfu_from_log(
                        str(log_file), 768, 8, bool(moe), seq, bs * gpus, gpu_model
                    )
                    rec["mfu_pct"] = mfu_result.get("mfu_pct")
                    rec["tokens_per_s"] = mfu_result.get("tokens_per_s")
                    rec["tflops_per_s"] = mfu_result.get("tflops_per_s")
                else:
                    rec.update(parse_inference_metrics(str(log_file)))

                results.append(rec)

    # Write JSON
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Write CSV
    csv_path = out_path.with_suffix(".csv")
    if results:
        all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in all_keys})

    print(f"Collected {len(results)} records → {out_path} and {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", default="paper/logs")
    parser.add_argument("--out", default="paper/tables/metrics.json")
    args = parser.parse_args()
    collect_all(args.log_root, args.out)
