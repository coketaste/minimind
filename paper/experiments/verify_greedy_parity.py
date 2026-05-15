#!/usr/bin/env python3
"""
Gate 6: Verify that the Qwen3-converted weights served via vLLM (or SGLang)
produce token-for-token identical greedy outputs to the native MiniMind model.

Requires the vLLM/SGLang server to already be running.
Usage:
    # Start vLLM server first:
    # python -m vllm.entrypoints.openai.api_server \
    #   --model out/qwen3_converted/full_sft_blockb_dense_g8 --port 8998
    python paper/experiments/verify_greedy_parity.py \
        --native_weight full_sft_blockb_dense_g8 \
        --conv_path out/qwen3_converted/full_sft_blockb_dense_g8 \
        --server_url http://127.0.0.1:8998 \
        --num_prompts 20
"""
import argparse
import sys
import os
import requests
import torch
from transformers import AutoTokenizer

sys.path.insert(0, '.')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


PROMPTS = [
    "你好，介绍一下自己。",
    "1+1等于多少？",
    "Python中如何定义一个类？",
    "什么是机器学习？",
    "解释一下深度学习的基本原理。",
    "北京是哪个国家的首都？",
    "请写一首关于春天的诗。",
    "如何计算圆的面积？",
    "什么是递归？",
    "解释什么是API。",
    "简述梯度下降算法。",
    "什么是过拟合？如何防止？",
    "解释Transformer模型的注意力机制。",
    "什么是混合专家模型（MoE）？",
    "如何在PyTorch中实现一个简单的神经网络？",
    "什么是迁移学习？",
    "解释反向传播算法。",
    "什么是批归一化？",
    "如何评估语言模型的质量？",
    "什么是RLHF？",
]


def generate_native(model, tokenizer, prompt: str, device: str, max_new: int = 64) -> str:
    """Greedy generate using native MiniMind model."""
    conv = [{"role": "user", "content": prompt}]
    inputs_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_server(server_url: str, prompt: str, max_new: int = 64) -> str:
    """Greedy generate via OpenAI-compatible server."""
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "minimind",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new,
            "temperature": 0,     # greedy
            "stream": False,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_weight", required=True)
    parser.add_argument("--conv_path", required=True)
    parser.add_argument("--server_url", default="http://127.0.0.1:8998")
    parser.add_argument("--num_prompts", type=int, default=20)
    parser.add_argument("--max_new", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--save_dir", default="out")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("model/")
    cfg = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = MiniMindForCausalLM(cfg)
    ckp = f"{args.save_dir}/{args.native_weight}_{args.hidden_size}.pth"
    model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
    model = model.half().eval().to(device)

    prompts = PROMPTS[:args.num_prompts]
    mismatches = 0

    for i, prompt in enumerate(prompts):
        native_out = generate_native(model, tokenizer, prompt, device, args.max_new)
        server_out = generate_server(args.server_url, prompt, args.max_new)

        if native_out.strip() != server_out.strip():
            print(f"[MISMATCH] prompt {i}: '{prompt[:40]}...'")
            print(f"  native : {native_out[:80]!r}")
            print(f"  server : {server_out[:80]!r}")
            mismatches += 1
        else:
            print(f"[MATCH]    prompt {i}: '{prompt[:40]}...'")

    print(f"\nGreedy parity: {len(prompts) - mismatches}/{len(prompts)} match")
    if mismatches > 0:
        print("PARITY FAIL")
        sys.exit(1)
    print("PARITY PASS")


if __name__ == "__main__":
    main()
