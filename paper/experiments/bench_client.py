#!/usr/bin/env python3
"""
Async HTTP benchmark client for OpenAI-compatible inference servers
(MiniMind serve_openai_api.py, vLLM, SGLang).

Usage:
    python paper/experiments/bench_client.py \
        --host 127.0.0.1 --port 8998 \
        --concurrency 32 --num_requests 200 --max_tokens 512
"""
import argparse
import asyncio
import json
import statistics
import time

import aiohttp

SAMPLE_PROMPTS = [
    "解释量子纠缠的基本原理。",
    "用Python写一个快速排序函数并分析其时间复杂度。",
    "比较深度学习和传统机器学习方法的优缺点。",
    "描述Transformer架构中注意力机制的工作原理。",
    "什么是混合精度训练？它如何加速神经网络训练？",
    "解释梯度消失问题及其常见解决方案。",
    "描述知识蒸馏技术在模型压缩中的应用。",
    "什么是强化学习？举例说明其在LLM对齐中的应用。",
]


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": "minimind",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    first_token_time = None
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            ttft = time.perf_counter() - t0  # non-streaming: TTFT ≈ total time
            body = await resp.json()
            total_time = time.perf_counter() - t0
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            n_tokens = len(content.split())  # rough proxy; replace with tiktoken if needed
    except Exception as e:
        return {"error": str(e), "ttft": None, "total_time": None, "n_tokens": 0}

    return {
        "ttft": ttft,
        "total_time": total_time,
        "n_tokens": n_tokens,
        "itl": (total_time - ttft) / max(n_tokens - 1, 1),  # inter-token latency approx
    }


async def run_benchmark(
    host: str,
    port: int,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
):
    url = f"http://{host}:{port}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)

    results = []
    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        async def bounded(prompt):
            async with sem:
                return await single_request(session, url, prompt, max_tokens)

        tasks = [
            bounded(SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)])
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

    wall = time.perf_counter() - start

    ok = [r for r in results if "error" not in r]
    errors = len(results) - len(ok)

    ttfts   = [r["ttft"]       for r in ok if r["ttft"] is not None]
    itls    = [r["itl"]        for r in ok if r["itl"]  is not None]
    tokens  = [r["n_tokens"]   for r in ok]

    total_tokens = sum(tokens)
    tps = total_tokens / wall if wall > 0 else 0.0
    rps = len(ok) / wall if wall > 0 else 0.0

    def p(lst, q):
        if not lst:
            return None
        s = sorted(lst)
        idx = int(len(s) * q / 100)
        return s[min(idx, len(s) - 1)]

    print(f"\n{'='*60}")
    print(f"Benchmark: {url}")
    print(f"  concurrency={concurrency}  requests={num_requests}  max_tokens={max_tokens}")
    print(f"  Wall time   : {wall:.2f} s")
    print(f"  Requests/s  : {rps:.2f}")
    print(f"  Throughput  : {tps:.1f} tok/s")
    print(f"  Errors      : {errors}")
    if ttfts:
        print(f"  TTFT p50    : {p(ttfts, 50)*1000:.1f} ms")
        print(f"  TTFT p95    : {p(ttfts, 95)*1000:.1f} ms")
    if itls:
        print(f"  ITL  p50    : {p(itls, 50)*1000:.1f} ms")
        print(f"  ITL  p95    : {p(itls, 95)*1000:.1f} ms")
    print(f"{'='*60}\n")

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "requests_per_s": round(rps, 2),
        "tokens_per_s": round(tps, 1),
        "errors": errors,
        "ttft_p50_ms": round(p(ttfts, 50) * 1000, 1) if ttfts else None,
        "ttft_p95_ms": round(p(ttfts, 95) * 1000, 1) if ttfts else None,
        "itl_p50_ms":  round(p(itls,  50) * 1000, 1) if itls  else None,
        "itl_p95_ms":  round(p(itls,  95) * 1000, 1) if itls  else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--num_requests", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(
        args.host, args.port, args.concurrency, args.num_requests, args.max_tokens
    ))
    print(json.dumps(result, indent=2))
