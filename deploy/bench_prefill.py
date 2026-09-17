#!/usr/bin/env python3
"""Lightweight prefill/decode benchmark for the FreeToken server.

Measures TTFT (time-to-first-token = prefill time) and decode tok/s using a
fixed long prompt, via the OpenAI-compatible streaming endpoint. No third-party
deps (stdlib only). Run on the host where the server listens.

Usage: python3 bench_prefill.py [port] [prompt_repeats] [max_tokens] [runs]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 1919
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 128
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# Fixed deterministic prompt (~80 tokens/para * REPEATS).
PARA = (
    "In the study of pump station operation and maintenance, the engineer must "
    "carefully monitor water levels, flow rates, and pressure differentials "
    "across the entire hydraulic system. Regular inspection of mechanical "
    "seals, bearings, and coupling alignment is essential to prevent "
    "unexpected failures during peak demand periods. The channel maintenance "
    "crew coordinates with the dispatch center to optimize gate openings and "
    "minimize energy consumption while ensuring safe operation. "
)
PROMPT = PARA * REPEATS


def get_model():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return "default"


def run_once(model):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j.get("choices", [{}])[0].get("delta", {})
                # model may stream thinking tokens in reasoning_content
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
            except Exception:
                pass
    total = time.time() - t0
    return ttft if ttft is not None else total, total, n


def main():
    model = get_model()
    print(f"model={model} prompt_chars={len(PROMPT)} max_tokens={MAX_TOKENS}")
    # warmup (populates MoE cache / JIT)
    run_once(model)
    print("warmup done, starting timed runs")
    ttfts, decs = [], []
    for i in range(RUNS):
        ttft, total, n = run_once(model)
        dec = (n - 1) / (total - ttft) if total > ttft and n > 1 else 0.0
        ttfts.append(ttft)
        decs.append(dec)
        print(f"run{i+1}: TTFT={ttft:.3f}s total={total:.3f}s tokens={n} decode={dec:.1f} tok/s")
    best = min(ttfts)
    print(f"BEST_TTFT={best:.3f}s  AVG_TTFT={sum(ttfts)/len(ttfts):.3f}s  "
          f"AVG_DECODE={sum(decs)/len(decs):.1f} tok/s")


if __name__ == "__main__":
    main()
