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

# Unique prompt per run (avoids radix cache hits, measures real prefill).
import random, string

def make_prompt(target_tokens, seed):
    """Generate a unique prompt of approximately target_tokens tokens."""
    rng = random.Random(seed)
    words = []
    # Calibrated: ~1.47 tokens per word (incl. space) on Qwen tokenizer
    # (measured: 1300 words -> 1912 tokens). So words = tokens / 1.47.
    n_words = int(target_tokens / 1.47)
    base_words = [
        "pump", "station", "hydraulic", "valve", "pressure", "flow", "water",
        "channel", "maintenance", "inspection", "bearing", "coupling", "seal",
        "discharge", "suction", "head", "efficiency", "turbine", "generator",
        "transformer", "circuit", "breaker", "relay", "protection", "monitor",
        "sensor", "actuator", "controller", "frequency", "voltage", "current",
        "temperature", "vibration", "alignment", "lubrication", "cooling",
        "intake", "outlet", "reservoir", "flood", "drought", "irrigation",
        "drainage", "culvert", "weir", "spillway", "gate", "aperture",
        "discharge", "capacity", "rating", "nominal", "actual", "measured",
        "calculated", "theoretical", "empirical", "statistical", "analysis",
        "diagnosis", "prediction", "optimization", "scheduling", "dispatch",
        "coordination", "regulation", "stabilization", "balancing", "matching",
    ]
    for i in range(n_words):
        w = rng.choice(base_words)
        # Add unique suffix to prevent cache hits
        if i % 10 == 0:
            w += str(rng.randint(1000, 9999))
        words.append(w)
    return " ".join(words)


def get_model():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=10) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return "default"


def run_once(model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    # REPEATS now interpreted as target prompt tokens (unique prompt per run)
    target_tokens = REPEATS
    print(f"model={model} target_prompt_tokens={target_tokens} max_tokens={MAX_TOKENS}")
    # warmup (populates MoE cache / JIT)
    run_once(model, make_prompt(target_tokens, seed=0))
    print("warmup done, starting timed runs")
    ttfts, decs = [], []
    for i in range(RUNS):
        prompt = make_prompt(target_tokens, seed=1000 + i)
        ttft, total, n = run_once(model, prompt)
        dec = (n - 1) / (total - ttft) if total > ttft and n > 1 else 0.0
        ttfts.append(ttft)
        decs.append(dec)
        print(f"run{i+1}: TTFT={ttft:.3f}s total={total:.3f}s tokens={n} decode={dec:.1f} tok/s")
    best = min(ttfts)
    print(f"BEST_TTFT={best:.3f}s  AVG_TTFT={sum(ttfts)/len(ttfts):.3f}s  "
          f"AVG_DECODE={sum(decs)/len(decs):.1f} tok/s")


if __name__ == "__main__":
    main()
