# Running FreeToken with Docker & Docker Compose

Docker support for FreeToken. Based on community PR
[FlashML-org/FreeToken#295](https://github.com/FlashML-org/FreeToken/pull/295),
adapted to install FreeToken **from source (git main)** so the image carries
the latest model-loading fixes (notably PR #438, which the PyPI 0.1.2 wheel
predates).

---

## Prerequisites

- **NVIDIA GPU** (Ada Lovelace / Hopper / Ampere / Turing / Pascal)
- **NVIDIA Driver** r580+ (CUDA 13 compatible)
- **Docker Engine** 24.0+ and **Docker Compose** v2+
- **NVIDIA Container Toolkit** configured as the default Docker runtime

Verify GPU passthrough first:

```bash
docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu26.04 nvidia-smi
```

---

## 1. Build the image

From the repository root:

```bash
docker build -t freetoken:latest .
```

The build installs `freetoken[accel]` from source and compiles the C++
extensions (pinned_tensor, cpu_moe, ple_store). Triton/CUDA kernels are still
JIT-compiled on first use, so the first request after boot is slower.

> Rebuild after pulling upstream changes: `git pull && docker build -t freetoken:latest .`

## 2. Configure (`.env`)

Optional `.env` next to `docker-compose.yml`:

```env
# Only for gated/private models
HF_TOKEN=hf_your_token_here

# Hugging Face repo id (or a local path mounted into the container)
MODEL_NAME=ornith-ai/Ornith-1.5-35B-A3B-FP8
```

## 3. Run with Docker Compose

```bash
docker compose up -d
docker compose logs -f freetoken
```

Wait for:

```text
API server is ready to serve on 0.0.0.0:1919
```

Stop:

```bash
docker compose down
```

## 4. Or run with `docker run`

```bash
docker run -d \
  --name freetoken-server \
  --gpus all \
  --ipc=host \
  -p 1919:1919 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME="/root/.cache/huggingface" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  freetoken:latest \
  --model "ornith-ai/Ornith-1.5-35B-A3B-FP8" \
  --host 0.0.0.0 \
  --port 1919
```

## 5. Test the API

OpenAI-compatible endpoint on port 1919:

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ornith-ai/Ornith-1.5-35B-A3B-FP8",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1919/v1", api_key="none")
r = client.chat.completions.create(
    model="ornith-ai/Ornith-1.5-35B-A3B-FP8",
    messages=[{"role": "user", "content": "Explain KV caching in two sentences."}],
)
print(r.choices[0].message.content)
```

---

## Notes & troubleshooting

- **`ipc: host` is required.** High-performance engines use shared memory
  (`/dev/shm`) for KV cache and multi-worker communication. Omitting it can
  cause immediate `Bus error` or `CUDA OOM` crashes.
- **Persistent cache.** Weights download to `~/.cache/huggingface` on the
  host; subsequent boots load from disk.
- **Local model paths.** Mount the directory and pass the path:
  `-v /path/to/models:/models` with `--model /models/your-model-folder`.
- **Memory.** The compose file sets no memory limit by design — MoE expert
  weights live in host RAM (e.g. ~35 GB for a 35B-A3B FP8 model). Make sure
  the host has enough free RAM.
- **Version tracking.** The image is built from the repo checkout, so it
  tracks whatever commit you built from. Rebuild to pick up upstream fixes.
