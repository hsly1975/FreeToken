#!/usr/bin/env bash
# Recreate freetoken-ornith with --moe-prefill-hit-d2d enabled.
# 1:1 replica of the original `docker run` (captured via docker inspect),
# plus the D2D flag. Old container is stopped+renamed (not removed) so it
# can be restored if the new one misbehaves.
set -euo pipefail

NAME=freetoken-ornith
OLD=freetoken-ornith-old

echo ">>> stopping old container"
docker stop "$NAME" 2>/dev/null || true
docker rename "$NAME" "$OLD" 2>/dev/null || true

echo ">>> starting new container with --moe-prefill-hit-d2d"
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --gpus device=1 \
  --shm-size 64m \
  -p 1919:8000 \
  -v /data/models:/models:ro \
  -e HTTPS_PROXY=http://192.168.31.10:1082 \
  -e https_proxy=http://192.168.31.10:1082 \
  -e HTTP_PROXY=http://192.168.31.10:1082 \
  -e http_proxy=http://192.168.31.10:1082 \
  -e NO_PROXY=localhost,127.0.0.1,172.17.0.0/16,192.168.0.0/16 \
  -e no_proxy=localhost,127.0.0.1,172.17.0.0/16,192.168.0.0/16 \
  freetoken:latest \
  --model /models/Ornith-1.5-35B-A3B-NVFP4 \
  --memory-ratio 0.85 \
  --kv-reserve-tokens 81920 \
  --moe-cache-auto \
  --max-prefill-length 8192 \
  --moe-prefill-hit-d2d \
  --port 8000 \
  --host 0.0.0.0

echo ">>> new container started. old kept as: $OLD"
