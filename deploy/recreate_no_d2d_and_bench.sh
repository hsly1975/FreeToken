#!/usr/bin/env bash
# Control test: fresh container WITHOUT --moe-prefill-hit-d2d, to attribute
# the decode-speed delta (D2D only touches the prefill prefetch path).
set -uo pipefail
NAME=freetoken-ornith
D2D=freetoken-ornith-d2d

echo "[$(date +%T)] stopping D2D container"
docker stop "$NAME" 2>/dev/null || true
docker rename "$NAME" "$D2D" 2>/dev/null || true

echo "[$(date +%T)] starting no-D2D container (fresh)"
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
  --port 8000 \
  --host 0.0.0.0

echo "[$(date +%T)] waiting for ready"
for i in $(seq 1 60); do
  st=$(curl -s http://127.0.0.1:1919/health 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("status","?"))' 2>/dev/null)
  if [ "$st" = "ok" ]; then echo "[$(date +%T)] READY after ~${i}0s"; break; fi
  sleep 10
done
echo "[$(date +%T)] health:"
curl -s http://127.0.0.1:1919/health; echo
echo "[$(date +%T)] running bench (no-D2D, fresh container)"
cd /tmp && python3 bench_prefill.py 1919 50 128 3
echo "[$(date +%T)] DONE"
