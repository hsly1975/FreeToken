#!/bin/bash
# Restart FreeToken with 256K context (kv-reserve 250000 to leave headroom for prefill activations)
set -e

# Stop and remove current container
docker stop freetoken-ornith 2>/dev/null || true
docker rm freetoken-ornith 2>/dev/null || true
echo "--- removed old container, starting with kv-reserve 250000 ---"

# Get model path from the 128k container (or use known path)
MODEL_PATH=$(docker inspect freetoken-ornith-128k --format '{{.Config.Cmd}}' 2>/dev/null | python3 -c "import sys,json; cmd=json.load(sys.stdin); i=cmd.index('--model'); print(cmd[i+1])" 2>/dev/null)
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/models/Ornith-1.5-35B-A3B-NVFP4"
fi
echo "model path: $MODEL_PATH"

docker run -d --name freetoken-ornith \
  --restart unless-stopped \
  --gpus '"device=1"' \
  -p 1919:8000 \
  -v /home/hslz123/models:/models \
  --memory 14g \
  freetoken:latest \
  --model "$MODEL_PATH" \
  --memory-ratio 0.93 \
  --kv-reserve-tokens 200704 \
  --max-seq-len-override 200704 \
  --moe-cache-auto \
  --max-prefill-length 4096 \
  --moe-prefill-hit-d2d \
  --port 8000 \
  --host 0.0.0.0

echo "--- container started, waiting for health ---"
for i in $(seq 1 60); do
  sleep 15
  R=$(curl -s -m 5 http://127.0.0.1:1919/health 2>/dev/null || true)
  ST=$(echo "$R" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo "?")
  echo "[$((i*15))s] status=$ST"
  if [ "$ST" = "ready" ]; then
    echo "=== READY ==="
    echo "$R"
    exit 0
  fi
done
echo "=== TIMEOUT waiting for ready ==="
docker logs freetoken-ornith 2>&1 | tail -20
exit 1
