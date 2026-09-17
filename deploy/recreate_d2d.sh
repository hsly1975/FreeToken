#!/bin/bash
# 重建 FreeToken 容器：196K 上下文 + D2D 优化
# 196K = 196 * 1024 = 200704 tokens
# 基于 128K 稳定配置，仅扩大上下文 + 保留 D2D
set -e

echo "=== 停止并删除旧容器 ==="
docker rm -f freetoken-ornith 2>/dev/null || true

echo "=== 重建容器 (196K + D2D) ==="
docker run -d \
  --name freetoken-ornith \
  --gpus '"device=1"' \
  --shm-size 64m \
  -p 1919:1919 \
  -v /data/models:/models \
  -e HTTP_PROXY=http://192.168.31.1:7890 \
  -e HTTPS_PROXY=http://192.168.31.1:7890 \
  -e http_proxy=http://192.168.31.1:7890 \
  -e https_proxy=http://192.168.31.1:7890 \
  -e no_proxy=localhost,127.0.0.1,192.168.31.0/24 \
  freetoken:latest \
  --model /models/Ornith-1.5-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 1919 \
  --memory-ratio 0.85 \
  --kv-reserve-tokens 200704 \
  --max-seq-len-override 200704 \
  --moe-cache-auto \
  --max-prefill-length 8192 \
  --moe-prefill-hit-d2d

echo "=== 容器已启动，等待加载 ==="
docker ps --filter name=freetoken-ornith --format '{{.Names}} | {{.Status}}'
