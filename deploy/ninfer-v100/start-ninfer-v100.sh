#!/usr/bin/env bash
# NInfer V100 手动启动脚本（在 192.168.31.9 上运行）
# 用法: ./start-ninfer-v100.sh [start|stop|status|logs]
#
# 双 CUDA 环境说明：
#   - GPU0 (V100 32G) 当前跑 llama.cpp 27B (qwen38-27b-server, 8081) —— 给 Hermes 供模型，切换前必须先停它
#   - 本脚本启动 ninfer-serve 容器，同样占用 GPU0，与 llama.cpp 27B 互斥
#   - 切换流程: docker stop qwen38-27b-server  ->  ./start-ninfer-v100.sh start
#   - 回退流程: ./start-ninfer-v100.sh stop    ->  docker start qwen38-27b-server
set -euo pipefail

MODEL_DIR="/data/models/Qwen3.8-27B NVFP4"
NINFER_FILE="${MODEL_DIR}/qwen3_8_27b.ninfer"
IMAGE="ninfer-v100:sm70"
CONTAINER="ninfer-v100-serve"
PORT=8081   # 与 llama.cpp 27B 同端口，切换后 Hermes 配置无需改动
GPU=0       # V100

cmd="${1:-status}"

case "$cmd" in
  start)
    if [ ! -f "$NINFER_FILE" ]; then
      echo "ERROR: 模型文件不存在: $NINFER_FILE" >&2
      exit 1
    fi
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
      echo "容器已在运行"
      exit 0
    fi
    # 端口占用检查（llama.cpp 27B 还占着 8081 的话起不来）
    if ss -tln | grep -q ":${PORT} "; then
      echo "ERROR: 端口 ${PORT} 被占用（llama.cpp 27B 还在跑？先 docker stop qwen38-27b-server）" >&2
      exit 1
    fi
    docker run -d \
      --name "$CONTAINER" \
      --gpus "device=${GPU}" \
      --restart unless-stopped \
      -p 127.0.0.1:${PORT}:${PORT} \
      -v "${MODEL_DIR}:/models:ro" \
      "$IMAGE" \
      ninfer-serve "/models/$(basename "$NINFER_FILE")" \
        --host 0.0.0.0 --port "$PORT" \
        --model-id qwen3.8-27b \
        --max-context 131072 --kv-capacity auto \
        --prefill-chunk 2048 --kv-dtype int8 \
        --spec mtp --draft-tokens 3 --lm-head-draft \
        --preserve-thinking --vision --max-concurrency 1
    echo "已启动: $CONTAINER (GPU${GPU}, 端口 ${PORT})"
    echo "查看日志: docker logs -f $CONTAINER"
    ;;
  stop)
    docker rm -f "$CONTAINER" 2>/dev/null || echo "容器未在运行"
    echo "已停止。回退: docker start qwen38-27b-server"
    ;;
  status)
    echo "=== 容器 ==="
    docker ps -a --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    echo "=== GPU0 显存 ==="
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv -i "$GPU"
    echo "=== 端口 ${PORT} ==="
    ss -tln | grep ":${PORT} " || echo "端口空闲"
    ;;
  logs)
    docker logs --tail 100 -f "$CONTAINER"
    ;;
  *)
    echo "用法: $0 [start|stop|status|logs]" >&2
    exit 1
    ;;
esac
