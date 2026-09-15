#!/usr/bin/env bash
# FreeToken 一键安装脚本（Docker 部署）
# 用法: ./install.sh
# 可覆盖: MODEL_DIR=/path PORT=1919 GPU=0 MEMORY_RATIO=0.85 KV_RESERVE=81920 ./install.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models}"
PORT="${PORT:-1919}"
GPU="${GPU:-0}"
IMAGE="${IMAGE:-freetoken:latest}"
MEMORY_RATIO="${MEMORY_RATIO:-0.85}"
KV_RESERVE="${KV_RESERVE:-81920}"

echo "==> [1/5] 环境检查"
command -v docker >/dev/null || { echo "错误: 未安装 Docker"; exit 1; }
docker info >/dev/null 2>&1 || { echo "错误: Docker 未运行或无权限"; exit 1; }
nvidia-smi >/dev/null 2>&1 || { echo "错误: 未检测到 NVIDIA 驱动"; exit 1; }
docker info --format '{{.Runtimes}}' | grep -q nvidia || { echo "错误: 缺少 nvidia-container-toolkit"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "错误: 模型目录 $MODEL_DIR 不存在"; exit 1; }
echo "    通过: Docker + NVIDIA 运行时 + 模型目录"

echo "==> [2/5] 检查镜像 $IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "    本地无镜像，尝试从 Docker Hub 拉取..."
  docker pull "$IMAGE" || {
    echo "    拉取失败。离线部署: 在有网机器上 docker save freetoken:latest -o freetoken.tar"
    echo "    传到本机后执行: docker load -i freetoken.tar"
    exit 1; }
fi
echo "    镜像就绪"

echo "==> [3/5] 生成 compose 文件"
COMPOSE_FILE="$(dirname "$0")/docker-compose.prod.yml"
[ -f "$COMPOSE_FILE" ] || { echo "错误: 找不到 $COMPOSE_FILE"; exit 1; }
sed -i.bak \
  -e "s|/data/models|$MODEL_DIR|g" \
  -e "s|device_ids: \[\"[0-9]*\"\]|device_ids: [\"$GPU\"]|" \
  "$COMPOSE_FILE"
rm -f "${COMPOSE_FILE}.bak"
echo "    已写入: MODEL_DIR=$MODEL_DIR GPU=$GPU"

echo "==> [4/5] 启动容器"
cd "$(dirname "$COMPOSE_FILE")"
docker compose -f docker-compose.prod.yml up -d
sleep 3
docker compose -f docker-compose.prod.yml ps

echo "==> [5/5] 等待服务就绪（首次加载权重约 1-3 分钟）"
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "    服务已就绪!"
    break
  fi
  sleep 5
  [ "$i" = 60 ] && { echo "    超时，查看日志: docker logs -f freetoken-ornith"; exit 1; }
done

echo ""
echo "================================================"
echo " 部署完成"
echo "   API:  http://<本机IP>:$PORT/v1"
echo "   模型: $(curl -s http://127.0.0.1:$PORT/v1/models | head -c 200)"
echo " 常用命令:"
echo "   日志:   docker logs -f freetoken-ornith"
echo "   停止:   docker compose -f docker-compose.prod.yml down"
echo "   重启:   docker compose -f docker-compose.prod.yml restart"
echo "================================================"
