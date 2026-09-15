#!/usr/bin/env bash
# 导出 FreeToken 镜像为离线 tar 包（在有网/已构建的机器上执行）
# 用法: ./export-image.sh [输出文件]   默认 freetoken-image.tar
set -euo pipefail
OUT="${1:-freetoken-image.tar}"
docker image inspect freetoken:latest >/dev/null 2>&1 || { echo "错误: 本地没有 freetoken:latest 镜像"; exit 1; }
echo "==> 导出 freetoken:latest -> $OUT （约 15GB，请耐心等待）"
docker save freetoken:latest -o "$OUT"
echo "==> 完成: $(du -h "$OUT" | cut -f1)"
echo "传到目标机器后执行:  docker load -i $OUT"
