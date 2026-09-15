#!/usr/bin/env bash
# 导入离线镜像包（在目标机器上执行）
# 用法: ./import-image.sh [镜像tar文件]   默认 freetoken-image.tar
set -euo pipefail
TAR="${1:-freetoken-image.tar}"
[ -f "$TAR" ] || { echo "错误: 找不到 $TAR"; exit 1; }
echo "==> 导入 $TAR （约 15GB，请耐心等待）"
docker load -i "$TAR"
echo "==> 完成，镜像列表:"
docker images | grep -i freetoken
