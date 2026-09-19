#!/usr/bin/env bash
# 守护脚本: 等待 bf16 下载完成 (DOWNLOAD_OK) -> 自动跑 CPU 转换
# 日志: /data/models/Qwen3.8-27B-bf16/auto_convert.log
set -uo pipefail

DEST="/data/models/Qwen3.8-27B-bf16"
DFLASH2="/data/models/dflash2-qwen3.8-27b"
OUTDIR="/data/models/Qwen3.8-27B NVFP4"
OUTFILE="${OUTDIR}/qwen3_8_27b.ninfer"
DLOG="${DEST}/download.log"
CLOG="${DEST}/auto_convert.log"
IMAGE="ninfer-convert:cpu"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CLOG"; }

log "=== 守护脚本启动 (PID $$) ==="
log "等待下载完成标记 DOWNLOAD_OK ..."

# 1) 等待下载完成
while true; do
  if grep -q "DOWNLOAD_OK" "$DLOG" 2>/dev/null; then
    log "检测到 DOWNLOAD_OK, 开始校验文件..."
    break
  fi
  if grep -q "DOWNLOAD_FAILED" "$DLOG" 2>/dev/null; then
    log "!!! 检测到 DOWNLOAD_FAILED, 守护脚本退出, 不转换"
    exit 1
  fi
  # 下载进程若已死且无 OK 标记, 也退出
  if ! pgrep -f "download-bf16-ms.sh" >/dev/null 2>&1; then
    # 给 curl 一点收尾时间, 再确认一次
    sleep 20
    if ! grep -q "DOWNLOAD_OK" "$DLOG" 2>/dev/null && ! pgrep -f "download-bf16-ms.sh" >/dev/null 2>&1; then
      log "!!! 下载进程已退出但无 DOWNLOAD_OK, 守护脚本退出"
      exit 1
    fi
  fi
  sleep 30
done

# 2) 校验 18 个分片齐全且非空
MISSING=0
for i in $(seq -w 1 18); do
  f="${DEST}/model-000${i}-of-00018.safetensors"
  if [ ! -s "$f" ]; then
    log "!!! 缺失或空文件: $f"
    MISSING=1
  fi
done
if [ "$MISSING" -ne 0 ]; then
  log "!!! 分片校验失败, 不转换"
  exit 1
fi
log "18 个分片校验通过"

# 3) 跑 CPU 转换
log "=== 开始 CPU 转换 ==="
log "bf16 源:   $DEST"
log "DFlash2:   $DFLASH2"
log "输出:      $OUTFILE"

docker run --rm \
  -v "${DEST}:/m:ro" \
  -v "${DFLASH2}:/d:ro" \
  -v "${OUTDIR}:/o" \
  -w /src \
  "$IMAGE" \
  python3 -m tools.convert.qwen3_8_27b.convert \
    --model /m \
    --dflash2-model /d \
    --out /o/qwen3_8_27b.ninfer \
    --device cpu 2>&1 | tee -a "$CLOG"

rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ] && [ -s "$OUTFILE" ]; then
  log "=== 转换成功 ==="
  ls -la "$OUTFILE"
  echo "CONVERT_OK" >> "$CLOG"
else
  log "!!! 转换失败 rc=$rc"
  echo "CONVERT_FAILED" >> "$CLOG"
  exit "$rc"
fi
