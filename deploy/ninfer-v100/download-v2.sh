#!/usr/bin/env bash
# 下载 v2 制品 (NINFER\x00\x02) 并校验、软链
# 按 deploy-v100 项目设计
set -uo pipefail

MODEL_DIR="/data/models/Qwen3.8-27B NVFP4"
LINK_DIR="/data/models/Qwen3.8-27B-NVFP4"
FILE="qwen3_8_27b_nvfp4.ninfer"
URL="https://hf-mirror.com/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/52907138a5d23a8f7f868ba7b773e721fd275405/${FILE}"
EXPECTED_SHA="552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c"
LOG="/data/models/ninfer-v100/download-v2.log"

mkdir -p "$MODEL_DIR" "$LINK_DIR"
cd "$MODEL_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 开始下载 v2 制品 ==="
log "URL: $URL"
log "目标: $MODEL_DIR/$FILE"

# 断点续传下载 (curl -C - 自动续传)
for attempt in 1 2 3 4 5; do
  log "下载尝试 #$attempt ..."
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors \
       --connect-timeout 30 --max-time 7200 \
       -o "$FILE" "$URL" 2>>"$LOG"
  rc=$?
  if [ $rc -eq 0 ]; then
    log "curl 完成 (rc=0)"
    break
  fi
  log "curl 失败 rc=$rc，60s 后续传..."
  sleep 60
done

# 校验大小
SIZE=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
log "文件大小: $SIZE 字节"

# 校验 magic (v2)
MAGIC=$(head -c 8 "$FILE" 2>/dev/null | xxd -p)
log "文件头: $MAGIC (期望 4e494e4645520002)"
if [ "$MAGIC" != "4e494e4645520002" ]; then
  log "!!! 文件头不是 v2，下载可能不完整或错误"
fi

# 校验 SHA256
log "计算 SHA256 (约需 1-2 分钟)..."
ACTUAL_SHA=$(sha256sum "$FILE" 2>/dev/null | awk '{print $1}')
log "实际 SHA256: $ACTUAL_SHA"
log "期望 SHA256: $EXPECTED_SHA"

if [ "$ACTUAL_SHA" = "$EXPECTED_SHA" ]; then
  log "=== SHA256 校验通过 ==="
  # 软链
  ln -sfn "$MODEL_DIR/$FILE" "$LINK_DIR/current.ninfer"
  log "软链已建立: $LINK_DIR/current.ninfer -> $MODEL_DIR/$FILE"
  log "=== 下载+校验+软链 全部完成 ==="
  echo "DOWNLOAD_OK" >> "$LOG"
else
  log "!!! SHA256 不匹配，请检查"
  echo "DOWNLOAD_SHA_MISMATCH" >> "$LOG"
fi
