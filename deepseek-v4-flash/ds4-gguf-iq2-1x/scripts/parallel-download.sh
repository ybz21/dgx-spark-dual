#!/usr/bin/env bash
# parallel-download.sh — 无依赖的多连接分片下载器（纯 curl/bash）。
# 用途：从 hf-mirror.com 拉 DeepSeek-V4-Flash 的 GGUF，单流慢时用 16 连接把
#       聚合速度从 ~4MB/s 提到 ~14MB/s；断点续传（每分片 curl -r range 追加）。
#
# 用法:  bash parallel-download.sh
#        下完后:  bash install.sh --no-download --start
set -uo pipefail
GDIR="${DS4_GGUF_DIR:-$HOME/gguf}"; mkdir -p "$GDIR"
log(){ printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

dl() {
  local repo="$1" file="$2" nchunks="${3:-16}"
  local url="https://hf-mirror.com/$repo/resolve/main/$file"
  local out="$GDIR/$file"
  local size
  size=$(curl -sIL --max-time 40 "$url" | awk -F': ' 'tolower($1)=="content-length"{v=$2+0} END{print v}')
  if [ -z "$size" ] || [ "$size" -le 0 ]; then log "[$file] 取大小失败"; return 1; fi
  log "[$file] size=$((size/1048576)) MB"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then log "[$file] 已完整"; return 0; fi
  local chunk=$(( (size + nchunks - 1) / nchunks )) pids=() i
  for i in $(seq 0 $((nchunks-1))); do
    local start=$(( i * chunk )); [ "$start" -ge "$size" ] && break
    local end=$(( start + chunk - 1 )); [ "$end" -ge "$size" ] && end=$((size-1))
    local part="$out.part$i"
    ( local want=$(( end - start + 1 ))
      while :; do
        local have=0; [ -f "$part" ] && have=$(stat -c%s "$part")
        [ "$have" -ge "$want" ] && break
        curl -sL -r $((start+have))-${end} "$url" >> "$part" 2>/dev/null
        sleep 1
      done ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  : > "$out"
  for i in $(seq 0 $((nchunks-1))); do
    local start=$(( i * chunk )); [ "$start" -ge "$size" ] && break
    cat "$out.part$i" >> "$out"
  done
  local final; final=$(stat -c%s "$out")
  if [ "$final" = "$size" ]; then rm -f "$out".part*; log "[$file] DONE"; return 0
  else log "[$file] 大小不符 $final != $size"; return 1; fi
}

dl antirez/deepseek-v4-gguf "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" 16
dl antirez/deepseek-v4-gguf "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf" 8
dl bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF "DSpark-drafter-Q2K-Q8.gguf" 8
log "全部完成"
