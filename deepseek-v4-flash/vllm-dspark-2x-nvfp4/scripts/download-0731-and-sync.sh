#!/usr/bin/env bash
# 定时下载 DeepSeek-V4-Flash-0731 正式版 -> 校验 -> 走 200G 光纤同步到 .12
# 所有已知坑都已规避（见 ~/ds4-0731.log）
set -u
MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
CACHE="$HOME/.cache/huggingface"
PY="$HOME/miniconda3/bin/python"
WORKER_FABRIC="10.0.0.3"
WORKER_MGMT="192.168.130.12"
SNAPDIR="$CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

log "=== 开始：下载 $MODEL ==="
df -h / | tail -1

# ---------- 1. 下载器（并发 8：24 会被 xet CDN 限流，6~10 是安全区） ----------
cat > "$HOME/dl-0731.py" <<'PYEOF'
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
p = snapshot_download("deepseek-ai/DeepSeek-V4-Flash-0731", max_workers=8)
print("SNAPSHOT_OK", p, flush=True)
PYEOF

# supervisor：进程静默死掉只表现为 du 不再增长，必须自动拉起
for attempt in $(seq 1 150); do
  log "--- 下载尝试 $attempt ---"
  "$PY" "$HOME/dl-0731.py" && { log "snapshot_download 报告完成"; break; }
  log "下载器退出，20s 后重试"
  sleep 20
done

# ---------- 2. 补 index.json（已知：这个 5.6MB 小文件极易反复失败） ----------
SNAP=$(ls -d "$SNAPDIR"/*/ 2>/dev/null | head -1)
if [ -z "$SNAP" ]; then log "!! 快照目录不存在，下载彻底失败"; exit 1; fi
if [ ! -f "${SNAP}model.safetensors.index.json" ]; then
  log "index.json 缺失，改从 modelscope 直取"
  curl -sL --max-time 300 -o /tmp/idx0731.json \
    "https://www.modelscope.cn/api/v1/models/${MODEL}/repo?Revision=master&FilePath=model.safetensors.index.json" \
    && cp /tmp/idx0731.json "${SNAP}model.safetensors.index.json" \
    && log "index.json 已补齐" || log "!! index.json 补齐失败"
fi

# ---------- 3. 清理 .incomplete 残留（重试会产生同 blob 的多份临时文件） ----------
BEFORE=$(du -sb "$CACHE" | cut -f1)
find "$CACHE" -name "*.incomplete" -delete 2>/dev/null
log "清理临时文件，回收 $(( (BEFORE - $(du -sb "$CACHE" | cut -f1)) / 1000000 )) MB"

# ---------- 4. 完整性校验：分片齐全 + safetensors 头部长度自洽 ----------
cat > /tmp/verify0731.py <<'PYEOF'
import json,struct,sys
from pathlib import Path
p=Path(sys.argv[1])
idx=p/"model.safetensors.index.json"
if not idx.exists(): print("VERIFY_FAIL 缺 index.json"); sys.exit(1)
d=json.loads(idx.read_text())
need=sorted(set(d["weight_map"].values()))
missing=[n for n in need if not (p/n).exists()]
if missing: print(f"VERIFY_FAIL 缺 {len(missing)} 个分片: {missing[:5]}"); sys.exit(1)
bad=[]
for n in need:
    f=(p/n).resolve()
    try:
        sz=f.stat().st_size
        with open(f,"rb") as fh:
            hl=struct.unpack("<Q",fh.read(8))[0]
            hdr=json.loads(fh.read(hl))
        end=max(v["data_offsets"][1] for k,v in hdr.items() if k!="__metadata__")
        if 8+hl+end != sz: bad.append(n)
    except Exception as e: bad.append(f"{n}:{e}")
if bad: print(f"VERIFY_FAIL 损坏 {len(bad)}: {bad[:5]}"); sys.exit(1)
print(f"VERIFY_OK 分片 {len(need)} 全部通过头部校验")
PYEOF
"$PY" /tmp/verify0731.py "$SNAP" || { log "!! 校验未通过，不同步到 worker"; exit 1; }
log "$("$PY" /tmp/verify0731.py "$SNAP")"

# ---------- 5. 同步到 .12（优先 200G 光纤；IP 是临时配的，重启会丢，先自愈） ----------
TARGET="$WORKER_FABRIC"
if ! ping -c1 -W2 "$WORKER_FABRIC" >/dev/null 2>&1; then
  log "光纤 IP 不通，尝试重新配置"
  sudo -n ip addr add 10.0.0.2/24 dev enp1s0f1np1 2>/dev/null
  sudo -n ip link set enp1s0f1np1 up 2>/dev/null
  ssh -o ConnectTimeout=10 "ai@$WORKER_MGMT" \
    "sudo -n ip addr add 10.0.0.3/24 dev enp1s0f1np1 2>/dev/null; sudo -n ip link set enp1s0f1np1 up" 2>/dev/null
  sleep 3
  ping -c1 -W2 "$WORKER_FABRIC" >/dev/null 2>&1 || { TARGET="$WORKER_MGMT"; log "光纤仍不通，退回管理网(慢 4 倍)"; }
fi
log "同步到 $TARGET ..."
rsync -a --info=progress2 \
  "$CACHE/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731" \
  "ai@$TARGET:$CACHE/hub/" && log "rsync 完成" || { log "!! rsync 失败"; exit 1; }

# ---------- 6. worker 侧同样校验 ----------
scp -q /tmp/verify0731.py "ai@$TARGET:/tmp/"
WSNAP=$(ssh "ai@$TARGET" "ls -d $SNAPDIR/*/ | head -1")
ssh "ai@$TARGET" "\$(ls ~/models/.msvenv/bin/python 2>/dev/null || which python3) /tmp/verify0731.py '$WSNAP'" \
  && log "worker 校验通过" || log "!! worker 校验失败"

log "=== 全部完成 ==="
log "下一步（需人工确认后执行）：改 .env.dspark 的 DSPARK_MODEL 指向 0731 + 挂载 Patch 4 + 重启"
