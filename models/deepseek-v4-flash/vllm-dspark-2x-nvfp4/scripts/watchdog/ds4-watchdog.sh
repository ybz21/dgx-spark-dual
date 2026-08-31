#!/usr/bin/env bash
# DS4 双机看门狗：保证重启/换口/GID 漂移后无人值守自动恢复。
# 每台机器都装（systemd timer 每 2 分钟）。/etc/ds4-node.conf 里 ROLE=head|worker。
# 逻辑：
#  1) roce-autoconf 实测当前光口/HCA/GID（幂等，正常时秒回）
#  2) 容器里烧的 NCCL 参数 / VLLM_HOST_IP 与实测不符，或容器不存在 → 按本机实测值重建
#  3) 容器 Exited → 拉起（补 restart 策略之外的死角）
#  4) head 专属：容器已起 >15 分钟 API 仍不通 → 重启 worker 容器 + 重建自己（解一切组网卡死）
set -u
CONF=/etc/ds4-node.conf; [ -f "$CONF" ] || { echo "缺 $CONF"; exit 1; }
. "$CONF"   # ROLE=head|worker  [WORKER_SSH=ai@10.0.0.3]
/usr/local/sbin/roce-autoconf.sh >/tmp/ds4-roce-autoconf.last 2>&1 || true
. /etc/ds4-roce.env
C=ds4-dspark-2x-vllm-dspark-1
DIR=/home/ai/ds4-dspark-2x
log(){ echo "[ds4-watchdog $(date +%F_%T)] $*"; }

# 本机刚开机的第一跑：对端容器里还是抱着死连接的旧进程，主动 restart 对端一次，
# 让两边同时进入新的组网窗口（单边重启场景从"等 15 分钟卡死判定"提速到分钟级）。
UP=$(cut -d. -f1 /proc/uptime)
STAMP=/run/ds4-watchdog-boot-poked
if [ "$UP" -lt 300 ] && [ ! -f "$STAMP" ] && [ -n "${PEER_SSH:-}" ]; then
  log "开机首跑：restart 对端容器（$PEER_SSH）对齐组网窗口"
  ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$PEER_SSH" "docker restart $C" 2>&1 | tail -1
  touch "$STAMP"
fi

recreate(){
  log "按本机实测参数重建容器：IF=$ROCE_IFNAME HCA=$ROCE_HCA GID=$ROCE_GID_INDEX IP=$ROCE_LOCAL_IP"
  # compose recreate 与 restart 策略有竞态（残留 <hash>_name 容器 / 名字冲突），先硬清同名容器
  docker ps -aq --filter "name=ds4-dspark-2x-vllm-dspark" | xargs -r docker rm -f >/dev/null 2>&1
  if [ "$ROLE" = head ]; then NR=0; HL=""; else NR=1; HL=1; fi
  ( cd "$DIR" && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=$NR HEADLESS=$HL \
      HF_CACHE=/home/ai/.cache/huggingface VLLM_HOST_IP="$ROCE_LOCAL_IP" \
      NCCL_IB_GID_INDEX="$ROCE_GID_INDEX" NCCL_IB_HCA="$ROCE_HCA" NCCL_SOCKET_IFNAME="$ROCE_IFNAME" \
      docker compose --env-file .env.dspark -f docker-compose.dspark.yml up -d --force-recreate )
}

if ! docker inspect "$C" >/dev/null 2>&1; then recreate; exit 0; fi

# 服务健康时绝不动手（参数漂移只记录，等下次真出问题再随重建生效）。
# 2026-08-31 教训：GID 运行中漂移，已建立的 RDMA 连接不受影响，重建反而杀了好服务。
# 健康 = 真能推理（1 token 补全）。换口实测教训：光口挪走后 /v1/models 仍 200（假活），
# 推理挂死——探针必须打穿 RDMA 数据面。25s 超时给足首 token 余量。
probe_infer(){ curl -s -o /dev/null -w '%{http_code}' -m 25 -X POST "http://$1:${VLLM_PORT:-8888}/v1/chat/completions"   -H 'Content-Type: application/json'   -d '{"model":"'"${SERVED_MODEL_NAME:-deepseek-v4-flash}"'","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"temperature":0}' || true; }
healthy=0
if [ "$ROLE" = head ]; then
  [ "$(probe_infer 127.0.0.1)" = 200 ] && healthy=1
else
  [ "$(probe_infer "$ROCE_PEER_IP")" = 200 ] && healthy=1
fi

envof(){ docker inspect "$C" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n "s/^$1=//p" | head -1; }
mismatch=""
[ "$(envof NCCL_IB_GID_INDEX)" = "$ROCE_GID_INDEX" ] || mismatch="GID $(envof NCCL_IB_GID_INDEX)->$ROCE_GID_INDEX"
[ "$(envof NCCL_IB_HCA)" = "$ROCE_HCA" ] || mismatch="$mismatch HCA"
[ "$(envof NCCL_SOCKET_IFNAME)" = "$ROCE_IFNAME" ] || mismatch="$mismatch IFNAME"
[ "$(envof VLLM_HOST_IP)" = "$ROCE_LOCAL_IP" ] || mismatch="$mismatch HOST_IP"
[ "$(envof GLOO_SOCKET_IFNAME)" = "$ROCE_IFNAME" ] || mismatch="$mismatch GLOO"
if [ -n "$mismatch" ]; then
  if [ "$healthy" = 1 ]; then log "参数漂移（$mismatch）但服务健康，只记录不动手"; else log "容器参数过期（$mismatch）"; recreate; exit 0; fi
fi

state=$(docker inspect "$C" --format '{{.State.Status}}')
[ "$state" = running ] || { log "容器 $state，拉起"; docker start "$C" >/dev/null; exit 0; }

# —— 窗口纪元对齐（head 专属，根治"两边组网窗口错位互相干等"）——
# head 容器每次换新实例（任何原因的重启）都是一个新的 10 分钟集合窗口。检测到新纪元且
# 该实例尚未完成组网（日志无 Init COMPLETE）时，立刻重启 worker 进同一窗口：2 分钟内
# 必对齐，窗口余量 ≥8 分钟。已组网/加载中的实例绝不打扰。
if [ "$ROLE" = head ] && [ -n "${WORKER_SSH:-}" ]; then
  EPOCH_FILE=/run/ds4-head-epoch
  cur_epoch=$(docker inspect "$C" --format '{{.State.StartedAt}}')
  if [ "$cur_epoch" != "$(cat "$EPOCH_FILE" 2>/dev/null)" ]; then
    echo "$cur_epoch" > "$EPOCH_FILE"
    if ! docker logs --since "$cur_epoch" "$C" 2>&1 | grep -aq "Init COMPLETE"; then
      log "head 新窗口（$cur_epoch）且未组网：同步重启 worker 对齐"
      ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$WORKER_SSH" "docker restart $C" 2>&1 | tail -1
    fi
  fi
fi

if [ "$ROLE" = head ]; then
  [ "$healthy" = 1 ] && exit 0
  started=$(docker inspect "$C" --format '{{.State.StartedAt}}')
  age=$(( $(date +%s) - $(date -d "$started" +%s) ))
  if [ "$age" -gt "${STUCK_SEC:-900}" ]; then
    log "API 不通且容器已跑 ${age}s：判定卡死，清两端 kernel 缓存、重启 worker、重建自己"
    # 卡死路径连缓存一起清（两端）：旧 vllm-cache + cudagraph 捕获会触发 Triton not-permitted 循环
    rm -rf /home/ai/.cache/huggingface/vllm-cache
    [ -n "${WORKER_SSH:-}" ] && ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 "$WORKER_SSH" "rm -rf ~/.cache/huggingface/vllm-cache; docker restart $C" 2>&1 | tail -1
    recreate
  else
    log "API 未就绪（容器 ${age}s，冷启/组网窗口内），继续观察"
  fi
fi
