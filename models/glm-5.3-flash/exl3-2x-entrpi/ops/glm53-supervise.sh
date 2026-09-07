#!/usr/bin/env bash
# GLM-5.3-Flash 双节点看门狗 / 编排器。跑在 head 上，worker 侧只需要 rail 服务。
#
# 为什么编排全放 head：启动顺序是硬约束——必须先删 head 容器、再起 worker、
# 最后起 head（新 worker 会跟旧 head 的 TCP store 会合，旧 head 一走就 reset 而死），
# 而且 worker->head 的间隔要小于 torch 的 600 秒 rendezvous 超时。
# 把顺序逻辑放在一个地方，比两台各自 systemd 互相 After= 要可靠得多。
#
# 覆盖的故障：
#   - 整机重启（systemd 拉起本脚本，走一次完整有序启动）
#   - worker 单独崩溃/重启（health 探到异常 -> 整体有序重启）
#   - head 容器崩溃（同上）
#   - 光缆换口（每轮重启前重跑 rail 脚本，重新探测并写入 RAIL_IF/HCA）
set -u
CFG=${CFG:-/etc/glm53-ops.env}
. "$CFG"
LAUNCH=${LAUNCH:-$HOME/launch-glm53-vllm-tp2.sh}
RAIL=${RAIL:-/usr/local/sbin/glm53-rail.sh}
INTERVAL=${INTERVAL:-30}          # 健康检查间隔
PROBE_TIMEOUT=${PROBE_TIMEOUT:-60} # 生成探针超时；负载重时 1 token 也可能慢
MODEL_ID=${MODEL_ID:-glm-5.3-flash}
BOOT_GRACE=${BOOT_GRACE:-900}     # 一次启动最多等多久到 healthy（1M 档要编 kernel）
FAIL_LIMIT=${FAIL_LIMIT:-3}       # 连续几次不健康才动手，避免抖动就重启
backoff=60                        # 重启失败后的退避，指数增长封顶 600

log(){ echo "$(date -Is) [sup] $*"; }
peer(){ ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
        "$SSH_USER@$WORKER_RAIL_IP" "$@"; }
knobs(){ echo "KV_DTYPE=$PROFILE_KV_DTYPE VLLM_NVFP4_MLA_DYNAMIC_SCALE=$PROFILE_NVFP4_DYNAMIC_SCALE"\
              "MAX_LEN=$PROFILE_MAX_LEN MNBT=$PROFILE_MNBT"\
              "KV_CACHE_MEMORY=$PROFILE_KV_CACHE_MEMORY MEM_USED_MAX_GB=$PROFILE_MEM_USED_MAX_GB"; }
# 活性判据：必须真发一次生成请求。
# 实测过的坑：worker 容器被 kill 之后，head 的 /health 依然稳定返回 200，
# 而任何真实请求都会挂死（60s 超时 http=000）。拿 /health 当判据等于没装看门狗。
# 先做一次极便宜的结构检查（worker 容器在不在），命中就直接判失败，
# 不用等探针超时，故障发现能快一个数量级。
healthy(){
  peer 'docker ps -q --filter name=vllm_glm53 --filter status=running' 2>/dev/null | grep -q . || return 1
  docker ps -q --filter name=vllm_glm53 --filter status=running | grep -q . || return 1
  curl -s -m "$PROBE_TIMEOUT" "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":1,\"temperature\":0}" \
    2>/dev/null | grep -q '"choices"'
}

restart_cluster(){
  log "开始有序重启"
  # 0) 光缆可能换过口，重新探测
  sudo -n "$RAIL" || log "rail 脚本返回非零，继续尝试"
  # 1) 先删 head —— 顺序不能反
  docker rm -f vllm_glm53 >/dev/null 2>&1
  # 2) 再删 worker
  peer 'docker rm -f vllm_glm53' >/dev/null 2>&1 || log "worker 暂时连不上（可能在重启），继续"
  # 3) 等 worker 侧 SSH 回来，最多 5 分钟
  for i in $(seq 1 30); do peer true >/dev/null 2>&1 && break; sleep 10; done
  peer true >/dev/null 2>&1 || { log "worker 不可达，本轮放弃"; return 1; }
  # 4) 起 worker
  log "起 worker (rank 1)"
  peer "env $(knobs) ~/launch-glm53-vllm-tp2.sh 1" || { log "worker 启动失败"; return 1; }
  # 5) 立刻起 head —— 间隔必须 < 600s rendezvous 超时
  log "起 head (rank 0)"
  env $(knobs) "$LAUNCH" 0 || { log "head 启动失败"; return 1; }
  # 6) 等健康
  for i in $(seq 1 $((BOOT_GRACE/10))); do
    healthy && { log "healthy，预热"; "$HOME/glm53-warmup.sh" >/dev/null 2>&1; return 0; }
    docker ps -q --filter name=vllm_glm53 | grep -q . || { log "head 容器退出了"; return 1; }
    sleep 10
  done
  log "超过 ${BOOT_GRACE}s 仍未 healthy"; return 1
}

log "supervisor 启动: profile MAX_LEN=$PROFILE_MAX_LEN KV=$PROFILE_KV_DTYPE"
fails=0
while :; do
  if healthy; then
    [ "$fails" -gt 0 ] && log "恢复正常"
    fails=0; backoff=60
  else
    fails=$((fails+1))
    log "健康检查失败 $fails/$FAIL_LIMIT"
    if [ "$fails" -ge "$FAIL_LIMIT" ]; then
      if restart_cluster; then
        log "重启成功"; fails=0; backoff=60
      else
        log "重启失败，退避 ${backoff}s"; sleep "$backoff"
        backoff=$(( backoff*2 > 600 ? 600 : backoff*2 ))
      fi
    fi
  fi
  sleep "$INTERVAL"
done
