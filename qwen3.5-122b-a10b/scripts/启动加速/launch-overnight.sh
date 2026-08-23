#!/bin/bash
# 把 PoC 目录传到 130.12，用 nohup + setsid 启动 run-all.sh，独立于控制端 ssh 链路
# 之后 ssh 链路断不影响后台进程。明早回来 ssh 看 results/
set -euo pipefail

SSH_HOST="${SSH_HOST:-115.190.152.1}"
SSH_PORT="${SSH_PORT:-39023}"
SSH_USER="${SSH_USER:-ai}"
SSH_OPTS=(-o ServerAliveInterval=15 -o ConnectTimeout=20 -o TCPKeepAlive=yes)

REMOTE_DIR=/opt/box-deploy/poc-startup
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[launch-overnight] uploading PoC dir → $REMOTE_DIR"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "echo bladeai | sudo -S mkdir -p $REMOTE_DIR && echo bladeai | sudo -S chown ai:ai $REMOTE_DIR"

# rsync 整个 PoC 目录（含 docs + scripts）
rsync -e "ssh ${SSH_OPTS[*]} -p $SSH_PORT" -av --delete \
  "$LOCAL_DIR/" \
  "${SSH_USER}@${SSH_HOST}:${REMOTE_DIR}/"

# 远程启动
echo "[launch-overnight] starting run-all.sh on target via setsid + nohup"
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "
    chmod +x ${REMOTE_DIR}/scripts/*.sh
    cd ${REMOTE_DIR}
    # 杀掉前面可能遗留的 run-all
    pkill -f 'run-all.sh' 2>/dev/null || true
    pkill -f 'A-runai.sh\\|B-shard.sh\\|C-custom.sh' 2>/dev/null || true
    # 启动新 run，setsid 让它脱离 controlling tty，nohup 抗 SIGHUP
    setsid nohup env USE_SSH=0 RESULTS_DIR=${REMOTE_DIR}/results \\
        ${REMOTE_DIR}/scripts/run-all.sh \\
        > ${REMOTE_DIR}/run-all.log 2>&1 < /dev/null &
    sleep 2
    echo '[launch-overnight] PID(s):'
    pgrep -af 'run-all.sh|A-runai.sh|B-shard.sh|C-custom.sh' | head -5
"

echo
echo "================================================="
echo "  启动成功。明早回来看："
echo "  ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
echo "  cat $REMOTE_DIR/results/A/result.json"
echo "  cat $REMOTE_DIR/results/B/result.json"
echo "  cat $REMOTE_DIR/results/C/result.json"
echo
echo "  实时进度："
echo "  tail -f $REMOTE_DIR/run-all.log"
echo "================================================="
