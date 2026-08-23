#!/bin/bash
# 三方案共享变量与函数
# Usage: source ./_common.sh
#
# 两种运行模式：
#   USE_SSH=1（默认）：从控制端 ssh 远程操作 130.12（依赖 frp 链路）
#   USE_SSH=0      ：直接在 130.12 本地跑（独立于 ssh，适合 overnight 运行）

set -uo pipefail

# ---------- 配置 ----------
SSH_HOST="${SSH_HOST:-115.190.152.1}"
SSH_PORT="${SSH_PORT:-39023}"
SSH_USER="${SSH_USER:-ai}"
SUDO_PW="${SUDO_PW:-bladeai}"
USE_SSH="${USE_SSH:-1}"

ALIYUN_PYPI="https://mirrors.aliyun.com/pypi/simple/"
IMAGE="192.168.130.23:5000/vllm-qwen35-v2:v26041616"
MODEL_HOST="/opt/box-deploy/models/host"
CACHE_HOST="${MODEL_HOST}/.vllm_cache"
PARSER_PY="/opt/box-deploy/models/llm-122b/qwen3_nothink_reasoning_parser_vllm.py"

SSH_OPTS=(-o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=20 -o TCPKeepAlive=yes)

RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../results}"
mkdir -p "$RESULTS_DIR"

# ---------- 工具函数 ----------
log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

remote() {
    # remote "shell command"
    if [ "$USE_SSH" = "0" ]; then
        bash -c "$1"
    else
        ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "$1"
    fi
}

remote_sudo() {
    if [ "$USE_SSH" = "0" ]; then
        echo "$SUDO_PW" | sudo -S bash -c "$1"
    else
        remote "echo $SUDO_PW | sudo -S sh -c $(printf '%q' "$1")"
    fi
}

# 把本地文件上传到目标机的 /tmp/<name>
upload_to_target() {
    local src="$1"
    local dst="$2"   # 目标路径（绝对）
    if [ "$USE_SSH" = "0" ]; then
        cp "$src" "$dst"
    else
        scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$src" "${SSH_USER}@${SSH_HOST}:$dst" >/dev/null
    fi
}

stop_122b() {
    log "Stop 122B production container"
    remote_sudo "cd /opt/box-deploy/models/llm-122b && docker compose down 2>&1 | tail -3" || true
}

start_122b() {
    log "Start 122B production container"
    remote_sudo "cd /opt/box-deploy/models/llm-122b && docker compose up -d 2>&1 | tail -3" || true
}

cleanup_test_container() {
    log "Remove test container"
    remote_sudo "docker rm -f vllm-qwen35-test 2>&1 | tail -1" || true
}

record_result() {
    local scheme="$1" boot_time="$2" oom="$3" healthy="$4" error="$5"
    local out="$RESULTS_DIR/$scheme"
    mkdir -p "$out"
    cat > "$out/result.json" <<EOF
{
  "scheme": "$scheme",
  "boot_time_s": ${boot_time:-null},
  "oom": $oom,
  "healthy": $healthy,
  "error": $(if [ "$error" = "null" ]; then echo null; else echo "\"$error\""; fi),
  "timestamp": "$(date -Iseconds)"
}
EOF
    log "result -> $out/result.json"
}

wait_test_container() {
    local timeout="${1:-1500}"
    remote "
        START=\$(date +%s)
        DEADLINE=\$((START + $timeout))
        while [ \$(date +%s) -lt \$DEADLINE ]; do
            if curl -sf -m 3 http://127.0.0.1:30001/health > /dev/null 2>&1; then
                END=\$(date +%s)
                echo \"BOOT_TIME=\$((END - START)) STATUS=healthy EXIT=0 OOM=false\"
                exit 0
            fi
            STATUS=\$(docker inspect vllm-qwen35-test --format '{{.State.Status}}' 2>/dev/null || echo missing)
            if [ \"\$STATUS\" = exited ]; then
                EXIT=\$(docker inspect vllm-qwen35-test --format '{{.State.ExitCode}}')
                OOM=\$(docker inspect vllm-qwen35-test --format '{{.State.OOMKilled}}')
                END=\$(date +%s)
                echo \"BOOT_TIME=\$((END - START)) STATUS=exited EXIT=\$EXIT OOM=\$OOM\"
                exit 0
            fi
            sleep 5
        done
        echo \"BOOT_TIME=$timeout STATUS=timeout EXIT=- OOM=-\"
    "
}

on_exit() {
    local rc=$?
    log "EXIT TRAP rc=$rc — cleaning up"
    cleanup_test_container || true
    start_122b || true
}
