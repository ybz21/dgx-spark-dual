#!/bin/bash
# 方案 A: runai_streamer
# 改 --load-format 即可，不需要预生成 cache
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

trap on_exit EXIT

SCHEME="A"
log "=== Scheme A: runai_streamer ==="

# ---------- wrapper script (上传到目标机 /tmp) ----------
WRAPPER=/tmp/run-A-runai.sh
cat > "$WRAPPER" <<EOF
#!/bin/bash
set -e
pip install -q -i $ALIYUN_PYPI runai-model-streamer 2>&1 | tail -2 || \
  pip install -q -i $ALIYUN_PYPI 'vllm[runai]' 2>&1 | tail -2 || true
exec /opt/entrypoint.sh \\
  serve /models/Qwen3.5-122B-A10B-int4-AutoRound-Intel \\
  --served-model-name qwen3.5-122b-int4 \\
  --port 30001 \\
  --max-model-len 131072 \\
  --gpu-memory-utilization 0.80 \\
  --reasoning-parser-plugin /custom/qwen3_nothink_reasoning_parser_vllm.py \\
  --reasoning-parser qwen3_nothink \\
  --attention-backend FLASHINFER \\
  --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder \\
  --trust-remote-code \\
  --cudagraph-capture-sizes 1 8 32 64 128 256 384 512 \\
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \\
  --load-format runai_streamer \\
  --model-loader-extra-config '{"concurrency":1,"memory_limit":4294967296}'
EOF
chmod +x "$WRAPPER"
upload_to_target "$WRAPPER" "/tmp/run-A.sh"

# ---------- 准备：停 122B + 清测试容器 ----------
stop_122b
cleanup_test_container

# ---------- 启动测试容器 ----------
log "Starting test container with --load-format runai_streamer"
remote_sudo "
    chmod +x /tmp/run-A.sh
    docker run -d --name vllm-qwen35-test --gpus all --ipc=host -p 30001:30001 \
      -v $MODEL_HOST:/models \
      -v $CACHE_HOST:/cache \
      -v $PARSER_PY:/custom/qwen3_nothink_reasoning_parser_vllm.py:ro \
      -v /tmp/run-A.sh:/run.sh:ro \
      -e VLLM_CACHE_ROOT=/cache \
      -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
      -e TRITON_CACHE_DIR=/cache/triton \
      --entrypoint /run.sh \
      $IMAGE
" || { record_result "$SCHEME" "" "false" "false" "container_start_failed"; exit 1; }

# ---------- 等结果 ----------
log "Waiting for healthy or exit (max 25 min)"
RESULT=$(wait_test_container 1500)
log "$RESULT"

eval "$RESULT"  # BOOT_TIME, STATUS, EXIT, OOM 注入到当前 shell

# ---------- 抓日志 ----------
mkdir -p "$RESULTS_DIR/$SCHEME"
remote "docker logs vllm-qwen35-test 2>&1" > "$RESULTS_DIR/$SCHEME/boot.log" 2>&1 || true
remote "docker inspect vllm-qwen35-test 2>&1" > "$RESULTS_DIR/$SCHEME/inspect.json" 2>&1 || true

# ---------- 记结果 ----------
HEALTHY=false
ERROR="null"
case "$STATUS" in
    healthy) HEALTHY=true ;;
    exited)  ERROR="exit_$EXIT" ;;
    timeout) ERROR="timeout" ;;
    *)       ERROR="unknown_$STATUS" ;;
esac
record_result "$SCHEME" "$BOOT_TIME" "${OOM:-false}" "$HEALTHY" "$ERROR"

log "Done $SCHEME: BOOT_TIME=$BOOT_TIME STATUS=$STATUS OOM=${OOM:-false}"
