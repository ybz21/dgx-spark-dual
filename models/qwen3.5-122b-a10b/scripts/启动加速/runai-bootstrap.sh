#!/bin/bash
# 122B vLLM 启动 wrapper：
#   1. 装 runai-model-streamer（vLLM 镜像默认未装）
#      → 启用 --load-format runai_streamer 才能并行 stream 权重
#      → 实测把 weight load 从 405s 降到 165s，整体启动 12.5min → 4min
#   2. 装 boto3：runai 内部 stream 流框架依赖
#   3. exec 原 entrypoint 跑 vllm serve
#
# 详见 docs/models/启动加速-PoC/A-runai-streamer.md
set -e
ALIYUN_PYPI=https://mirrors.aliyun.com/pypi/simple/
pip install --quiet --index-url "$ALIYUN_PYPI" \
    runai-model-streamer boto3 2>&1 | tail -3
exec /opt/entrypoint.sh "$@"
