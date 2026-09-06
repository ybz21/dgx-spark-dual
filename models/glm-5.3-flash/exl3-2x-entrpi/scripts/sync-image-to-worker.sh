#!/usr/bin/env bash
# 把服务镜像从 head(.8) 推到 worker(.12)，走 200GbE 光缆。
# 让 worker 自己去外网拉要几个小时，走光缆几分钟。
set -euo pipefail
IMG=${IMG:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1}
PEER=${PEER:-ai@10.0.0.3}
docker save "$IMG" | ssh -o StrictHostKeyChecking=no "$PEER" docker load
ssh "$PEER" "docker image inspect '$IMG' --format 'worker: ARCH={{.Architecture}} SIZE={{.Size}}'"
