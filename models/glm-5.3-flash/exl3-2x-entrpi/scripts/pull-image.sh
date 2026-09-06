#!/usr/bin/env bash
# 拉 Entrpi 服务镜像（~18 GB，arm64）。
#
# 为什么不直接让 install.sh 拉：这两台 Spark 到 ghcr.io / ghcr.nju.edu.cn /
# ghcr.dockerproxy.net 实测都只有 0.1 MB/s，install.sh 会在 pull 阶段卡死
# （实测跑了 50 分钟，19 层拉完、4 层无限重试）。ghcr.chenby.cn 是目前唯一
# 快的：docker 多层并行实测 10.13 MiB/s，约 40 分钟。
#
# 拉完 retag 成 ghcr.io/... 让 launcher 的默认镜像名本地命中，不用改 .env。
# 之后用 sync-image-to-worker.sh 走光缆把镜像推给 worker，比让 worker 自己
# 去外网拉快得多。
set -u
SRC=${SRC:-ghcr.chenby.cn/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1}
DST=${DST:-ghcr.io/entrpi/glm-5.3-flash-exl3-2x-spark:v2.3-tier1}
log=${LOG:-$HOME/glm53-img.log}
: > "$log"
# 外层重试：单次 unexpected EOF 会把 docker pull 整个带走，重进时已完成的层会复用
for i in $(seq 1 200); do
  echo "=== attempt $i $(date -Is) ===" >> "$log"
  if docker pull "$SRC" >> "$log" 2>&1; then
    docker tag "$SRC" "$DST" && echo "=== TAGGED $(date -Is) ===" >> "$log"
    # 本机若是 x86_64 中转，务必确认这行是 arm64，否则拉到的是废件
    docker image inspect "$DST" --format '=== ARCH={{.Architecture}} OS={{.Os}} SIZE={{.Size}} ===' | tee -a "$log"
    echo "=== COMPLETE ===" >> "$log"; exit 0
  fi
  sleep 10
done
echo "=== GAVE UP ===" >> "$log"; exit 1
