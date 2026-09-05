#!/usr/bin/env bash
# 把 head(.8) 上的 EXL3 权重推给 worker(.12)，走 200GbE 光缆。
# 分 6 路并行是因为单条 ssh 流会先卡在加解密的单核上，远到不了光缆带宽。
# 实测 164 GiB / 96 s ≈ 1.7 GB/s。
set -euo pipefail
d="${MODEL_DIR:-$HOME/models/GLM-5.3-Flash-EXL3-TR3-4bpw}"
peer="${PEER:-ai@10.0.0.3}"
cd "$d"
ssh -o StrictHostKeyChecking=no "$peer" "mkdir -p '$d'"
# 先推元数据和配置，这样半途中断时一眼能看出来
rsync -a --partial --exclude 'model-*.safetensors' --exclude '._____temp' \
      --exclude '.msc' --exclude '.mv' ./ "$peer:$d/"
ls model-*-of-00120.safetensors | xargs -P 6 -I{} rsync -a --partial {} "$peer:$d/{}"
ssh "$peer" "echo -n 'worker shards: '; ls '$d'/model-*-of-00120.safetensors | wc -l"
