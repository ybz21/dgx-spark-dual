#!/usr/bin/env bash
# 从 ModelScope 拉 GLM-5.3-Flash EXL3 4bpw 主模型 + DFlash2 MXFP8 drafter。
# 在目标机上跑（head 或 worker 都行）。主模型 164 GiB，一台拉完之后建议用
# sync-to-worker.sh 走光缆推给另一台，比两台各拉一次 ModelScope 快得多。
set -euo pipefail
MS=${MS:-$HOME/.local/bin/modelscope}   # tmux/非交互 shell 里 PATH 上没有它
MODELS=${MODELS:-$HOME/models}

"$MS" download --model Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw \
  --local_dir "$MODELS/GLM-5.3-Flash-EXL3-TR3-4bpw"

"$MS" download --model local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8 \
  --local_dir "$MODELS/glm53-dflash2-mxfp8"

echo "shards: $(ls "$MODELS"/GLM-5.3-Flash-EXL3-TR3-4bpw/model-*-of-00120.safetensors | wc -l)/120"
