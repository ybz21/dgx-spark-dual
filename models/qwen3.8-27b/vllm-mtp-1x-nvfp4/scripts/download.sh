#!/usr/bin/env bash
# download.sh — 从 ModelScope（国内 CDN）下载 Qwen3.8-27B-NVFP4 主模型（含内嵌 MTP 头）。
# 单一 checkpoint，~23 GB，无需单独草稿模型。
#
# ⚠️ 为什么用 ModelScope 而不是 hf-mirror：
#    该权重的大文件 model.safetensors(22.6GB) 存在 HuggingFace 的 Xet CDN
#    (us.aws.cdn.hf.co)。hf-mirror 不代理 Xet，会 302 到美国 AWS CDN，国内拉大文件
#    必断连 + 预签名过期(403)，下不完。ModelScope 有国内 CDN，可稳定拉取。
# ⚠️ 若目标机外网被限速(box010 实测 ~245KB/s)，改在外网快的机器下好再 rsync 到
#    目标机 ~/models/（本仓库 quick-start.sh 亦是此路线）。
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
VENV="$MODELS_DIR/.msvenv"
mkdir -p "$MODELS_DIR"
if [ ! -x "$VENV/bin/modelscope" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install -U pip modelscope
fi
"$VENV/bin/modelscope" download --model unsloth/Qwen3.8-27B-NVFP4 \
  --local_dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
echo "[✓] 主模型在 $MODELS_DIR/Qwen3.8-27B-NVFP4"
