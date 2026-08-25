#!/usr/bin/env bash
# download.sh — 从 ModelScope（国内 CDN）下载 SGLang/DSpark 方案所需两份权重：
#   主模型 unsloth/Qwen3.8-27B-NVFP4    (~23 GB，与 vLLM 方案共用同一份)
#   草稿模型 RadixArk/Qwen3.8-27B-DSpark (~2.7 GB，DSpark 投机解码用)
#
# ⚠️ 为什么用 ModelScope：大权重存 HF Xet CDN，hf-mirror 不代理，国内下不完。详见
#    ../../vllm-mtp-1x-nvfp4/scripts/download.sh 注释。外网慢就异地下好再 rsync。
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
VENV="$MODELS_DIR/.msvenv"
mkdir -p "$MODELS_DIR"
if [ ! -x "$VENV/bin/modelscope" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install -U pip modelscope
fi
MS="$VENV/bin/modelscope"
echo "[*] 主模型 unsloth/Qwen3.8-27B-NVFP4"
"$MS" download --model unsloth/Qwen3.8-27B-NVFP4    --local_dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
echo "[*] 草稿模型 RadixArk/Qwen3.8-27B-DSpark"
"$MS" download --model RadixArk/Qwen3.8-27B-DSpark  --local_dir "$MODELS_DIR/RadixArk-Qwen3.8-27B-DSpark"
echo "[✓] 权重在 $MODELS_DIR/{Qwen3.8-27B-NVFP4, RadixArk-Qwen3.8-27B-DSpark}"
