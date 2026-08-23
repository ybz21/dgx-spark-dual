#!/usr/bin/env bash
# download.sh — 从 ModelScope 下载 Laguna-S-2.1-NVFP4 主模型 + DFlash 草稿模型。
# 用独立 venv 装 modelscope，避免动系统 Python（部分机器 apt 依赖冲突）。
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
VENV="$MODELS_DIR/.msvenv"

mkdir -p "$MODELS_DIR"
if [ ! -x "$VENV/bin/modelscope" ]; then
  echo "[*] 创建 venv 并安装 modelscope ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install -U pip modelscope
fi
MS="$VENV/bin/modelscope"

echo "[*] 下载 DFlash 草稿模型 (~2.2 GB) ..."
"$MS" download --model poolside/Laguna-S-2.1-DFlash-NVFP4 \
  --local_dir "$MODELS_DIR/Laguna-S-2.1-DFlash-NVFP4"

echo "[*] 下载主模型 NVFP4 (~72 GB, 15 分片) ..."
"$MS" download --model poolside/Laguna-S-2.1-NVFP4 \
  --local_dir "$MODELS_DIR/Laguna-S-2.1-NVFP4"

echo "[✓] 完成。模型在 $MODELS_DIR/{Laguna-S-2.1-NVFP4, Laguna-S-2.1-DFlash-NVFP4}"
