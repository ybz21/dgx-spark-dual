#!/bin/bash
# 方案 C: 手写 custom dump + custom loader
# 1. 跑一次完整 vLLM 启动（等 healthy），让模型走完 INC dispatch + Marlin repack + warmup + capture
# 2. 在 GPU 上的最终 state_dict 按 transformer block 拆分 dump 到 N 个 .pt 文件
# 3. 二次启动用 custom_layer_pt loader：跳过 INC + repack，直接 torch.load 每个 layer.pt → setattr
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

trap on_exit EXIT

SCHEME="C"
DUMP_DIR="$CACHE_HOST/layer-pt"

log "=== Scheme C: custom dump + custom load ==="

# ---------- 准备 custom_layer_pt loader patch ----------
LOADER_PATCH=/tmp/custom_layer_pt_patch.py
cat > "$LOADER_PATCH" <<'PYEOF'
"""注册一个 custom load-format: custom_layer_pt
跳过 INC dispatch + Marlin repack（因为 dump 是 repack 后的 state）。
"""
import gc
import os
from pathlib import Path

import torch
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader import get_model_loader as _orig_get_loader


class CustomLayerPtLoader(BaseModelLoader):
    """从 N 个 layer-{i:03d}.pt + top.pt 加载已 repack 的 state."""
    def __init__(self, load_config):
        super().__init__(load_config)
        cfg = load_config.model_loader_extra_config or {}
        self.dump_dir = Path(cfg["dump_dir"])

    def download_model(self, model_config):
        pass

    def load_weights(self, model, model_config):
        # 找 transformer blocks
        if hasattr(model, "language_model"):
            blocks = model.language_model.model.layers
        else:
            blocks = model.model.layers

        # 加载 top
        top_path = self.dump_dir / "top.pt"
        if top_path.exists():
            print(f"[custom_layer_pt] loading {top_path}", flush=True)
            top = torch.load(top_path, map_location="cuda:0")
            model.load_state_dict(top, strict=False, assign=True)
            del top
            gc.collect(); torch.cuda.empty_cache()

        # 按 layer 加载
        for i, block in enumerate(blocks):
            layer_path = self.dump_dir / f"layer-{i:03d}.pt"
            print(f"[custom_layer_pt] loading {layer_path.name}", flush=True)
            sd = torch.load(layer_path, map_location="cuda:0")
            block.load_state_dict(sd, strict=False, assign=True)
            del sd
            gc.collect(); torch.cuda.empty_cache()

        # 标记 model 已经是 repack 状态，让 process_weights_after_loading 跳过
        for m in model.modules():
            m._blade_skipped_repack = True
        print(f"[custom_layer_pt] all layers loaded, _blade_skipped_repack set", flush=True)

    def load_model(self, vllm_config, model_config, prefix=""):
        from vllm.model_executor.model_loader.utils import set_default_torch_dtype, initialize_model
        device_config = vllm_config.device_config
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
        self.load_weights(model, model_config)
        return model


# 注册 LoadFormat 字符串映射
from vllm.config.load import LoadFormat
LoadFormat.CUSTOM_LAYER_PT = "custom_layer_pt"

import vllm.model_executor.model_loader as ml_mod
_orig_factory = ml_mod.get_model_loader

def _patched_get_loader(load_config):
    if str(load_config.load_format).lower() in ("custom_layer_pt", "loadformat.custom_layer_pt"):
        return CustomLayerPtLoader(load_config)
    return _orig_factory(load_config)

ml_mod.get_model_loader = _patched_get_loader
print("[custom_layer_pt_patch] CustomLayerPtLoader registered", flush=True)


# patch process_weights_after_loading：检测到 _blade_skipped_repack 跳过
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod, GPTQMarlinMoEMethod,
)

for klass in [GPTQMarlinLinearMethod, GPTQMarlinMoEMethod]:
    _orig = klass.process_weights_after_loading
    def make_wrapper(orig):
        def wrapper(self, layer):
            if getattr(layer, "_blade_skipped_repack", False):
                return
            return orig(self, layer)
        return wrapper
    klass.process_weights_after_loading = make_wrapper(_orig)
print("[custom_layer_pt_patch] process_weights_after_loading patched to honor _blade_skipped_repack", flush=True)
PYEOF

# ---------- 准备 dump patch（hook 到正常启动后 dump state_dict）----------
DUMP_PATCH=/tmp/dump_layer_pt_patch.py
cat > "$DUMP_PATCH" <<'PYEOF'
"""注入 GPUWorker 的 dump_layer_pt：跑完正常加载后，按 transformer block 拆分 dump."""
from pathlib import Path
import torch
from vllm.v1.worker.gpu_worker import Worker


def dump_layer_pt(self, dump_dir: str):
    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = self.model_runner.model

    if hasattr(model, "language_model"):
        blocks = model.language_model.model.layers
        block_attr = "language_model.model.layers"
    else:
        blocks = model.model.layers
        block_attr = "model.layers"

    # 顶层 state_dict 减去 blocks
    full_sd = model.state_dict()
    block_prefixes = [f"{block_attr}.{i}." for i in range(len(blocks))]
    top_sd = {k: v for k, v in full_sd.items() if not any(k.startswith(p) for p in block_prefixes)}
    torch.save(top_sd, out / "top.pt")
    print(f"[dump_layer_pt] saved top.pt ({len(top_sd)} tensors)", flush=True)

    # 每个 block
    for i, block in enumerate(blocks):
        sd = block.state_dict()
        torch.save(sd, out / f"layer-{i:03d}.pt")
        print(f"[dump_layer_pt] saved layer-{i:03d}.pt ({len(sd)} tensors)", flush=True)

    print(f"[dump_layer_pt] all {len(blocks)+1} files written under {out}", flush=True)


Worker.dump_layer_pt = dump_layer_pt
print("[dump_layer_pt_patch] Worker.dump_layer_pt registered", flush=True)
PYEOF

# ---------- Phase 1 wrapper：跑 dump ----------
WRAPPER_DUMP=/tmp/run-C-dump.sh
cat > "$WRAPPER_DUMP" <<EOF
#!/bin/bash
set -e
mkdir -p /tmp/blade_patches
cp /dump_layer_pt_patch.py /tmp/blade_patches/
export PYTHONPATH=/tmp/blade_patches:\${PYTHONPATH:-}
mkdir -p $DUMP_DIR

# 先 import patch（让 spawn 子进程 sys.path 包含 /tmp/blade_patches）
python3 -u -c "import dump_layer_pt_patch"

# 跑一个简短的 vLLM 启动 → dump → 退出
python3 -u <<PYRUN
import os
import dump_layer_pt_patch  # 注入 hook

from vllm import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine

engine_args = EngineArgs(
    model="/models/Qwen3.5-122B-A10B-int4-AutoRound-Intel",
    served_model_name=["qwen3.5-122b-int4"],
    max_model_len=131072,
    gpu_memory_utilization=0.80,
    trust_remote_code=True,
    attention_backend="FLASHINFER",
    speculative_config={"method":"mtp","num_speculative_tokens":1},
    compilation_config={"cudagraph_capture_sizes":[1,8,32,64,128,256,384,512]},
)
engine_config = engine_args.create_engine_config()
print("Creating LLMEngine (full INC + repack + warmup + capture)...", flush=True)
engine = LLMEngine.from_vllm_config(engine_config)
print("Triggering dump via collective_rpc...", flush=True)
engine.collective_rpc("dump_layer_pt", kwargs={"dump_dir": "$DUMP_DIR"})
print("Dump complete!", flush=True)
PYRUN
EOF
chmod +x "$WRAPPER_DUMP"

# ---------- Phase 2 wrapper：custom load ----------
WRAPPER_LOAD=/tmp/run-C-load.sh
cat > "$WRAPPER_LOAD" <<EOF
#!/bin/bash
set -e
mkdir -p /tmp/blade_patches
cp /custom_layer_pt_patch.py /tmp/blade_patches/
export PYTHONPATH=/tmp/blade_patches:\${PYTHONPATH:-}
python3 -u -c "import custom_layer_pt_patch"
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
  --load-format custom_layer_pt \\
  --model-loader-extra-config '{"dump_dir":"$DUMP_DIR"}'
EOF
chmod +x "$WRAPPER_LOAD"

# 上传所有
for f in "$LOADER_PATCH" "$DUMP_PATCH" "$WRAPPER_DUMP" "$WRAPPER_LOAD"; do
    upload_to_target "$f" "/tmp/$(basename "$f")"
done

# ---------- 准备 ----------
stop_122b
cleanup_test_container

# ---------- Phase 1: dump ----------
log "Phase 1: full startup + dump layer-{i}.pt + top.pt to $DUMP_DIR"
remote_sudo "
    chmod +x /tmp/run-C-dump.sh
    docker run --rm --gpus all --ipc=host \
      -v $MODEL_HOST:/models \
      -v $CACHE_HOST:/cache \
      -v $PARSER_PY:/custom/qwen3_nothink_reasoning_parser_vllm.py:ro \
      -v /tmp/run-C-dump.sh:/run.sh:ro \
      -v /tmp/dump_layer_pt_patch.py:/dump_layer_pt_patch.py:ro \
      -e VLLM_CACHE_ROOT=/cache \
      -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
      -e TRITON_CACHE_DIR=/cache/triton \
      --entrypoint /run.sh \
      $IMAGE 2>&1 | tail -30
" > "$RESULTS_DIR/$SCHEME/dump.log" 2>&1 || {
    record_result "$SCHEME" "" "false" "false" "dump_failed"
    exit 1
}

DUMP_COUNT=$(remote "ls $DUMP_DIR/layer-*.pt 2>/dev/null | wc -l")
log "Dumped layer files: $DUMP_COUNT (expected ~48)"
if [ "$DUMP_COUNT" -lt 40 ]; then
    record_result "$SCHEME" "" "false" "false" "dump_layers_missing_$DUMP_COUNT"
    exit 1
fi

# ---------- Phase 2: custom load ----------
log "Phase 2: custom_layer_pt load, skipping INC + Marlin repack"
remote_sudo "
    chmod +x /tmp/run-C-load.sh
    docker run -d --name vllm-qwen35-test --gpus all --ipc=host -p 30001:30001 \
      -v $MODEL_HOST:/models \
      -v $CACHE_HOST:/cache \
      -v $PARSER_PY:/custom/qwen3_nothink_reasoning_parser_vllm.py:ro \
      -v /tmp/run-C-load.sh:/run.sh:ro \
      -v /tmp/custom_layer_pt_patch.py:/custom_layer_pt_patch.py:ro \
      -e VLLM_CACHE_ROOT=/cache \
      -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
      -e TRITON_CACHE_DIR=/cache/triton \
      --entrypoint /run.sh \
      $IMAGE
" || { record_result "$SCHEME" "" "false" "false" "container_start_failed"; exit 1; }

log "Waiting for healthy or exit (max 25 min)"
RESULT=$(wait_test_container 1500)
log "$RESULT"
eval "$RESULT"

mkdir -p "$RESULTS_DIR/$SCHEME"
remote "docker logs vllm-qwen35-test 2>&1" > "$RESULTS_DIR/$SCHEME/boot.log" 2>&1 || true
remote "docker inspect vllm-qwen35-test 2>&1" > "$RESULTS_DIR/$SCHEME/inspect.json" 2>&1 || true

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
