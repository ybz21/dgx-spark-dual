#!/bin/bash
# 方案 B: tensorizer 多文件分片
# 1. patch tensorize_vllm_model 让它写 N 个 shard 文件
# 2. patch tensorizer_loader 让它顺序读 N 个 shard，每读完一个就 close + munmap
# 3. tensorize 阶段：~13min 写出 10 个 ~6.7GB 文件
# 4. load 阶段：每次 mmap 1 个 shard，CPU 峰值 ~7GB，不 OOM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

trap on_exit EXIT

SCHEME="B"
N_SHARDS="${N_SHARDS:-10}"
SHARD_DIR="$CACHE_HOST/tensorized-shards"

log "=== Scheme B: tensorizer N=$N_SHARDS shards ==="

# ---------- 准备 tensorize-shard.py（patched 版本）----------
TENSORIZE_PY=/tmp/tensorize_shard_122b.py
cat > "$TENSORIZE_PY" <<'PYEOF'
"""按 module name 平均分组 tensorize 122B 到 N 个 .tensors 文件。

每个 shard 单独 serialize（不用 vLLM 的 single-file 路径）。
load 时 N 次 deserialize + 关 stream，避免 67GB mmap 一直 hold。
"""
import os
import sys
from pathlib import Path

import torch

from vllm import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine


def main():
    n_shards = int(os.environ.get("N_SHARDS", "10"))
    out_dir = Path(os.environ.get("SHARD_DIR", "/cache/tensorized-shards"))
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_args = EngineArgs(
        model="/models/Qwen3.5-122B-A10B-int4-AutoRound-Intel",
        served_model_name=["qwen3.5-122b-int4"],
        max_model_len=131072,
        gpu_memory_utilization=0.80,
        trust_remote_code=True,
        attention_backend="FLASHINFER",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        compilation_config={
            "cudagraph_capture_sizes": [1, 8, 32, 64, 128, 256, 384, 512],
        },
    )
    engine_config = engine_args.create_engine_config()
    print("Creating LLMEngine (full INC dispatch + Marlin repack + warmup)...", flush=True)
    engine = LLMEngine.from_vllm_config(engine_config)

    # 通过 collective_rpc 在 worker 进程里执行 dump（model 只在 worker 里）
    engine.collective_rpc(
        "save_sharded_tensorizer",
        kwargs={"out_dir": str(out_dir), "n_shards": n_shards},
    )
    print("Sharded tensorize complete!", flush=True)


if __name__ == "__main__":
    main()
PYEOF

# ---------- 准备 worker patch（新增 save_sharded_tensorizer 方法）----------
WORKER_PATCH=/tmp/worker_shard_patch.py
cat > "$WORKER_PATCH" <<'PYEOF'
"""注入 GPUWorker 的 save_sharded_tensorizer 方法。
启动 vLLM 前 import 此 module，patch 生效。
"""
from pathlib import Path
import torch
from tensorizer import TensorSerializer
from vllm.v1.worker.gpu_worker import Worker  # vLLM v1 worker class


def save_sharded_tensorizer(self, out_dir: str, n_shards: int):
    """按 transformer block 平均分 N 组 dump."""
    model = self.model_runner.model
    out = Path(out_dir)

    # 找到 transformer blocks 列表（适配 Qwen3_5MoeForConditionalGeneration）
    if hasattr(model, "language_model"):
        blocks = model.language_model.model.layers
        top_prefix = "language_model"
    else:
        blocks = model.model.layers
        top_prefix = ""

    n_blocks = len(blocks)
    per_shard = (n_blocks + n_shards - 1) // n_shards
    print(f"[save_sharded] {n_blocks} blocks → {n_shards} shards × ~{per_shard} blocks", flush=True)

    # Top-level 非 block 部分：embedding, lm_head, norm, MTP draft 等
    full_sd = model.state_dict()
    block_keys_per_shard = []
    for i in range(n_shards):
        start, end = i * per_shard, min((i + 1) * per_shard, n_blocks)
        prefix_match = []
        for layer_idx in range(start, end):
            prefix = f"{top_prefix + '.' if top_prefix else ''}model.layers.{layer_idx}."
            prefix_match.append(prefix)
        block_keys_per_shard.append(prefix_match)

    # Top-level keys (非 transformer blocks)
    layer_prefixes = sum(block_keys_per_shard, [])
    top_sd = {k: v for k, v in full_sd.items() if not any(k.startswith(p) for p in layer_prefixes)}

    # 写 top
    top_path = out / "top.tensors"
    print(f"[save_sharded] writing top.tensors ({len(top_sd)} tensors)", flush=True)
    with open(top_path, "wb+") as f:
        ser = TensorSerializer(f)
        for k, v in top_sd.items():
            ser.write_tensor(0, k, 0, v)
        ser.close()

    # 写每个 shard
    for i, prefixes in enumerate(block_keys_per_shard):
        shard_sd = {k: v for k, v in full_sd.items() if any(k.startswith(p) for p in prefixes)}
        shard_path = out / f"shard-{i:03d}.tensors"
        print(f"[save_sharded] writing {shard_path.name} ({len(shard_sd)} tensors)", flush=True)
        with open(shard_path, "wb+") as f:
            ser = TensorSerializer(f)
            for k, v in shard_sd.items():
                ser.write_tensor(0, k, 0, v)
            ser.close()

    # Marker
    (out / "MANIFEST.json").write_text(
        f'{{"n_shards": {n_shards}, "version": "blade-shard-v1"}}'
    )
    print(f"[save_sharded] all shards written under {out}", flush=True)


# Monkey-patch
Worker.save_sharded_tensorizer = save_sharded_tensorizer
print("[worker_shard_patch] Worker.save_sharded_tensorizer registered", flush=True)
PYEOF

# ---------- 准备 entrypoint wrapper ----------
WRAPPER_TENSORIZE=/tmp/run-B-tensorize.sh
cat > "$WRAPPER_TENSORIZE" <<EOF
#!/bin/bash
set -e
pip install -q -i $ALIYUN_PYPI tensorizer libnacl 2>&1 | tail -2
mkdir -p $SHARD_DIR
# 让 patch 在主进程 import，并通过 spawn 子进程也能继承（PYTHONSTARTUP 不行，要走 sys.path）
mkdir -p /tmp/blade_patches
cp /worker_shard_patch.py /tmp/blade_patches/
export PYTHONPATH=/tmp/blade_patches:\${PYTHONPATH:-}
# 然后 import patch（在主进程激活），调 tensorize
python3 -u -c "import worker_shard_patch" || true
N_SHARDS=$N_SHARDS SHARD_DIR=$SHARD_DIR python3 -u /tensorize-shard.py
EOF
chmod +x "$WRAPPER_TENSORIZE"

# 上传所有文件
upload_to_target "$TENSORIZE_PY" "/tmp/tensorize-shard.py"
upload_to_target "$WORKER_PATCH" "/tmp/worker_shard_patch.py"
upload_to_target "$WRAPPER_TENSORIZE" "/tmp/run-B-tensorize.sh"

# ---------- 准备 load wrapper ----------
WRAPPER_LOAD=/tmp/run-B-load.sh
cat > "$WRAPPER_LOAD" <<EOF
#!/bin/bash
set -e
pip install -q -i $ALIYUN_PYPI tensorizer libnacl 2>&1 | tail -2
# patch loader：替换 tensorizer_loader.load_weights 走 sharded 路径
mkdir -p /tmp/blade_patches
cp /loader_shard_patch.py /tmp/blade_patches/
export PYTHONPATH=/tmp/blade_patches:\${PYTHONPATH:-}
python3 -u -c "import loader_shard_patch" || true
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
  --load-format tensorizer \\
  --model-loader-extra-config '{"tensorizer_uri":"$SHARD_DIR/MANIFEST.json","_blade_sharded":true}'
EOF
chmod +x "$WRAPPER_LOAD"

# ---------- loader patch ----------
LOADER_PATCH=/tmp/loader_shard_patch.py
cat > "$LOADER_PATCH" <<'PYEOF'
"""patch vLLM tensorizer_loader：检测 _blade_sharded 标记走 sharded load."""
import gc
import json
from pathlib import Path

import torch
from tensorizer import TensorDeserializer
from tensorizer.stream_io import open_stream
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerLoader


_orig_load_weights = TensorizerLoader.load_weights


def load_weights_sharded(self, model, model_config):
    cfg = self.tensorizer_config
    extra = self.load_config.model_loader_extra_config or {}
    if not extra.get("_blade_sharded"):
        return _orig_load_weights(self, model, model_config)

    manifest_path = Path(extra["tensorizer_uri"])
    manifest = json.loads(manifest_path.read_text())
    n_shards = manifest["n_shards"]
    out_dir = manifest_path.parent

    # 加载 top
    print(f"[blade_sharded] loading top.tensors", flush=True)
    with open_stream(str(out_dir / "top.tensors"), mode="rb") as stream, \
         TensorDeserializer(stream, device="cuda:0") as deser:
        deser.load_into_module(model)
    gc.collect(); torch.cuda.empty_cache()

    # 加载 N 个 shard
    for i in range(n_shards):
        shard_path = out_dir / f"shard-{i:03d}.tensors"
        print(f"[blade_sharded] loading {shard_path.name}", flush=True)
        with open_stream(str(shard_path), mode="rb") as stream, \
             TensorDeserializer(stream, device="cuda:0") as deser:
            deser.load_into_module(model)
        gc.collect(); torch.cuda.empty_cache()

    print(f"[blade_sharded] all {n_shards} shards loaded", flush=True)


TensorizerLoader.load_weights = load_weights_sharded
print("[loader_shard_patch] TensorizerLoader.load_weights patched", flush=True)
PYEOF
upload_to_target "$LOADER_PATCH" "/tmp/loader_shard_patch.py"
upload_to_target "$WRAPPER_LOAD" "/tmp/run-B-load.sh"

# ---------- 准备：停 122B + 清测试容器 ----------
stop_122b
cleanup_test_container

# ---------- Phase 1: tensorize 写多 shard ----------
log "Phase 1: tensorize → $N_SHARDS shards (~13min)"
remote_sudo "
    chmod +x /tmp/run-B-tensorize.sh
    docker run --rm --gpus all --ipc=host \
      -v $MODEL_HOST:/models \
      -v $CACHE_HOST:/cache \
      -v $PARSER_PY:/custom/qwen3_nothink_reasoning_parser_vllm.py:ro \
      -v /tmp/run-B-tensorize.sh:/run.sh:ro \
      -v /tmp/tensorize-shard.py:/tensorize-shard.py:ro \
      -v /tmp/worker_shard_patch.py:/worker_shard_patch.py:ro \
      -e VLLM_CACHE_ROOT=/cache \
      -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
      -e TRITON_CACHE_DIR=/cache/triton \
      --entrypoint /run.sh \
      $IMAGE 2>&1 | tail -30
" > "$RESULTS_DIR/$SCHEME/tensorize.log" 2>&1 || {
    record_result "$SCHEME" "" "false" "false" "tensorize_failed"
    exit 1
}

# 验证 shard 文件
log "Verifying shards under $SHARD_DIR"
SHARD_COUNT=$(remote "ls $SHARD_DIR/shard-*.tensors 2>/dev/null | wc -l")
log "Shard files found: $SHARD_COUNT (expected $N_SHARDS)"
if [ "$SHARD_COUNT" -ne "$N_SHARDS" ]; then
    record_result "$SCHEME" "" "false" "false" "shards_missing_$SHARD_COUNT"
    exit 1
fi

# ---------- Phase 2: load via sharded ----------
log "Phase 2: load via sharded tensorizer"
remote_sudo "
    chmod +x /tmp/run-B-load.sh
    docker run -d --name vllm-qwen35-test --gpus all --ipc=host -p 30001:30001 \
      -v $MODEL_HOST:/models \
      -v $CACHE_HOST:/cache \
      -v $PARSER_PY:/custom/qwen3_nothink_reasoning_parser_vllm.py:ro \
      -v /tmp/run-B-load.sh:/run.sh:ro \
      -v /tmp/loader_shard_patch.py:/loader_shard_patch.py:ro \
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

# 抓日志
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
