# 方案 B：tensorizer 多文件分片

> **状态：设计可行，未真测；A 已胜，B 留作存档**

## 思路

把 67 GB tensorize cache 拆成 N 个 6.7 GB 文件，加载时**每次只 mmap 一个文件 → 灌入对应 layer GPU tensor → munmap → 下一个**。

每次 mmap 占用 = 6.7 GB（chunk size），CPU + GPU 总占用 ≈ 6.7 + 67 = 73.7 GB < 119 GB → **不 OOM**。

## 为什么单文件 tensorizer 不行

vLLM 的 `tensorize_vllm_model(...)` 把整个 model state_dict 写到一个 67 GB 文件。
load 时 `TensorDeserializer(stream)` mmap 整个文件，**context manager 一直开着 = mmap 一直在 = page cache 累积 67GB**。

DGX Spark unified memory 119GB → mmap 67GB + GPU 67GB + system + INC placeholder = > 119GB → **OOM**（实测两次 21-27 min 后被 kernel kill）。

`lazy_load=True` 也救不了——它只控制"何时读 page"，不影响 mmap 区域的存活时间。

## 设计

### Tensorize 阶段（写多文件）

跑一次完整 vLLM init（包含 INC dispatch + Marlin repack + warmup + capture），然后用 `collective_rpc` 在 worker 进程把 model state_dict 按 transformer block 平均分 N 组，每组单独 `TensorSerializer.write_tensor(...)` 到 `shard-{i:03d}.tensors`。

```python
# worker_shard_patch.py monkey-patch worker
def save_sharded_tensorizer(self, out_dir, n_shards):
    model = self.model_runner.model
    blocks = model.language_model.model.layers  # 48 blocks
    per_shard = (len(blocks) + n_shards - 1) // n_shards

    full_sd = model.state_dict()
    # 顶层（embedding / lm_head / norm / MTP draft）→ top.tensors
    layer_prefixes = [f"language_model.model.layers.{i}." for i in range(len(blocks))]
    top_sd = {k: v for k, v in full_sd.items() if not any(k.startswith(p) for p in layer_prefixes)}
    write_to(top_sd, out_dir / "top.tensors")

    # 每 N 个 block → shard-{i}.tensors
    for i in range(n_shards):
        start, end = i * per_shard, min((i+1) * per_shard, len(blocks))
        prefixes = [f"language_model.model.layers.{idx}." for idx in range(start, end)]
        shard_sd = {k: v for k, v in full_sd.items() if any(k.startswith(p) for p in prefixes)}
        write_to(shard_sd, out_dir / f"shard-{i:03d}.tensors")

    # MANIFEST.json 标记
    (out_dir / "MANIFEST.json").write_text(json.dumps({"n_shards": n_shards}))
```

### Load 阶段（读多文件）

patch vLLM 的 `TensorizerLoader.load_weights`：检测 `_blade_sharded` 标记 → 顺序 N 次 `with TensorDeserializer(...) as deser`：

```python
# loader_shard_patch.py
def load_weights_sharded(self, model, model_config):
    extra = self.load_config.model_loader_extra_config
    if not extra.get("_blade_sharded"):
        return _orig_load_weights(self, model, model_config)

    manifest = json.loads(Path(extra["tensorizer_uri"]).read_text())
    out_dir = Path(extra["tensorizer_uri"]).parent

    # 加载 top
    with open_stream(str(out_dir / "top.tensors")) as stream, \
         TensorDeserializer(stream, device="cuda:0") as deser:
        deser.load_into_module(model)
    gc.collect(); torch.cuda.empty_cache()  # 关键：强制释放

    # 顺序加载每个 shard，with 块结束自动 close → mmap 释放
    for i in range(manifest["n_shards"]):
        shard_path = out_dir / f"shard-{i:03d}.tensors"
        with open_stream(str(shard_path)) as stream, \
             TensorDeserializer(stream, device="cuda:0") as deser:
            deser.load_into_module(model)
        gc.collect(); torch.cuda.empty_cache()
```

启动时：

```yaml
command:
  - --load-format
  - tensorizer
  - --model-loader-extra-config
  - '{"tensorizer_uri":"/cache/tensorized-shards/MANIFEST.json","_blade_sharded":true}'
```

## 期望收益

| 阶段 | 估算 | vs A+B 基线 |
|---|---|---|
| weight load（10 个 shard 顺序） | 60-100 s | 405 s → 节省 300-345 s |
| 其他阶段（cache 命中下） | ~140 s | 同 |
| 总启动 | **~200-240 s** | 比 A 实测的 241 s 略好或相当 |

## 实测情况

### 自动化跑了一次：脚本 bug 没跑到方案逻辑

[`scripts/启动加速/B-shard.sh`](../../scripts/启动加速/B-shard.sh) 在 `USE_SSH=0` 模式（在目标机本地跑）下两个 bug：

1. `upload_to_target` 自我覆盖：源 = 目标 → cp 报错
2. `results/B/` 子目录没预先 mkdir → redirect 失败 → exit 1

整个 phase 1 (tensorize 阶段) **没真正启动 docker run**，没跑到 worker patch / monkey-patch 路径。

### 修脚本 + 重跑代价

修 ~10 行：

```bash
mkdir -p "$RESULTS_DIR/$SCHEME"   # 加在脚本最开始

upload_to_target() {
    [ "$1" = "$2" ] && return 0   # 跳过自我覆盖
    ...
}
```

重跑：tensorize 13 min + load 测试 5-10 min。

### 为什么不修

A (runai_streamer) 已实测 4 min 启动：
- A 不需要预生成 cache（B 要 13 min tensorize）
- A 实施零代码改动，B 要维护 monkey-patch
- A 的 runai_streamer 是 vLLM 官方支持，B 要自己跟 vLLM 升级保持兼容

**B 留作存档**，未来如果 A 失效（vLLM 升级 break runai_streamer），可修脚本重启 B PoC。

## 风险

| 风险 | 应对 |
|---|---|
| `Worker.save_sharded_tensorizer` monkey-patch 注入到 spawn worker 子进程 | PYTHONPATH=/tmp/blade_patches 让所有进程能 import |
| `TensorSerializer.write_tensor(0, k, 0, v)` 调用 API 跟实际签名不符 | 跑前 `inspect.signature` 确认 |
| layer 分组按 count 不均匀 | 改成按 byte size 平衡 |
| INC dispatch 仍跑（建 placeholder + 反量化），不会跳过 | 接受——它本来就只占 ~3 s |

## 复现脚本

[`scripts/启动加速/B-shard.sh`](../../scripts/启动加速/B-shard.sh) 含完整 monkey-patch 代码（worker_shard_patch.py + loader_shard_patch.py + tensorize-shard.py）。修两个 bug 后可一键跑。
