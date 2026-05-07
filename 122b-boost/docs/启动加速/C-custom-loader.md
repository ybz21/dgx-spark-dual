# 方案 C：手写 custom dump + load

> **状态：设计可行但复杂度最高，未真测；A 已胜，C 仅作"理论参考"**

## 思路

A、B 都依赖 tensorizer 库（A 不用 cache，B 用分片 cache）。**C 完全 bypass tensorizer**：

- 第一次启动：完整跑 vLLM init（含 INC dispatch + Marlin repack + warmup + capture）
- 启动完成后 hook 一个 callback：把 GPU 上每个 transformer layer 的 state（已 repack 完）单独 `torch.save` 到一个 .pt 文件
- 二次启动：写 custom `LoadFormat`，**跳过 vLLM 的 default load + INC dispatch + Marlin repack**，直接 `torch.load` 每个 layer.pt → setattr GPU tensor

二次启动**完全不走** vLLM 的 weight load + repack 路径。理论上 67 GB / NVMe 6 GB/s ≈ **11 s 加载完**，加上其他阶段总 < 100 s。

## Dump 阶段

vLLM v1 没有现成的 "post-init save" hook，monkey-patch `Worker.dump_layer_pt`：

```python
def dump_layer_pt(self, dump_dir):
    out = Path(dump_dir); out.mkdir(parents=True, exist_ok=True)
    model = self.model_runner.model
    blocks = model.language_model.model.layers  # 48 transformer blocks

    full_sd = model.state_dict()
    block_prefixes = [f"language_model.model.layers.{i}." for i in range(len(blocks))]
    top_sd = {k: v for k, v in full_sd.items() if not any(k.startswith(p) for p in block_prefixes)}
    torch.save(top_sd, out / "top.pt")

    for i, block in enumerate(blocks):
        torch.save(block.state_dict(), out / f"layer-{i:03d}.pt")
```

通过 `engine.collective_rpc("dump_layer_pt", kwargs={"dump_dir": ...})` 触发。

## Load 阶段

注册新 LoadFormat `custom_layer_pt`：

```python
class CustomLayerPtLoader(BaseModelLoader):
    def load_weights(self, model, model_config):
        dump_dir = self.load_config.model_loader_extra_config["dump_dir"]
        # top
        top = torch.load(Path(dump_dir) / "top.pt", map_location="cuda:0")
        model.load_state_dict(top, strict=False, assign=True)
        del top; gc.collect(); torch.cuda.empty_cache()
        # 每个 layer
        for i, block in enumerate(model.language_model.model.layers):
            sd = torch.load(Path(dump_dir) / f"layer-{i:03d}.pt", map_location="cuda:0")
            block.load_state_dict(sd, strict=False, assign=True)
            del sd; gc.collect(); torch.cuda.empty_cache()
        # 标记跳过 repack
        for m in model.modules():
            m._blade_skipped_repack = True
```

## 跳过 INC dispatch + repack

最 tricky 的点：vLLM 的 `model_runner.load_model` 调用流程：

1. `init_model_class()` → 实例化 model 类（**INC dispatch 在这里被调用**，决定 kernel + 建 placeholder）
2. `loader.load_weights(model)` → 灌入 weight
3. `process_weights_after_loading(model)` → Marlin repack

我们的 dump 是**第 3 步之后**的状态。重启时第 1 步 INC dispatch **会被重新调用**——但只是建 placeholder + 选 kernel，几秒就完。

第 2 步用我们的 custom loader 直接 set 已 repack 的 tensor。

第 3 步会再次调用 `process_weights_after_loading`：**关键问题**——它会把已经 repack 过的 weight **再 repack 一次**，导致 corruption。

解决：在 custom loader 里给 model 加 attribute `_blade_skipped_repack=True`，patch `process_weights_after_loading` 检测到就 skip：

```python
for klass in [GPTQMarlinLinearMethod, GPTQMarlinMoEMethod]:
    _orig = klass.process_weights_after_loading
    def make_wrapper(orig):
        def wrapper(self, layer):
            if getattr(layer, "_blade_skipped_repack", False):
                return
            return orig(self, layer)
        return wrapper
    klass.process_weights_after_loading = make_wrapper(_orig)
```

## 实测情况

### 自动化跑了一次：脚本 bug 没跑到方案逻辑

跟 [B](./B-tensorizer-shard.md) 同根源 bug：
- `upload_to_target` 自我覆盖
- `results/C/` 子目录未预先 mkdir → redirect 失败 → exit 1

dump 阶段**没启动 docker run**。

## 4 个潜在失败点（即使脚本修了也未必通）

C 比 A、B 多了几个不确定性：

1. **`Worker.dump_layer_pt` monkey-patch 注入到 spawn worker 子进程**
   - `collective_rpc("dump_layer_pt", ...)` 要求方法在 worker 进程的 Worker 实例上存在
   - patch 只在主进程 import → spawn worker 不继承 → method 不存在 → rpc 失败
   - 缓解：PYTHONPATH 让所有进程都能 import patch

2. **`_blade_skipped_repack` 标记必须设到所有相关 module，含嵌套 sublayer**
   - 如果遗漏某个 layer → 这层 repack 跑两次 → tensor corrupt → 推理结果错误（不是 crash）
   - 难调试：除非跑准确性测试否则发现不了

3. **`assign=True` 的 `load_state_dict`**
   - 直接替换 tensor 引用（不 copy_）
   - 但 vLLM 的 layer 可能持有原 tensor 的 view（如 `g_idx_sort_indices`）→ assign 后 view 失效

4. **MTP draft model 状态怎么处理**
   - dump 时只覆盖了主 model.layers，没覆盖 MTP draft
   - load 后 MTP draft 还是 dummy → 推理时 spec decoding 报错

## 跟 B 的对比

| 项 | B | C |
|---|---|---|
| 修脚本难度 | 同（10 行） | 同 |
| 重跑代价 | tensorize 13 min + load 10 min | dump 12 min + load 5 min |
| 方案本身复杂度 | 中（patch tensorizer 2 处） | **大**（5+ 个 patch 点 + 跳过 vLLM 标准流程） |
| 不确定性风险 | 中 | 高（list 4 项潜在失败） |

## 是否值得修

**不值得**，原因同 B + 复杂度更高：

1. A 已经把启动从 596 s 降到 **241 s**
2. C 即使理论收益最大（跳过 INC + repack ≈ 节省 165 s），也只能再省 100-150 s（241 → 100 s 左右）
3. C 的 patch 触及 vLLM 内部机制多，**长期维护成本高**（vLLM 升级容易 break）
4. 投入产出比远低于 A

## 结论

C 方案**未真正测试**，技术上可探索但**不应作为生产方案**。
脚本留作"理论参考"——未来如果 A、B 都死、需要把 122B 启动压到 < 100 s 时再考虑。

## 复现脚本

[`scripts/启动加速/C-custom.sh`](../../scripts/启动加速/C-custom.sh) 含完整 monkey-patch 代码（dump_layer_pt_patch.py + custom_layer_pt_patch.py）。修脚本 bug + 解决 4 个潜在失败点后可一键跑。
