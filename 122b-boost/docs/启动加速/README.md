# Qwen3.5-122B 启动加速方案

> 把 vLLM 容器启动时间从 **12.5 min 压到 4 min**（实测）。
>
> 主要瓶颈是 INT4 + INC dispatch + Marlin repack 反量化 405 s。本目录包含三个尝试方案：
>
> - **A. runai_streamer** — ✅ 实测 4 min 启动（**已固化到生产 compose**）
> - B. tensorizer 多文件分片 — 设计可行，未实测（A 已胜，无需继续）
> - C. 手写 custom dump+load — 设计最复杂，理论 < 100s，但工程量大

## 实测结果汇总（2026-05-06 单节点 DGX Spark）

| 方案 | Boot | OOM | Healthy | 状态 |
|---|---|---|---|---|
| 原始基线（默认 51 capture sizes，无 cache） | ~750 s (12.5 min) | false | true | 基线 |
| 优化 A+B（cache + 减少 capture sizes）| 596 s (9.9 min, cache 命中) | false | true | 早期固化 |
| **A. runai_streamer** | **241 s (4 min)** | **false** | **true** | **🏆 推荐** |
| B. tensorizer 多文件分片 | — | — | — | 未实测，详见 [B](./B-tensorizer-shard.md) |
| C. 手写 custom dump+load | — | — | — | 未实测，详见 [C](./C-custom-loader.md) |
| 试过的死路：单文件 tensorizer | — | **true** | false | 27 min OOM kill |
| 试过的死路：tensorizer + lazy_load | — | **true** | false | 21 min OOM kill |
| 试过的死路：换 NVFP4 权重 | — | — | — | SM121 上 FP4 CUTLASS 不完整，反而比 INT4 慢 |

## 为什么 INT4 加载这么慢？阶段拆解

启动 596 s（A+B 优化下，cache 命中）的去向：

| # | 阶段 | 耗时 | 占比 | 谁在干活 |
|---|---|---|---|---|
| 1 | Python init / arg / 架构识别 | 13 s | 2 % | CPU |
| 2 | EngineCore / NCCL init | 7 s | 1 % | CPU |
| 3 | INC GPTQ dispatch（47 layers）| 3 s | <1 % | CPU |
| **4** | **主模型 weight load**（INT4 → BF16 反量化 + Marlin repack） | **405 s** | **68 %** | **CPU bound** ⭐ |
| 5 | MTP draft weight load | 52 s | 9 % | CPU |
| 6 | MTP weight share / encoder cache | 10 s | 2 % | CPU |
| 7 | torch.compile 主模型（AOT cache hit） | 8 s | 1 % | CPU |
| 8 | profiling/warmup 主模型 | 36 s | 6 % | GPU |
| 9 | torch.compile MTP（cache hit） | 0.1 s | <1 % | — |
| 10 | profiling/warmup MTP | 0.2 s | <1 % | — |
| 11 | KV cache 配置 | 3 s | <1 % | — |
| 12 | CUDA graph profile + capture（8 sizes） | 4 s | <1 % | GPU |
| 13 | API server / chat template / parser | 22 s | 4 % | CPU |
| 14 | Multi-modal warmup | 6 s | 1 % | GPU |
| 15 | docker healthcheck 探测延迟 | ~20 s | 3 % | — |
| | **合计** | **596 s** | 100 % | |

**阶段 4 是无解大头**——除非：
- 缓存反量化结果 → 我们试过 [tensorizer](./C-custom-loader.md#path-3-tensorizer)，DGX Spark 上 OOM
- 用 streaming + 并行 IO → **方案 A 用 runai_streamer 干这个**，实测 405 s → 165 s

## 三方案对比

| 维度 | A | B | C |
|---|---|---|---|
| 思路 | runai_streamer 并行 stream，跳过 default loader | 把 67GB 拆成 N 个 6.7GB shard，每次 mmap 一片 | 完整 dump GPU 上 repack 后 state_dict，跳过 INC + repack |
| 工作量 | 极小（compose 改 flag） | 中（patch tensorize + custom loader） | 大（fork vLLM + 多个 hook） |
| 不需要 cache | ✅ 直接读原始 safetensors | ❌ 13 min 预 tensorize | ❌ 12 min 预 dump |
| 不 OOM | ✅ chunk 读不 mmap | ✅（理论） | ✅（理论） |
| 启动时间 | **4 min（实测）** | ~3-4 min（估） | < 100 s（估） |
| 维护成本 | 低（vLLM 官方支持） | 中（自维护 patch） | **高（patch 触及 vLLM 内部）** |

A 是 **最优 ROI**。除非 vLLM 把 runai_streamer 移除或 break 量化兼容，否则不需要 B/C。

## 详细方案

- [A. runai_streamer](./A-runai-streamer.md) — 实测报告 + 固化方法
- [B. tensorizer 多文件分片](./B-tensorizer-shard.md) — 设计 + 未实测原因
- [C. 手写 custom dump+load](./C-custom-loader.md) — 设计 + 风险分析

## 一键复现

把 PoC 脚本套放到目标机，跑 `run-all.sh`：

```bash
# 在控制端：上传脚本 + 远程启动
./scripts/启动加速/launch-overnight.sh

# 在目标机本地：直接跑（USE_SSH=0 模式）
USE_SSH=0 ./scripts/启动加速/run-all.sh

# 或单独跑某方案
./scripts/启动加速/A-runai.sh
```

跑完看 `results/<scheme>/result.json` 对比 boot time。
