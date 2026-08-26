# Qwen3.8-27B

单节点 DGX Spark (GB10) 部署 **Qwen3.8-27B**（27B 稠密 + 混合线性/全注意力，多模态；架构 `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`，与本仓库 Qwen3.5 系列同族）。两套投机解码方案，**共用同一份 NVFP4 主模型权重**，都用 `:8000`，**不能同时跑**。

| 方案 | 目录 | 引擎 | 投机解码 | 单流 decode | 定位 |
|---|---|---|---|---|---|
| **vLLM + MTP** | [`vllm-mtp-1x-nvfp4/`](vllm-mtp-1x-nvfp4) | vLLM 0.27.1 | 内嵌 MTP 头 | 代码30/数学29 tok/s，接受率44-49% | ✅ 生产(含 Function Call + 256k) |
| **SGLang + DSpark** | [`sglang-dspark-1x-nvfp4/`](sglang-dspark-1x-nvfp4) | SGLang（`lmsysorg/sglang:dspark`） | DSpark 草稿（1.36B, block=7） | 实测 15.3(散文)/31.4(代码)/33.5(数学) tok/s | ✅已验证 |

> 方案出处：[NVIDIA DGX Spark 论坛版块](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
> · A：[vLLM+MTP 帖](https://forums.developer.nvidia.com/t/qwen3-8-27b-nvfp4-on-a-single-dgx-spark-up-to-1m-context-vllm-mtp-measurements/380244)
> · B：[SGLang+DSpark 一键帖](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)

## 怎么选

- **要最快 → SGLang + DSpark**：单流快 ~40%，社区一键方案的默认配置。代价是多一份 2.7 GB 草稿模型 + torch.compile 首启编译较久。
- **要稳/省事 → vLLM + MTP**：直接复用本仓库既有的 vLLM 镜像，无需额外草稿模型，MTP 头已内嵌在主权重里。

## 权重

| 用途 | 仓库 | 大小 | 位置 |
|---|---|---|---|
| 主模型（两套共用） | `unsloth/Qwen3.8-27B-NVFP4` | ~23 GiB | `~/models/Qwen3.8-27B-NVFP4` |
| DSpark 草稿（仅 B） | `RadixArk/Qwen3.8-27B-DSpark` | ~2.7 GiB | `~/models/RadixArk-Qwen3.8-27B-DSpark` |

⚠️ **权重从 ModelScope（国内 CDN）下**，不要用 hf-mirror：主权重 `model.safetensors`(22.6GB) 存在 HF 的 Xet CDN（`us.aws.cdn.hf.co`），hf-mirror 不代理它，国内拉大文件必断连 + 预签名过期(403)。各子目录 `scripts/download.sh` 已用 ModelScope。

## 目标机

`ai@192.168.130.48`（`box010`，DGX Spark / GB10 / aarch64 / 119 GiB）。这是台生产机（:30001 跑 Qwen3.5-122B、:30002 跑 ASR，占 ~108 GiB），部署本模型（~55 GiB）前需先 `docker stop vllm-qwen35 qwen3-asr` 腾内存，用完 `docker start` 恢复。镜像走 LAN 私仓 `192.168.130.23:5000`（ghcr.io/docker hub 本网被墙）。

## 评测

跨模型质量/速度对比见 [`../../eval/`](../../eval)。**本模型两套方案的实测见各子目录 `docs/deploy-report.md`。**

## 并发实测（box010 GB10，单机，max_tokens=256，关思考，本地压测）

> TTFT=首 token 延迟；per-req=单请求 decode；agg=聚合吞吐（总 token / 批墙钟）。
> 测试脚本 `scripts/bench_concurrency.py`（流式，ThreadPool 并发）。

| 并发 | A: vLLM(plain, 无投机) TTFT / per-req / agg | B: SGLang+DSpark TTFT / per-req / agg |
|---|---|---|
| 1 | 0.20s / 11.3 / 11.3 | 0.21s / 12.7 / 12.7 |
| 2 | 0.20s / 10.9 / 21.8 | 0.44s / 12.0 / 22.9 |
| 4 | 0.25s / 10.6 / 40.4 | 0.44s / 10.3 / 39.4 |
| 6 | 0.26s / 10.4 / 58.1 | 0.37s / 10.0 / **59.8** |
| 8 | 0.29s / 10.1 / **74.8** | 5.42s* / 8.8 / 43.9 |

\* **DSpark 并发上限 6**（受混合线性注意力的 mamba 状态缓存 + DSpark cuda graph 捕获限制；提高到 8 会在 cuda graph 捕获阶段 `illegal memory access` 崩溃）。c8 时 2 个请求排队 → TTFT 尾延迟飙到 ~21s，聚合吞吐反降。

### 投机解码在并发下的限制（Qwen3.8 这类新模型，实测）
- **vLLM + MTP**：单流很好（代码/数学 27–31 tok/s），但**并发 ≥4 会崩**（vLLM 0.27.1 的 MTP 投机在此混合模型下 `CUDA illegal memory access`，spec token 变 -1）。上表 A 用的是**关掉 MTP 的纯 vLLM**，稳定可扩到 c8。
- **SGLang + DSpark**：**并发 ≤6 稳定**，c6 聚合峰值 ~60 tok/s；投机在结构化内容（代码/数学）上单流最快（33.5 tok/s）。

### 选型（并发角度）
- **要高并发/高吞吐** → vLLM 纯解码（无投机），可扩到 c8（~75 tok/s 聚合），TTFT 稳 <0.3s。
- **要单/低并发下最快单请求** → SGLang+DSpark 或 vLLM+MTP（投机，结构化内容 27–33 tok/s），但并发受限。
