# Qwen3.8-27B

单节点 DGX Spark (GB10) 部署 **Qwen3.8-27B**（27B 稠密 + 混合线性/全注意力，多模态；架构 `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`，与本仓库 Qwen3.5 系列同族）。两套投机解码方案，**共用同一份 NVFP4 主模型权重**，都用 `:8000`，**不能同时跑**。

| 方案 | 目录 | 引擎 | 投机解码 | 单流 decode | 定位 |
|---|---|---|---|---|---|
| **vLLM + MTP** | [`vllm-mtp-1x-nvfp4/`](vllm-mtp-1x-nvfp4) | vLLM（复用仓库 `vllm-qwen35-v2` 镜像） | 内嵌 MTP 头 | ~24–26 tok/s | 稳，零额外权重 |
| **SGLang + DSpark** | [`sglang-dspark-1x-nvfp4/`](sglang-dspark-1x-nvfp4) | SGLang（`sglang-dev-cu13-accel`） | DSpark 草稿模型（1.36B, block=7） | **~34–38 tok/s，峰值 46.7** | 快，社区一键方案 |

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
