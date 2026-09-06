# GLM-5.3-Flash

GLM-5.3-Flash（320B MoE / 18B 激活，zai-org）在 DGX Spark 上的部署。
FP8 原始权重约 328 GiB，单台 Spark 只有 128 GiB 统一内存，所以**必须量化**，
两台 TP=2 才是能兼顾质量和速度的档位。

本仓库采用 **[`exl3-2x-entrpi/`](exl3-2x-entrpi)**：EXL3 4bpw + DFlash2 投机，
双节点 TP=2（`.8` head + `.12` worker）。选它的原因见下。

## 路线对比（都是 2026-09 前后社区在 GB10 上跑出来的）

| 路线 | 节点 | 权重 | 结构化 decode | 散文 decode | TTFT | 上下文 |
|---|---|---|---|---|---|---|
| **EXL3 4bpw + DFlash2（Entrpi）** ← 本仓库选型 | 2 | EXL3 4bpw 164 GiB | **72.4** tok/s | **27.4** tok/s | **0.43–0.47 s** | 524K 默认，可开 1M |
| EXL3 4bpw + DFlash2（MiaAI-Lab） | 2 | 同一份 EXL3 4bpw | 61.7–62.9 tok/s | 26.9 tok/s | ~0.72 s | 900K |
| SGLang + DFlash2 | 2 | NVFP4 | 29.4（代码） | 23.4 tok/s | — | 131K |
| vLLM + NVFP4 + MTP | 2 | NVFP4 182 GiB | 21.8 tok/s | — | 0.29 s | 262K |
| EXL3 2.05bpw 单机 | 1 | 85 GiB | 64 tok/s | 25 tok/s | — | 262K |

- **为什么不是单机 2.05bpw**：数字好看，但 2-bit 与全精度 top-1 一致率只有 88.9%，
  论坛里多人报"复杂任务进死循环 / 并发时挂住"。我们要跑评测和真实业务，质量不能这么让。
- **为什么不是 MiaAI-Lab**：同一份权重、同一个 drafter，两个栈独立做出来的。
  Entrpi 结构化快 17%、TTFT 快约 40%，同预算下 KV 池大 46%（1,435,070 vs 982,612 token）。
  论坛原帖里两边用户的结论也是 Entrpi 在 PP 和 TG 上都更快。
- **为什么不是 vLLM NVFP4 / SGLang**：这两条是 day-0 抢跑的路线，慢一半以上，
  且需要自己维护七八个 sm_121 补丁。Entrpi 把补丁都打进发行镜像了。

两个 EXL3 栈用的是同一份量化权重：`brandonmusic/GLM-5.3-Flash-tr3-4bpw`，
ModelScope 上有字节一致的镜像（见下）。

## 权重

| 用途 | 仓库 | 大小 | 落盘位置（`.8` / `.12` 两台都有） |
|---|---|---|---|
| 主模型 | ModelScope `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` | 164 GiB / 120 shard | `~/models/GLM-5.3-Flash-EXL3-TR3-4bpw` |
| 投机 drafter | ModelScope `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` | 1.2 GiB | `~/models/glm53-dflash2-mxfp8` |

**两个都在 ModelScope 上有**，不用碰 huggingface.co（国内 DNS 污染），也不用 hf-mirror。

- 主模型：`Mia-AiLab/...` 是 `brandonmusic/GLM-5.3-Flash-tr3-4bpw`
  （upstream revision `5ab363a8`）的字节一致再分发，仓库内 `MIRROR.json` 自证。
- drafter：ModelScope 上 `model.safetensors` 的 sha256 是
  `c033e03d…1614e58`，与 HuggingFace 上 Entrpi 钉住的 revision
  `62f758c0a0e19b9cb76fc098c911b8ed76daff5b` 的 LFS oid 完全一致 —— 换源不换内容。
  DFlash2 是 CC BY-NC-ND 4.0（研究/评测用），不要再分发。

下载与校验脚本见 [`exl3-2x-entrpi/scripts/`](exl3-2x-entrpi/scripts)。

## 参考

- Entrpi 方案：<https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark>
- MiaAI-Lab 方案：<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks>
- 论坛横评：<https://forums.developer.nvidia.com/t/deepseek-v4-flash-glm-5-3-flash-qwen3-8-flash-next/381832>
- 单机 EXL3 2.05bpw：<https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140>
- SGLang 路线：<https://forums.developer.nvidia.com/t/glm-5-3-flash-dflash2-on-sglang-2x-dgx-spark-first-sglang-path-recipe-4-gb10-fixes-honest-numbers-concurrency-curve/381703>
- vLLM NVFP4 路线：<https://forums.developer.nvidia.com/t/glm-5-3-flash-running-on-2x-dgx-spark-sm-121-day-0-24-7-30-3-tok-s-with-mtp-5-two-silent-gb10-gotchas-worth-knowing/381433>
