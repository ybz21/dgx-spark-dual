# DeepSeek-V4-Flash

本仓库有两套 DeepSeek-V4-Flash 部署方案，**共用 `192.168.130.12` 这台机器，不能同时跑**。

| 方案 | 目录 | 节点 | 引擎 | 上下文 | 单流 decode | 并发聚合 |
|---|---|---|---|---|---|---|
| **vLLM + DSpark + NVFP4 KV** | [`vllm-dspark-2x-nvfp4/`](vllm-dspark-2x-nvfp4) | 2×（`.8` head + `.12` worker，TP=2） | vLLM 定制镜像 | **1M** | **69–85 tok/s** | **244 tok/s @ c6** |
| **ds4 引擎 + IQ2 GGUF** | [`ds4-gguf-iq2-1x/`](ds4-gguf-iq2-1x) | 1×（`.12`） | antirez/ds4（自研 CUDA） | 32K–512K | ~38 tok/s | ~50 tok/s @ c8 |

## 怎么选

- **两台机器都空着 → 用 vLLM DSpark 方案**：速度快一倍以上，上下文大一个量级，
  官方 0731 全精度权重（NVFP4 KV，权重本身未再降精度）。代价是占两台机器 + 156 GiB 权重 ×2。
- **只有一台机器 → 用 ds4 GGUF 方案**：IQ2XXS ~2-bit，81 GiB 单机塞得下，
  长上下文靠原生稀疏注意力（实测 500K+）。质量上限低于上面那套。

## 权重

| 方案 | 权重 | 大小 | 位置 |
|---|---|---|---|
| vLLM DSpark | `deepseek-ai/DeepSeek-V4-Flash-0731` | 156 GiB | 两台的 `~/.cache/huggingface` |
| ds4 GGUF | `antirez/deepseek-v4-gguf` IQ2XXS + MTP + DSpark drafter | 81 + 3.6 + 6.5 GiB | `.12` 的 `~/gguf` |

两者的权重都下载自 `hf-mirror.com`（国内 `huggingface.co` 有 DNS 污染）。

## 评测

跨模型质量/速度对比见 [`../../eval/`](../../eval)，其中
[`../../eval/reports/ds4-vs-qwen-performance.md`](../../eval/reports/ds4-vs-qwen-performance.md)
是针对 ds4 GGUF 方案的。**vLLM DSpark 方案尚未进入 eval 流程。**
