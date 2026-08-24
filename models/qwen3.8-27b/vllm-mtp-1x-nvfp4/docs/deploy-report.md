# Qwen3.8-27B · vLLM + MTP · 部署验证报告

> 目标机：`ai@192.168.130.48`（box010，DGX Spark / GB10 / aarch64 / 119 GiB）
> 日期：2026-08-24 · 状态：**✅ 部署成功并实测**

## 配置

| 项 | 值 |
|---|---|
| 引擎 | vLLM **0.27.1**（`vllm/vllm-openai:latest`，arm64）|
| 模型 | `unsloth/Qwen3.8-27B-NVFP4`（compressed-tensors NVFP4，~22 GiB 权重）|
| 架构 | `Qwen3_5ForConditionalGeneration`（qwen3_5，混合 GDN 线性注意力 + 全注意力，多模态）|
| 投机解码 | **MTP**（`--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`；vLLM 自动检测内嵌 MTP 头，与主模型共享 embedding/lm_head）|
| 量化 | NVFP4（FlashInferCutlassNvFp4LinearKernel），lm_head 也量化（需 vLLM ≥0.27）|
| gpu-memory-utilization | 0.45 |
| max-model-len | 32768（起步值，可调大）|
| 端口 | :9001（box010 上 :8000/:8010 被生产服务占用）|

## 实测（GB10，单流，greedy）

| 任务 | 输出 tok | 墙钟 | Decode |
|---|---|---|---|
| 中文散文（自由生成）| 221 | 12.6s | **17.6 tok/s** |
| 代码（快排+注释）| 256 | 9.5s | **27.0 tok/s** |
| 数学（求和推导）| 256 | 8.3s | **30.9 tok/s** |

- 权重加载 ~25s，模型占用 22.1 GiB；含 torch.compile 冷启约 4–5 分钟。
- MTP 投机在结构化内容（代码/数学）上接受率高，decode 27–31 tok/s，达到/超过论坛参考值（24–26 tok/s）；自由散文接受率低，~17.6 tok/s。

## 关键坑

1. **必须用较新 vLLM（≥0.27）**：Qwen3.8 的 NVFP4 权重把 `lm_head` 也量化了；本仓库 LAN 的旧 vLLM 镜像（0.19，v26041616）加载会报 `no module named lm_head.weight_scale`。用 `vllm/vllm-openai:latest`(0.27.1) 才行。
2. **镜像获取**：ghcr 的 spark-arena nightly 本网被墙；`vllm/vllm-openai:latest`(docker hub) 可用，但 box010 外网被限速——在开发机(走代理)拉 arm64 → `docker save --platform linux/arm64`(9.9GB tar) → 走局域网传到 box010 → `docker load`。
3. **入口**：`vllm/vllm-openai` 的 entrypoint 是 `vllm serve`，模型路径是**位置参数**（不是 `--model`）。
