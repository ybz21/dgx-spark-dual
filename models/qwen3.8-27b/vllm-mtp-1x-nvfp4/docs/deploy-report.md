# Qwen3.8-27B · vLLM + 官方 MTP + Function Call · 部署验证报告

> 目标机：`ai@192.168.130.48`（box010，DGX Spark / GB10 / aarch64）
> 日期：2026-08-26 · 状态：**✅ 生产配置，已实测**（agent/多人场景推荐）

## 配置

| 项 | 值 |
|---|---|
| 引擎 | vLLM **0.27.1**（`vllm/vllm-openai:latest`，arm64）|
| 模型 | `unsloth/Qwen3.8-27B-NVFP4`（compressed-tensors NVFP4）|
| 投机解码 | **官方 MTP**（模型自带 `model_mtp.safetensors`，`num_speculative_tokens=5`）|
| **Function Call** | ✅ `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| 思考解析 | `--reasoning-parser qwen3` |
| 上下文 | **256k**（原生上限 `max_position_embeddings=262144`）|
| 并发 | `--max-num-seqs 4`（MTP 稳定并发，超出排队，实测扛到 8）|

## 实测

**单流 decode（tok/s）**：散文 13.9 / 代码 30.5 / 数学 28.8。

**MTP 接受率：44–49%**（mean acceptance length 3.2–3.4）——远高于社区 DSpark 的 15–25%。

**并发**：1/2/3/4/8 全稳、零崩溃（4 路并行 + 超出排队）。

**Function Call**：正确返回结构化 `tool_calls`（`get_weather{"city":"北京"}`、`read_file{"path":"main.py"}`）——修复了 agent「输出一句就停、要人工点继续」的问题。

## 官方 MTP vs 社区 DSpark（同机实测）

| | DSpark(社区第三方) | **MTP(Qwen 官方)** |
|---|---|---|
| 接受率 | 15–25% | **44–49%** |
| 代码/数学单流 | 31–33 | 29–31 tok/s |
| 并发稳定性 | 上限 6、cuda graph 易崩 | **稳到 8** |
| 工具调用 | 需另配 | ✅ |
| 出身 | RadixArk 第三方草稿 + 补丁 SGLang | Qwen 官方随模型发布 |

结论：**官方 MTP 完胜**——接受率高一倍、并发更稳、开箱工具调用。

## 关键坑

1. **必须 vLLM ≥0.27**：Qwen3.8 NVFP4 的 lm_head 也量化，旧镜像加载报 `no module named lm_head.weight_scale`。
2. **Function Call 必须配 `--tool-call-parser qwen3_coder`**：否则模型的工具调用变纯文本，agent 认不出 → 每步卡住要人工点「继续」。
3. **MTP 并发**：max-num-seqs=4 + 干净 GPU 稳；开很大且同机有进程 churn GPU 时 MTP 可能触发 CUDA illegal memory access。
4. **1M 上下文**：非原生（config 上限 256k），需 RoPE 外推有质量风险，未开。
