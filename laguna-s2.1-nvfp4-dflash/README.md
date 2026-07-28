# Laguna-S-2.1-NVFP4 · vLLM · 单节点

单节点 DGX Spark (GB10) 用 **vLLM v0.26.0** 跑 poolside 的 **Laguna-S-2.1-NVFP4**（代码专精 MoE 模型），并用 **DFlash（EAGLE3 草稿模型）做投机解码**。NVFP4 正好吃 GB10 的 Blackwell FP4。

|  | 本方案 (laguna-s2.1-nvfp4-dflash/) |
|---|---|
| 引擎 | vLLM v0.26.0 (arm64) |
| 模型 | Laguna-S-2.1-NVFP4（MoE 256 专家 / 10 激活 / 48 层，67 GiB） |
| 量化 | compressed-tensors `nvfp4-pack-quantized`（vLLM 自动识别） |
| 投机解码 | **DFlash = EAGLE3 草稿模型**，`num_speculative_tokens=7`，实测接受率 ~40% |
| 上下文 | 128K（本配置起 32K，可调大） |
| 定位 | 代码/数学专精：MBPP 93% / HumanEval 90% / GSM8K 95% |

> **能开 MTP 吗？** 不能——Laguna 主模型没有 MTP/nextn 模块（145153 个权重里无相关层）。它的官方投机方案就是 DFlash（EAGLE3），已在本配置默认开启。

## 硬件要求

- NVIDIA GB10（DGX Spark），**aarch64**、CUDA 13、128 GiB 统一内存
- ≥80 GiB 空闲磁盘；运行时占 ~110 GiB 统一内存（与同机其它大模型**互斥**）

## 快速开始

### 1. 下载模型（ModelScope，国内快）

```bash
bash scripts/download.sh          # 下到 ~/models/Laguna-S-2.1-NVFP4 和 ...-DFlash-NVFP4
```

### 2. 拉 vLLM 镜像

```bash
# docker.1ms.run 的 blob CDN 常 TLS 超时，用 daocloud 更稳
docker pull docker.m.daocloud.io/vllm/vllm-openai:v0.26.0
docker tag  docker.m.daocloud.io/vllm/vllm-openai:v0.26.0 vllm/vllm-openai:v0.26.0
```

### 3. 起服务

```bash
cd deploy && docker compose up -d
docker compose logs -f            # 冷启加载 72GB + 构建投机栈，约 10 分钟
```

`deploy/docker-compose.yml` 关键参数：`--trust-remote-code`（Laguna 自定义架构 `LagunaForCausalLM`）、`--speculative-config '{"model":".../DFlash-NVFP4","num_speculative_tokens":7}'`、`--max-model-len 32768`、prefix caching、`0.0.0.0:8000`。

## API

OpenAI 兼容，`model=laguna-s-2.1`：

```bash
curl http://<host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"写个快排"}],"max_tokens":1024}'
```

> Laguna 是**重推理型**模型：答题前会先长篇 `<think>…</think>` 再给结果，单请求 token 消耗和延迟偏高。评测/调用时给足 `max_tokens`（代码任务建议 ≥4096，否则可能写代码前就被截断）。

## 实测（Laguna vs Qwen3.5-122B，同批）

| benchmark | Laguna | Qwen3.5-122B |
|---|---|---|
| HumanEval (pass@1) | 90% | 100% |
| MBPP (pass@1) | **93%** | 77% |
| GSM8K 数学 | 95% | 97.5% |
| MMLU 英文知识 | 62% | 88% |
| CMMLU 中文知识 | 60% | 78% |
| decode TPS（单流+DFlash） | ~38 tok/s | — |

结论：**代码/数学是 Laguna 强项（MBPP 反超）；广域学科知识弱于通用模型 Qwen**。

## 注意事项

- **统一内存互斥**：占 ~110 GiB，和同机 ds4-flash-iq2-dspark / 其它大模型不能同时跑。切换前 `docker compose down`。
- 冷启约 10 分钟（加载 72 GiB + EAGLE3 草稿 + KV profiling）；compose 已配 `restart: unless-stopped` 和 600s health start_period。
- ModelScope CLI 建议装独立 venv（部分机器系统 apt 有依赖冲突）。

## 参考

- 模型：<https://modelscope.cn/models/poolside/Laguna-S-2.1-NVFP4> · DFlash：<https://modelscope.cn/models/poolside/Laguna-S-2.1-DFlash-NVFP4>
- 引擎：<https://github.com/vllm-project/vllm>
