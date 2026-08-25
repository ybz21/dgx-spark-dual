# Qwen3.8-27B · vLLM + MTP · 单节点 NVFP4

单节点 DGX Spark (GB10) 用 **vLLM** 跑 **Qwen3.8-27B-NVFP4**，**内嵌 MTP 头做投机解码**。主模型架构 `Qwen3_5ForConditionalGeneration`（`model_type=qwen3_5`），与本仓库 Qwen3.5 部署同族——直接复用既有 vLLM 镜像。

| | |
|---|---|
| 引擎 | vLLM（LAN 镜像 `192.168.130.23:5000/vllm-qwen35-v2:v26041616`，已服务 qwen3_5+NVFP4） |
| 模型 | `unsloth/Qwen3.8-27B-NVFP4`（27B 稠密 / 64 层 / 混合线性+全注意力 / 多模态，~23 GiB） |
| 量化 | compressed-tensors `mixed-precision`（NVFP4，vLLM 自动识别） |
| 投机解码 | **MTP**（`model_mtp.safetensors` 内嵌头，`num_speculative_tokens=5`） |
| 上下文 | 本配置起 32K；内存宽裕可调到 256K（最高 ~1M） |

## 硬件

NVIDIA GB10（DGX Spark），aarch64、CUDA 13、128 GiB 统一内存；`gpu-memory-utilization=0.45` 时运行占 ~55 GiB。与同机其它大模型**互斥**。

## 部署

```bash
# 1. 下模型（ModelScope，~23GB；外网慢就异地下好 rsync 到 ~/models/）
bash scripts/download.sh

# 2. 起服务（复用 LAN 镜像，无需联网拉镜像）
cd deploy && docker compose up -d && docker compose logs -f    # 冷启 3–6 分钟

# 3. 自测
bash scripts/smoke-test.sh 192.168.130.48 8000
```

`deploy/docker-compose.yml` 关键参数：`--trust-remote-code`、`--gpu-memory-utilization 0.45`、`--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`、`--max-model-len 32768`、prefix caching、`0.0.0.0:8000`。

> **若镜像不支持 MTP**：删掉 compose 里 `--speculative-config` 两行即退到纯连续解码（功能不受影响，decode 略慢）。

## API

OpenAI 兼容，`model=qwen3.8-27b`：

```bash
curl http://192.168.130.48:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
```

## 实测

见 [`docs/deploy-report.md`](docs/deploy-report.md)。论坛参考：decode ~24–26 tok/s（MTP n=5），prefill 4K ~1700 / 48K ~850 tok/s，TTFT ~0.4s。

## 参考

模型 <https://modelscope.cn/models/unsloth/Qwen3.8-27B-NVFP4> · 引擎 <https://github.com/vllm-project/vllm>
