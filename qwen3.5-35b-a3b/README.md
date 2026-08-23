# Qwen3.5-35B-A3B-NVFP4

单节点 DGX Spark (GB10) 跑 Qwen3.5-35B-A3B 的 NVFP4 量化版（txn545 检查点），vLLM + FLASHINFER。

比 122B 小得多（权重 ~22 GB），适合单机常驻或和其它服务共存。

| 项 | 值 |
|---|---|
| 权重 | `~/models/Qwen3.5-35B-A3B-NVFP4-txn545`，~22 GB |
| 镜像 | `vllm-qwen35-v2`（和 122B 共用，SM121 编译） |
| 端口 | 30000 |
| 上下文 | 32768 |
| `gpu-memory-utilization` | 0.60（统一内存 119 GB，给系统留余量） |
| served model name | `qwen3.5-35b-a3b-nvfp4` |
| 实测速度 | ~30 tok/s（见根 README 已验证模型表） |

> ⚠️ **和 122B 互斥**，共用同一块统一内存，不能同时启动。

## 部署

compose 依赖 [`../services/services/qwen3_nothink_reasoning_parser_vllm.py`](../services/services/qwen3_nothink_reasoning_parser_vllm.py)
（`qwen3_nothink` reasoning parser，和 122B 共享）：

```bash
scp deploy/docker-compose-model-qwen35-35b-nvfp4-vllm.yaml ai@<host>:~/lm_scripts/
scp -r ../services/services ai@<host>:~/lm_scripts/
ssh ai@<host> 'cd ~/lm_scripts && docker compose -f docker-compose-model-qwen35-35b-nvfp4-vllm.yaml up -d'
```
