# Qwen3.5-122B-A10B

单节点 DGX Spark (GB10) 跑 Qwen3.5-122B-A10B INT4 量化版，vLLM + FLASHINFER + **MTP 投机解码**。

和仓库根目录 `quick-start.sh` 的 **多节点 TP=2 NVFP4** 方案互为补充：

|  | 多节点 (根目录 quick-start.sh) | **本方案 (qwen3.5-122b-a10b/)** |
|---|---|---|
| 模型量化 | NVFP4 | INT4 AutoRound (Intel) |
| 部署 | 2× DGX Spark，TP=2 over ConnectX-7 | **单节点** |
| 跨机通信 | Ray + NCCL | 无 |
| MTP 投机 | 否 | **是 (k=2)** |
| 上下文 | 32K | **128K** |
| 首 token 延迟 @ 32k | — | 17s |
| Decode 吞吐 | — | 38–46 tok/s |
| **启动时间** | — | **4 min（实测，A+B+C+D 启动加速生效后）** ⚡ |
| 适合场景 | 跨节点显存扩展 | 单机极致吞吐 + 长上下文 |

参考上游：<https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4>

## 目录结构

```
qwen3.5-122b-a10b/
├── README.md                 本文件（方案总览 + 快速开始）
├── docs/
│   ├── 部署指南.md           从零部署、参数调优、故障排查
│   ├── INT4-单机部署指南.md  内存分析 + SM121 镜像构建 + MTP-2 细节
│   ├── TTFT瓶颈分析.md       TTFT 瓶颈分析 + 上下文长度经验指南
│   └── 启动加速/             ⭐ 启动时间从 12.5min → 4min 的优化方案
│       ├── README.md         三方案对比 + 阶段时间拆解 + 实测汇总
│       ├── A-runai-streamer.md   方案 A（已固化）：runai_streamer 并行 stream
│       ├── B-tensorizer-shard.md 方案 B（未实测）：tensorizer 多文件分片
│       └── C-custom-loader.md    方案 C（未实测）：手写 custom dump+load
├── deploy/
│   ├── docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml  ⭐ 已合入 A+B+C+D 启动加速
│   └── runai-bootstrap.sh           entrypoint wrapper：装 runai-model-streamer
├── presets/                  quick-start.sh 的 .env 参数参考（7 个 122B 变体）
├── scripts/
│   ├── bench_llm.py                 needle + latency + 并发压测
│   ├── soak_test.py                 长久稳定性（混合负载 + 正确性探针 + 分时漂移）
│   └── 启动加速/                    PoC 自动化脚本（A/B/C 一键复现 + 对比）
│       ├── run-all.sh, A-runai.sh, B-shard.sh, C-custom.sh
│       └── _common.sh, launch-overnight.sh, runai-bootstrap.sh
└── reports/
    ├── 基准报告-128k.md + .json       128k 上下文基准（.12）
    ├── 基准报告-256k.md + .json       256k 上下文对比（.8）
    ├── TTFT曲线测试.md + .json        细粒度 TTFT 曲线（2k→96k）
    └── 稳定性测试报告-1h.md + .json   1 小时稳定性结果（1393 reqs · 100% 成功）
```

## 快速使用

**前置**：模型 `Qwen3.5-122B-A10B-int4-AutoRound-Intel` 放在 `~/models/` 下（72GB），镜像 `vllm-qwen35-v2` 已就绪。

```bash
# 1. 部署（目标机器上）
scp deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml ai@<host>:~/lm_scripts/
scp -r ../services/services ai@<host>:~/lm_scripts/      # 依赖 qwen3_nothink parser
ssh ai@<host> 'cd ~/lm_scripts && docker compose up -d'

# 2. 等待健康（≈ 10–15 min，加载 72GB 权重 + 编译 FLASHINFER）
ssh ai@<host> 'docker ps --filter name=vllm-qwen35 --format "{{.Status}}"'

# 3. 调用
curl http://<host>:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-122b-int4","messages":[{"role":"user","content":"你好"}]}'

# 4. 基准测试
python3 scripts/bench_llm.py --base http://<host>:30000 --out reports/基准报告-latest.md
```

详见 [部署指南](./docs/部署指南.md) 和 [TTFT 瓶颈分析](./docs/TTFT瓶颈分析.md)。

## 当前已部署节点

| 节点 | IP | 状态 |
|---|---|---|
| spark-c915 | `192.168.130.12` | 运行中 · `http://192.168.130.12:30000/v1` |

## 关键参数速查

- 端口：`30000`（host network）
- 模型名：`qwen3.5-122b-int4`
- 上下文上限：**131072** (128K)
- 显存预算：`--gpu-memory-utilization 0.90`（GB10 统一内存 128GB，约 101GB 被占）
- 投机解码：`mtp` · `num_speculative_tokens=2`（模型只有 1 层 MTP，再调大收益递减）
- Tool calling：`qwen3_coder` parser
- Reasoning：`qwen3` parser，关闭 thinking 传 `chat_template_kwargs.enable_thinking=false`
