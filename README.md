# DGX Spark 大模型推理部署

DGX Spark (GB10) 上的大模型部署方案合集：单节点 / 双节点（ConnectX-7 光缆 + TP=2）都有。
**所有模型部署集中在 [`models/`](models) 下，按模型名建文件夹**，
每个模型目录下是该模型的一套或多套部署方案 + 实测报告。

## 模型索引

| 模型 | 目录 | 方案 | 节点 | 上下文 | 单流 decode |
|---|---|---|---|---|---|
| **DeepSeek-V4-Flash** | [`models/deepseek-v4-flash/`](models/deepseek-v4-flash) | vLLM + DSpark + NVFP4 KV | 2 | 1M | 69–85 tok/s |
| | | ds4 引擎 + IQ2 GGUF | 1 | 32K–512K | ~38 tok/s |
| **GLM-5.3-Flash** | [`models/glm-5.3-flash/`](models/glm-5.3-flash) | EXL3 4bpw + DFlash2 投机（Entrpi） | 2 | 524K–1M | 30–72 tok/s（待实测） |
| **Qwen3.5-122B-A10B** | [`models/qwen3.5-122b-a10b/`](models/qwen3.5-122b-a10b) | vLLM INT4 AutoRound + MTP-2 | 1 | 128K | 38–46 tok/s |
| | | vLLM NVFP4 TP=2（`quick-start.sh`） | 2 | 32K | ~17 tok/s |
| **Qwen3.5-35B-A3B** | [`models/qwen3.5-35b-a3b/`](models/qwen3.5-35b-a3b) | vLLM NVFP4 | 1 | 32K | ~30 tok/s |
| **Qwen3.8-27B** | [`models/qwen3.8-27b/`](models/qwen3.8-27b) | vLLM + MTP（NVFP4） | 1 | 32K–1M | ~24–26 tok/s |
| | | SGLang + DSpark（NVFP4） | 1 | 32K | ~34–38 tok/s |
| **Qwen3.5-397B-A17B** | [`models/qwen3.5-397b-a17b/`](models/qwen3.5-397b-a17b) | vLLM INT4 AutoRound TP=2 | 2 | — | 仅配置留档 |
| **Laguna-S-2.1** | [`models/laguna-s-2.1/`](models/laguna-s-2.1) | vLLM NVFP4 + DFlash 投机 | 1 | — | 见目录 |
| **Gemma 4 26B-A4B** | [`models/gemma4-26b-a4b/`](models/gemma4-26b-a4b) | vLLM BF16 | 1 | — | 仅配置留档 |
| **BGE Embedding** | [`models/bge-embedding/`](models/bge-embedding) | embedding-server | 1 | — | 未验证 |

- 跨模型评测：[`eval/`](eval)
- 配套服务（open-webui / 代理层 / 共享 vLLM 插件）：[`services/`](services)

## 双节点通用入口 (quick-start.sh)

`quick-start.sh` 是 NVFP4 双节点 TP=2 的一键脚本，容器运行时在 [`runtime/`](runtime)。
各模型的 `.env` 预设放在**各自模型文件夹的 `presets/` 下**（脚本不自动加载，是参数参考）。
其它模型走各自目录下的部署文件。

```bash
bash quick-start.sh <工作节点IP> <模型路径>
```

```bash
# 示例: 部署 Qwen3.5-122B (双节点 TP=2)
bash quick-start.sh 192.168.130.8 ~/models/Qwen3___5-122B-A10B-NVFP4

# 指定镜像
bash quick-start.sh 192.168.130.8 ~/models/YOUR_MODEL --image your-image:tag

# 仅校验不执行
bash quick-start.sh 192.168.130.8 ~/models/YOUR_MODEL --dry-run
```

脚本自动完成全部流程:

1. **预检校验** — Docker、镜像、模型文件、光口检测、高速网连通性
2. **同步镜像** — 通过高速网将 Docker 镜像传输到工作节点
3. **同步模型** — 通过高速网 rsync 模型到工作节点
4. **生成配置** — 自动检测量化方式、网络接口、NCCL/GLOO 参数
5. **启动服务** — Ray Head + Worker 组建集群，vLLM TP=2 推理
6. **验证部署** — 健康检查 + 推理测试

### 管理命令

```bash
bash quick-start.sh --status                    # 查看服务状态
bash quick-start.sh --stop 192.168.130.8        # 停止服务
docker logs -f --tail 50 vllm-spark-head        # 查看日志
```

### 选项

| 选项 | 说明 |
|------|------|
| `--image IMAGE` | 指定 Docker 镜像 (默认 ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603) |
| `--port PORT` | API 端口 (默认 30000) |
| `--max-len LEN` | 最大上下文长度 (默认 8192) |
| `--no-sync-model` | 跳过模型同步 |
| `--no-sync-image` | 跳过镜像同步 |
| `--dry-run` | 仅校验，不执行 |
| `--status` | 查看服务状态 |
| `--stop WORKER_IP` | 停止服务 |

## 自动检测

脚本自动检测以下配置，无需手动指定:

- **光口**: 遍历 ConnectX-7 四个口，找到有光缆连接 (carrier=1) 的接口
- **高速网 IP**: 从活跃光口读取已配置的 IP 地址
- **量化方式**: 从模型 config.json / hf_quant_config.json 自动识别
- **GLOO 接口**: 从管理网 IP 反推网口名
- **光口适配**: 两台光口名不同时自动切换 NCCL 检测模式

## 硬件

| 节点 | 角色 | GPU | 内存 | 互联 |
|------|------|-----|------|------|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |

## 架构

```
spark01 (head)                    spark02 (worker)
+-----------------------+        +-----------------------+
|  Ray Head (6379)      |        |  Ray Worker           |
|  vLLM API (:30000)    |<------>|                       |
|  GB10 GPU             | RoCE   |  GB10 GPU             |
|  TP rank 0            | 200G   |  TP rank 1            |
+-----------------------+        +-----------------------+
```

## 已验证模型

| 模型 | 量化 | TP | 镜像 | 速度 |
|------|------|----|------|------|
| Qwen3.5-122B-A10B-NVFP4 | compressed-tensors | 2 | vllm-spark:v019-ngc2603 | ~17 t/s |
| Qwen3.5-35B-A3B-NVFP4 | modelopt_fp4 | 1 | sglang-dev-cu13-accel | ~30 t/s |
| DeepSeek-V4-Flash-0731 | NVFP4 KV + DSpark | 2 | vllm-dspark-runtime:dspark-nvfp4-stage-c | 69–85 t/s |

## 模型预设

`quick-start.sh` 的 `.env` 预设按模型分散在各模型文件夹的 `presets/` 下:

```
models/qwen3.5-122b-a10b/presets/
├── qwen3.5-122b-nvfp4.env       — Qwen3.5 122B NVFP4 (TP1)
├── qwen3.5-122b-nvfp4-tp2.env   — Qwen3.5 122B NVFP4 (TP2)
├── qwen3.5-122b-fp8.env         — Qwen3.5 122B FP8 (TP2)
├── intel-122b-int4.env          — Intel INT4 AutoRound (TP1)
├── redhatai-122b-nvfp4.env      — RedHatAI NVFP4 (TP1)
├── wangzhang-122b-fp8.env       — abliterated FP8 (TP2)
└── wangzhang-122b-nvfp4.env     — abliterated NVFP4 (TP1)
models/qwen3.5-397b-a17b/presets/qwen3.5-397b-int4.env   — Qwen3.5 397B INT4 (TP2)
models/gemma4-26b-a4b/presets/gemma4-26b-a4b.env         — Gemma 4 26B MoE (TP1)
```

> ⚠️ `quick-start.sh` **不会自动读取**这些 `.env`，它们是参数参考；
> 实际用的是 [`runtime/.env.example`](runtime/.env.example) 和命令行参数。

## API

兼容 OpenAI 格式:

```bash
curl http://192.168.130.16:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3___5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 500
  }'
```

## 目录结构

```
dgx-spark-multinode/
├── README.md
├── quick-start.sh              # 双节点 NVFP4 TP=2 一键入口
├── runtime/                    # quick-start.sh 的容器运行时
│   ├── docker-compose.yml      # vLLM head + worker 编排
│   ├── entrypoint.sh           # 容器入口 (TP1直连/TP2 Ray)
│   ├── .env.example            # 配置模板
│   └── patches/                # DGX Spark SM121 兼容补丁
│
├── models/                     # ── 所有模型部署，按模型名分目录 ──
│   │                           #    注：这里放的是部署方案，不是权重；
│   │                           #    权重在目标机的 ~/models/ 下
│   ├── deepseek-v4-flash/
│   │   ├── README.md           #    两套方案对比 + 选型
│   │   ├── vllm-dspark-2x-nvfp4/   # 双节点 vLLM + DSpark，1M 上下文
│   │   └── ds4-gguf-iq2-1x/        # 单节点 ds4 引擎 + IQ2 GGUF
│   ├── glm-5.3-flash/          #    EXL3 4bpw + DFlash2，双节点 TP=2
│   │   └── exl3-2x-entrpi/         # .8 head + .12 worker，524K 上下文
│   ├── qwen3.5-122b-a10b/      #    INT4 AutoRound 单机 + MTP-2
│   │   └── README.md · deploy/ docs/ scripts/ reports/ presets/
│   ├── qwen3.5-35b-a3b/        #    NVFP4 单机，和 122B 互斥
│   ├── qwen3.5-397b-a17b/      #    仅配置留档
│   ├── laguna-s-2.1/
│   ├── gemma4-26b-a4b/         #    仅配置留档
│   └── bge-embedding/          #    未验证
│
├── services/                   # 配套服务：open-webui / 代理层 / 共享 vLLM 插件
├── eval/                       # 跨模型质量 / 速度评测
└── legacy/                     # 旧版 SGLang 部署脚本（已废弃）
```

## 前置准备

### SSH 免密

```bash
ssh-copy-id ai@192.168.130.16
ssh-copy-id ai@192.168.130.8
# Spark 之间也要互通
ssh ai@192.168.130.16 "ssh-copy-id ai@192.168.130.8"
```

### 高速网 IP

两台 DGX Spark 用光缆直连同一组 ConnectX-7 网口，手动配置 IP:

```bash
# spark01
sudo ip addr add 10.0.0.1/24 dev enp1s0f0np0
# spark02
sudo ip addr add 10.0.0.2/24 dev enp1s0f0np0
```

### Docker 镜像

```bash
# 从南大镜像拉取 (国内快)
docker pull ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603
# 或从 GitHub 原站
docker pull ghcr.io/bjk110/vllm-spark:v019-ngc2603
```

## 注意事项

- DGX Spark 是统一内存架构，122B 模型需双节点 TP=2
- 当前使用 NCCL TCP Socket (`NCCL_IB_DISABLE=1`)，后续可配置 RoCE
- 模型文件需在两台机器的相同路径下
- `runtime/docker-compose.yml` 配置了 `restart: unless-stopped`

## 致谢

- [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) — DGX Spark vLLM 适配和 SM121 补丁
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
