# DGX Spark 双节点大模型推理部署

通过光缆 (ConnectX-7) 连接两台 DGX Spark (GB10)，使用 vLLM + Ray 实现跨节点 Tensor Parallelism 大模型推理。

## 硬件

| 节点 | 角色 | GPU | 内存 | 互联 |
|------|------|-----|------|------|
| spark01 | Ray Head + vLLM API | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |
| spark02 | Ray Worker | NVIDIA GB10 (Blackwell) | 128 GiB 统一内存 | 200Gbps RoCE |

## 快速开始

只需三个参数即可完成部署：**工作节点 IP**、**Docker 镜像**、**模型路径**。

```bash
bash quick-start.sh <工作节点IP> <Docker镜像> <模型路径>
```

### 示例

```bash
# 部署 Qwen3.5-122B (compressed-tensors NVFP4, 双节点 TP=2)
bash quick-start.sh 192.168.130.8 \
  ghcr.nju.edu.cn/bjk110/vllm-spark:v019-ngc2603 \
  ~/models/Qwen3___5-122B-A10B-NVFP4
```

脚本自动完成：
1. **预检校验** — Docker、镜像、模型文件、高速网络、SSH 连通性
2. **同步镜像** — 通过高速网将 Docker 镜像传输到工作节点
3. **同步模型** — 通过高速网 rsync 模型文件到工作节点
4. **生成配置** — 自动检测量化方式、网络接口、NCCL 参数
5. **启动服务** — Ray Head → Worker 加入 → vLLM TP=2 推理
6. **健康检查** — 等待模型加载完成并测试推理

### 选项

```bash
# 仅校验，不执行
bash quick-start.sh 192.168.130.8 IMAGE MODEL --dry-run

# 镜像/模型已在工作节点上，跳过同步
bash quick-start.sh 192.168.130.8 IMAGE MODEL --no-sync-image --no-sync-model

# 指定 API 端口和上下文长度
bash quick-start.sh 192.168.130.8 IMAGE MODEL --port 8000 --max-len 32768

# 停止服务
bash quick-start.sh --stop 192.168.130.8

# 查看状态
bash quick-start.sh --status
```

## 已验证模型

| 模型 | 量化 | TP | 镜像 | 速度 |
|------|------|----|------|------|
| Qwen3.5-122B-A10B-NVFP4 | compressed-tensors | 2 | vllm-spark:v019-ngc2603 | ~17 t/s |
| Qwen3.5-35B-A3B-NVFP4 | modelopt_fp4 | 1* | sglang-dev-cu13-accel | ~30 t/s |

\* 35B 模型单机即可运行，无需双节点

## 架构

```
spark01 (head)                    spark02 (worker)
┌─────────────────────┐          ┌─────────────────────┐
│  Ray Head (6379)    │          │  Ray Worker          │
│  vLLM API (:30000)  │◄────────►│                      │
│  GB10 GPU           │ 200Gbps │  GB10 GPU            │
│  TP rank 0          │  RoCE   │  TP rank 1           │
└─────────────────────┘          └─────────────────────┘
```

## API 使用

兼容 OpenAI 格式：

```bash
curl http://192.168.130.16:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3___5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 500
  }'
```

## 文件结构

```
dgx-spark-multinode/
├── quick-start.sh          # 一键部署入口 (校验+传输+启动)
├── docker-compose.yml      # vLLM head/worker compose 配置
├── entrypoint.sh           # 智能入口 (TP1 直连 / TP2 Ray)
├── .env.example            # 配置模板
├── patches/                # DGX Spark SM121 兼容补丁
│   ├── apply_sm121_patches.py
│   ├── fix_cuda13_memcpy_batch.py
│   ├── fix_pytorch211_compat.py
│   └── ...
├── deploy.sh               # SGLang 部署管理 (旧版)
├── auto-pull.sh            # 镜像自动拉取 (失败重试+源切换)
├── auto-deploy-vllm.sh     # 镜像拉完自动部署
└── scripts/
    ├── common.sh           # 共享函数库
    ├── setup-network.sh    # 高速网络配置
    └── test-connection.sh  # 连接测试
```

## 网络配置

### 高速网 (ConnectX-7 光缆)

两台 DGX Spark 用 QSFP112 光缆直连同一组网口（如 `enp1s0f0np0`），手动配置 IP：

```bash
# spark01
sudo ip addr add 10.0.0.1/24 dev enp1s0f0np0

# spark02
sudo ip addr add 10.0.0.2/24 dev enp1s0f0np0
```

### SSH 免密

```bash
# 控制机到两台 Spark
ssh-copy-id ai@192.168.130.16
ssh-copy-id ai@192.168.130.8

# Spark 之间互通
ssh ai@192.168.130.16 "ssh-copy-id ai@192.168.130.8"
```

## 注意事项

- DGX Spark 是统一内存架构 (128GB GPU+CPU 共享)，122B 模型需要双节点才能放下
- 当前使用 NCCL TCP Socket 通信 (`NCCL_IB_DISABLE=1`)，RoCE/IB 需要额外配置 GID
- `docker-compose.yml` 中的 `restart: unless-stopped` 会自动重启崩溃的容器
- 模型文件需要在两台机器的相同路径下

## 致谢

- [bjk110/spark_vllm_docker](https://github.com/bjk110/spark_vllm_docker) — DGX Spark vLLM 适配和 SM121 补丁
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理引擎
- [SGLang](https://github.com/sgl-project/sglang) — 结构化生成语言推理框架
