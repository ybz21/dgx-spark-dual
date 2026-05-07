# 方案 A：runai_streamer ✅ 实测可行

> **状态：实测 4 min 启动（vs 12.5 min 基线，加速 60%），不 OOM，已固化到生产 compose**

## 思路

vLLM 0.19+ 自带 `--load-format runai_streamer`：
- RunAI（NVIDIA 收购）的 streaming weight loader
- 设计目标：从远程对象存储（S3 / HTTP）按 chunk 顺序读 weight，**不 mmap 整个文件**
- 本地文件场景下也可用：用 `pread()` 顺序读 chunk → buffer → 写 GPU → 释放 buffer
- INC dispatch + Marlin repack 仍在跑，但跟 IO 流水线**重叠**

## 为什么避免 OOM

vLLM 默认 tensorizer 加载 67GB tensorized cache 时会 mmap 整个文件，DGX Spark unified memory（CPU+GPU 共 119GB）撑不住 67GB mmap + 67GB GPU = 134GB → OOM。

runai_streamer 单次内存峰值 = chunk size（默认 16-64 MB）+ 当前 layer GPU 数据，**远小于 67GB**。

## 实施

### Docker compose 改两处

```yaml
services:
  vllm:
    # 1. entrypoint 改成 wrapper，先装 runai-model-streamer 再 exec 原 entrypoint
    entrypoint: ["/runai-bootstrap.sh"]
    volumes:
      - ./runai-bootstrap.sh:/runai-bootstrap.sh:ro
      # ... 其它 mount 不变
    command:
      # ... 原参数不变
      # 2. 加 --load-format
      - --load-format
      - runai_streamer
      - --model-loader-extra-config
      - '{"concurrency":1,"memory_limit":4294967296}'
```

`runai-bootstrap.sh`：

```bash
#!/bin/bash
set -e
pip install --quiet --index-url https://mirrors.aliyun.com/pypi/simple/ \
    runai-model-streamer boto3 2>&1 | tail -3
exec /opt/entrypoint.sh "$@"
```

### 配置说明

| 参数 | 含义 | 调整建议 |
|---|---|---|
| `concurrency` | 并行 reader 数 | 1 起步；调到 2-4 可能更快但要测 OOM |
| `memory_limit` | streaming buffer 上限（字节） | 4GB 已够；不要超过 8GB 否则跟 GPU 抢内存 |

## 实测结果（2026-05-06，DGX Spark `192.168.130.12`）

| 阶段 | A (runai) | A+B 基线（cache 命中） | 节省 |
|---|---|---|---|
| Python init / arg / 架构识别 | 13 s | 13 s | 0 |
| EngineCore / NCCL init | 12 s | 7 s | +5 s |
| **weight load + INC + Marlin repack** | **165 s** | **405 s** | **-240 s** ⭐ |
| MTP draft load + setup | 0.5 s | ~52 s | -51 s |
| torch.compile 主模型（AOT cache hit） | 8.40 s | 8.26 s | 0 |
| Initial profiling/warmup 主模型 | 33.71 s | 35.96 s | -2 s |
| torch.compile MTP（cache hit） | 0.10 s | 0.11 s | 0 |
| profiling/warmup MTP | 0.17 s | 0.18 s | 0 |
| CUDA graph capture（8 sizes） | 1 s | 4 s | -3 s |
| Multi-modal warmup | 4.99 s | 5.89 s | -1 s |
| API server start | 4 s | 30 s | -26 s |
| **总计** | **241 s (4 min)** | **596 s (9.9 min)** | **-355 s (60%)** |

## 关键观察

### 1. 没出现 "Loading weights took XXX seconds" 日志

runai_streamer 完全绕过 `default_loader.load_weights` 路径。它走自己的 fast-path，并行 stream 所有 safetensors shards 到 GPU。

实测 165 s 完成原本 405 s 的 INC + repack 工作，约 **2.5x 加速**——并行 IO + GPU pipeline 跟 CPU 反量化 overlap 了。

### 2. MTP draft load 几乎为 0

原默认路径 MTP draft 要 52 s（再跑一遍 default_loader）。runai_streamer 一次性 stream 所有 weights（含 MTP），所以 MTP 阶段只剩共享 embedding/lm_head 的 setup（0.5 s）。

### 3. 不 OOM

不像 tensorizer 那样 mmap 整个文件。按 chunk pread → buffer → GPU → release buffer，**单次 buffer < 4GB**。

### 4. 不需要预生成 cache

A 直接读原始 `.safetensors` 文件。对比 path 3（tensorizer cache）：
- tensorizer 要 13 min 预生成 67GB cache
- 启动时仍 OOM
- A 完胜：**零额外磁盘、零预处理、不 OOM、4 min 启动**

## 完整 compose 示例

见 [`deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml`](../../deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml)（已合入主 compose）。

如需手动复现：

```bash
# 在目标机上
mkdir -p ~/lm_scripts
cp deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml ~/lm_scripts/
cp scripts/启动加速/runai-bootstrap.sh ~/lm_scripts/
cd ~/lm_scripts
docker compose -f docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml up -d

# 等 healthy（应该 4 min 左右）
watch -n 5 'docker inspect vllm-qwen35 --format "{{.State.Health.Status}}"'
```

## 风险与维护

| 风险 | 缓解 |
|---|---|
| `runai-model-streamer` pypi 拉不到 | 用 nexus mirror（box 自带）/ aliyun mirror |
| 长期稳定性较 vLLM 默认路径测试覆盖少 | 每次升级 vLLM 镜像后跑一次启动时间回归 |
| `concurrency=1` 不是最快配置 | 后续可以试 2-4 看是否触发 OOM |
| pip install 每次启动多耗 5-10 s | 长期可 build 自定义镜像内置 `runai-model-streamer + boto3` |

## 推荐：build 自定义镜像（可选）

```dockerfile
FROM vllm-qwen35-v2:v26041616
RUN pip install --no-cache-dir \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    runai-model-streamer boto3
```

build 后 entrypoint 不需要 wrapper，直接用原始 `/opt/entrypoint.sh`，再省 5-10 s。

## 复现脚本

[`scripts/启动加速/A-runai.sh`](../../scripts/启动加速/A-runai.sh) 把所有步骤自动化（启容器 + 等 healthy + 抓日志 + 输出 result.json）。
