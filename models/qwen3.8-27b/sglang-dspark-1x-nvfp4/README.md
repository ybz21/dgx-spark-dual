# Qwen3.8-27B · SGLang + DSpark · 单节点 NVFP4

单节点 DGX Spark (GB10) 用 **SGLang** 跑 **Qwen3.8-27B-NVFP4**，用 **DSpark 草稿模型做投机解码**（论坛社区一键方案，实测最快）。

| | |
|---|---|
| 引擎 | SGLang（LAN 镜像 `192.168.130.23:5000/sglang-dev-cu13-accel:latest`，cu13 加速版） |
| 主模型 | `unsloth/Qwen3.8-27B-NVFP4`（~23 GiB，与 vLLM 方案共用同一份） |
| 草稿模型 | `RadixArk/Qwen3.8-27B-DSpark`（1.36B，~2.7 GiB） |
| 投机解码 | **DSpark**，`--speculative-dspark-block-size 7` |
| 上下文 | 本配置起 32K；内存宽裕可调大 |

## 硬件

NVIDIA GB10（DGX Spark），aarch64、CUDA 13、128 GiB 统一内存；`--mem-fraction-static 0.50` 时占 ~64 GiB。与同机其它大模型**互斥**。

## 部署

```bash
# 1. 下权重（主模型 + DSpark 草稿；ModelScope，外网慢就异地下好 rsync）
bash scripts/download.sh

# 2. 先确认镜像支持 DSpark（DSPARK 是打过补丁的 SGLang 才有的算法）
docker run --rm 192.168.130.23:5000/sglang-dev-cu13-accel:latest \
  python3 -m sglang.launch_server --help | grep -i dspark

# 3. 起服务
cd deploy && docker compose up -d && docker compose logs -f    # torch.compile 首启编译较久

# 4. 自测
bash scripts/smoke-test.sh 192.168.130.48 8000
```

`deploy/docker-compose.yml` 关键参数：`--speculative-algorithm DSPARK`、`--speculative-draft-model-path .../RadixArk-Qwen3.8-27B-DSpark`、`--speculative-dspark-block-size 7`、`--mem-fraction-static 0.50`、`--attention-backend flashinfer`、`--enable-torch-compile`、`--max-mamba-cache-size 96`（混合线性注意力需要）、`0.0.0.0:8000`。

> **若镜像不支持 `DSPARK`**：改 `--speculative-algorithm EAGLE3`（换对应草稿模型），或直接用 [`../vllm-mtp-1x-nvfp4/`](../vllm-mtp-1x-nvfp4) 方案。

## API

OpenAI 兼容，`model=qwen3.8-27b`：

```bash
curl http://192.168.130.48:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}],"max_tokens":256}'
```

## 实测

见 [`docs/deploy-report.md`](docs/deploy-report.md)。论坛参考：~34 tok/s 实测、38 平均、峰值 46.7（GSM8K 类），约 92% 带宽上限。

## 参考

一键仓库 <https://github.com/hasso5703/dgx-spark-qwen38> · 引擎 <https://github.com/sgl-project/sglang> · 草稿 <https://modelscope.cn/models/RadixArk/Qwen3.8-27B-DSpark>
