# DeepSeek-V4-Flash · ds4 引擎 · 单节点

单节点 DGX Spark (GB10) 用 [antirez/ds4](https://github.com/antirez/ds4) 的性能分支 [Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) 跑 **DeepSeek-V4-Flash**（IQ2XXS ~2-bit GGUF），内置 **MTP + DSpark 无损投机解码**。

和本仓库其它方案的区别：这不是 vLLM，而是 antirez 手写的 CUDA 推理引擎（C/CUDA），为 GB10/SM121 专门优化，主打**极长上下文**（原生稀疏注意力，实测 500K+）和 in-process 快速路径。

|  | 本方案 (ds4-flash-iq2-dspark/) | vLLM 方案 (../laguna-s2.1-nvfp4-dflash, ../122b-boost) |
|---|---|---|
| 引擎 | antirez/ds4（自研 CUDA） | vLLM |
| 模型 | DeepSeek-V4-Flash | Laguna / Qwen 等 |
| 量化 | IQ2XXS / Q2_K（~2-bit，81 GiB） | NVFP4 / INT4 |
| 投机解码 | MTP + DSpark（EAGLE 类，无损） | DFlash / MTP |
| 上下文 | 32K–512K（稀疏注意力） | 32K–128K |
| 并发 | 连续批处理（自动） | 连续批处理 |

## 硬件要求

- NVIDIA GB10（DGX Spark）/ 或 Blackwell `sm_120`，**aarch64**
- CUDA **13.x**、≥110 GiB 空闲磁盘、128 GiB 统一内存
- 单卡即可；模型加载后占 ~104 GiB 统一内存（与同机其它大模型**互斥**）

## 快速开始

官方一键脚本（含编译 + 下载 + smoke test + 起服务）：

```bash
curl -sSL https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh | bash -s -- --start
```

### 国内网络（huggingface.co 被污染）

`huggingface.co` 在国内会 DNS 污染，需改用 `hf-mirror.com`：

```bash
curl -sSL -o install.sh https://raw.githubusercontent.com/entrpi/ds4-on-spark/main/install.sh
sed -i 's#https://huggingface.co/#https://hf-mirror.com/#g' install.sh
bash install.sh --start          # 或 --no-download 先只编译
```

单流下载慢（~4 MB/s）时，用 `scripts/parallel-download.sh`（16 连接分片续传，可拉到 ~14 MB/s）先把 GGUF 下好，再 `install.sh --no-download --start`。

需要下载的文件（都来自 `antirez/deepseek-v4-gguf` + `bleysg/...DSpark-drafter`）：

| 文件 | 大小 | 作用 |
|---|---|---|
| `DeepSeek-V4-Flash-IQ2XXS-...-chat-v2-imatrix.gguf` | 81 GiB | 主模型 |
| `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | 3.6 GiB | MTP 投机头 |
| `DSpark-drafter-Q2K-Q8.gguf` | 6.5 GiB | DSpark 草稿模型 |

## 启动服务

```bash
# 满血 DSpark 投机解码栈，绑 0.0.0.0:8000
~/.local/bin/ds4-serve --host 0.0.0.0 --port 8000 -c 32768

# 更大上下文（内存够时）
~/.local/bin/ds4-serve --host 0.0.0.0 --port 8000 -c 131072
```

- `-c` = 上下文预算（并发请求共享）。KV 用 FP8 压缩，128K 约 3 GiB。
- 默认开 DSpark；`--no-dspark` 退到 MTP，`--no-spec` 纯连续解码。
- 日志 `~/ds4-server.log`。

## API

OpenAI 兼容，`model=deepseek-v4-flash`：

```bash
curl http://<host>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"think":false,"max_tokens":256}'
```

`"think": false` 关思考模式；`reasoning_effort: "max"` 开 Think-Max（需 ctx ≥ 384K）。

## 实测性能（GB10，单流）

| 上下文 | Prefill | Decode(裸引擎) |
|---|---|---|
| 4K | ~880 tok/s | ~19 tok/s |
| 32K | ~690 tok/s | ~18 tok/s |

- 开 DSpark 投机后单流 decode ~38 tok/s；并发聚合到 8 路约 ~50 tok/s（128K 下 max_seq≈10）。
- 长上下文大海捞针：至 60K 深度 9/9 命中。

## 注意事项

- **统一内存互斥**：占 ~104 GiB，和同机其它大模型（如 vLLM-Qwen/Laguna）**不能同时跑**。切换前 `pkill ds4-server`。
- apt 在部分机器有依赖冲突，装工具优先用纯 bash/curl 或独立 venv。
- 源码 `~/code/ds4`、模型 `~/gguf`、启动器 `~/.local/bin/ds4-serve`。

## 参考

- 引擎：<https://github.com/antirez/ds4> · 分支：<https://github.com/Entrpi/ds4-on-spark>
