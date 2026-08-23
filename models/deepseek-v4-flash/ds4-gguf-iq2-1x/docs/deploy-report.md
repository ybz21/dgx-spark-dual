# DeepSeek-V4-Flash 部署与性能测试报告

> 部署项目：[Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) · DwarfStar 4 (ds4) 引擎
> 日期：2026-07-24 · 硬件：NVIDIA DGX Spark (GB10)

---

## 1. 部署概况

| 项目 | 内容 |
|---|---|
| 目标机 | `ai@192.168.130.12`（主机名 `spark-c915`） |
| 硬件 | NVIDIA GB10 / aarch64 / 统一内存 128 GiB / CUDA 13.0 |
| 引擎 | ds4-server，native 编译 `sm_121`（`make cuda CUDA_ARCH=sm_121`） |
| 模型 | DeepSeek-V4-Flash · IQ2XXS/Q2K 量化 GGUF（80.8 GiB） |
| 投机解码 | MTP（3.6 GiB）+ DSpark drafter（6.5 GiB），默认开启（无损） |
| 服务地址 | `http://192.168.130.12:8000`（OpenAI 兼容） |
| model id | `deepseek-v4-flash` |
| 上下文窗口 | 131072（128K） |
| GPU 内存占用 | ~104 GiB / 128 GiB |

### 目录 / 文件

| 用途 | 路径 |
|---|---|
| 引擎源码（已编译） | `~/code/ds4/` |
| 模型权重 | `~/gguf/` |
| 启动器 | `~/.local/bin/ds4-serve` |
| 服务日志 | `~/ds4-server.log` |

### 启动 / 停止

```bash
# 启动（128K 上下文，局域网可访问，满血 DSpark 投机解码）
~/.local/bin/ds4-serve --host 0.0.0.0 --port 8000 -c 131072

# 停止
pkill -f ds4-server
```

---

## 2. 接口用法

```bash
curl http://192.168.130.12:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role":"user","content":"你好"}],
    "think": false,
    "max_tokens": 256
  }'
```

- `GET /v1/models` — 模型信息
- `POST /v1/chat/completions` — 对话补全
- 默认 thinking 模式；`"think": false` 关闭；`reasoning_effort: "max"` 开 Think-Max（需 ctx ≥ 384K）
- 支持 `tools` / `tool_choice`（工具调用也走投机解码路径）

---

## 3. 功能验证

| 项目 | 结果 |
|---|---|
| 英文问答（capital of France） | ✅ "The capital of France is Paris." |
| 中文推理（鸡兔同笼） | ✅ 6 只，算式正确 |
| 数学（进水/放水） | ✅ 12 小时，推导清晰 |
| 中文写作（秋天小诗） | ✅ 通顺 |
| 长上下文大海捞针（43K token，65% 深度埋针） | ✅ 精确命中 `SPARK-GB10-778-蓝鲸` |
| 局域网访问（从 130.19 直连） | ✅ 通 |

---

## 4. 性能测试

### 4.1 Prefill / Decode（`ds4-bench` 官方口径，单流，裸引擎不含投机解码）

| 上下文深度 | Prefill (prompt) | Decode (生成) | 首 token 延迟* |
|---:|---:|---:|---:|
| 2K | 582 tok/s（冷启） | 21.9 tok/s | 56 ms |
| 4K | **880 tok/s**（峰值） | 19.1 tok/s | 48 ms |
| 8K | 829 tok/s | 18.8 tok/s | 55 ms |
| 16K | 765 tok/s | 18.5 tok/s | 56 ms |
| 24K | 725 tok/s | 18.2 tok/s | 57 ms |
| 32K | 689 tok/s | 18.0 tok/s | 62 ms |

\* 单个 2048-token prefill chunk 的处理时间

- **Prefill**：~690–880 tok/s（符合 README 标称 GB10 ~800 tok/s）
- **Decode**：裸引擎贪心 ~18–22 tok/s；32K 深度相比 4K 仅掉 ~6%，长上下文友好（稀疏注意力 + FP8 压缩 KV）
- 服务态默认开 **DSpark 投机解码**，结构化/代码内容 decode 更快（README 标称 1.2–1.7×）

### 4.2 多并发吞吐（服务态，128K + DSpark，中文散文负载）

| 并发 | 聚合吞吐 | 单请求均速 | 相对 1 路 |
|---:|---:|---:|---:|
| 1 | 13.1 tok/s | 13.1 tok/s | 1.0× |
| 2 | 26.6 tok/s | 14.6 tok/s | 2.0× |
| 4 | 39.8 tok/s | 10.7 tok/s | 3.0× |
| 8 | **49.5 tok/s** | 7.2 tok/s | 3.8× |
| 16 | 48.4 tok/s | 3.8 tok/s | 3.7×（饱和） |

- 总吞吐随并发上升，**~8 路封顶 ≈ 50 tok/s**；8→16 聚合不再涨，只增加排队延迟
- 瓶颈：128K 下每路 KV bank ~560 MiB，内存只够 `max_seq≈10` 槽位
- 1→2 路近乎线性，性价比最高

---

## 5. 上下文 vs 并发权衡

上下文窗口越大，可并发路数越少（共享统一内存）：

| 配置 | 单路 KV | 大致可并发槽位 | 适合场景 |
|---|---|---|---|
| `-c 131072`（当前） | ~560 MiB | ~10 | 少数长文档会话 |
| `-c 32768` | ~130 MiB | ~30+ | 高并发短对话 |
| `-c 8192` | ~35 MiB | 更多 | 大批量短请求 |

---

## 6. 运维注意事项

1. **与 vLLM-Qwen 内存互斥**：本机原有生产服务 `vllm-qwen35` / `qwen3-asr`（Qwen3.5-122B-A10B，占 ~92 GiB）已停以腾内存。DS4 占 ~104 GiB，**两者不能同时运行**。切换：
   ```bash
   pkill ds4-server && docker start vllm-qwen35 qwen3-asr   # 切回 Qwen
   ```
2. **当前为 nohup 进程**：可扛 SSH 断开，但**机器重启不会自动拉起**。如需开机自启，建议做成 systemd 服务。
3. **下载来源**：`huggingface.co` 被 DNS 污染，走 `hf-mirror.com`（直连，无 proxy）；用 16 连接并行分片下载器把速度从 4 MB/s 提到 ~14 MB/s。
