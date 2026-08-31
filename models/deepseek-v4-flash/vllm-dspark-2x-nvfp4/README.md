# DeepSeek-V4-Flash-0731 · vLLM TP=2 · DSpark 投机解码 · NVFP4 KV · 1M 上下文

双节点 DGX Spark (GB10) 用 vLLM TP=2 跑 **DeepSeek-V4-Flash-0731 官方权重**，
配 **DSpark 投机解码**（k=5）+ 实验性 `nvfp4_ds_mla` KV cache，**1M token 上下文**。

上游配方：[tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)（MIT）
本仓库记录的是**我们集群上的落地配置和实测数据**，不是上游的复制品。

和同模型另一方案的区别：

|  | 本方案 `vllm-dspark-2x-nvfp4/` | [`../ds4-gguf-iq2-1x/`](../ds4-gguf-iq2-1x) |
|---|---|---|
| 引擎 | vLLM（DSpark overlay 定制镜像） | antirez/ds4（自研 CUDA） |
| 节点 | **2 台，TP=2，RoCE 直连** | 单台 |
| 权重 | `deepseek-ai/DeepSeek-V4-Flash-0731`（156 GiB，48 分片 safetensors） | IQ2XXS GGUF（81 GiB） |
| KV | `nvfp4_ds_mla` | FP8 |
| 上下文 | **1M（1048576，YaRN 校准上限）** | 32K–512K |
| 投机 | DSpark k=5（+Patch 4） | MTP + DSpark drafter |
| 单流 decode | **69–85 tok/s** | ~38 tok/s |
| c6 聚合 | **244 tok/s** | ~50 tok/s |
| 100K prefill | **2605 tok/s** | — |

> 本方案速度和上下文都明显更好，代价是要占**两台机器**。单机可用时选 ds4 GGUF 方案。

---

## 我们的集群

| 角色 | 主机名 | 管理网 | 光纤 (RoCE) | 说明 |
|---|---|---|---|---|
| head / rank 0 | `spark-e9d7` | `192.168.130.8` | `10.0.0.2` | 跑 API，`:8888` |
| worker / rank 1 | `spark-c915` | `192.168.130.12` | `10.0.0.3` | `--headless` |

- 光口 `enp1s0f1np1`（HCA `rocep1s0f1`，GID index 3），`MASTER_PORT=25000`
- 部署目录两台都是 `~/ds4-dspark-2x`，HF cache 两台都是 `~/.cache/huggingface`
- 运行时镜像 `vllm-dspark-runtime:dspark-nvfp4-stage-c`（两台都已 load）

> 光纤 IP / 光口 / GID 推荐用 [`scripts/roce-autoconf/`](scripts/roce-autoconf) 自动管理（systemd 开机自启）：
> 扫描所有有光的 CX7 口，选出对端可达的那个，把点对点 IP 写成 NetworkManager 静态连接（重启不丢），
> 并把实测的 `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` 就地同步进 `.env.dspark`。
> 安装（head/worker 各一次，root）：
> ```bash
> scripts/roce-autoconf/install.sh 10.0.0.2 10.0.0.3   # head
> scripts/roce-autoconf/install.sh 10.0.0.3 10.0.0.2   # worker
> ```
> 解决的三个实测坑（46/141 双 Spark 集群部署与重启测试踩出来的）：
> 1. 手工 `ip addr add` 会被 NetworkManager 刷掉（"重启会丢"的根因，且不重启也可能丢）；
> 2. **GID index 不能写死**：它是网卡 GID 表的行号，跟口上的地址数量/顺序走，机器之间不同
>    （我们两台一台 5 一台 6），重启后还会漂移，选错 NCCL 直接 `ibv_modify_qp EINVAL` 崩环；
> 3. 光纤插错口/换口后无需改任何配置，重跑脚本即自动收敛。
> 另建议：把光口上 NM 自动生成的"有线连接"配置 `autoconnect no`——它们反复抢 DHCP 加删地址，
> 是 GID 表运行中重排的元凶。
>
> **完整的六坑实录与自愈看门狗见 [`docs/实操注意事项.md`](docs/实操注意事项.md) 与 [`scripts/watchdog/`](scripts/watchdog)。**

## 前置状态（截至 2026-08-23 已就绪，不用重做）

| 项 | 状态 |
|---|---|
| `~/ds4-dspark-2x` 检出（上游 `d728fae`） | ✅ 两台都有 |
| 0731 权重 156 GiB（HF cache） | ✅ 两台都有，已过分片头部校验 |
| `vllm-dspark-runtime:dspark-nvfp4-stage-c` 镜像 | ✅ 两台都有，且与 `recipe/overlay/` 校验一致（Patch 4 已烘进镜像） |
| `.env.dspark` | ✅ head 上已按本集群改好，见 [`deploy/.env.dspark`](deploy/.env.dspark) |
| 服务 | ⛔ **当前停着**（无容器、8888 未监听） |

## 启动 / 停止

全部在 **head（192.168.130.8）** 上执行，脚本会自己 ssh 到 worker 先起 rank 1：

```bash
cd ~/ds4-dspark-2x
./start-deepseek-v4-flash-dspark.sh      # worker 先起，再起 head，最后跑 smoke
./status-deepseek-v4-flash-dspark.sh
./logs-deepseek-v4-flash-dspark.sh
./stop-deepseek-v4-flash-dspark.sh
```

启动脚本会先拿镜像里的文件和 `recipe/overlay/` 逐个比 sha256，不一致就自动重建
（这是防"镜像里的 proposer 和 vLLM 版本对不上直接崩"的护栏）。已知一致时可以
`SKIP_OVERLAY_CHECK=1` 跳过。**冷启约 6 分钟**（156 GiB 权重 + FlashInfer autotune）。

## API

```bash
curl http://192.168.130.8:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"你好"}],
       "max_tokens":256}'
```

- `SERVED_MODEL_NAME=deepseek-v4-flash-0731`，端口 `8888`，绑 `0.0.0.0`
- 默认**关思考**（`--default-chat-template-kwargs '{"thinking":false}'`）
- 带 `--enable-auto-tool-choice` + `deepseek_v4` 的 tool-call / reasoning parser

## 实测性能（本集群，0731 + Patch 4）

单流 decode，temp 0，warm，按 server 返回的 `completion_tokens` 算：

| 提示词类型 | tokens | 秒 | tok/s |
|---|---:|---:|---:|
| 数到 300 | 600 | 7.04 | **85.2** |
| 12×12 乘法表 | 900 | 11.25 | 80.0 |
| 60 元素 JSON | 800 | 10.05 | 79.6 |
| 二叉搜索树（代码） | 600 | 9.02 | 66.5 |
| 200 词叙事散文 | 289 | 9.55 | 30.3 |
| **峰值 / 均值** | | | **85.2 / 68.3** |

并发（同一提示词，各 400 token）：

| 并发 | 聚合 tok/s | 单流 tok/s |
|---:|---:|---:|
| 1 | 69.0 | 69.0 |
| 2 | 103.8 | 51.9 |
| 4 | 172.9 | 44.3 |
| 6 | **244.2** | 41.3 |

Prefill（TTFT 法，输出 1 token）：

| 深度 | prompt tokens | 秒 | tok/s |
|---:|---:|---:|---:|
| 8K | 6,234 | 3.6 | 1,740 |
| 32K | 24,900 | 10.0 | 2,498 |
| 100K | 77,790 | 29.9 | **2,605** |

> **30 → 85 tok/s 的落差是同一个服务同一份配置**。DSpark 的 decode 速度 =
> `steps/s × 每步接受 token 数`，而接受率由内容决定（结构化/代码高、散文低）。
> 容量规划按均值，别按峰值。上游在混合真实 agent 流量下测到 c4 聚合 ~88 tok/s、
> 单流 ~22 tok/s，比这里的基准数低一半——那个才是排产用的数。

## 三个能让你白折腾几天的坑

### 1. Patch 4：不打就只有一半速度，而且输出质量完全正常

vLLM 的 DSpark **draft** 权重加载器漏了 12 个 tensor
（`model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.{weight,weight_scale_inv}`）：
loader 只重命名了 `w2`，`w1`/`w3` 匹配不到任何规则，落进 `logger.debug("Skipping unknown
DSpark weight")`——INFO 级别下**完全看不见**。这个 shared expert 是 `n_shared_experts: 1`
且**常开**，于是三级 draft 全都带着一个没初始化的专家在跑。

因为 target 模型仍然逐 token 验证，**输出质量一点不掉**，塌的只有接受率：

| | 接受率 | tok/step | mean tok/s |
|---|---:|---:|---:|
| 0731 原版 loader | 25.7% | 2.28 | 32.7 |
| **0731 + Patch 4** | **60.2%** | **4.01** | **55.4** |

"换了 0731 权重后速度腰斩但质量没变"就是这个 bug 的指纹。
**本集群的镜像已经把 Patch 4 烘进去了**（在 `recipe/overlay/vllm/v1/spec_decode/dspark.py`），
不需要额外 bind-mount。判断依据：启动脚本的 overlay sha256 校验通过。

### 2. 压测必须 `stream: false`

投机解码下 vLLM 每个 decode **step** 最多发一个 SSE chunk，里面装着这一步接受的所有
token。数流式 content delta 数出来的是 **steps/s 不是 tokens/s**，同一个请求会被低估到
14.7 vs 60.1 tok/s。读 `usage.completion_tokens`，或者用服务端
`vllm:generation_tokens_total` 除以墙钟时间。

### 3. `k` 只能是 5

`dspark_block_size = 5`。DeepSeek 官方 model card 推荐 `num_speculative_tokens: 7` —— 在这里
不成立：把启动时的整除性检查 patch 掉之后，第一次生成就会崩
（`size of tensor a (7) must match tensor b (5)`），因为 drafter 每趟只吐
`dspark_block_size` 个 token，多 block 草稿没实现。`k=10` 同理。准确规则：**`k ≤ 5`，或 5 的倍数**。

另外 `k=3` 是 2026-07-03 garble 修复时代的遗留值，Patch 3 之后不需要了，**降回 3 只是白丢 ~24% decode**。

### 已经被实验排除的（别再试一遍）

| 试过 | 结果 |
|---|---|
| `draft_sample_method` greedy vs probabilistic | 无差别——DSpark 路径上是 **no-op**（proposer 不设 `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` 就不填 draft probs） |
| `fp8_ds_mla` vs `nvfp4_ds_mla` KV | 对接受率无影响，只影响 KV 池大小 |
| temperature 0 vs 0.7 | 无差别 |
| 关掉 B12X kernels | 更慢（steps/s 14.4 → 10.7），然后崩 |
| 独占节点、零竞争流量 | 无差别——从来不是资源争抢问题 |

## 配置要点

完整配置见 [`deploy/.env.dspark`](deploy/.env.dspark) 和
[`deploy/docker-compose.dspark.yml`](deploy/docker-compose.dspark.yml)（compose 与上游一致，
留档是因为它才是启动 flag 的真正来源）。几个关键值：

| 变量 | 本集群值 | 说明 |
|---|---|---|
| `DSPARK_MODEL` | `deepseek-ai/DeepSeek-V4-Flash-0731` | 官方 0731 正式版 |
| `MAX_MODEL_LEN` | `1048576` | 1M 是 config.json 里 YaRN 的真实校准上限（65536 × factor 16）。上游历史配置的 1500000 靠 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 强开，能启动也能跑分，但超过 1M 是外推，输出连贯性无保证 |
| `MTP_NUM_TOKENS` | `5` | 见坑 3 |
| `MAX_NUM_SEQS` | `12` | ⚠️ 见下方"待确认" |
| `GPU_MEMORY_UTILIZATION` | `0.85` | ⚠️ 见下方"待确认" |
| `VLLM_USE_B12X_MOE` | `1` | **速度的全部来源**。=0 会静默退回 DEEPGEMM_MXFP4，decode 掉到 ~29 tok/s。启动日志里应看到 `Using 'B12X' Mxfp4 MoE backend` |
| `VLLM_USE_FLASHINFER_SAMPLER` | `1` | 2026-07-03 garble 修复的一部分 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内 `huggingface.co` DNS 污染 |

不能设的：

- `VLLM_USE_V2_MODEL_RUNNER=1` —— 和 DSpark 投机解码不兼容，启动直接拒绝
- `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` —— 这个镜像上没这个 backend 名，留 AUTO
- `--override-generation-config` —— 它带的 `repetition_penalty=1.05` 是已知的 DSpark 崩溃诱因（illegal memory access）

### ⚠️ 待确认：`MAX_NUM_SEQS` / `GPU_MEMORY_UTILIZATION` 与上游建议不一致

本集群 `.env.dspark` 用的是 `MAX_NUM_SEQS=12` + `GPU_MEMORY_UTILIZATION=0.85`，
而上游 `DEFAULT-CONFIG.md` 的"当前最佳"是 **6 + 0.78**，理由是：

- 投机解码的 buffer 是在**第一个真实请求**时分配的，不是启动时。`0.80` 能启动、能过
  smoke test，然后在真实流量下挂掉（上游 issue #8）。
- `max_num_seqs=12` 是同一类内存压力崩溃的已知触发条件。

上面那份 c1–c6 基准是在 12 + 0.85 下跑出来的且跑通了，但**没有做长时间 soak**。
真上生产流量前建议先按 6 + 0.78 跑一轮 soak 对比，或者至少盯着第一小时的容器状态。

## 权重下载

156 GiB × 2 台。head 上下完再走 200G 光纤同步到 worker，脚本见
[`scripts/download-0731-and-sync.sh`](scripts/download-0731-and-sync.sh)。里面规避了几个实测到的坑：

- 走 `hf-mirror.com`，`HF_HUB_DISABLE_XET=1`
- `max_workers=8`——24 会被 xet CDN 限流，6~10 是安全区
- 下载器会静默死掉（表现为 `du` 不再增长），必须 supervisor 循环拉起
- `model.safetensors.index.json` 这个 5.6MB 小文件极易反复失败，兜底从 modelscope 直取
- 同步前逐分片做 safetensors 头部长度自洽校验，不过就不同步

## 与其它服务的内存互斥

156 GiB 权重跑在 128 GiB 统一内存的机器上靠的是 TP=2 分片，**两台各占大半**。
起这个服务前 head 和 worker 上的其它大模型都得停：

- head `192.168.130.8`：`vllm-qwen35`、`vllm-ornith15-35b-nvfp4`、`laguna-vllm`（当前均已 Exited）
- worker `192.168.130.12`：ds4 GGUF 单机方案（`pkill ds4-server`），见 [`../ds4-gguf-iq2-1x/`](../ds4-gguf-iq2-1x)

> worker 这台同时也是 ds4 GGUF 单机方案的宿主。**两个 DeepSeek-V4-Flash 方案不能同时跑。**

## 参考

- 上游配方：<https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark>（MIT，Tony Deangelo）
- DSpark 并发 patch：[drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash](https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash)
- 权重：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
