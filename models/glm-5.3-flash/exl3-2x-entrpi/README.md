# GLM-5.3-Flash · EXL3 4bpw + DFlash2 · 双节点 TP=2

上游方案：[Entrpi/glm-5.3-flash-exl3-2x-spark](https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark)
（本仓库记录的是 `v2.3-tier1`）。为什么选它见 [`../README.md`](../README.md)。

| | |
|---|---|
| 节点 | `192.168.130.8`（head，API）+ `192.168.130.12`（worker），TP=2 |
| 互联 | 200GbE 直连，`enp1s0f1np1` / HCA `rocep1s0f1`，`10.0.0.2` ↔ `10.0.0.3` |
| 权重 | EXL3 4bpw 164 GiB（120 shard），两台各存一份 |
| 投机 | DFlash2 MXFP8 drafter 1.2 GiB，k=7，与主模型一起 TP2 分片 |
| 引擎 | Entrpi 的 vLLM fork + ExLlamaV3 kernel，发行镜像已含全部 sm_121 补丁 |
| 上下文 | 默认 524K（KV 池 1,287,194 token），可切 1M |
| 端口 | `8000`，OpenAI 兼容，模型名 `glm-5.3-flash` |

上游实测（单请求、greedy、5 次中位数）：结构化 71–72.4、JSON 51、数学 45、
代码 42、散文 30 tok/s；TTFT 0.3–0.47 s；读文档 1,490 tok/s @133K。
并发 10 路聚合 77 tok/s；关掉 drafter 的 agent 档 12–16 路聚合 88–103 tok/s。
质量：math_500 91%、GPQA-diamond 70%、133K 文档检索 10/10。

## 当前进度

- [x] 选型：四条路线横评，定 Entrpi EXL3 4bpw
- [x] 权重下载：ModelScope 拉到 `.8`，光缆推到 `.12`（164 GiB / 96 s ≈ 1.7 GB/s）
- [x] 权重校验：两台各 133 个文件 sha256 全过（120 shard 全对，见下方"校验"）
- [x] drafter 下载：两台 `~/models/glm53-dflash2-mxfp8`
- [x] **服务已上线** 2026-09-06：`http://192.168.130.8:8000/v1`，模型名 `glm-5.3-flash`，
      `install.sh EXIT=0`，`max_model_len=524288`
- [x] 单流速度实测（见下方"实测速度"）
- [ ] 进 [`../../../eval/`](../../../eval) 流程出质量分
- [ ] 与 NVFP4 方案 A/B（权重已在两台就位，见 [`../README.md`](../README.md)）

## 实测速度（2026-09-06，本地两台实机）

单流、`temperature=0`、`max_tokens=400`：

| 负载 | 本地实测 | 上游 Entrpi |
|---|---:|---:|
| 结构化（计数 1–200） | **67.0** tok/s | 72.4 |
| 代码（LRU cache 实现） | **56.3** tok/s | 42 |
| JSON（40 个对象） | **56.0** tok/s | 51 |
| 英文散文 | 23.2 tok/s | 27.4 |
| **中文散文** | **14.9** tok/s | 上游未测 |
| 首 token 往返（短 prompt） | **0.21** s | 0.43–0.47 |

代码、JSON、TTFT 都比上游好；**中文散文只有英文的 64%**，是本地新发现的一条：
DFlash2 drafter 对中文的接受率明显低于英文，上游从没测过中文。
跑中文自由对话的实际体感会比宣传数字差不少；结构化 / 代码 / JSON 类负载则对得上。

启动实测：权重加载 82.59 GiB / 38 秒，NCCL 2.31.2 双机建链，预热 32 秒。

## 部署步骤

### 1. 腾机器（必须先做）

两台都要退到系统内存占用 < 6 GB。至少要停掉：

```bash
# 两台上的 DeepSeek-V4-Flash 双节点服务
ssh ai@192.168.130.8  'docker stop ds4-dspark-2x-vllm-dspark-1'
ssh ai@192.168.130.12 'docker stop ds4-dspark-2x-vllm-dspark-1'
```

`docker stats` 里这两台所有容器加起来才 2.5 GiB 左右，112 GiB 是 DS4 的 vLLM
占着的统一内存 —— 那部分不进容器 RSS 统计，只在 `free` 里能看到。所以**停掉 DS4
基本就够了**，`.8` 上那 20 个 blade-* 业务容器（gateway / bos / agent / hub / oauth /
gitea / grafana / prometheus / nexus…）总共不到 700 MiB，可以留着。
DS4 那套还有评测没跑完，别顺手删容器，`docker stop` 就够，回头 `start` 即可。

> Spark 是统一内存，超售不会优雅降级，是直接崩。宁可先把 `KV_CACHE_MEMORY`
> 调小（每 1 GB ≈ 9 万 token 池），也别去关 `MEM_USED_MAX_GB` 那道检查。

### 2. 装 kit

在 head(`.8`) 上：

```bash
git clone https://github.com/Entrpi/glm-5.3-flash-exl3-2x-spark
cd glm-5.3-flash-exl3-2x-spark
cp <本仓库>/models/glm-5.3-flash/exl3-2x-entrpi/deploy/.env.entrpi .env
./install.sh --skip-download        # 权重已就位，别让它再拉一遍 176 GiB
```

`install.sh` 会体检两台、拉 ~25 GiB 服务镜像、装启动脚本、先起 worker 再起 head、
预热并打一次测试请求。每一步都可重跑。

### 3. 用

```bash
curl -s http://192.168.130.8:8000/health
curl -s http://192.168.130.8:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "你好"}]
}'
```

思考模式默认关，要开传 `"chat_template_kwargs": {"enable_thinking": true}`。
支持 tool calling 和图片输入。

### 4. 重启顺序

**永远是 head 先停、worker 先起**：

```bash
ssh ai@192.168.130.8  'docker rm -f vllm_glm53'
ssh ai@192.168.130.12 '~/launch-glm53-vllm-tp2.sh 1'
ssh ai@192.168.130.8  '~/launch-glm53-vllm-tp2.sh 0'
ssh ai@192.168.130.8  '~/glm53-warmup.sh'      # 不预热第一个请求要多等 ~7 s
```

到 API 可用约 4 分钟。

## 权重与校验

见 [`../README.md#权重`](../README.md)。本目录 [`scripts/`](scripts) 下三个脚本
就是这次实际用的：

```bash
bash scripts/download-model.sh      # 在 .8 上跑：ModelScope 拉主模型 + drafter
bash scripts/sync-to-worker.sh      # 在 .8 上跑：6 路并行 rsync 走光缆推给 .12
bash scripts/verify-checkpoint.sh   # 两台各跑一次：对 SHA256SUMS 校验
```

校验结果两台一致：**ok=133 / failed=2**，失败的两个是 `LICENSE` 和 `README.md` ——
ModelScope 镜像换掉了这两个文件，120 个 shard 和全部配置文件逐字节对上，推理无影响。

`SHA256SUMS` 里另有 193 条 `runtime/`（量化流水线源码）和 `.materialization/`
条目本地没有，ModelScope 镜像没带这两块，推理也用不到，脚本已跳过。

## 配置档位

默认档就是上游的生产配置，改哪个旋钮看要什么：

| 想要 | 设 | 得到 | 代价 |
|---|---|---|---|
| 长文档、少并发（默认） | 不动 | 1,287,194 token 池，单流最快 | — |
| 更大上下文池 | `KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1` | 池 +32% | math 91→88 |
| 1M 单请求 | 上一行 + `MAX_LEN=1048576 MNBT=4096` | 池 2,144,814 | 读满 1M 要 ~15 分钟 |
| 多路 agent | `MAX_LEN=131072 MAX_SEQS=12 MNBT=4096 SPEC=none` + 两个 `MIXED_PREFILL_*` | 12–16 路聚合 88–103 tok/s | 单流速度、上限 131K |
| drafter 出问题时兜底 | `MTP=4` | 走模型自带多 token 头 | decode 慢约 20% |

旋钮写在两台的 `.env` / `~/.glm53-serve.env`，命令行变量优先级最高。

## 坑

- **光口是 f1 不是 f0**。kit 的 `.env.example` 默认 `enp1s0f0np0`/`rocep1s0f0`，
  我们这两台插的是第二个口，必须改成 `enp1s0f1np1`/`rocep1s0f1`，否则 NCCL 建不起来。
  IP 也沿用本仓库既有的 `10.0.0.2`/`10.0.0.3`，不是 kit 示例的 `10.200.0.x`。
- **`modelscope` CLI 不在非交互 shell 的 PATH 上**，在 tmux 里跑要写全
  `~/.local/bin/modelscope`，否则 `command not found`。
- **中文 locale 会让 `sha256sum -c` 的统计失灵**：它打的是"成功/失败"不是 OK/FAILED。
  校验脚本里已经 `export LC_ALL=C`。
- **swap 只有 16 GiB**，够用：默认的 `LOAD_FORMAT=instanttensor` 直接 I/O 加载不吃 swap
  （~3.6 分钟到 API）。**别切回 page-cache 加载器**，那个会吃光 swap，16 GiB 会在加载到
  九成时把 head 打死，要切先把两台 swap 扩到 32 GiB。
- **drafter 是 CC BY-NC-ND 4.0**，研究/评测可用，不要再分发。
- **`KV_CACHE_MEMORY` 只吃纯整数**。写 `12.4e9` 会被 vLLM 拒：
  `argument --kv-cache-memory-bytes: Value 12.4e9 cannot be converted`。
  必须写 `12400000000`。坑在于 kit 自己的 `.env.example` 和 README 里
  通篇是 `14.4e9` 这种写法（那个走的是 launcher 的默认分支，不经过这条校验）。
- **内存预检卡在 7 GB（限 6 GB）**。`.8` 是常驻业务机，停掉 DS4 之后仍有
  java(nexus) 1.1 GB、dockerd 0.4 GB、gnome-shell 0.2 GB 等。
  处理办法是把 `MEM_USED_MAX_GB` 抬到 8 **同时**把 `KV_CACHE_MEMORY` 从
  14.4 GB 降到 12.4 GB 补偿回去，用自己的池子换系统余量；
  **不要** `MEM_USED_MAX_GB=0` 直接关检查。
- **国内拉不动 ghcr 系镜像**。这台机器到 `ghcr.io` / `ghcr.nju.edu.cn` /
  `ghcr.dockerproxy.net` 实测都只有 0.1 MB/s，25 GiB 镜像要十几小时，
  `install.sh` 会在 pull 阶段卡死。可用的是 **`ghcr.chenby.cn`**：
  docker 多层并行实测 **10.13 MiB/s**，约 40 分钟拉完。做法见
  [`scripts/pull-image.sh`](scripts/pull-image.sh)，拉完 retag 成
  `ghcr.io/entrpi/...:v2.3-tier1` 让 launcher 的默认镜像名本地命中，不用改 `.env`。
- **测镜像源速度别只看 HTTP 码和 curl 的 speed_download**。代理站会秒返
  26 字节的 `{"message":"UNAUTHORIZED"}`，`%{http_code}` 是 200、算出来的速度
  高达 4.4 MiB/s，全是假的。必须真下几十 MB 再用 `file` 确认是 gzip 二进制。
  另外代理站要走**它自己**的 token 认证（`www-authenticate` 里的 realm），
  只对 ghcr.io 做握手会把可用的源误判成 401。
- **镜像同步到 worker 走光缆**。`docker save | ssh | docker load` 传 18 GB
  比让 worker 自己去外网拉快得多。
