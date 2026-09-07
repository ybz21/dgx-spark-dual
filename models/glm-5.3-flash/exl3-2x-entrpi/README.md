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
- [x] 进 [`../../../eval/`](../../../eval) 流程出质量分（2026-09-06，见下）
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

## 质量评测（2026-09-06）

用仓库的 [`eval/harness.py`](../../../eval/harness.py) 对准 `:8000`，每集 40 题、6 并发、思考关。
完整报告在 [`../../../eval/reports/glm-5.3-flash/`](../../../eval/reports/glm-5.3-flash)。

| 数据集 | 准确率 | 用时 |
|---|---:|---:|
| HumanEval（代码） | **100.0%** (40/40) | 104s |
| MBPP（代码） | **95.0%** (38/40) | 293s |
| GSM8K（数学） | 90.0% (36/40) | 60s |
| CMMLU（中文知识） | **87.5%** (35/40) | 102s |
| toolcall（函数调用） | 90.0% (18/20) | 10s |
| demo_math_zh | 100.0% (5/5) | 1s |

MMLU 这次没跑成：`eval/scripts/download-datasets.sh` 里的伯克利源
`people.eecs.berkeley.edu/~hendrycks/data.tar` 已 404，拉回来是 nginx 错误页。
是仓库既有问题，另行修。

> ⚠️ 这轮跑之前修了 `harness.py` 的 `_extract_code`（详见该函数注释）。
> 旧逻辑盲取最后一个代码块，话多的模型会被抽到解法后面的说明性片段，
> GLM 的 MBPP 因此被判成 25%，修完是 95%；HumanEval 90% → 100%。
> **仓库里 Laguna / Qwen3.5-122B / DS4 的历史代码分是旧逻辑跑的，没有回填，
> 跨模型比代码分时要注意这条边界。**

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

## 运维层（开机自启 / 崩溃自愈 / 光口自适应）

目标是插上就能用、中间出问题自己恢复。装一次，之后开机、崩溃、换光口都不用管：

```bash
cd ops && bash install-ops.sh      # 在 head 上跑一次，它会把 worker 侧也装好
```

| 文件 | 干什么 |
|---|---|
| [`ops/glm53-ops.env`](ops/glm53-ops.env) | 两台共用配置：角色、光口候选、serving 档位（当前 1M） |
| [`ops/glm53-rail.sh`](ops/glm53-rail.sh) | 光口自适应：探测并配好 200GbE 直连地址 |
| [`ops/systemd/glm53-rail.timer`](ops/systemd/glm53-rail.timer) | 每 30 秒重探测，支持运行中插拔/换口 |
| [`ops/glm53-supervise.sh`](ops/glm53-supervise.sh) | 看门狗 + 启动顺序编排，跑在 head |
| [`ops/systemd/`](ops/systemd) | 三个单元，两台开机自启 |

### 光口自适应

这里使用的是双节点直连所需的**固定点对点地址**（不是 DHCP）；脚本负责自动选择接口并补配地址。
节点角色通过 `NODE_ROLE=master/slave` 显式指定，管理网地址只用于安装时通过 `WORKER_HOST`（主机名/DNS）找到从节点，不参与角色判断。
判据是**「配上地址后能不能 ping 通对端」，不是「有没有光」**。实机上每台插了两对缆、
四个口里有两个 carrier=1，只看 carrier 会挑错口。脚本把四个口全列为候选逐个试，
成功后从 sysfs 反查对应的 RoCE HCA 名写进 `/run/glm53-rail.env`。

幂等：已配好且对端可达就直接返回，不动网络（服务不会被踢断）。实测三种场景都过：
已配好 → 不动；地址被删 → 第 1 轮找回；地址在别的口 → 正确识别并写出对应 HCA。

### 启动顺序

硬约束：**先删 head 容器 → 再删 worker → 起 worker → 起 head**，且 worker→head
间隔要小于 torch 的 600 秒 rendezvous 超时（新 worker 会跟旧 head 的 TCP store 会合，
旧 head 一走就 connection-reset 而死）。

所以编排全部收在 head 的 supervisor 里，**没有让两台各自开机自启容器**——那样顺序必乱。
worker 侧的 `glm53-worker-boot.service` 只做一件事：开机清掉上次的残留容器，然后等 head 来编排。

### ⚠️ 看门狗不能用 `/health` 判活

**实测过的坑**：`docker kill` 掉 worker 容器之后，head 的 `/health` 依然稳定返回 200，
而任何真实请求都挂死（60 秒超时 `http=000`）。第一版看门狗拿 `/health` 当判据，
worker 死了 50 分钟一次都没触发。

改成两级判据：

1. **结构检查**（便宜）：两台容器是不是都在 `running`。命中直接判失败，不等探针超时。
2. **生成探针**（真实）：发 `max_tokens=1` 的请求，必须拿到 `"choices"` 才算活。

连续 3 次失败才动手（避免抖动误触发），重启失败按 60→600 秒指数退避。
每轮重启前重跑一次 rail 探测，所以"崩溃期间光缆被换了口"这种组合也覆盖。

实测一次真实故障恢复（`docker kill` worker）：

```
10:45:54  健康检查失败 1/3      ← 结构检查第一次轮询就命中
10:46:55  健康检查失败 3/3 → 开始有序重启
10:46:59  起 worker (rank 1)
10:47:15  起 head (rank 0)      ← 间隔 16s
10:51:22  重启成功
```

**故障发现到服务恢复 5 分 28 秒，无人介入**，恢复后 `max_model_len` 和 KV 池与故障前一致。

### 改 serving 旋钮不要走 install.sh

改 `.env` 后重跑 `install.sh` 会失败，head 报 `this host does not own 10.0.0.2`，
但手动跑它自己那条检查（`ip -o addr show | grep "10.0.0.2/"`）是命中的。
更糟的是 install.sh 判定失败后**自动回滚，把新写的旋钮一起还原掉**。

直接调 launcher，命令行变量优先级最高（supervisor 就是这么干的）：

```bash
docker rm -f vllm_glm53                              # 先删 head
ssh ai@10.0.0.3 docker rm -f vllm_glm53              # 再删 worker
K="KV_DTYPE=nvfp4_ds_mla VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 MAX_LEN=1048576 MNBT=4096"
ssh ai@10.0.0.3 "env $K ~/launch-glm53-vllm-tp2.sh 1"   # 起 worker
env $K ~/launch-glm53-vllm-tp2.sh 0                     # 起 head
```

## 1M 上下文档（当前生效）

| | 512K 档 | **1M 档（现用）** |
|---|---:|---:|
| `max_model_len` | 524,288 | **1,048,576** |
| KV 池 | 1,051,936 token | **1,613,717 token**（+53%） |
| 满长并发 | 2.01x | 1.54x |
| KV 格式 | fp8_ds_mla | nvfp4_ds_mla |
| MNBT | 8192 | 4096 |

fp8 KV 撑不住 1M，必须**同时**开 NVFP4 注意力内存（`KV_DTYPE=nvfp4_ds_mla` +
`VLLM_NVFP4_MLA_DYNAMIC_SCALE=1`），代价是上游实测 math_500 从 91 降到 88。

速度几乎没掉（单流 temp 0）：

| 负载 | 1M 档 | 512K 档 |
|---|---:|---:|
| 结构化 | **68.8** tok/s | 67.0 |
| 代码 | **57.8** tok/s | 56.3 |
| JSON | 55.4 tok/s | 56.0 |
| 英文散文 | 19.8 tok/s | 23.2 |

长上下文实测：30,038 token 大海捞针**答对**，prefill 1,283 tok/s。

换回 512K：改 `ops/glm53-ops.env` 里的 `PROFILE_*` 三项，重启 supervisor 即可。
