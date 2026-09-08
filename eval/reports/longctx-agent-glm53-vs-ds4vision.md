# 长上下文 + 迭加 diff：编码 agent 负载压测

模拟真实写代码场景：先把整个代码库塞进上下文，之后每轮追加一个 unified diff
再提问。会话单调增长、前缀完全一致，正是 prefix cache 该发挥作用的形态。

- 脚本：[`../scripts/bench_longctx_agent.py`](../scripts/bench_longctx_agent.py)
- 日期：2026-09-07 / 09-08
- 硬件：2× DGX Spark (GB10)，TP=2，200GbE 直连

## 量什么、怎么量

| | |
|---|---|
| **TTFT** | 用**流式**量首个 content delta 的时间。整轮耗时被 decode 长度污染，没有可比性 |
| **prefix 命中率** | 每轮请求**前后**各抓一次 `/metrics` 的 `prefix_cache_queries_total` / `hits_total` 算增量。直接读累计值会被历史请求稀释 |
| **上下文** | 合成代码库，刻意做出变化，避免高度重复文本让 tokenizer 和 cache 行为失真 |
| **diff** | 真改两个文件并把改动落回 repo 状态，保证前后一致，不是造假的 diff 文本 |

## 结果一：两个模型同负载对比（ctx≈38k，6 轮迭加）

| | GLM-5.3-Flash (EXL3 4bpw) | **DeepSeek-V4-Flash-Vision** |
|---|---:|---:|
| 冷启 TTFT | 29.36 s | **21.69 s** |
| 迭加轮 TTFT 中位 | 4.44 s | **0.49 s** |
| TTFT 范围 | 4.26–5.01 s | 0.48–0.55 s |
| prefix 命中率 | 86–87% | **99.3–99.6%** |
| decode 中位 | 13.8 tok/s | **39.6 tok/s** |
| 冷/热 TTFT 比 | 6.6× | **44.6×** |

**热轮 TTFT 差 9 倍。** 对编码 agent 这是决定性的——每追加一次 diff 等 0.5 秒还是
4.4 秒，体感完全两回事。

根因是 prefix 命中率：GLM 卡在 86%，DS4 是 99.6%。GLM 启动日志里那句

```
Mamba cache mode is set to 'align' for Glm5NextForConditionalGeneration
by default when prefix caching is enabled
```

说明它的混合架构（sparse + linear attention）里 mamba 那部分状态没法像纯注意力
那样按前缀复用，每轮都要重算一截。**这是架构性的，不是调参能解决的。**

## 结果二：DS4-Vision 大规模验证（27 万 token）

| | |
|---|---|
| 打底代码库 | 200 文件 × 140 行 = 802,977 字符 |
| 实际 prompt_tokens | **286,351** |
| 冷 prefill | **104.4 s**（≈2,744 tok/s） |
| 缓存热后同一份上下文 | **TTFT 1.28 s，总耗时 3.9 s** |
| 冷/热差距 | **27×** |
| KV 池 | 2,708,932 token |

对照 100 文件：143,102 token 冷 prefill 87.8 s。

这就是编码 agent 场景的核心价值：**第一次读代码库贵，之后每轮追加 diff 几乎免费。**

## 未完成

长时（小时级）压测没跑成——中途服务端把 8888 从 `deepseek-v4-flash-vision-exp`
换成了 `qwen3.8-flash-next`（`/opt/qwen/launch_secure.py`）并开启了鉴权，
20 轮请求全部 401。拿到 API key 后可续跑：

```bash
python3 bench_longctx_agent.py --base http://192.168.130.8:8888/v1 \
  --model <model-id> --api-key <key> --ctx-files 200 --duration 3600
```

要看的是：TTFT 会不会随时间劣化、命中率能否长期维持、KV 池被占满触发驱逐后的行为。

## 踩到的坑

1. **`/health` 和 `/v1/models` 都不能用来判活。** 对端节点整机掉线后，head 的
   `/health` 持续返回 200、`/v1/models` 正常返回，而真实推理 45–60 秒超时无响应。
   GLM 和 DS4-Vision 两个不同模型、不同部署脚本上都复现了，是 **vLLM 多节点 TP 的
   通病**。判活必须发真实生成请求。
2. **`nohup ... &` 在某些调用环境下会被连带杀掉**，长压测要用 `setsid` 才真正脱离。
3. **裸 `HTTPError` 什么都看不出来。** 排查那次 401 时，原始错误只有
   `HTTPError`，看不出是鉴权、超长还是别的。已改成带状态码、body 和实际耗时。
