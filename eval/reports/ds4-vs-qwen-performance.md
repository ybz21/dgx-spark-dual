# 模型对比报告：DeepSeek-V4-Flash vs Qwen3.5-122B-A10B

- **A. DeepSeek-V4-Flash** — `192.168.130.12:8000`，ds4-server，IQ2XXS(~2-bit)量化，DSpark 投机解码
- **B. Qwen3.5-122B-A10B** — `192.168.130.45:30001`，vLLM，INT4 量化，enable-prefix-caching
- 两台均为 NVIDIA GB10 (DGX Spark)，128K 上下文，单并发。日期 2026-07-26。
- 测试脚本：`model_compare.py`（纯标准库）

---

## 一句话结论

**Qwen3.5-122B-A10B 在这套对比里综合更强**：TTFT/prefill 快 2–3×、准确率更高、长上下文延迟更低；**DeepSeek-V4-Flash 的 decode 速度与之持平、prefix-cache 提速更猛、长上下文检索同样满分**，但受 2-bit 量化 + 更重的 prefill 拖累，首 token 慢、准确率略低。

---

## 1. 性能（TTFT / decode TPS，单并发）

| 上下文 | 指标 | DeepSeek-V4-Flash | Qwen3.5-122B | 谁快 |
|---:|---|---:|---:|:--:|
| ~260 tok | TTFT | 583 ms | **397 ms** | Qwen |
| | decode | **46.3 tok/s** | 40.1 tok/s | DS4 |
| ~5K tok | TTFT | 5394 ms | **1459 ms** | Qwen 3.7× |
| | decode | 39.2 | 39.9 | 持平 |
| ~20K tok | TTFT | 18870 ms | **9949 ms** | Qwen 1.9× |
| | decode | 38.3 | 38.2 | 持平 |
| ~40K tok | TTFT | 33296 ms | **15726 ms** | Qwen 2.1× |
| | decode | 37.2 | 36.8 | 持平 |

- **Prefill/TTFT：Qwen 稳定快 2–3×**（40K 时等效 prefill ≈ Qwen 2.8K tok/s vs DS4 1.2K tok/s）。原因：Qwen 仅激活 10B 参数且 INT4 走 vLLM；DS4 是更重的 MoE 且 2-bit 重打包。
- **Decode TPS：几乎持平**（都 ~37–46 tok/s）。DS4 在可预测内容上靠 DSpark 投机解码略占优（小 ctx 46 vs 40）。
- **随上下文加深，两者 decode 都只轻微下滑**，都算长上下文友好。

## 2. Prefix Cache 效果（16K 前缀，冷→热）

| | 冷 TTFT | 热 TTFT | 提速 |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 23251 ms | 1341 ms | **17.3×** |
| Qwen3.5-122B | 10925 ms | 1592 ms | 6.9× |

- **两者前缀缓存都极有效**，命中后热 TTFT 都压到 ~1.3–1.6s。
- DS4 冷启更慢 → 相对提速更夸张（17×）；但**热态两者 TTFT 接近**。多轮同前缀（RAG/长 system prompt/agent）场景，缓存收益都很大。

## 3. 长上下文·大海捞针（检索准确率）

| 长度档 | DeepSeek-V4-Flash | Qwen3.5-122B |
|---|:--:|:--:|
| 4K（3 深度） | 3/3 ✅ | 3/3 ✅ |
| 20K（3 深度） | 3/3 ✅ | 3/3 ✅ |
| 60K（3 深度） | 3/3 ✅ | 3/3 ✅ |
| **合计** | **9/9 满分** | **9/9 满分** |

- **检索准确率打平，都满分**（测到 ~60–67K token 深度）。
- **但延迟差距大**：60K 档 DS4 单次 ~88–91s，Qwen ~28–40s（Qwen 快 2–3×，还是 prefill 差距）。

## 4. Mini-Benchmark（18 题：数学/MCQ/推理，自动判分）

| | 总分 | 数学 | 常识MCQ | 推理 |
|---|:--:|:--:|:--:|:--:|
| DeepSeek-V4-Flash | 14/18 (78%) | 5/8 | 6/6 | 3/4 |
| **Qwen3.5-122B** | **17/18 (94%)** | 7/8 | 6/6 | 4/4 |

- **Qwen 明显更准**，主要赢在数学和推理。
- ⚠️ 注意公平性：**DS4 是 IQ2(~2-bit)极限量化，Qwen 是 INT4(更高精度)**，这个准确率差里有相当一部分来自量化，不完全代表底座模型强弱。
- ⚠️ 这只是 18 题的 mini-bench，**指示性参考，不能替代 MMLU/GSM8K 全量**。

---

## 关键取舍

| 你的场景 | 更推荐 | 原因 |
|---|---|---|
| 交互式/低延迟（聊天、agent 首响） | **Qwen3.5** | TTFT 快 2–3× |
| 高准确率（数学/推理/知识） | **Qwen3.5** | mini-bench 94% vs 78% |
| 长文档检索准确性 | 平手 | 都 9/9，但 Qwen 延迟更低 |
| 多轮同前缀缓存复用 | 平手 | 热 TTFT 都 ~1.3–1.6s |
| 纯 decode 吞吐 | 平手 | 都 ~37–46 tok/s |

> 若要更公平地评 DS4 底座实力，应换更高精度量化（Q4/Q8）再比；当前 2-bit 是为塞进单机 128GB 做的激进压缩。

---

## 附：原始输出

```
==============================================================================
模型对比:  DeepSeek-V4-Flash   vs   Qwen3.5-122B-A10B
  - DeepSeek-V4-Flash    http://192.168.130.12:8000/v1  (model=deepseek-v4-flash)
  - Qwen3.5-122B-A10B    http://192.168.130.45:30001/v1  (model=qwen3.5-122b-int4)
==============================================================================

## 1. 性能（单并发，流式测 TTFT / decode TPS）

  [DeepSeek-V4-Flash   ] ctx≈   266 tok | TTFT     583 ms | decode   46.3 tok/s | out 140 | e2e 3.6s
  [Qwen3.5-122B-A10B   ] ctx≈   267 tok | TTFT     397 ms | decode   40.1 tok/s | out 130 | e2e 3.6s
  [DeepSeek-V4-Flash   ] ctx≈  4910 tok | TTFT    5394 ms | decode   39.2 tok/s | out 139 | e2e 8.9s
  [Qwen3.5-122B-A10B   ] ctx≈  5247 tok | TTFT    1459 ms | decode   39.9 tok/s | out 134 | e2e 4.8s
  [DeepSeek-V4-Flash   ] ctx≈ 19652 tok | TTFT   18870 ms | decode   38.3 tok/s | out 139 | e2e 22.5s
  [Qwen3.5-122B-A10B   ] ctx≈ 21627 tok | TTFT    9949 ms | decode   38.2 tok/s | out 134 | e2e 13.4s
  [DeepSeek-V4-Flash   ] ctx≈ 40191 tok | TTFT   33296 ms | decode   37.2 tok/s | out 141 | e2e 37.1s
  [Qwen3.5-122B-A10B   ] ctx≈ 44347 tok | TTFT   15726 ms | decode   36.8 tok/s | out 136 | e2e 19.4s

## 2. Prefix Cache 效果（前缀≈16000 tok，冷→热两次同一前缀）

  [DeepSeek-V4-Flash   ] 冷 TTFT   23251 ms → 热 TTFT    1341 ms | 提速 17.3×
  [Qwen3.5-122B-A10B   ] 冷 TTFT   10925 ms → 热 TTFT    1592 ms | 提速  6.9×

## 3. 长上下文·大海捞针（命中=答案含密钥）

  [DeepSeek-V4-Flash   ] len≈  4000 depth 15% | ctx 4944 tok |   6.0s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈  4000 depth 15% | ctx 5280 tok |   2.8s | ✅命中
  [DeepSeek-V4-Flash   ] len≈  4000 depth 50% | ctx 4945 tok |   6.0s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈  4000 depth 50% | ctx 5281 tok |   1.9s | ✅命中
  [DeepSeek-V4-Flash   ] len≈  4000 depth 85% | ctx 4945 tok |   3.3s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈  4000 depth 85% | ctx 5280 tok |   1.9s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 16000 depth 15% | ctx 19687 tok |  21.2s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 16000 depth 15% | ctx 21661 tok |  11.4s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 16000 depth 50% | ctx 19686 tok |  23.9s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 16000 depth 50% | ctx 21660 tok |   7.4s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 16000 depth 85% | ctx 19686 tok |  23.9s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 16000 depth 85% | ctx 21661 tok |   4.3s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 48000 depth 15% | ctx 61308 tok |  91.5s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 48000 depth 15% | ctx 67644 tok |  40.2s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 48000 depth 50% | ctx 61309 tok |  88.5s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 48000 depth 50% | ctx 67645 tok |  27.7s | ✅命中
  [DeepSeek-V4-Flash   ] len≈ 48000 depth 85% | ctx 61308 tok |  88.4s | ✅命中
  [Qwen3.5-122B-A10B   ] len≈ 48000 depth 85% | ctx 67644 tok |  27.7s | ✅命中

  小结: DeepSeek-V4-Flash 9/9 | Qwen3.5-122B-A10B 9/9

## 4. Mini-Benchmark（数学/MCQ/推理，自动判分）

  [DeepSeek-V4-Flash   ] 总 14/18 (78%)  |  math:5/8 mcq:6/6 reason:3/4
  [Qwen3.5-122B-A10B   ] 总 17/18 (94%)  |  math:7/8 mcq:6/6 reason:4/4

完成。
```
