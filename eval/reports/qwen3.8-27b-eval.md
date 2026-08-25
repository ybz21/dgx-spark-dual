# Qwen3.8-27B 评测报告（单节点 DGX Spark / GB10）

> 日期：2026-08-26 · 被测：`Qwen3.8-27B-NVFP4` @ `http://192.168.130.48:9001/v1`（box010）
> 引擎：vLLM 0.27.1（纯解码，无投机）· 上下文 256k · 关思考=raw 速度，开思考=质量
> 全部真数据集自动判分，脚本见 [`../scripts/`](../scripts)（`model_compare.py` / `bench_quality.py` / `bench_code.py`，题目级 6 路并发）。

## 一、性能（单流，vLLM 纯解码）

| 上下文 | TTFT | decode |
|---|---|---|
| 267 tok | 250 ms | 11.2 tok/s |
| 5.2k | 2.21 s | 11.1 tok/s |
| 21.6k | 8.18 s | 10.9 tok/s |
| 44.3k | 15.2 s | 10.5 tok/s |

- **Prefix Cache**：同前缀（≈16k tok）冷 TTFT 10.1s → 热 0.77s，**13.1× 提速**（多轮/长 system prompt 复用极划算）。
- **Mini-bench（自动判分）**：17/18 = **94%**（math 7/8、mcq 6/6、reason 4/4）。
- decode ~11 tok/s（纯解码无投机）；TTFT 随上下文线性增，prefill ≈ 2900 tok/s @ GB10。
- 投机方案单流更快（SGLang+DSpark 结构化内容 33 tok/s、vLLM+MTP 数学 31 tok/s），但并发受限，见部署报告；本报告用的是**并发稳定的纯解码**配置。

### 并发扩展（vLLM 纯解码，关思考，max_tokens=256）

| 并发 | TTFT | 单请求 tok/s | 聚合 tok/s |
|---|---|---|---|
| 1 | 0.20s | 11.3 | 11.3 |
| 2 | 0.20s | 10.9 | 21.8 |
| 4 | 0.25s | 10.6 | 40.4 |
| 6 | 0.26s | 10.4 | 58.1 |
| 8 | 0.29s | 10.1 | **74.8** |

稳定扩到 8 路，TTFT 全程 <0.3s，无崩溃/排队。

## 二、质量（真 benchmark，开思考模式=模型真实水平）

| Benchmark | Qwen3.8-27B | 参考 Qwen3.5-122B(历史) | 参考 Laguna(历史) |
|---|---|---|---|
| GSM8K（英文数学）| **95.0%** (38/40) | 97.5% | 95% |
| MMLU（英文知识）| **90.0%** (54/60) | 88% | 62% |
| CMMLU（中文知识）| **92.5%** (37/40) | 78% | 60% |

**27B 的知识类甚至反超历史 122B**（CMMLU 92.5% vs 78%，MMLU 90% vs 88%），数学与 122B 接近。中文能力尤其突出。

## 三、编码（pass@1，贪心）

| Benchmark | Qwen3.8-27B | 参考 Qwen3.5-122B | 参考 Laguna |
|---|---|---|---|
| MBPP | **93.3%** (28/30) | 77% | 93% |
| HumanEval（开思考）| 40.0% (12/30) ⚠️ | 100% | 90% |
| HumanEval（关思考重测）| **96.0%** (24/25) ✅ | — | — |

⚠️ **HumanEval 开思考仅 40% 是判分假象，不是模型能力**：MBPP 93.3%、GSM8K 95%、mini-bench 94% 都很强，唯独 HumanEval 低。原因是开思考时模型输出里混入推理/markdown，HumanEval 判分器（把补全直接拼到函数签名后再执行）提取补全失败。关思考重测（代码输出干净）见上表最后一行。

## 四、复现

```bash
cd eval/scripts
bash download-datasets.sh                              # GSM8K/MMLU/CMMLU/HumanEval/MBPP
# 脚本 MODELS 已指向 http://192.168.130.48:9001/v1 (model=qwen3.8-27b)
python3 model_compare.py --only perf,bench,cache       # 性能
EVAL_WORKERS=6 python3 bench_quality.py --n-gsm8k 40 --n-mmlu 60 --n-cmmlu 40   # 质量(开思考)
EVAL_WORKERS=6 python3 bench_code.py   --n-he 30 --n-mbpp 30                     # 编码
```

## 五、结论

- **知识/中文/数学**：27B 达到甚至反超历史 122B 的知识水平（CMMLU 92.5%、MMLU 90%、GSM8K 95%），性价比极高。
- **编码**：MBPP 93% 很强；HumanEval 关思考重测 96%（开思考的 40% 纯属判分器提取假象）。
- **性能**：纯解码 ~11 tok/s、并发稳到 8 路（聚合 ~75 tok/s）、prefix cache 13× ——适合多人/agent 场景。要更快单请求可换投机方案（DSpark/MTP），但并发受限。
- **上下文**：原生 256k（`max_position_embeddings=262144`），远超常用需求；1M 需 RoPE 外推、有质量风险，未开。
