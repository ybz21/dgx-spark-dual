# Laguna-S-2.1-NVFP4 vs Qwen3.5-122B-A10B 对比评测

> 2026-07-28 · 两台 DGX Spark(GB10) · Laguna@192.168.130.12:8000(vLLM+DFlash投机) · Qwen@192.168.130.45:30001(vLLM)
> 判分全自动、真跑用例/真数据集。token 上限 10k(不截断)。Laguna 推理输出经抽查确认判分公平。
> ⚠️ DS4 与 Laguna 在 .12 内存互斥，本轮 DS4 未在线，其分数为**历史参考**(非同批)。样本偏小(30-60题)，差距 <5% 视为噪声。

## 编码能力 (pass@1，真跑测试用例)

| benchmark | Laguna-S-2.1 | Qwen3.5-122B | DS4(历史参考) |
|---|---|---|---|
| HumanEval | 90.0% (27/30) | **100%** (30/30) | 98% (49/50) |
| MBPP | **93.3%** (28/30) | 76.7% (23/30) | 76% (38/50) |

## 质量 (真 benchmark)

| benchmark | Laguna-S-2.1 | Qwen3.5-122B | DS4(历史参考) |
|---|---|---|---|
| GSM8K 英文数学 | 95.0% (38/40) | **97.5%** (39/40) | 95% (38/40) |
| MMLU 英文知识 | 61.7% (37/60) | **88.3%** (53/60) | 未测 |
| CMMLU 中文知识 | 60.0% (24/40) | **77.5%** (31/40) | 85% (34/40) |

## 性能 (Laguna，单流+DFlash投机)

| 指标 | 值 |
|---|---|
| decode TPS | ~38 tok/s |
| DFlash 投机接受率 | ~40% |
| 特点 | 重推理型：答题前长篇 `<think>`，单题墙钟长(慢在 token 多，不是速度慢) |

## 结论

- **Laguna = 代码/数学专精**：MBPP 反超 Qwen(93 vs 77)、HumanEval 90%、GSM8K 95% 与 Qwen 齐平。是很强的编码模型。
- **广域知识是 Laguna 短板**：MMLU 62% / CMMLU 60%，明显低于通用模型 Qwen(88/78)。符合"代码专精 vs 通用"的定位。
- **Qwen = 均衡强通才**：知识类领先，HumanEval 满分，各项无短板。
- **选型建议**：纯编码/代码 agent 场景 Laguna 有竞争力(且 NVFP4+DFlash 延迟低)；需要广域知识问答/通用助手选 Qwen。

## 附：评测脚本(可复现)

- `bench_code.py` — HumanEval/MBPP，子进程执行判分
- `bench_quality.py` — GSM8K/MMLU/CMMLU，对推理型输出鲁棒的判分
- 数据集在 `datasets/`；token 上限均 10k
