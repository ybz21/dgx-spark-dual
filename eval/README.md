# 模型评测

在两台 DGX Spark (GB10) 上对本仓库部署的模型做的**性能 + 质量 + 编码**对比评测。全部自动判分、可复现。

被测端点(OpenAI 兼容):

| 模型 | 部署 | 端点 | model id |
|---|---|---|---|
| DeepSeek-V4-Flash | [../ds4-flash-iq2-dspark](../ds4-flash-iq2-dspark) | `:8000` | `deepseek-v4-flash` |
| Laguna-S-2.1-NVFP4 | [../laguna-s2.1-nvfp4-dflash](../laguna-s2.1-nvfp4-dflash) | `:8000` | `laguna-s2.1` |
| Qwen3.5-122B-A10B | vLLM (INT4) | `:30001` | `qwen3.5-122b-int4` |

> ⚠️ DS4 与 Laguna 在同一台机的统一内存里**互斥**，无法同时在线；因此三方并非全程同批。报告中会标注哪些是同批、哪些是历史参考。

## 脚本

| 脚本 | 测什么 |
|---|---|
| `scripts/model_compare.py` | 性能：TTFT、decode TPS、跨上下文、**prefix cache 冷/热**、长上下文大海捞针、mini-bench |
| `scripts/bench_quality.py` | 质量：**GSM8K**(数学) / **MMLU**(英文知识) / **CMMLU**(中文知识)，真数据集自动判分 |
| `scripts/bench_code.py` | 编码：**HumanEval / MBPP** pass@1，子进程真跑测试用例 |

均为纯 Python 标准库(无需 pip)。两模型在不同机器 → 并发评测。判分对推理型输出(如 Laguna 的 `<think>`)做了鲁棒处理。

## 复现

```bash
cd eval/scripts
bash download-datasets.sh          # 拉 GSM8K/MMLU/CMMLU/HumanEval/MBPP 到 ./datasets/
# 按需改脚本顶部 MODELS 的 base/model 指向你的端点
python3 model_compare.py           # 性能
python3 bench_quality.py --n-gsm8k 40 --n-mmlu 60 --n-cmmlu 40
python3 bench_code.py   --n-he 30 --n-mbpp 30   # Laguna 重推理，token 上限已设 10k
```

> `datasets/` 不入库(体积大、为公开数据)，用 `download-datasets.sh` 现拉。

## 结果速览

### 编码 (pass@1)
| | Laguna | Qwen3.5-122B | DS4(历史参考) |
|---|---|---|---|
| HumanEval | 90% | 100% | 98% |
| MBPP | **93%** | 77% | 76% |

### 质量
| | Laguna | Qwen3.5-122B | DS4(历史参考) |
|---|---|---|---|
| GSM8K 数学 | 95% | 97.5% | 95% |
| MMLU 英文知识 | 62% | **88%** | — |
| CMMLU 中文知识 | 60% | **78%** | 85% |

### 性能(单流)
- DS4：prefill ~690–880 tok/s、decode 裸引擎 ~18 / DSpark 投机 ~38 tok/s；prefix cache 冷→热 17×
- Laguna：decode ~38 tok/s(DFlash 投机接受率 ~40%)；重推理型，单请求 token 多、延迟高
- Qwen：TTFT 最低(比 DS4 快 2–3×)、知识类最强

**结论**:Laguna 代码/数学专精(MBPP 反超),广域知识弱;Qwen 均衡通才;DS4 长上下文与 prefix cache 强。详见 `reports/`。

## 报告

- [`reports/laguna-vs-qwen.md`](reports/laguna-vs-qwen.md) — Laguna vs Qwen 完整对比(编码+质量+性能)
- [`reports/ds4-vs-qwen-performance.md`](reports/ds4-vs-qwen-performance.md) — DS4 vs Qwen 性能(TTFT/TPS/prefix-cache/大海捞针/mini-bench)
- [`../ds4-flash-iq2-dspark/docs/deploy-report.md`](../ds4-flash-iq2-dspark/docs/deploy-report.md) — DS4 部署 + 基准详报

## 判分方法说明

- **GSM8K**:允许链式推理，取末行 `#### <数字>` 或最后一个数字，数值比对。
- **MMLU/CMMLU**:允许推理，取"答案/Answer 后"或末尾的独立字母 A–D。token 上限 10k 防止推理型模型没输出答案就被截断。
- **HumanEval/MBPP**:提取 ```python 代码块 → 拼 test/断言 → 子进程执行(15s 超时)判 pass。token 上限 10k(Laguna 写代码前会长推理，太小会被截断误判)。
