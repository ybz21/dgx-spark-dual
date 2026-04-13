# vLLM Benchmark Report

- **Endpoint**: `http://192.168.130.12:30000`
- **Model**: `qwen3.5-122b-int4`
- **Timestamp**: 2026-04-13 11:31:57 +0800
- **Script**: `bench_llm.py`

## 1. Long-context needle-in-a-haystack

- Target prompt: **125,000** tokens (actual: 122,503)
- Needle: `MAGENTA-7834-GORILLA`

| Depth | Found | Prompt tok | TTFT (s) | Total (s) | Answer |
|------:|:-----:|-----------:|---------:|----------:|--------|
| 0% | ✅ | 122567 | 95.99 | 96.19 | MAGENTA-7834-GORILLA |
| 50% | ✅ | 122568 | 97.19 | 97.36 | MAGENTA-7834-GORILLA |
| 99% | ✅ | 122568 | 97.06 | 97.21 | MAGENTA-7834-GORILLA |

**Pass rate**: 3/3

## 2. Latency sweep (single request, greedy decode)

| Prompt tok | TTFT (s) | Total (s) | Out tok | Prefill tok/s | Decode tok/s |
|-----------:|---------:|----------:|--------:|--------------:|-------------:|
| 936 | 0.63 | 1.43 | 36 | 1477 | 45.1 |
| 7,879 | 3.80 | 4.55 | 33 | 2072 | 44.1 |
| 31,390 | 16.89 | 17.65 | 33 | 1859 | 43.5 |
| 122,629 | 96.93 | 97.78 | 33 | 1265 | 38.9 |

## 3. Concurrency stress (prompt=506 tok, max_output=128 tok)

| Conc | Req | Dur (s) | OK | Fail | p50 TTFT (s) | p95 TTFT (s) | p50 Total (s) | Agg out tok/s | RPS |
|----:|----:|--------:|---:|----:|-------------:|-------------:|-------------:|--------------:|----:|
| 1 | 16 | 40.1 | 16 | 0 | 0.48 | 0.48 | 2.61 | 28.9 | 0.40 |
| 4 | 16 | 30.2 | 16 | 0 | 0.89 | 5.09 | 7.30 | 36.4 | 0.53 |
| 8 | 16 | 25.4 | 16 | 0 | 1.77 | 2.65 | 11.98 | 45.7 | 0.63 |

