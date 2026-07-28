#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_compare.py — 对比两个 OpenAI 兼容 LLM 服务的性能与效果。

对比对象（默认）:
  A. DeepSeek-V4-Flash   @ http://192.168.130.12:8000/v1   (ds4-server)
  B. Qwen3.5-122B-A10B   @ http://192.168.130.45:30001/v1  (vLLM, INT4)

测试维度:
  1) 性能  : TTFT(首token延迟) / decode TPS / 端到端，单并发，跨上下文长度
  2) prefix cache : 同一长前缀 冷/热 两次，看 TTFT 提升（前缀缓存效果）
  3) 效果-长上下文 : 大海捞针（不同长度 x 不同深度），命中率
  4) 效果-benchmark: 内置 mini-bench（数学/常识MCQ/推理），自动判分

只用 Python 标准库，无需 pip。流式读取测 TTFT。
用法:
  python3 model_compare.py                 # 默认全套（中等规模，约 10-20 分钟）
  python3 model_compare.py --quick         # 快速冒烟（小规模，验证连通）
  python3 model_compare.py --only perf,bench
  python3 model_compare.py --out report.md
"""
import json, time, argparse, urllib.request, urllib.error, re, sys, statistics

# ----------------------------------------------------------------------------
# 被测模型配置：extra 是各自“关闭思考模式”的私有参数（互不发送）
# ----------------------------------------------------------------------------
MODELS = [
    {"name": "DeepSeek-V4-Flash", "base": "http://192.168.130.12:8000/v1",
     "model": "deepseek-v4-flash", "extra": {"think": False}},
    {"name": "Qwen3.5-122B-A10B", "base": "http://192.168.130.45:30001/v1",
     "model": "qwen3.5-122b-int4", "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
]

# ----------------------------------------------------------------------------
# HTTP 基础
# ----------------------------------------------------------------------------
def _req(url, body, timeout, stream=False):
    data = json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)

def chat(m, messages, max_tokens=200, temperature=0.0, timeout=600):
    """非流式：返回 (text, usage_dict, elapsed)。"""
    body = {"model": m["model"], "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    body.update(m.get("extra", {}))
    t0 = time.time()
    resp = _req(m["base"] + "/chat/completions", body, timeout)
    d = json.load(resp)
    dt = time.time() - t0
    text = d["choices"][0]["message"]["content"]
    return text, d.get("usage", {}), dt

def stream_metrics(m, messages, max_tokens=200, temperature=0.0, timeout=600):
    """流式：返回 dict(ttft, total, decode_tps, out_tokens, prompt_tokens, text)。"""
    body = {"model": m["model"], "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}}
    body.update(m.get("extra", {}))
    t0 = time.time()
    ttft = None
    out_tokens = 0
    prompt_tokens = None
    usage_out = None
    chunks = []
    resp = _req(m["base"] + "/chat/completions", body, timeout, stream=True)
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("usage"):
            u = obj["usage"]
            prompt_tokens = u.get("prompt_tokens", prompt_tokens)
            usage_out = u.get("completion_tokens", usage_out)
        for ch in obj.get("choices", []):
            delta = ch.get("delta", {}) or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                out_tokens += 1
                chunks.append(piece)
    total = time.time() - t0
    if usage_out:  # 用服务上报的真实 completion_tokens 更准
        out_tokens = usage_out
    decode_span = max(total - (ttft or 0), 1e-6)
    decode_tps = (out_tokens - 1) / decode_span if out_tokens > 1 else 0.0
    return {"ttft": ttft or total, "total": total, "decode_tps": decode_tps,
            "out_tokens": out_tokens, "prompt_tokens": prompt_tokens, "text": "".join(chunks)}

# ----------------------------------------------------------------------------
# 上下文填充：生成约 target_tokens 个 token 的英文背景文本
# ----------------------------------------------------------------------------
_LINE = ("Section {i}: routine telemetry for subsystem {i} shows nominal operation, "
         "all sensors within tolerance, no anomalies detected during this interval. ")
def make_context(target_tokens):
    approx_tok_per_line = 22
    n = max(1, target_tokens // approx_tok_per_line)
    return "".join(_LINE.format(i=i) for i in range(1, n + 1))

# ----------------------------------------------------------------------------
# 1) 性能：TTFT / decode TPS，跨上下文
# ----------------------------------------------------------------------------
def test_perf(models, ctx_points, gen_tokens):
    print("\n## 1. 性能（单并发，流式测 TTFT / decode TPS）\n")
    rows = {}
    for tgt in ctx_points:
        prefix = make_context(tgt) if tgt > 50 else ""
        prompt = (prefix + "\n\n" if prefix else "") + \
                 "Summarize the above in exactly one sentence, then count from 1 to 40."
        for m in models:
            try:
                r = stream_metrics(m, [{"role": "user", "content": prompt}], max_tokens=gen_tokens)
                key = (m["name"], tgt)
                rows[key] = r
                print(f"  [{m['name']:<20}] ctx≈{r['prompt_tokens'] or tgt:>6} tok | "
                      f"TTFT {r['ttft']*1000:7.0f} ms | decode {r['decode_tps']:6.1f} tok/s | "
                      f"out {r['out_tokens']} | e2e {r['total']:.1f}s")
            except Exception as e:
                print(f"  [{m['name']:<20}] ctx≈{tgt}: ERROR {e}")
    return rows

# ----------------------------------------------------------------------------
# 2) prefix cache：同一长前缀 冷/热
# ----------------------------------------------------------------------------
def test_prefix_cache(models, ctx_tokens):
    print(f"\n## 2. Prefix Cache 效果（前缀≈{ctx_tokens} tok，冷→热两次同一前缀）\n")
    out = {}
    for m in models:
        base_ctx = make_context(ctx_tokens)
        # 冷：加唯一 nonce 保证前缀此前未见
        nonce = f"[run-{int(time.time()*1000)%100000}] "
        cold_prompt = nonce + base_ctx + "\n\nReply with just the word READY."
        try:
            cold = stream_metrics(m, [{"role": "user", "content": cold_prompt}], max_tokens=8)
            # 热：立刻发完全相同的前缀（命中前缀缓存）
            warm = stream_metrics(m, [{"role": "user", "content": cold_prompt}], max_tokens=8)
            spd = cold["ttft"] / warm["ttft"] if warm["ttft"] > 0 else 0
            out[m["name"]] = (cold["ttft"], warm["ttft"], spd)
            print(f"  [{m['name']:<20}] 冷 TTFT {cold['ttft']*1000:7.0f} ms → "
                  f"热 TTFT {warm['ttft']*1000:7.0f} ms | 提速 {spd:4.1f}×")
        except Exception as e:
            print(f"  [{m['name']:<20}] ERROR {e}")
    return out

# ----------------------------------------------------------------------------
# 3) 效果-长上下文：大海捞针
# ----------------------------------------------------------------------------
def test_needle(models, sizes, depths):
    print("\n## 3. 长上下文·大海捞针（命中=答案含密钥）\n")
    CODE = "ZEBRA-GB10-4471-NEON"
    needle = f" 【IMPORTANT】The secret pass code for this drill is {CODE}. Remember it well. "
    q = ("\n\nQuestion: What is the secret pass code mentioned somewhere above? "
         "Answer with ONLY the code, nothing else.")
    score = {m["name"]: [0, 0] for m in models}
    for size in sizes:
        for dep in depths:
            body = make_context(size)
            pos = int(len(body) * dep)
            hay = body[:pos] + needle + body[pos:]
            for m in models:
                try:
                    text, usage, dt = chat(m, [{"role": "user", "content": hay + q}], max_tokens=32)
                    hit = CODE in text.replace(" ", "").upper() or "4471" in text
                    score[m["name"]][1] += 1
                    if hit:
                        score[m["name"]][0] += 1
                    pt = usage.get("prompt_tokens", "?")
                    print(f"  [{m['name']:<20}] len≈{size:>6} depth {int(dep*100):>2}% | "
                          f"ctx {pt} tok | {dt:5.1f}s | {'✅命中' if hit else '❌漏 -> '+text.strip()[:30]}")
                except Exception as e:
                    print(f"  [{m['name']:<20}] len≈{size} depth {dep}: ERROR {e}")
    print("\n  小结: " + " | ".join(f"{n} {s[0]}/{s[1]}" for n, s in score.items()))
    return score

# ----------------------------------------------------------------------------
# 4) 效果-mini benchmark：数学 / 常识MCQ / 推理，自动判分
# ----------------------------------------------------------------------------
BENCH = [
    # (type, question, answer)
    ("math", "A shop sells pens at 3 for $2. How much do 12 pens cost in dollars? Reply with just the number.", "8"),
    ("math", "If 5 machines make 5 widgets in 5 minutes, how many minutes for 100 machines to make 100 widgets? Reply with just the number.", "5"),
    ("math", "A train travels 60 km in 45 minutes. What is its speed in km/h? Reply with just the number.", "80"),
    ("math", "The sum of three consecutive integers is 72. What is the largest? Reply with just the number.", "25"),
    ("math", "12% of 250 is what? Reply with just the number.", "30"),
    ("math", "A rectangle is 8 by 5. A square has the same perimeter. What is the square's area? Reply with just the number.", "42.25"),
    ("math", "You buy an item for $80 and sell for $100. What is the profit percentage? Reply with just the number (no % sign).", "25"),
    ("math", "Rabbits and chickens: 10 heads, 28 legs. How many rabbits? Reply with just the number.", "4"),
    ("mcq", "Which planet is largest in our solar system? A) Earth B) Jupiter C) Saturn D) Mars. Reply with just the letter.", "B"),
    ("mcq", "What is the capital of Australia? A) Sydney B) Melbourne C) Canberra D) Perth. Reply with just the letter.", "C"),
    ("mcq", "Which is a prime number? A) 21 B) 27 C) 29 D) 33. Reply with just the letter.", "C"),
    ("mcq", "In which layer does photosynthesis mainly occur? A) Root B) Leaf C) Stem D) Flower. Reply with just the letter.", "B"),
    ("mcq", "Big-O of binary search on a sorted array? A) O(n) B) O(n^2) C) O(log n) D) O(1). Reply with just the letter.", "C"),
    ("mcq", "Which gas do plants absorb for photosynthesis? A) Oxygen B) Nitrogen C) Carbon dioxide D) Hydrogen. Reply with just the letter.", "C"),
    ("reason", "Tom is taller than Sam. Sam is taller than Bob. Who is shortest? Reply with just the name.", "bob"),
    ("reason", "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball in cents? Reply with just the number.", "5"),
    ("reason", "If today is Wednesday, what day is it 100 days later? Reply with just the weekday name.", "friday"),
    ("reason", "All roses are flowers. Some flowers fade quickly. Can we conclude some roses fade quickly? Reply yes or no.", "no"),
]
def _norm_num(s):
    mnums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return mnums[-1] if mnums else s.strip().lower()
def _check(typ, ans, gold):
    a = ans.strip()
    if typ == "math":
        return _norm_num(a) == gold or _norm_num(a) == gold.rstrip("0").rstrip(".")
    if typ == "mcq":
        letters = re.findall(r"\b([ABCD])\b", a.upper())
        return bool(letters) and letters[0] == gold
    # reason: 数字 or 关键词
    g = gold.lower()
    if g.isdigit():
        return _norm_num(a) == g
    return g in a.lower()
def test_bench(models):
    print("\n## 4. Mini-Benchmark（数学/MCQ/推理，自动判分）\n")
    res = {}
    for m in models:
        cats = {}
        for typ, q, gold in BENCH:
            try:
                text, _, _ = chat(m, [{"role": "user", "content": q}], max_tokens=64)
                ok = _check(typ, text, gold)
            except Exception as e:
                text, ok = f"ERROR {e}", False
            cats.setdefault(typ, [0, 0])
            cats[typ][1] += 1
            if ok:
                cats[typ][0] += 1
        tot = [sum(v[0] for v in cats.values()), sum(v[1] for v in cats.values())]
        res[m["name"]] = (cats, tot)
        detail = " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in cats.items())
        print(f"  [{m['name']:<20}] 总 {tot[0]}/{tot[1]} ({100*tot[0]/tot[1]:.0f}%)  |  {detail}")
    return res

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="perf,cache,needle,bench",
                    help="逗号分隔: perf,cache,needle,bench")
    ap.add_argument("--quick", action="store_true", help="小规模冒烟")
    ap.add_argument("--gen", type=int, default=200, help="性能测试生成 token 数")
    ap.add_argument("--out", default="", help="把摘要写到 markdown 文件")
    args = ap.parse_args()
    only = set(args.only.split(","))

    if args.quick:
        ctx_points = [200, 4000]
        cache_ctx = 4000
        needle_sizes, needle_depths = [4000], [0.5]
    else:
        ctx_points = [200, 4000, 16000, 32000]
        cache_ctx = 16000
        needle_sizes = [4000, 16000, 48000]
        needle_depths = [0.15, 0.5, 0.85]

    print("="*78)
    print("模型对比:  " + "   vs   ".join(m["name"] for m in MODELS))
    for m in MODELS:
        print(f"  - {m['name']:<20} {m['base']}  (model={m['model']})")
    print("="*78)

    if "perf" in only:   test_perf(MODELS, ctx_points, args.gen)
    if "cache" in only:  test_prefix_cache(MODELS, cache_ctx)
    if "needle" in only: test_needle(MODELS, needle_sizes, needle_depths)
    if "bench" in only:  test_bench(MODELS)
    print("\n完成。")

if __name__ == "__main__":
    main()
