from concurrent.futures import ThreadPoolExecutor
import os as _os
WORKERS=int(_os.environ.get('EVAL_WORKERS','6'))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_quality.py — 用真·标准 benchmark 评两个模型的质量。

数据集（本地 datasets/ 下）:
  - GSM8K   英文小学数学（推理+数值判分）  gsm8k_test.jsonl
  - MMLU    英文知识 MCQ（字母判分）        data/test/*_test.csv   (来自 mmlu.tar)
  - CMMLU   中文知识 MCQ（字母判分）        cmmlu/*.csv

两模型在不同机器 -> 并发评测（墙钟减半）。判分全自动、可复现（固定采样种子）。

用法:
  python3 bench_quality.py                    # 默认样本量
  python3 bench_quality.py --n-gsm8k 40 --n-mmlu 80 --n-cmmlu 40
  python3 bench_quality.py --only gsm8k,mmlu
  python3 bench_quality.py --out quality_report.md
"""
import os, re, csv, json, glob, random, argparse, urllib.request, threading, time

# 改 base/model 指向你的端点; 多模型对比就在下面加条目
MODELS = [
    {"name": "Qwen3.8-27B", "base": "http://192.168.130.48:9001/v1",
     "model": "qwen3.8-27b", "extra": {}},
]
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

def chat(m, prompt, max_tokens=512, temperature=0.0, timeout=600):
    body = {"model": m["model"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    body.update(m.get("extra", {}))
    r = urllib.request.Request(m["base"] + "/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(r, timeout=timeout))
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                return f"__ERR__ {e}"
            time.sleep(2)

# ---------- 判分 ----------
def num_of(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    return m[-1] if m else None
def grade_gsm8k(pred, gold):
    g = num_of(gold)
    # pred 优先取 #### 之后
    tail = pred.split("####")[-1] if "####" in pred else pred
    p = num_of(tail) or num_of(pred)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-4
    except Exception:
        return p == g
def letter_of(s):
    up = s.upper()
    # 推理型模型答案在末尾：优先取“答案/ANSWER 后”的字母，否则取最后一个独立字母
    m = re.search(r"(?:答案|ANSWER|正确选项)[^A-D]{0,6}([ABCD])\b", up)
    if m: return m.group(1)
    allm = re.findall(r"\b([ABCD])\b", up)
    if allm: return allm[-1]
    allm = re.findall(r"([ABCD])", up)
    return allm[-1] if allm else None
def grade_mcq(pred, gold):
    return letter_of(pred) == gold.strip().upper()

# ---------- 数据加载 ----------
def load_gsm8k(n, seed=13):
    path = os.path.join(DS, "gsm8k_test.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    items = []
    for r in rows[:n]:
        prompt = ("Solve this math problem. Think step by step, then on the last line "
                  "write the final answer as '#### <number>'.\n\n" + r["question"])
        items.append((prompt, r["answer"], grade_gsm8k, 10000))
    return items

def _mcq_prompt(q, a, b, c, d, lang):
    if lang == "zh":
        return (f"回答下面的单选题，只输出正确选项的字母（A/B/C/D），不要解释。\n\n"
                f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n答案：")
    return (f"Answer this multiple-choice question. Reply with ONLY the letter (A/B/C/D).\n\n"
            f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")

def load_mmlu(n, seed=13):
    files = sorted(glob.glob(os.path.join(DS, "data", "test", "*_test.csv")))
    pool = []
    for f in files:
        subj = os.path.basename(f).replace("_test.csv", "")
        for row in csv.reader(open(f, encoding="utf-8")):
            if len(row) >= 6:
                q, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                pool.append((_mcq_prompt(q, a, b, c, d, "en"), ans, grade_mcq, 10000, subj))
    random.Random(seed).shuffle(pool)
    return [x[:4] for x in pool[:n]]

def load_cmmlu(n, seed=13):
    files = sorted(glob.glob(os.path.join(DS, "cmmlu", "*.csv")))
    pool = []
    for f in files:
        rdr = csv.reader(open(f, encoding="utf-8"))
        next(rdr, None)  # header
        for row in rdr:
            if len(row) >= 7:
                q, a, b, c, d, ans = row[1], row[2], row[3], row[4], row[5], row[6]
                pool.append((_mcq_prompt(q, a, b, c, d, "zh"), ans, grade_mcq, 10000))
    random.Random(seed).shuffle(pool)
    return pool[:n]

# ---------- 评测（两模型并发）----------
def run_bench(name, items):
    print(f"\n## {name}  ({len(items)} 题)")
    results = {m["name"]: {"ok": 0, "n": 0, "err": 0} for m in MODELS}
    lock = threading.Lock()
    def worker(m):
        def do(it):
            prompt, gold, grader, mx = it[0], it[1], it[2], it[3]
            out = chat(m, prompt, max_tokens=mx)
            ok = (not out.startswith("__ERR__")) and grader(out, gold)
            with lock:
                results[m["name"]]["n"] += 1
                if out.startswith("__ERR__"): results[m["name"]]["err"] += 1
                if ok: results[m["name"]]["ok"] += 1
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(do, items))
    ths = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    dt = time.time() - t0
    for m in MODELS:
        r = results[m["name"]]
        acc = 100 * r["ok"] / max(r["n"], 1)
        print(f"  [{m['name']:<20}] {r['ok']:>3}/{r['n']:<3} = {acc:5.1f}%"
              + (f"  (err {r['err']})" if r["err"] else ""))
    print(f"  用时 {dt:.0f}s")
    return {m["name"]: (results[m["name"]]["ok"], results[m["name"]]["n"]) for m in MODELS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="gsm8k,mmlu,cmmlu")
    ap.add_argument("--n-gsm8k", type=int, default=40)
    ap.add_argument("--n-mmlu", type=int, default=80)
    ap.add_argument("--n-cmmlu", type=int, default=40)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    only = set(args.only.split(","))

    print("=" * 70)
    print("质量评测（真·benchmark）:", "  vs  ".join(m["name"] for m in MODELS))
    print("=" * 70)

    summary = {}
    if "gsm8k" in only and args.n_gsm8k:
        summary["GSM8K(EN数学)"] = run_bench("GSM8K — 英文小学数学", load_gsm8k(args.n_gsm8k))
    if "mmlu" in only and args.n_mmlu:
        summary["MMLU(EN知识)"] = run_bench("MMLU — 英文知识 MCQ", load_mmlu(args.n_mmlu))
    if "cmmlu" in only and args.n_cmmlu:
        summary["CMMLU(ZH知识)"] = run_bench("CMMLU — 中文知识 MCQ", load_cmmlu(args.n_cmmlu))

    print("\n" + "=" * 70 + "\n汇总（准确率%）")
    hdr = f"  {'benchmark':<16}" + "".join(f"{m['name']:>22}" for m in MODELS)
    print(hdr)
    for b, r in summary.items():
        line = f"  {b:<16}"
        for m in MODELS:
            ok, n = r[m["name"]]
            line += f"{f'{100*ok/max(n,1):.1f}% ({ok}/{n})':>22}"
        print(line)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# 质量评测报告（真·benchmark）\n\n")
            f.write("| benchmark | " + " | ".join(m["name"] for m in MODELS) + " |\n")
            f.write("|---|" + "---|" * len(MODELS) + "\n")
            for b, r in summary.items():
                f.write(f"| {b} | " + " | ".join(
                    f"{100*r[m['name']][0]/max(r[m['name']][1],1):.1f}% ({r[m['name']][0]}/{r[m['name']][1]})"
                    for m in MODELS) + " |\n")
        print(f"\n报告已写入 {args.out}")

if __name__ == "__main__":
    main()
