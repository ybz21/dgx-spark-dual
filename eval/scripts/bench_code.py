#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_code.py — 编码能力评测（HumanEval + MBPP，pass@1，真跑测试用例）。

- HumanEval: 补全函数 -> 拼 test -> 跑 check(entry_point)
- MBPP     : 按描述写函数 -> 跑 test_list 断言
两模型在不同机器 -> 并发。每份生成代码在**独立子进程 + 超时**里执行判分。

⚠️ 会执行模型生成的代码（benchmark 标准做法），已限制超时；请在可信环境运行。

用法:
  python3 bench_code.py                       # 默认 HumanEval 50 + MBPP 50
  python3 bench_code.py --n-he 60 --n-mbpp 60
  python3 bench_code.py --only humaneval --out code_report.md
"""
import os, re, json, time, argparse, threading, subprocess, tempfile, urllib.request

MODELS = [
    # .12 现在跑的是 Laguna(DS4 已停，两者内存互斥)
    {"name": "Laguna-S-2.1-NVFP4", "base": "http://192.168.130.12:8000/v1",
     "model": "laguna-s-2.1", "extra": {}},
    {"name": "Qwen3.5-122B-A10B", "base": "http://192.168.130.45:30001/v1",
     "model": "qwen3.5-122b-int4", "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
]
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

def chat(m, prompt, max_tokens=768, temperature=0.0, timeout=600):
    body = {"model": m["model"], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    body.update(m.get("extra", {}))
    r = urllib.request.Request(m["base"] + "/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    for att in range(3):
        try:
            return json.load(urllib.request.urlopen(r, timeout=timeout))["choices"][0]["message"]["content"]
        except Exception as e:
            if att == 2: return f"__ERR__ {e}"
            time.sleep(2)

def extract_code(text):
    """取第一个 ```python``` 代码块；没有就取整段。"""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m: return m.group(1)
    # 退化：去掉可能的散文，保留看起来像代码的部分
    return text

def run_program(src, timeout=15):
    """在子进程执行；返回 True=通过（退出码0，无异常）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run(["python3", path], capture_output=True, timeout=timeout, text=True)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass

# ---------- HumanEval ----------
def load_humaneval(n):
    rows = [json.loads(l) for l in open(os.path.join(DS, "HumanEval.jsonl"), encoding="utf-8")]
    return rows[:n]
def grade_humaneval(row, out):
    code = extract_code(out)
    ep = row["entry_point"]
    if f"def {ep}" in code:
        prog = code + "\n\n" + row["test"] + f"\n\ncheck({ep})\n"
    else:  # 模型只给了函数体，拼回签名+docstring
        prog = row["prompt"] + code + "\n\n" + row["test"] + f"\n\ncheck({ep})\n"
    return run_program(prog)
def prompt_humaneval(row):
    return ("Complete the following Python function. "
            "Return ONLY the complete function (with signature) inside a ```python code block, no explanation.\n\n"
            + row["prompt"])

# ---------- MBPP ----------
def load_mbpp(n):
    rows = [json.loads(l) for l in open(os.path.join(DS, "mbpp.jsonl"), encoding="utf-8")]
    return rows[:n]
def grade_mbpp(row, out):
    code = extract_code(out)
    setup = row.get("test_setup_code", "") or ""
    prog = (setup + "\n" if setup else "") + code + "\n" + "\n".join(row["test_list"]) + "\n"
    return run_program(prog)
def prompt_mbpp(row):
    return ("Write a Python function for the task below. "
            "Return ONLY the function inside a ```python code block, no explanation.\n\n"
            f"Task: {row['text']}\n"
            f"It must satisfy this test: {row['test_list'][0]}")

BENCHES = {
    "humaneval": ("HumanEval", load_humaneval, prompt_humaneval, grade_humaneval),
    "mbpp":      ("MBPP",      load_mbpp,      prompt_mbpp,      grade_mbpp),
}

def run_bench(key, n):
    title, load, mkprompt, grade = BENCHES[key]
    rows = load(n)
    print(f"\n## {title}  ({len(rows)} 题, pass@1 贪心)")
    res = {m["name"]: {"ok": 0, "n": 0, "err": 0} for m in MODELS}
    lock = threading.Lock()
    def worker(m):
        for row in rows:
            out = mkprompt(row)
            gen = chat(m, out, max_tokens=10000)  # 基本不设限，保证 Laguna 长推理不被截断
            ok = (not gen.startswith("__ERR__")) and grade(row, gen)
            with lock:
                res[m["name"]]["n"] += 1
                if gen.startswith("__ERR__"): res[m["name"]]["err"] += 1
                if ok: res[m["name"]]["ok"] += 1
    ths = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    for m in MODELS:
        r = res[m["name"]]
        print(f"  [{m['name']:<20}] {r['ok']:>3}/{r['n']:<3} = {100*r['ok']/max(r['n'],1):5.1f}%"
              + (f"  (err {r['err']})" if r['err'] else ""))
    print(f"  用时 {time.time()-t0:.0f}s")
    return {m["name"]: (res[m["name"]]["ok"], res[m["name"]]["n"]) for m in MODELS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="humaneval,mbpp")
    ap.add_argument("--n-he", type=int, default=50)
    ap.add_argument("--n-mbpp", type=int, default=50)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    only = args.only.split(",")
    print("=" * 70)
    print("编码评测（pass@1）:", "  vs  ".join(m["name"] for m in MODELS))
    print("=" * 70)
    summary = {}
    if "humaneval" in only: summary["HumanEval"] = run_bench("humaneval", args.n_he)
    if "mbpp" in only:      summary["MBPP"] = run_bench("mbpp", args.n_mbpp)
    print("\n" + "=" * 70 + "\n汇总 pass@1")
    print(f"  {'benchmark':<14}" + "".join(f"{m['name']:>22}" for m in MODELS))
    for b, r in summary.items():
        line = f"  {b:<14}"
        for m in MODELS:
            ok, n = r[m["name"]]; line += f"{f'{100*ok/max(n,1):.1f}% ({ok}/{n})':>22}"
        print(line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# 编码评测报告 (pass@1)\n\n| benchmark | " + " | ".join(m["name"] for m in MODELS) + " |\n")
            f.write("|---|" + "---|" * len(MODELS) + "\n")
            for b, r in summary.items():
                f.write(f"| {b} | " + " | ".join(f"{100*r[m['name']][0]/max(r[m['name']][1],1):.1f}% ({r[m['name']][0]}/{r[m['name']][1]})" for m in MODELS) + " |\n")
        print(f"\n报告已写入 {args.out}")

if __name__ == "__main__":
    main()
