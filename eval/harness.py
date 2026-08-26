#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harness.py — 可扩展的 OpenAI 接口评测框架。

特性:
  1. 纯 OpenAI 兼容接口驱动 (--base/--model/--api-key)，任何 OpenAI 兼容端点都能测。
  2. 数据集注册表 + JSON 声明式动态加数据集 (datasets.d/*.json)，加数据集无需改代码。
  3. 每个数据集出一份独立报告 (reports/<model>/<dataset>.md) + 一份汇总 SUMMARY.md。
  4. 题目级并发 (--workers)，纯标准库无需 pip。

用法:
  # 内置数据集(需先 bash scripts/download-datasets.sh)
  python3 harness.py --base http://192.168.130.48:9001/v1 --model qwen3.8-27b --datasets all
  python3 harness.py --base ... --model ... --datasets gsm8k,mmlu --workers 6 --n 40
  python3 harness.py --base ... --model ... --datasets all --think   # 开思考模式

动态加数据集: 在 datasets.d/ 放一个 <name>.json (见 datasets.d/README 或 _example.json)，
  再把数据文件放到 scripts/datasets/ 下，即可 --datasets <name> 直接测，无需改代码。
"""
import os, re, csv, json, glob, time, random, argparse, subprocess, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "scripts", "datasets")      # 复用现有数据目录
EXTDIR = os.path.join(HERE, "datasets.d")             # JSON 声明式数据集

# ----------------------------------------------------------------------------- OpenAI 调用
def _post(cfg, body, timeout=600):
    body = dict(body); body.setdefault("temperature", 0.0)
    body.update(cfg.get("extra", {}))
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]
    req = urllib.request.Request(cfg["base"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    last = None
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            return d["choices"][0]["message"]
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(2)
    return {"__err__": str(last)}

def chat(cfg, prompt, max_tokens=1024, timeout=600):
    m = _post(cfg, {"model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens}, timeout)
    if "__err__" in m:
        return "__ERR__ " + m["__err__"]
    return m.get("content") or ""

def chat_tool(cfg, query, tools, max_tokens=512, timeout=120):
    return _post(cfg, {"model": cfg["model"],
                       "messages": [{"role": "user", "content": query}],
                       "tools": tools, "tool_choice": "auto",
                       "max_tokens": max_tokens}, timeout)

# ----------------------------------------------------------------------------- 判分器
def _num(s):
    s = (s or "").replace(",", "")
    m = re.findall(r"-?\d+\.?\d*", s)
    return m[-1] if m else None

def g_numeric(pred, gold):
    p, g = _num(pred), _num(str(gold))
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-4
    except Exception:
        return p == g

def _letter(s):
    s = (s or "").strip()
    m = re.search(r"\b([ABCD])\b", s.upper())
    if m:
        return m.group(1)
    m = re.search(r"[（(]([ABCD])[)）]", s.upper())
    return m.group(1) if m else None

def g_mcq(pred, gold):
    return _letter(pred) == str(gold).strip().upper()

def g_exact(pred, gold):
    return (pred or "").strip().lower() == str(gold).strip().lower()

def g_contains(pred, gold):
    return str(gold).strip().lower() in (pred or "").lower()

def g_regex(pred, gold):
    try:
        return re.search(str(gold), pred or "", re.S) is not None
    except Exception:
        return False

def g_toolcall(msg, expect):
    """判分工具调用: 模型是否调对函数 + 关键参数匹配(子集, 宽松)。"""
    if not isinstance(msg, dict) or "__err__" in msg:
        return False
    calls = msg.get("tool_calls") or []
    want_name = expect.get("name")
    want_args = expect.get("args", {}) or {}
    for c in calls:
        fn = c.get("function", {})
        if fn.get("name") != want_name:
            continue
        try:
            got = json.loads(fn.get("arguments") or "{}")
        except Exception:
            got = {}
        ok = True
        for k, v in want_args.items():
            gv = got.get(k)
            if gv is None or str(v).strip().lower() not in str(gv).strip().lower():
                ok = False; break
        if ok:
            return True
    return False

def _extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m[-1] if m else (text or "")).strip()

def _run_py(src, timeout=15):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(path)

# ----------------------------------------------------------------------------- 内置数据集加载
def _mcq_prompt(q, a, b, c, d, lang):
    tip = "只回答选项字母(A/B/C/D)。" if lang == "zh" else "Answer with just the option letter (A/B/C/D)."
    return f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n{tip}"

def load_gsm8k(n, seed=13):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "gsm8k_test.jsonl"), encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        gold = r["answer"].split("####")[-1].strip()
        out.append((r["question"] + "\nPlease reason, then give the final numeric answer.", gold))
    return out

def load_mmlu(n, seed=13):
    pool = []
    for f in sorted(glob.glob(os.path.join(DATA, "data", "test", "*_test.csv"))):
        for row in csv.reader(open(f, encoding="utf-8")):
            if len(row) >= 6:
                pool.append((_mcq_prompt(row[0], row[1], row[2], row[3], row[4], "en"), row[5]))
    random.Random(seed).shuffle(pool)
    return pool[:n]

def load_cmmlu(n, seed=13):
    pool = []
    for f in sorted(glob.glob(os.path.join(DATA, "cmmlu", "*.csv"))):
        rd = csv.reader(open(f, encoding="utf-8")); next(rd, None)
        for row in rd:
            if len(row) >= 7:
                pool.append((_mcq_prompt(row[1], row[2], row[3], row[4], row[5], "zh"), row[6]))
    random.Random(seed).shuffle(pool)
    return pool[:n]

def load_humaneval(n):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "HumanEval.jsonl"), encoding="utf-8")][:n]
    return [(r["prompt"], r) for r in rows]

def load_mbpp(n):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "mbpp.jsonl"), encoding="utf-8")][:n]
    return [(f"{r['text']}\n请用 Python 实现，并保证通过:\n" + "\n".join(r["test_list"]), r) for r in rows]

def grade_he(pred, row):
    return _run_py(row["prompt"] + _extract_code(pred) + "\n" + row["test"] +
                   f"\ncheck({row['entry_point']})")

def grade_mbpp(pred, row):
    return _run_py(_extract_code(pred) + "\n" + "\n".join(row["test_list"]))

# 内置注册表: name -> spec
BUILTIN = {
    "gsm8k":     {"group": "math",     "mode": "text", "load": load_gsm8k,   "grade": g_numeric, "max_tokens": 8192,  "n": 40},
    "mmlu":      {"group": "知识-EN",  "mode": "text", "load": load_mmlu,    "grade": g_mcq,     "max_tokens": 8192,  "n": 80},
    "cmmlu":     {"group": "知识-ZH",  "mode": "text", "load": load_cmmlu,   "grade": g_mcq,     "max_tokens": 8192,  "n": 40},
    "humaneval": {"group": "code",     "mode": "code", "load": load_humaneval,"grade": grade_he, "max_tokens": 4096,  "n": 40},
    "mbpp":      {"group": "code",     "mode": "code", "load": load_mbpp,    "grade": grade_mbpp,"max_tokens": 4096,  "n": 40},
}

# ----------------------------------------------------------------------------- JSON 声明式数据集
GRADERS = {"numeric": g_numeric, "mcq": g_mcq, "exact": g_exact, "contains": g_contains, "regex": g_regex}

def load_json_dataset(spec):
    """从 datasets.d/*.json 的声明构造 (load, grade) — 动态加数据集，无需改代码。"""
    path = os.path.join(EXTDIR, spec["file"])
    if not os.path.exists(path):
        path = os.path.join(DATA, spec["file"])
    qf, af = spec.get("question_field", "question"), spec.get("answer_field", "answer")
    tmpl = spec.get("prompt_template", "{question}")
    opts = spec.get("option_fields")  # mcq: ["A","B","C","D"] 字段名
    def loader(n, seed=13):
        rows = []
        if spec.get("format", "jsonl") == "jsonl":
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        else:
            rd = csv.DictReader(open(path, encoding="utf-8")); rows = list(rd)
        random.Random(seed).shuffle(rows)
        out = []
        for r in rows[:n]:
            if opts:
                q = _mcq_prompt(r[qf], *[r[o] for o in opts], spec.get("lang", "zh"))
            else:
                q = tmpl.format(**r)
            out.append((q, r[af]))
        return out
    if spec.get("kind") == "toolcall":
        def tc_loader(n, seed=13):
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            random.Random(seed).shuffle(rows)
            return [(r["query"], {"tools": r["tools"], "expect": r["expect"]}) for r in rows[:n]]
        return {"group": spec.get("group", "function-call"), "mode": "toolcall",
                "load": tc_loader, "grade": g_toolcall,
                "max_tokens": spec.get("max_tokens", 512), "n": spec.get("n_default", 20)}
    return {"group": spec.get("group", "custom"), "mode": "text", "load": loader,
            "grade": GRADERS[spec.get("kind", "exact")],
            "max_tokens": spec.get("max_tokens", 4096), "n": spec.get("n_default", 40)}

def discover_datasets():
    reg = dict(BUILTIN)
    for jf in sorted(glob.glob(os.path.join(EXTDIR, "*.json"))):
        if os.path.basename(jf).startswith("_"):
            continue
        try:
            spec = json.load(open(jf, encoding="utf-8"))
            reg[spec["name"]] = load_json_dataset(spec)
            reg[spec["name"]]["_ext"] = True
        except Exception as e:
            print(f"[warn] 跳过 {jf}: {e}")
    return reg

# ----------------------------------------------------------------------------- 运行 + 报告
def run_one(cfg, name, spec, n, workers):
    items = spec["load"](n)
    total = len(items)
    res = {"ok": 0, "n": 0, "err": 0}
    lock = __import__("threading").Lock()
    grade = spec["grade"]
    mode = spec.get("mode", "text")
    def do(it):
        payload, gold = it
        if mode == "toolcall":
            msg = chat_tool(cfg, payload, gold["tools"], max_tokens=spec["max_tokens"])
            err = isinstance(msg, dict) and "__err__" in msg
            ok = (not err) and grade(msg, gold["expect"])
        else:
            out = chat(cfg, payload, max_tokens=spec["max_tokens"])
            err = out.startswith("__ERR__")
            ok = (not err) and grade(out, gold)
        with lock:
            res["n"] += 1
            if err: res["err"] += 1
            if ok: res["ok"] += 1
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(do, items))
    dt = time.time() - t0
    acc = 100.0 * res["ok"] / max(res["n"], 1)
    return {"name": name, "group": spec["group"], "acc": acc, "ok": res["ok"],
            "n": res["n"], "err": res["err"], "sec": dt}

def write_report(out_dir, cfg, r):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{r['name']}.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# {r['name']} 评测报告\n\n")
        f.write(f"- 模型: `{cfg['model']}` @ `{cfg['base']}`\n")
        f.write(f"- 类别: {r['group']} · 思考模式: {'开' if not cfg.get('extra',{}).get('chat_template_kwargs',{}).get('enable_thinking',True)==False else '关'}\n")
        f.write(f"- 日期: {cfg['date']}\n\n")
        f.write("| 指标 | 值 |\n|---|---|\n")
        f.write(f"| 准确率 | **{r['acc']:.1f}%** ({r['ok']}/{r['n']}) |\n")
        f.write(f"| 出错(网络/超时) | {r['err']} |\n")
        f.write(f"| 用时 | {r['sec']:.0f}s |\n")
    return p

def main():
    ap = argparse.ArgumentParser(description="OpenAI 接口可扩展评测框架")
    ap.add_argument("--base", required=True, help="OpenAI 兼容端点, 如 http://host:9001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--datasets", default="all", help="逗号分隔; all=全部(含动态)")
    ap.add_argument("--n", type=int, default=0, help="每个数据集题数(0=用各自默认)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("EVAL_WORKERS", "6")))
    ap.add_argument("--think", action="store_true", help="开思考模式(默认关)")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    reg = discover_datasets()
    names = list(reg) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",")]
    bad = [x for x in names if x not in reg]
    if bad:
        print(f"[err] 未知数据集: {bad}\n可用: {list(reg)}"); return

    extra = {} if args.think else {"chat_template_kwargs": {"enable_thinking": False}}
    cfg = {"base": args.base, "model": args.model, "api_key": args.api_key,
           "extra": extra, "date": time.strftime("%Y-%m-%d")}
    out_dir = args.out_dir or os.path.join(HERE, "reports", re.sub(r"[^\w.-]", "_", args.model))

    print("=" * 74)
    print(f"评测 {args.model} @ {args.base}  | 思考={'开' if args.think else '关'} | 并发={args.workers}")
    print(f"数据集: {names}  | 报告 → {out_dir}/")
    print("=" * 74)

    rows = []
    for nm in names:
        spec = reg[nm]
        n = args.n or spec["n"]
        print(f"\n## {nm} ({spec['group']}, {n} 题) ...", flush=True)
        r = run_one(cfg, nm, spec, n, args.workers)
        rows.append(r)
        rp = write_report(out_dir, cfg, r)
        tag = " [动态]" if spec.get("_ext") else ""
        print(f"  [{nm}{tag}] {r['ok']}/{r['n']} = {r['acc']:.1f}%  ({r['sec']:.0f}s"
              + (f", err {r['err']}" if r['err'] else "") + f")  → {os.path.relpath(rp, HERE)}")

    # 汇总报告
    os.makedirs(out_dir, exist_ok=True)
    sp = os.path.join(out_dir, "SUMMARY.md")
    with open(sp, "w", encoding="utf-8") as f:
        f.write(f"# {args.model} 评测汇总\n\n")
        f.write(f"- 端点: `{args.base}` · 思考模式: {'开' if args.think else '关'} · 日期: {cfg['date']}\n\n")
        f.write("| 数据集 | 类别 | 准确率 | 题数 | 用时 |\n|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['group']} | **{r['acc']:.1f}%** | {r['ok']}/{r['n']} | {r['sec']:.0f}s |\n")
    print(f"\n{'='*74}\n汇总 → {os.path.relpath(sp, HERE)}")
    for r in rows:
        print(f"  {r['name']:<14} {r['acc']:5.1f}%  ({r['ok']}/{r['n']})")

if __name__ == "__main__":
    main()
