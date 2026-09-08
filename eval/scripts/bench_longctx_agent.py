#!/usr/bin/env python3
"""长上下文 + 迭加下文的编码 agent 压测。

模拟真实写代码场景：先把一个"代码库"整个塞进上下文，之后每一轮追加一个
unified diff（模拟一次编辑）再提问。会话单调增长，前缀完全一致，
所以 prefix cache 应当让 TTFT 基本持平，而不是随上下文线性上涨。

量的是**流式首 token 时间**，不是整轮耗时——后者被 decode 长度污染。

用法:
  python3 bench_longctx_agent.py --base http://192.168.130.8:8000/v1 \
      --model glm-5.3-flash --ctx-files 60 --turns 20
  # 长时压测（跑够 60 分钟，中途不断追加 diff）
  python3 bench_longctx_agent.py --base ... --model ... --duration 3600

纯标准库，无需 pip。
"""
import argparse, json, os, random, re, statistics, sys, time, urllib.request, urllib.error

# --------------------------------------------------------------- 合成代码库
FUNCS = ["parse", "encode", "resolve", "merge", "validate", "flush", "compact",
         "reindex", "shard", "rebalance", "checkpoint", "replay", "prune", "gc"]
NOUNS = ["segment", "manifest", "journal", "tablet", "bloom", "arena", "slab",
         "cursor", "epoch", "quorum", "lease", "digest", "ledger", "batch"]

def make_file(rng, name, lines):
    """生成一段看着像真代码的 Python。刻意做出变化，避免高度重复的文本
    让 tokenizer 和 cache 的行为失真。"""
    out = [f'"""{name} — {rng.choice(NOUNS)} 层实现。"""',
           "from __future__ import annotations", "import dataclasses, typing as t", ""]
    while len(out) < lines:
        fn, noun = rng.choice(FUNCS), rng.choice(NOUNS)
        n = rng.randint(3, 9)
        out.append(f"def {fn}_{noun}(state: dict, *, limit: int = {rng.randint(2,4096)}) -> int:")
        out.append(f'    """{fn} the {noun} under a {rng.choice(NOUNS)} budget."""')
        out.append(f"    total = 0")
        for _ in range(n):
            k = rng.choice(["acc", "cur", "hi", "lo", "seen", "carry"])
            out.append(f"    {k} = state.get({rng.randint(0,99)!r}, {rng.randint(0,999)})")
            out.append(f"    total += {k} * {rng.randint(2,17)} % {rng.randint(3,251)}")
        out.append(f"    return total")
        out.append("")
    return "\n".join(out[:lines])

def make_repo(n_files, lines, seed=7):
    rng = random.Random(seed)
    return {f"src/{rng.choice(NOUNS)}_{i:03d}.py": make_file(rng, f"module {i}", lines)
            for i in range(n_files)}

def repo_blob(repo):
    return "\n".join(f"===== FILE: {p} =====\n{c}" for p, c in repo.items())

def make_diff(repo, rng, turn):
    """造一个改两个文件的 unified diff，并把改动落回 repo（保持状态一致）。"""
    hunks = []
    for path in rng.sample(list(repo), k=min(2, len(repo))):
        body = repo[path].split("\n")
        i = rng.randrange(max(1, len(body) - 4))
        old = body[i]
        new = f"    total += {rng.randint(2,97)}  # turn {turn} tune"
        body[i] = new
        repo[path] = "\n".join(body)
        hunks.append(f"--- a/{path}\n+++ b/{path}\n@@ -{i+1},1 +{i+1},1 @@\n-{old}\n+{new}")
    return "\n".join(hunks)

# --------------------------------------------------------------- HTTP
def post_stream(base, model, msgs, max_tokens, api_key=None, timeout=1800, retries=1):
    """返回 (ttft 秒, 生成 token 数, decode 秒, 首个错误)。用流式量首 token。"""
    body = json.dumps({"model": model, "messages": msgs, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    hdr = {"Content-Type": "application/json"}
    if api_key: hdr["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body, headers=hdr)
    t0 = time.time(); ttft = None; n = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"): continue
                payload = line[5:].strip()
                if payload == "[DONE]": break
                try: d = json.loads(payload)
                except Exception: continue
                delta = (d.get("choices") or [{}])[0].get("delta", {}) or {}
                if delta.get("content"):
                    if ttft is None: ttft = time.time() - t0
                    n += 1
    except urllib.error.HTTPError as e:
        # 带上状态码和 body。裸 HTTPError 什么都看不出来——排查一次 401
        # （服务端悄悄换了模型并开了鉴权）就是被这个拖慢的。
        try: detail = e.read().decode()[:200]
        except Exception: detail = ""
        return None, 0, time.time() - t0, f"HTTP {e.code} after {time.time()-t0:.0f}s: {detail}"
    except Exception as e:
        # 冷 prefill 打满 27 万 token 时单次抖动就会废掉整轮，重试一次再判死。
        if retries > 0:
            time.sleep(5)
            return post_stream(base, model, msgs, max_tokens, api_key, timeout, retries - 1)
        return None, 0, time.time() - t0, f"{type(e).__name__} after {time.time()-t0:.0f}s"
    t_end = time.time()
    if ttft is None:
        # 流正常结束却一个 content delta 都没有：可能整轮都被 reasoning 吃掉、
        # 或 max_tokens 太小、或服务端静默截断。当作失败记录，别让 round(None) 崩掉。
        return None, 0, t_end - t0, f"no content delta in {t_end-t0:.1f}s"
    return ttft, n, (t_end - t0 - ttft), None

def metrics(base):
    """抓 prefix cache 累计计数，用于算每轮增量命中率。"""
    url = base.rstrip("/").rsplit("/v1", 1)[0] + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            txt = r.read().decode()
    except Exception:
        return None, None
    def g(name):
        m = re.search(rf"^vllm:{name}\{{[^}}]*\}}\s+([0-9.e+-]+)$", txt, re.M)
        return float(m.group(1)) if m else None
    return g("prefix_cache_queries_total"), g("prefix_cache_hits_total")

# --------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="长上下文 + 迭加 diff 的编码 agent 压测")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--ctx-files", type=int, default=60, help="打底代码库文件数")
    ap.add_argument("--ctx-lines", type=int, default=120, help="每个文件行数")
    ap.add_argument("--turns", type=int, default=20, help="追加轮数（--duration 优先）")
    ap.add_argument("--duration", type=int, default=0, help="压测秒数，>0 时忽略 --turns")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=int, default=1800, help="单次请求超时秒数")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    repo = make_repo(a.ctx_files, a.ctx_lines)
    rng = random.Random(11)
    blob = repo_blob(repo)
    print(f"打底代码库: {a.ctx_files} 文件 × {a.ctx_lines} 行 ≈ {len(blob):,} 字符", flush=True)

    msgs = [{"role": "user", "content":
             "你是代码维护 agent。下面是整个代码库，先通读，后续我会不断给你 diff。\n\n"
             + blob + "\n\n请用一句话说明这个库大致在做什么。"}]
    rows = []
    q0, h0 = metrics(a.base)
    t_start = time.time(); turn = 0
    while True:
        if a.duration: 
            if time.time() - t_start >= a.duration: break
        elif turn >= a.turns: break

        if turn > 0:   # 追加一轮 diff
            diff = make_diff(repo, rng, turn)
            msgs.append({"role": "user", "content":
                         f"第 {turn} 次改动：\n```diff\n{diff}\n```\n"
                         "这次改动有没有引入问题？一句话回答。"})
        qa, ha = metrics(a.base)
        ttft, ntok, dec, err = post_stream(a.base, a.model, msgs, a.max_tokens, a.api_key, a.timeout)
        qb, hb = metrics(a.base)
        hit = None
        if None not in (qa, ha, qb, hb) and qb > qa:
            hit = (hb - ha) / (qb - qa) * 100
        approx_ctx = sum(len(m["content"]) for m in msgs) // 3   # 粗估 token
        if err:
            print(f"  turn {turn:3d}  错误: {err}", flush=True)
            rows.append(dict(turn=turn, err=err, ctx_tok_approx=approx_ctx)); turn += 1; continue
        tps = ntok / dec if dec > 0 else 0
        rows.append(dict(turn=turn, ctx_tok_approx=approx_ctx, ttft=round(ttft, 3),
                         gen_tok=ntok, decode_tps=round(tps, 1),
                         prefix_hit_pct=None if hit is None else round(hit, 1)))
        print(f"  turn {turn:3d}  ctx≈{approx_ctx:>8,}  TTFT {ttft:6.2f}s  "
              f"decode {tps:5.1f} tok/s  prefix命中 {('%.1f%%'%hit) if hit is not None else '?'}",
              flush=True)
        # 把回答也接进会话，前缀才继续单调增长
        msgs.append({"role": "assistant", "content": "(ok)"})
        turn += 1

    ok = [r for r in rows if "ttft" in r]
    if ok:
        first, rest = ok[0], ok[1:]
        print("\n=== 汇总 ===")
        print(f"  首轮(冷)   TTFT {first['ttft']}s  ctx≈{first['ctx_tok_approx']:,}")
        if rest:
            ts = [r["ttft"] for r in rest]
            print(f"  迭加轮(热) TTFT 中位 {statistics.median(ts):.2f}s  "
                  f"min {min(ts):.2f}  max {max(ts):.2f}  n={len(ts)}")
            print(f"  末轮 ctx≈{rest[-1]['ctx_tok_approx']:,}  TTFT {rest[-1]['ttft']}s")
            print(f"  冷/热 TTFT 比 {first['ttft']/statistics.median(ts):.1f}×")
            ds = [r["decode_tps"] for r in rest]
            print(f"  decode 中位 {statistics.median(ds):.1f} tok/s")
    if a.out:
        with open(a.out, "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  明细 → {a.out}")

if __name__ == "__main__":
    main()
