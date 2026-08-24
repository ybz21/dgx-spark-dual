#!/usr/bin/env python3
# 并发压测: 对 OpenAI 兼容端点测 concurrency 1/2/4/8 下的 TTFT + TPS（流式）。
import sys, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9001"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
LEVELS = [1, 2, 4, 6, 8]
PROMPT = "请写一篇关于人工智能如何改变日常生活的短文，包含引言、三个方面和结论，约300字。"

def one_request(idx):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAXTOK, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; ntok = 0; comp = 0
    resp = urllib.request.urlopen(req, timeout=300)
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        try: j = json.loads(data)
        except Exception: continue
        ch = j.get("choices") or []
        if ch:
            delta = ch[0].get("delta", {})
            c = delta.get("content") or delta.get("reasoning_content")
            if c:
                if ttft is None: ttft = time.time() - t0
                ntok += 1
        if j.get("usage"): comp = j["usage"].get("completion_tokens", 0)
    dt = time.time() - t0
    toks = comp or ntok
    return {"ttft": ttft or dt, "wall": dt, "toks": toks}

def run_level(c):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(one_request, range(c)))
    wall = time.time() - t0
    ttfts = sorted(r["ttft"] for r in res)
    tot_toks = sum(r["toks"] for r in res)
    per_req_tps = [r["toks"] / r["wall"] for r in res if r["wall"] > 0]
    return {
        "c": c,
        "ttft_avg": sum(ttfts) / len(ttfts),
        "ttft_p50": ttfts[len(ttfts)//2],
        "ttft_max": ttfts[-1],
        "per_req_tps_avg": sum(per_req_tps) / len(per_req_tps),
        "agg_tps": tot_toks / wall,
        "tot_toks": tot_toks, "wall": wall,
    }

print("# warmup..."); 
for _ in range(2):
    try: one_request(0)  # WARMUP
    except Exception: pass
print(f"# endpoint={BASE} model={MODEL} max_tokens={MAXTOK}")
print(f"{'conc':>4} | {'TTFT_avg':>9} {'TTFT_p50':>9} {'TTFT_max':>9} | {'per-req TPS':>11} | {'aggregate TPS':>13}")
print("-" * 78)
for c in LEVELS:
    r = run_level(c)
    print(f"{r['c']:>4} | {r['ttft_avg']:>8.2f}s {r['ttft_p50']:>8.2f}s {r['ttft_max']:>8.2f}s | "
          f"{r['per_req_tps_avg']:>10.1f} | {r['agg_tps']:>12.1f}")
