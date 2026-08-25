#!/usr/bin/env bash
# smoke-test.sh — 对已起的 Qwen3.8-27B 服务做健康 + 推理 + 简易 decode TPS 测试。
# 用法: bash smoke-test.sh [host] [port]
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"
BASE="http://$HOST:$PORT"
MODEL="qwen3.8-27b"

echo "== 1. /health =="
curl -fsS -m5 "$BASE/health" && echo "  OK" || { echo "  FAIL"; exit 1; }

echo "== 2. /v1/models =="
curl -fsS -m5 "$BASE/v1/models" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  served:", [m["id"] for m in d["data"]])'

echo "== 3. chat 推理（中文）=="
curl -fsS -m120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"用一句话解释什么是张量并行。\"}],
  \"max_tokens\": 128,
  \"temperature\": 0.6
}" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  ->", d["choices"][0]["message"]["content"][:400])'

echo "== 4. decode TPS（256 tok 贪心）=="
python3 - "$BASE" "$MODEL" <<'PY'
import sys, json, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "写一段关于夏天海边的中文散文，约200字。"}],
    "max_tokens": 256, "temperature": 0.0, "stream": False,
}).encode()
req = urllib.request.Request(base + "/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t0 = time.time()
d = json.load(urllib.request.urlopen(req, timeout=180))
dt = time.time() - t0
ct = d.get("usage", {}).get("completion_tokens", 0)
print(f"  completion_tokens={ct}  wall={dt:.1f}s  decode≈{ct/dt:.1f} tok/s")
PY

echo "== 全部通过 =="
