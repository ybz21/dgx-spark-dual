#!/usr/bin/env bash
# 用权重仓库自带的 SHA256SUMS 校验 120 个 shard + 配置文件。
# 跳过 runtime/ 和 .materialization/ 两组条目：那是量化流水线的源码和中间产物，
# ModelScope 镜像没带，推理也用不到。
# LICENSE 和 README.md 必然对不上 —— ModelScope 镜像换过这两个文件，不影响推理。
set -u
export LC_ALL=C          # 否则中文 locale 下 sha256sum -c 打的是"成功/失败"，统计会全为 0
d="${MODEL_DIR:-$HOME/models/GLM-5.3-Flash-EXL3-TR3-4bpw}"
log="${LOG:-$HOME/glm53-verify.log}"
cd "$d" || exit 1
: > "$log"
grep -vE '^\S+  (runtime/|\.materialization/)' SHA256SUMS \
  | while read -r h f; do [ -f "$f" ] && printf '%s  %s\n' "$h" "$f"; done > /tmp/glm53-sums.want
echo "checking $(wc -l < /tmp/glm53-sums.want) files" | tee -a "$log"
rm -f /tmp/glm53-sums.part.*
split -n l/8 /tmp/glm53-sums.want /tmp/glm53-sums.part.
for p in /tmp/glm53-sums.part.*; do sha256sum -c "$p" >> "$log" 2>&1 & done
wait
echo "RESULT ok=$(grep -c ': OK' "$log") failed=$(grep -c ': FAILED' "$log")" | tee -a "$log"
grep ': FAILED' "$log" || true
