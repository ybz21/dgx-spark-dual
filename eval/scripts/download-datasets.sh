#!/usr/bin/env bash
# download-datasets.sh — 拉取评测用的公开 benchmark 数据集到 ./datasets/
# 脚本 bench_quality.py / bench_code.py 会在同目录的 datasets/ 下找数据。
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p datasets/cmmlu
cd datasets

echo "[*] GSM8K (英文数学)"
curl -sL -o gsm8k_test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

echo "[*] HumanEval (代码)"
curl -sL -o HumanEval.jsonl.gz \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
gunzip -kf HumanEval.jsonl.gz

echo "[*] MBPP (代码)"
curl -sL -o mbpp.jsonl \
  https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl

echo "[*] CMMLU (中文知识 MCQ, 部分学科)"
for s in astronomy college_medicine chinese_history high_school_physics marketing \
         world_religions economics elementary_mathematics philosophy law professional_psychology; do
  curl -sL -o "cmmlu/$s.csv" \
    "https://raw.githubusercontent.com/haonan-li/CMMLU/master/data/test/$s.csv"
done

echo "[*] MMLU (英文知识 MCQ, 57 学科) — 伯克利源较慢，耐心等"
curl -sL -o mmlu.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xf mmlu.tar && rm -f mmlu.tar   # 解出 data/test/*_test.csv

echo "[✓] 数据集就绪于 $(pwd)"
