# 评测框架 harness.py（OpenAI 接口 · 动态数据集 · 每集独立报告）

在 [`scripts/`](scripts) 那套单模型对比脚本之外，`harness.py` 是**可扩展的通用评测框架**：

- **OpenAI 接口驱动**：`--base/--model/--api-key`，任何 OpenAI 兼容端点都能测。
- **动态加数据集（零代码）**：往 [`datasets.d/`](datasets.d) 丢一个 `<name>.json` 声明即可，见 `datasets.d/README.md`。
- **每个数据集出独立报告**：`reports/<model>/<dataset>.md` + 一份 `SUMMARY.md`。
- **题目级并发**：`--workers`（默认 6 / 环境变量 `EVAL_WORKERS`）；纯标准库，无需 pip。

## 用法

```bash
# 先下内置数据集
bash scripts/download-datasets.sh

# 全量（内置 + datasets.d 里的动态数据集）
python3 harness.py --base http://192.168.130.48:9001/v1 --model qwen3.8-27b --datasets all

# 指定数据集 / 题数 / 开思考模式
python3 harness.py --base <endpoint>/v1 --model <id> --datasets gsm8k,cmmlu --n 40 --workers 6 --think

# 带鉴权
python3 harness.py --base https://api.example.com/v1 --model gpt-x --api-key sk-xxx --datasets all
```

## 内置数据集
`gsm8k`(数学) · `mmlu`(英文知识) · `cmmlu`(中文知识) · `humaneval` / `mbpp`(代码，子进程真跑测试用例)。

## 加自己的数据集（举例）
```bash
# 1) 数据文件放 datasets.d/（或 scripts/datasets/）
echo '{"question":"1+1=?","answer":"2"}' > datasets.d/mytest.jsonl
# 2) 写声明
cat > datasets.d/mytest.json <<JSON
{"name":"mytest","group":"custom","kind":"numeric","file":"mytest.jsonl",
 "question_field":"question","answer_field":"answer",
 "prompt_template":"{question}\n只给数字。","n_default":50}
JSON
# 3) 直接测（自动出 reports/<model>/mytest.md）
python3 harness.py --base <endpoint>/v1 --model <id> --datasets mytest
```
`kind` 支持 `numeric/mcq/exact/contains/regex`；MCQ 用 `option_fields`。详见 `datasets.d/README.md`。
