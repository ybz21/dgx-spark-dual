# 动态加数据集（无需改代码）

在本目录放一个 `<name>.json`（下划线开头的会被忽略，如 `_example.json`），
数据文件放到 `../scripts/datasets/` 下，即可 `harness.py --datasets <name>` 直接测。

## JSON 字段

| 字段 | 说明 |
|---|---|
| `name` | 数据集名（= --datasets 里用的名字）|
| `group` | 类别标签（math / 知识 / code / custom 等；code 组会跑代码判分）|
| `kind` | 判分方式：`numeric`(数值) / `mcq`(选择题字母) / `exact`(全等) / `contains`(含答案) / `regex`(正则) |
| `file` | 数据文件名（相对 scripts/datasets/）|
| `format` | `jsonl` 或 `csv` |
| `question_field` / `answer_field` | 问题/答案字段名 |
| `prompt_template` | 用 `{字段名}` 拼 prompt，如 `"{q}\n只给数字"` |
| `option_fields` | MCQ 选项字段名数组 `["A","B","C","D"]`（配 kind=mcq）|
| `lang` | MCQ 提示语言 `zh`/`en` |
| `max_tokens` / `n_default` | 生成上限 / 默认题数 |

## 示例：加一个中文数学数据集 CMATH

1. 下 `cmath_test.jsonl` 到 `../scripts/datasets/`（每行 `{"question":...,"answer":...}`）
2. 建 `cmath.json`：
```json
{"name":"cmath","group":"math-zh","kind":"numeric","file":"cmath_test.jsonl",
 "question_field":"question","answer_field":"answer",
 "prompt_template":"{question}\n只输出最终数字答案。","max_tokens":4096,"n_default":40}
```
3. 跑：`python3 harness.py --base ... --model ... --datasets cmath`
