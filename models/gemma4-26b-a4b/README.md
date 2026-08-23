# Gemma 4 26B-A4B

`google/gemma-4-26B-A4B-it`，MoE 26B 总参 / 4B 激活，BF16，单节点 TP=1。

只有一份 `quick-start.sh` 预设，**没有实测记录**——这是配置留档，不是验证过的部署方案。

| 项 | 值 |
|---|---|
| 镜像 | `vllm-spark:gemma4` |
| 量化 | 无（BF16） |
| TP | 1（单节点） |
| 预设 | [`presets/gemma4-26b-a4b.env`](presets/gemma4-26b-a4b.env) |

## 用法

预设文件是给根目录 [`../../quick-start.sh`](../../quick-start.sh) 的运行时（[`../../runtime/`](../../runtime)）用的参数参考：
`quick-start.sh` 本身不会自动加载它，需要手工把里面的值填进 `runtime/.env` 或命令行参数。
