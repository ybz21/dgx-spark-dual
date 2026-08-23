# Qwen3.5-397B-A17B INT4

`Intel/Qwen3.5-397B-A17B-int4-AutoRound`，MoE 397B 总参 / 17B 激活，AutoRound INT4，双节点 TP=2。

本仓库里最大的模型。预设注释标着 *currently in production on spark01 (head) + spark02 (worker)*，
但**没有实测报告**——只有配置留档。

| 项 | 值 |
|---|---|
| 镜像 | `vllm-spark:v018-ngc2603-staging` |
| 量化 | INT4 AutoRound (Intel) |
| TP | 2（双节点） |
| 预设 | [`presets/qwen3.5-397b-int4.env`](presets/qwen3.5-397b-int4.env) |

## 用法

预设文件是给根目录 [`../quick-start.sh`](../quick-start.sh) 的运行时（[`../runtime/`](../runtime)）用的参数参考：
`quick-start.sh` 本身不会自动加载它，需要手工把里面的值填进 `runtime/.env` 或命令行参数。
