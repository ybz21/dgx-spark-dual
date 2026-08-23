# legacy — 旧版 SGLang 部署脚本

**已废弃，别用。** 保留只为翻历史。

这是仓库早期基于 **SGLang** 的双节点部署方案，后来被根目录的
[`../quick-start.sh`](../quick-start.sh)（vLLM + Ray）整体取代。

| 文件 | 干什么的 |
|---|---|
| `deploy.sh` | 旧版部署主入口 |
| `scripts/common.sh` | 公共函数 |
| `scripts/setup-network.sh` | ConnectX-7 光口 IP 配置 |
| `scripts/test-connection.sh` | 跨节点连通性测试 |

## 还有参考价值的部分

`setup-network.sh` 和 `test-connection.sh` 里的**光口/RoCE 排查思路**仍然适用——
两台 Spark 之间的光缆连通性问题和引擎无关。`quick-start.sh` 已经把光口自动检测
（遍历 ConnectX-7 四个口找 `carrier=1`）内建了，所以正常情况下用不到这两个脚本。

其余部分（SGLang 启动参数、进程编排）和现在的 vLLM 方案完全对不上，不要照抄。
