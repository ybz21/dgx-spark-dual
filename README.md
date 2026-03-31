# DGX Spark 多节点推理部署

通过光缆 (ConnectX-7) 连接多台 NVIDIA DGX Spark，使用 SGLang Tensor Parallelism 实现跨节点大模型推理。

## 硬件要求

- 2+ 台 NVIDIA DGX Spark (GB10, 128GB 显存)
- 光缆连接 ConnectX-7 网口 (每台 4 口, 最高 200Gbps/口)
- 各节点可通过管理网 SSH 免密互通

## 快速开始

### 1. 配置

```bash
cp .env.example .env
vim .env   # 修改 NODE_LIST、MODEL_PATH 等
```

`NODE_LIST` 格式：`管理网IP,高速网IP,SSH用户名,主机名,光口接口名`

```bash
NODE_LIST=(
    "192.168.1.100,10.10.10.1,ai,spark-node1,enp1s0f0np0"   # HEAD
    "192.168.1.101,10.10.10.2,ai,spark-node2,enp1s0f1np1"   # WORKER
)
```

每台机器光缆可能插在不同的口，查看哪个口有光缆：

```bash
for dev in /sys/class/net/enp*/carrier /sys/class/net/enP*/carrier; do
  [ -f "$dev" ] && echo "$(dirname $dev | xargs basename): $(cat $dev)"
done
# carrier=1 表示该口有光缆
```

### 2. 配置高速网络

插上光缆后，一键配置所有节点的 ConnectX-7 网卡 IP 和 MTU：

```bash
bash deploy.sh network
# 输入 sudo 密码，自动配置所有节点
```

### 3. 测试连接

```bash
# 在节点 1 上运行
bash test-connection.sh 1 2
```

预期结果：
- Ping 延迟 < 1ms
- Jumbo Frame (MTU 9000) 通过
- `iperf3` TCP ~40 Gbps
- `ib_write_bw` RDMA ~100 Gbps

### 4. 启动推理

```bash
bash deploy.sh start
# 等待 2-3 分钟模型加载完成

bash deploy.sh status   # 查看状态
bash deploy.sh test     # 发送测试请求
```

API 默认监听在 HEAD 节点的 30000 端口，兼容 OpenAI 格式：

```bash
curl http://<HEAD节点IP>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/model","messages":[{"role":"user","content":"你好"}]}'
```

## 管理命令

```bash
bash deploy.sh start       # 启动多节点推理
bash deploy.sh stop        # 停止
bash deploy.sh restart     # 重启
bash deploy.sh status      # 查看状态 + 健康检查
bash deploy.sh logs [N]    # 查看节点 N 的日志 (默认 1)
bash deploy.sh test        # 发送测试请求
bash deploy.sh network     # 配置高速网络 (需 sudo)
bash deploy.sh sync        # 同步配置到所有节点
```

## .env 配置说明

### 节点配置

| 字段 | 说明 | 示例 |
|------|------|------|
| 管理网IP | SSH 连接用的 IP | `192.168.1.100` |
| 高速网IP | ConnectX-7 光缆互联 IP (自行分配) | `10.10.10.1` |
| SSH用户名 | 免密 SSH 的用户 | `ai` |
| 主机名 | 可选，用于日志显示 | `spark-node1` |
| 光口接口名 | 该节点光缆实际插入的接口 | `enp1s0f0np0` |

- 第一个节点自动作为 HEAD (rank 0)
- 添加行即可增加 WORKER 节点
- `TP_SIZE` 需等于总节点数
- RDMA HCA 名从接口名自动推导 (如 `enp1s0f1np1` -> `rocep1s0f1`)

### 换机器

修改 `NODE_LIST` 中的 IP 和接口名，然后：

```bash
bash deploy.sh network     # 配置新机器网络
bash deploy.sh restart     # 重启服务
```

### 换模型

```bash
MODEL_PATH=/home/ai/models/YOUR_MODEL
SGLANG_IMAGE=nvcr.io/nvidia/sglang:26.03-py3
TP_SIZE=2
QUANTIZATION=                # 留空=bf16, 或 modelopt_fp4
EXTRA_ARGS="--trust-remote-code"
```

### 自定义 Patch (可选)

支持挂载自定义脚本到容器中，用于 patch SGLang 行为。例如自定义 reasoning detector：

```bash
CUSTOM_SCRIPTS_DIR=/home/ai/scripts
CUSTOM_REASONING_DETECTOR=my_reasoning_detector.py
CUSTOM_ENTRYPOINT=my_entrypoint.sh
```

entrypoint 脚本会在 SGLang 启动前执行，可用于替换/patch SGLang 内部文件。

### 网络参数

```bash
FAST_IFACE_DEFAULT=enp1s0f0np0   # 默认 ConnectX-7 接口名
FAST_MTU=9000                    # Jumbo Frame
```

查看可用接口: `ip -br link show`
查看 RDMA 设备: `ibv_devinfo`

## 文件结构

```
dgx-spark-dual/
├── .env.example          # 配置模板 (复制为 .env 使用)
├── .env                  # 实际配置 (git忽略)
├── .gitignore
├── deploy.sh             # 管理脚本 (start/stop/status/...)
├── setup-network.sh      # 网络配置 (由 deploy.sh network 调用)
├── test-connection.sh    # 连接测试
└── README.md
```

`docker-compose.nodeN.yml` 在 `deploy.sh start` 时根据 `.env` 自动生成，不纳入版本管理。

## 注意事项

- 所有节点的模型路径 (`MODEL_PATH`) 必须一致，模型文件需提前拷贝到每台机器
- 节点间需免密 SSH，可用 `ssh-copy-id` 配置
- 每台机器光缆可能插在不同的 ConnectX-7 口，`NODE_LIST` 第 5 个字段用于指定
- 单机能装下的模型 (如 35B)，多节点不会更快 (通信开销)；多节点的价值在于跑 128GB 以上的大模型
- NCCL 通信自动选择最优传输方式 (Socket / RoCE)

## 前置准备

### SSH 免密

```bash
# 在控制机上，对每台 DGX Spark 执行
ssh-copy-id user@node1-mgmt-ip
ssh-copy-id user@node2-mgmt-ip

# DGX Spark 之间也需要互通
ssh user@node1 "ssh-copy-id user@node2"
ssh user@node2 "ssh-copy-id user@node1"
```

### 光缆连接

DGX Spark 有 4 个 ConnectX-7 网口 (2 组各 2 口)。
用 QSFP112 光缆或 DAC 铜缆连接两台机器的网口。
不要求两台插同一编号的口，脚本会根据 `NODE_LIST` 中的配置自动适配。
