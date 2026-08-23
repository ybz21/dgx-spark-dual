# BGE Embedding

`bge-large-zh-v1.5` 中文 embedding 服务，独立于各个大模型运行（体量小，不抢统一内存）。

| 项 | 值 |
|---|---|
| 权重 | `~/models/bge-large-zh-v1.5` |
| 镜像 | `embedding-server:latest` |
| 端口 | 30010 |
| 网络 | `network_mode: host` |

> ⚠️ **未验证**：`embedding-server:latest` 这个镜像当时在目标机上并不存在，需要先 pull 或自行 build，
> 且镜像要支持 `PORT` 环境变量。这份 compose 是写好但没跑通的状态。

## 部署

```bash
scp deploy/docker-compose-embedding-bge.yaml ai@<host>:~/lm_scripts/
ssh ai@<host> 'cd ~/lm_scripts && docker compose -f docker-compose-embedding-bge.yaml up -d'
```
