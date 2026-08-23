# 配套服务

不是模型，是围着模型转的东西：前端、代理层、以及被多个模型 compose 共享的 vLLM 插件。

| 服务 | 文件 | 端口 | 上游 |
|---|---|---|---|
| open-webui | `docker-compose-chat-open-webui.yaml` | 30030 → 内部 8080 | `http://localhost:30000/v1` |
| cc-proxy（Claude Code 代理） | `docker-compose-proxy-cc.yaml` + `services/cc_proxy.py` | 30021 | `http://localhost:30000/v1` |
| sysfix-proxy（system message 修正） | `docker-compose-proxy-sysfix.yaml` + `services/sysfix_proxy.py` | 30020 | `http://localhost:30000` |

## 共享 vLLM 插件

`services/qwen3_nothink_reasoning_parser_vllm.py` —— `qwen3_nothink` reasoning parser，
被 **两个模型的 compose 同时引用**：

- [`../qwen3.5-122b-a10b/deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml`](../qwen3.5-122b-a10b/deploy/docker-compose-model-qwen35-122b-a10b-int4-vllm.yaml)
- [`../qwen3.5-35b-a3b/deploy/docker-compose-model-qwen35-35b-nvfp4-vllm.yaml`](../qwen3.5-35b-a3b/deploy/docker-compose-model-qwen35-35b-nvfp4-vllm.yaml)

改它会同时影响这两个模型。

## 为什么这里有个嵌套的 `services/`

**是故意的。** 所有 compose 都是 copy 到目标机的 `~/lm_scripts/` 下再 `docker compose up` 的，
里面的挂载路径写的是 `./services/xxx.py`，相对的是**目标机上的 `~/lm_scripts/`**，不是仓库目录。
这里保留同名嵌套目录，是为了让仓库布局和目标机布局一一对应，compose 里的路径不用改。

部署时把需要的 compose 和 `services/` 一起放过去：

```bash
scp docker-compose-proxy-cc.yaml ai@<host>:~/lm_scripts/
scp -r services ai@<host>:~/lm_scripts/
ssh ai@<host> 'cd ~/lm_scripts && docker compose -f docker-compose-proxy-cc.yaml up -d'
```
