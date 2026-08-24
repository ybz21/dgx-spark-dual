# Qwen3.8-27B · SGLang + DSpark · 部署验证报告

> 目标机：`ai@192.168.130.48`（box010，DGX Spark / GB10 / aarch64 / 119 GiB）
> 日期：2026-08-24 · 状态：**✅ 部署成功并实测**

## 配置

| 项 | 值 |
|---|---|
| 引擎 | SGLang（`lmsysorg/sglang@sha256:febfb971…`，arm64，DSpark 补丁版）|
| 主模型 | `RadixArk/Qwen3.8-27B-NVFP4`（**modelopt** NVFP4，3 shards，~21 GiB）|
| 草稿模型 | `RadixArk/Qwen3.8-27B-DSpark`（1.36B，~2.6 GiB）|
| 投机解码 | **DSPARK**（`--speculative-algorithm DSPARK --speculative-dspark-block-size 7`）|
| 其它 | `--mem-fraction-static 0.50 --attention-backend flashinfer --enable-torch-compile --torch-compile-max-bs 4 --num-continuous-decode-steps 2` |
| 端口 | :9001（box010 上 :8000/:8010 被生产服务占用）|

## 实测（GB10，单流，greedy）

| 任务 | 输出 tok | 墙钟 | Decode |
|---|---|---|---|
| 中文散文 | 256 | 16.8s | **15.3 tok/s** |
| 代码（快排+注释）| 256 | 8.1s | **31.4 tok/s** |
| 数学（求和推导）| 256 | 7.6s | **33.5 tok/s** |

- 冷启约 11 分钟（`scheduler_e2e≈651s`：torch.compile + DSpark 目标校验/草稿 cuda graph 捕获）。
- DSpark 在结构化内容（代码/数学）上快，33.5 tok/s 接近论坛参考（34–38，峰值 46.7）；自由散文接受率低 ~15 tok/s。
- 注：本模型默认开思考模式（`<think>…`），单请求含推理 token。

## 与 vLLM+MTP 方案对比（同机同任务）

| 任务 | vLLM+MTP | **SGLang+DSpark** |
|---|---|---|
| 散文 | 17.6 | 15.3 |
| 代码 | 27.0 | **31.4** |
| 数学 | 30.9 | **33.5** |

DSpark 在代码/数学更快；散文略慢。两者量级接近，结构化任务 DSpark 占优。

## 关键坑

1. **DSPARK 算法只在特定 SGLang 补丁镜像里**：本仓库 LAN 的 `sglang-dev-cu13-accel` 只有 EAGLE/EAGLE3/NEXTN/NGRAM，**没有 DSPARK**；且它加载 RadixArk modelopt 会报 fp8 block_n 尺寸错。必须用 `lmsysorg/sglang@sha256:febfb971…`（docker hub）。
2. **vLLM 跑不了 Qwen3.8 的 DSpark**：vLLM 0.27.1 的 DSpark 实现是 DeepSeek-V4 专用（`deepseek_v4/nvidia/dspark.py`，要 `hc_mult`），加载 Qwen3 DSpark 草稿报 `Qwen3Config has no attribute hc_mult`。
3. **DSpark 主模型要用 RadixArk（modelopt）**，不是 unsloth（compressed-tensors，SGLang 会因 actorder 报错）。
4. **镜像获取**：docker hub 可达但 box010 外网慢——开发机(代理)拉 arm64 → `docker save --platform linux/arm64`(17GB tar) → 局域网传 → `docker load` → `docker tag`（按 digest load 后是 <none>，需手动打 tag）。
