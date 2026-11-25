# vLLM gRPC 批量服务并发压测报告（CosyVoice3 / 昇腾 910B3）

## 结论

- 在 **vLLM 0.30 后端 + 单线程预热修复**后，MindIE 风格 gRPC 批量服务可以稳定并发服务：
  各并发档位 **成功率 100%（8/8）**，无 fork 崩溃、无异常、无 `main process disappeared`。
- 并发带来明显吞吐提升（同模型同文本，固定 8 请求，**单进程单卡**）：

  | 并发 | 总 wall | 首包 TTFB | 相对 conc1 的 wall 提速 |
  | ---: | ---: | ---: | ---: |
  | 1 | 91.45 s | 1.45 s | 1.00× |
  | 2 | 71.72 s | 1.56 s | 1.27× |
  | 4 | 43.07 s | 1.65 s | 2.12× |
  | 8 | 19.40 s | 1.72 s | **4.71×** |

- 首包时延 TTFB 随并发略升（1.45s→1.72s），属正常排队；整体仍保持在 ~1.5–1.7s。
- **文本长度补充结论**：TTFB 与句长几乎无关（~1.5s，首包固定 20 token）；总耗时随句长增加
  （conc4 每请求 wall：短 2.15s / 中 3.68s / 长 9.79s）。
- 说明：vLLM 的收益主要来自**并发/批处理**；单请求场景它反而比 HF 后端慢（逐 token 轮询开销）。

---

## 1. 测试环境

| 项 | 值 |
| :--- | :--- |
| 设备 | Ascend 910B3 单卡（`ASCEND_RT_VISIBLE_DEVICES=4`，空闲卡） |
| 模型 | Fun-CosyVoice3-0.5B（CosyVoice3，DiT Flow） |
| LLM 后端 | vLLM 0.30 + vllm_ascend（in-process V1 引擎，`enforce_eager=True`） |
| 服务 | `runtime/python/grpc/server.py`，`--max_conc 8`，`--speaker_info_dir`（zero_shot_by_id，spk1） |
| 关键修复 | gRPC 启动前**单线程预热**（避免 CANN TBE 多线程 fork 崩，见 `docs/bugfixes/04`）+ 除零修复 + vLLM 0.30 兼容 |
| 客户端 | `runtime/python/grpc/batch_test_client.py`，`--stream --mode zero_shot_by_id`，每组 **total_req=8**，分别 concurrency=1/2/4/8 |
| 采样 | 同一服务进程内依次跑 4 个并发档位（避免重复启动/预热的影响） |

> 注解：`batch_test_client` 打印的 “平均RTF / 系统吞吐量” 是基于每请求收到的音频时长估算，
> 与端到端 wall 时间口径不完全一致，故本报告以 **wall 时间 / 每请求 wall 延迟 / TTFB** 为主，
> 客户端吞吐数字仅作参考。

---

## 2. 结果明细

### 2.1 主表

| 并发 | 成功率 | 总 wall | 每请求平均 wall | 首包 TTFB | 客户端 RTF | 客户端吞吐(audio/wall) |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 91.45 s | 11.43 s | 1446 ms | 15.82 | 0.15× |
| 2 | 8/8 | 71.72 s | 8.97 s | 1557 ms | 3.58 | 1.09× |
| 4 | 8/8 | 43.07 s | 5.38 s | 1645 ms | 1.67 | 2.24× |
| 8 | 8/8 | 19.40 s | 2.43 s | 1721 ms | 1.58 | 4.56× |

- 每请求平均 wall = 总 wall / 8。
- conc8 相对 conc1：wall 缩短 **4.71×**（91.45s → 19.40s）。
- 服务端全程 `main process disappeared` 计数 = **0**；无 `ZeroDivision` / `Traceback`。

### 2.2 趋势解读

- **wall 时间**：91.45 → 71.72 → 43.07 → 19.40（并发↑，总时间↓），在 conc8 处吞吐最优。
- **TTFB**：1446 → 1557 → 1645 → 1721 ms，随并发缓升（排队/共享 NPU 计算）。
- **RTF（客户端口径）**：单并发因“逐请求串行 + 每句全流程开销”数值虚高（~15.8），
  并发后回落到 ~1.6；这与“vLLM 单请求慢、批处理快”的预期一致。
- conc2 的 wall 提速偏弱（仅 1.27×），但达到 conc4/8 后线性变好——符合 NPU 上
  LLM 解码 + Flow/HiFi 流水线在批处理时才充分并行/重叠的特征。

### 2.3 文本长度维度（补充）

固定 spk1、同一服务进程（`max_conc=4`），扫描短/中/长三种文本（约 7 / 40 / 120 字）：

| 文本长度 | 并发 | 请求数 | 总 wall | 每请求 wall | 首包 TTFB | 成功率 |
| :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| 短(~7字) | 1 | 2 | 10.85 s | 5.43 s | 1471 ms | 2/2 |
| 中(~40字) | 1 | 2 | 22.98 s | 11.49 s | 1481 ms | 2/2 |
| 长(~120字) | 1 | 2 | 22.95 s | 11.48 s | 1468 ms | 2/2 |
| 短(~7字) | 4 | 4 | 8.61 s | 2.15 s | 1627 ms | 4/4 |
| 中(~40字) | 4 | 4 | 14.71 s | 3.68 s | 1716 ms | 4/4 |
| 长(~120字) | 4 | 4 | 39.16 s | 9.79 s | 1733 ms | 4/4 |

观察：
- **首包时延 TTFB 与文本长度几乎无关**：各长度均在 **~1.47–1.48s（conc1）/ ~1.63–1.73s（conc4）**。
  原因：服务端首包固定按 `first_token_len=20` 个 token 出第一块音频，TTFB 主要由“首块固定
  计算量 + 模型流水线启动”决定，不随整句长度线性增长。
- **总耗时随文本长度增长（并发下更清晰）**：conc4 每请求 wall：短 2.15s → 中 3.68s → 长 9.79s
  （长文本音频更长、LLM 生成更多 token）。conc1 时中/长文本 wall 接近（~23s），是因为请求数少时
  **固定流水线开销占比大**、掩盖了长度差异，故长度趋势应以并发档位为准。
- 全档位 8/8、2/2、4/4 均 **100% 成功**，无 fork 崩溃/异常。
- 同上：客户端打印的“平均RTF/吞吐量”在低请求数下受固定开销影响较大，仅作参考；本表以
  **wall / 每请求 wall / TTFB** 为准。

---

## 3. 复现步骤

```sh
# 0) 打补丁后的源码仓（apply.sh 生成），工作目录：
cd <repo>/runtime/python/grpc
bash update_grpc_file.sh          # 重新生成 pb2（proto 已扩展 zero_shot_by_id 等）

# 1) 启动服务（单卡空闲 NPU，max_conc=8；启动时会先单线程预热再 listening）
export ASCEND_RT_VISIBLE_DEVICES=4
python3 server.py --port 50150 \
  --model_dir /path/Fun-CosyVoice3-0.5B \
  --max_conc 8 \
  --speaker_info_dir /path/Fun-CosyVoice3-0.5B/speaker_info \
  > /tmp/grpc_bench_srv.log 2>&1 &
# 等待日志出现 "server listening"（含预热，约 3 分钟）

# 2) 依次压测不同并发（同一服务进程内）
for C in 1 2 4 8; do
  python3 batch_test_client.py --host 127.0.0.1 --port 50150 --stream \
    --mode zero_shot_by_id --spk_id spk1 --total_req 8 --concurrency $C \
    > /tmp/grpc_bench_c${C}.log 2>&1
  grep -E "成功率|总运行时间|平均首包时延|平均RTF|吞吐量" /tmp/grpc_bench_c${C}.log
done
```

`speaker_info/speaker_info.json` 示例（`prompt_wav` 建议绝对路径）：
```json
{ "spk1": { "prompt_wav": "/abs/path/asset/zero_shot_prompt.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" } }
```

---

## 4. 原始输出（4 个档位）

```
########## concurrency=1 total=8 ##########
成功率: 8/8
总运行时间: 91.45s
平均首包时延: 1446
平均RTF: 15.82
系统吞吐量(audio_sec / wall_sec): 0.15x

########## concurrency=2 total=8 ##########
成功率: 8/8
总运行时间: 71.72s
平均首包时延: 1557
平均RTF: 3.58
系统吞吐量(audio_sec / wall_sec): 1.09x

########## concurrency=4 total=8 ##########
成功率: 8/8
总运行时间: 43.07s
平均首包时延: 1645
平均RTF: 1.67
系统吞吐量(audio_sec / wall_sec): 2.24x

########## concurrency=8 total=8 ##########
成功率: 8/8
总运行时间: 19.40s
平均首包时延: 1721
平均RTF: 1.58
系统吞吐量(audio_sec / wall_sec): 4.56x
```

---

## 5. 注意事项

- 本测试是**单卡单进程**；多卡/多进程（`--num_worker`、`--use_process_pool`）未覆盖。
- GPU/NPU 显存占用：vLLM `gpu_memory_utilization=0.5`（MindIE 默认），建议在空闲卡上测，
  否则会出现 “Free memory < desired gpu_memory_utilization” 启动失败。
- 文本/发音人固定（spk1），未覆盖不同长度文本对 TTFB/RTF 的影响；如需更严谨，可扫文本长度。
- 与 HF 后端的“同口径”对比未做（MindIE 服务强绑定 vLLM）；本报告聚焦 vLLM 的**并发扩展性**。

---

## 6. 用 MindIE 部署方式（run_server_npu.sh）实测

为了对齐 MindIE single_timbre_fm_batch 的**部署方式**，新增了脚本
`runtime/python/grpc/run_server_npu.sh`：沿用其 `run_server.sh` 的环境变量
（`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF`、
`VLLM_ASCEND_ENABLE_MLP_OPTIMIZE`、`VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION`、
`LD_PRELOAD=libjemalloc`），但把作者写死的模型路径/卡号改为参数化（`MODEL_DIR`、
`SPEAKER_INFO_DIR`、`ASCEND_RT_VISIBLE_DEVICES`、`MAX_CONC`、`PORT`、`GRAPH_MODE`）。

用法（6 卡、8 并发）：
```sh
cd runtime/python/grpc
MODEL_DIR=/path/Fun-CosyVoice3-0.5B SPEAKER_INFO_DIR=/path/.../speaker_info \
ASCEND_RT_VISIBLE_DEVICES=6 MAX_CONC=8 PORT=50192 GRAPH_MODE=0 bash run_server_npu.sh
# 客户端：
python3 batch_test_client.py --host 127.0.0.1 --port 50192 --stream \
  --mode zero_shot_by_id --spk_id spk1 --total_req 12 --concurrency 8 --save_task_id 9
```

实测（910B3 单卡 6，CosyVoice3，12 请求 / 并发 8，MindIE 那套环境变量）：

| 配置 | 成功率 | 总 wall | 首包 TTFB | RTF(客户端) | 吞吐(audio/wall) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| MindIE 环境变量 + `GRAPH_MODE=0` | 12/12 | 14.92 s | 799 ms | 0.66 | 9.14× |
| （对照）不加这些 env、`graph_mode=0` | 12/12 | 14.33 s | 762 ms | 0.61 | 9.78× |

`--save_task_id 9` 正常保存音频（`task_9.wav`）。服务端 `main process disappeared` 计数 0。

**关于 MindIE 默认的 `--graph_mode`（GRAPH_MODE=1）——已打通（v3）：**
按 vLLM 导出架构分流后（v3 的 `vllm/config.json` 是 `Qwen2ForCausalLM`）：v3 采用 MindIE 同款
「spawn 多进程 V1 + grpc 启动后 warmup」（无启动前 prewarm、无自定义注册），并配合运行时替代
（`TORCH_NPU_DEVICE_CAPABILITY`、nvidia triton 置非活跃、ATB source）。实测（910B3 单卡，max_conc2）：
- 架构解析为 Qwen2ForCausalLM、prewarm 跳过、grpc 后 warmup 完成 → listening；
- 客户端 **2/2 成功，wall 5.44s，首包 682ms，RTF 0.32，吞吐 4.44×**，保存 task_1.wav，
  `main process disappeared` = 0。
注意：graph_mode 启动明显更慢（vLLM cudagraph capture + 全流水线 warmup，约 5 分钟）。
v2（`CosyVoice2ForCausalLM`）仍用 in-process + 注册（非 graph 路径，早前压测稳定）。
结论：**v3 的 graph_mode 现已可用（与 MindIE 行为一致）**；默认仍建议 `GRAPH_MODE=0`（启动更快），
需要 graph 模式时显式设 `GRAPH_MODE=1`。此前“graph_mode 不兼容”的结论作废。

---

## 7. 成对的“启动服务 / 测试客户端”脚本

- **起服务**：`runtime/python/grpc/run_server_npu.sh`（MindIE 风格环境变量，参数化；见第 6 节）。
- **测试客户端**：`runtime/python/grpc/test_client_npu.sh`（**单请求 + 批量压测二合一**）。

### 与 MindIE 旧脚本的关系（吸收映射）
| 旧脚本 | 作用 | 处理 |
| :--- | :--- | :--- |
| `start.sh` | **单机离线推理**：`taskset -c 192-223 python infer.py`（infer.py: transfer shim + set_compile_mode + `AutoModel(load_vllm=True)`，写死路径） | 不属于“服务/客户端”，该场景已由仓库 `run_npu.py` 覆盖；**不并入本对**（需要绑核可自行 `taskset ... python3 run_npu.py ...`） |
| `run_client.sh` | 只是一组**示例命令清单**（`client.py` 单请求 + `batch_test_client.py` 批量），写死端口 50099 | **吸收为 `test_client_npu.sh`**（参数化，单请求/批量二选一） |
| `client.py` | **单请求**客户端（发送一个请求、打印首包时延/RTF、保存单个 wav），支持 sft/zero_shot/cross_lingual/instruct/zero_shot_by_id | 保留为底层实现，由 `test_client_npu.sh` 的 `BENCH=0` 调用 |
| `batch_test_client.py` | **批量/并发**压测客户端（total_req/concurrency、可选 text_file、save_task_id/save_audio_dir，打印成功率/TTFB/RTF/吞吐） | 保留为底层实现，由 `test_client_npu.sh` 的 `BENCH=1` 调用 |

> 差异一句话：`client.py` 偏“单条请求 + 时延/RTF 观察”；`batch_test_client.py` 偏“多请求并发吞吐”。
> `run_client.sh` 只是把这两者的命令示例堆在一起；我们把它收敛成一个带参数的 `test_client_npu.sh`。

### 用法
```sh
cd runtime/python/grpc

# 批量压测（默认 BENCH=1）：zero_shot_by_id，12 请求 / 8 并发，保存第 9 号音频
MODE=zero_shot_by_id SPK_ID=spk1 TOTAL_REQ=12 CONCURRENCY=8 SAVE_TASK_ID=9 \
  HOST=127.0.0.1 PORT=50000 bash test_client_npu.sh

# 单请求（BENCH=0）：保存一个 wav
BENCH=0 MODE=zero_shot_by_id SPK_ID=spk1 TTS_WAV=./demo.wav \
  HOST=127.0.0.1 PORT=50000 bash test_client_npu.sh
```
（对应地起服务用：`MODEL_DIR=... SPEAKER_INFO_DIR=... ASCEND_RT_VISIBLE_DEVICES=6 MAX_CONC=8 PORT=50000 GRAPH_MODE=0 bash run_server_npu.sh`）

### 校验（910B3 单卡 6，GRAPH_MODE=0）
- 批量（2 请求 / 并发 2，save_task_id=1）：2/2 成功，wall 6.32s，首包 703ms，吞吐 3.82×，保存 `task_1.wav`。
- 单请求（BENCH=0）：RTF 0.40，成功保存 wav；服务端 `main process disappeared` 计数 0。

### 6.2 补充回归（arch 分流后）

910B3 单卡、`run_server_npu.sh`：

| 场景 | 结果 |
| :--- | :--- |
| v3 graph（`GRAPH_MODE=1`，max_conc=8，12 请求/并发 8） | **12/12 成功，wall 11.99s，首包 778ms，RTF 0.47，吞吐 11.91×**，`task_9.wav` 已存，disappeared=0；启动约 6 分钟（engine init+cudagraph capture+8 路 warmup ~3min） |
| v3 非 graph（`GRAPH_MODE=0`，spawn + 启动前 prewarm，max_conc=2） | 2/2 成功，wall 5.60s，首包 702ms，RTF 0.33，吞吐 4.32× |
| v2 gRPC（CosyVoice2 + 本 server） | **不支持**：报 `CosyVoice2.__init__() got an unexpected keyword argument 'speaker_info_dir'`（zero_shot_by_id/speaker 预加载与 graph_mode 仅加在 CosyVoice3 上，属设计限制）；v2 的 vLLM 路径请用 `run_npu.py --vllm` |
