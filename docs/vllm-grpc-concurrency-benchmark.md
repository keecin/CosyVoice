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
