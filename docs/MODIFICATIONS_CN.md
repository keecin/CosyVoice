# CosyVoice 昇腾 NPU 适配与优化 —— 修改说明（中文）

本文档汇总本仓库 `main` 分支相对上游的全部修改：**每处改了什么、为什么改**。
基线为上游 `074ca6d`（Fun-CosyVoice3-0.5B-2512 发布版）。目标环境：昇腾 910B3 容器
（`torch 2.10 + torch_npu + vllm_ascend + onnxruntime-cann`，Python 3.12，无 conda）。

> 修改分三层：**① 采用 MindIE single_timbre_fm_batch 的 NPU 优化** → **② 我们修复的通用 bug**
> → **③ 我们自己的优化**（含 vLLM 0.30 / gRPC 服务打通）。后有**工程脚本/文档**。
> 详细 bug 报告见 `docs/bugfixes/`，压测见 `docs/vllm-grpc-concurrency-benchmark.md`。

---

## 一、采用 MindIE 的 NPU 优化（主要是 gRPC 流式优化）

> **来源**：MindIE 的 cosyvoice3 模型仓 `single_timbre_fm_batch` ——
> <https://modelers.cn/models/MindIE/cosyvoice3/tree/main/single_timbre_fm_batch>。
>
> 说明：这批修改**主要是面向 gRPC 流式服务的优化**（降低首包延迟、批量/并发吞吐、流式分块，
> 以及配套的昇腾算子/服务适配），并非单纯的单机离线推理提速。下面的 DiT 融合注意力、istft
> 回退、掩码优化、vLLM NPU 配置、首包 hop/sleep、zero_shot_by_id、gRPC 批处理服务等，
> 都是围绕“**流式 + 批量并发**”这条链路引入的。

| 文件 | 修改内容 | 原因 |
| :--- | :--- | :--- |
| `cosyvoice/flow/DiT/modules.py` | `F.scaled_dot_product_attention` → `torch_npu.npu_fusion_attention`；`nn.Mish` 改为自定义 Mish | 昇腾融合注意力 kernel 更快；规避该环境下 `nn.Mish` 的兼容/融合差异 |
| `cosyvoice/hifigan/generator.py` | NPU 上 `torch.istft` 回退到 CPU 计算（复数张量/窗函数搬 CPU，结果搬回 NPU） | `torch.istft` 在 NPU 上支持差/慢 |
| `cosyvoice/utils/mask.py` | `add_optional_chunk_mask` 加 `@torch.inference_mode()`；把 `.sum().item()` 判断改为向量化 `logical_or` | `.item()` 会强制主机同步、打断图执行；向量化避免原地写，利于 NPU 图模式。**⚠️ 注意：`@torch.inference_mode()` 只适用于推理**——该函数也被训练路径调用（`transformer/encoder.py`、`transformer/upsample_encoder.py`、`flow/decoder.py`、`flow/DiT/dit.py` 的 forward），训练时会因 inference tensor 无法反传而报错。**本分支当前只关注推理，故保留原样**；如需训练，请先删除该装饰器 |
| `cosyvoice/cli/model.py` `load_vllm` | `EngineArgs` 改为 NPU 调优配置：`dtype='float16'`、`gpu_memory_utilization=0.5`、`torchair_graph_config.enabled`、`cudagraph_capture_sizes`+`cudagraph_mode=FULL`、`enable_prefix_caching=True` | 面向昇腾 vLLM 的图模式/显存/前缀缓存，提升 LLM 阶段吞吐 |
| `cosyvoice/cli/model.py` `llm_job` | 去掉 `self.llm_context`（npu stream）上下文，仅保留 autocast | 减少一条 NPU stream 切换开销、简化执行 |
| `cosyvoice/cli/model.py` `tts`（流式） | 轮询 `sleep` 0.1→0.05；首包 `this_token_hop_len=10` | **降低首包延迟（TTFB）**（实测 TTFB 减半以上） |
| `cosyvoice/flow/flow.py` | DiT flow 的 `inference` 增加 `n_timesteps` 参数（默认 10） | 便于按需减少扩散步数提速（质量/速度权衡） |
| `cosyvoice/cli/cosyvoice.py` + `cosyvoice/cli/frontend.py` | 新增 `zero_shot_by_id` 接口、启动时预加载 `speaker_info.json` 中说话人、`add_sft_spk`、`frontend_from_spk2info`；`tokenizer.encode` 加线程锁 | 固定发音人服务无需每次传/解析音频，降低单请求开销；多线程/多请求下 tokenizer 并发安全 |
| `cosyvoice/utils/file_utils.py` | `load_wav` 健壮化（支持 Tensor/ndarray/路径；对 torchaudio 2.9 用 soundfile 读取）；按 torchaudio 版本分支 | 提升兼容性与健壮性 |
| `runtime/python/grpc/*`（server/client/proto、`batch_test_client.py`、`update_grpc_file.sh`） | gRPC 批处理并发服务（`--max_conc`、流式）、`zero_shot_by_id` 协议。原 `run_server.sh` 的 NPU/vLLM 吞吐环境变量（`TASK_QUEUE_ENABLE`、`VLLM_ASCEND_ENABLE_MLP_OPTIMIZE`、jemalloc 等）改为按需手动设置，作者专用样例脚本已移除 | 批量/并发以提升吞吐；固定发音人服务接口 |
| `requirements.txt` | 保持上游原版（含 CUDA 版 torch/onnxruntime-gpu/tensorrt），本分支未修改 | **⚠️ 严禁在本镜像直接 `pip install -r requirements.txt`（会破坏 NPU 栈）；安装缺失依赖请用 `requirements-npu.txt`** |

---

## 二、我们修复的通用 bug（详见 `docs/bugfixes/`）

| 编号 | 文件 | 修改内容 | 原因 / 现象 |
| :--- | :--- | :--- | :--- |
| 01 | `cosyvoice/llm/llm.py` | `forward_one_step` 改为按 **past+current 总长度**构造 `attention_mask`；新增 `_cache_len()` 兼容 legacy tuple cache 与 `DynamicCache`；`inference_bistream` 去掉 `cache[0][0].size(2)` | **transformers ≥ 5** 下增量解码掩码长度只算了“当前 1 个 token”，与 KV-cache 不对齐 → 生成 40 秒乱读。通用问题 |
| 02 | `cosyvoice/llm/llm.py` | `forward_one_step` 在存在 cache 时把增量输入 `xs` cast 为 cache 的 K/V dtype | 双流（bistream）会喂入纯 **bf16** 文本块，而 KV-cache 是 **fp32** → SDPA 报 dtype 不一致崩溃；通用问题 |
| 03 | `run_npu.py`（调用层） | CosyVoice3 不传 `load_jit`（仅 v2 传） | `CosyVoice3.__init__` 不接受 `load_jit`，统一入口下会 `TypeError` |

---

## 三、我们自己的优化（vLLM 0.30 / gRPC 服务打通）

| 文件 | 修改内容 | 原因 |
| :--- | :--- | :--- |
| `cosyvoice/cli/frontend.py` | speech_tokenizer 的 onnxruntime provider 按可用性自适应 `CANN > CUDA > CPU` | 昇腾镜像没有 `CUDAExecutionProvider`，写死 CUDA 会告警/回退 |
| `cosyvoice/vllm/cosyvoice2.py` | 显式 `from typing import Iterable, Optional, Union`；新增 `embed_input_ids()`；`AutoWeightsLoader(skip_prefixes=...)` → `ignore_unexpected_prefixes=...` | 适配 **vLLM 0.30** 的模型接口/权重加载 API（否则报 `--runner generate` 不支持 / 关键字错误） |
| `cosyvoice/cli/model.py` | `EngineArgs` **不设置** `enforce_eager`（保留 MindIE 的 torchair/AscendCompiler 图编译 + cudagraph 配置） | 早期曾加 `enforce_eager=True` 规避 dynamo 取 `torch.cuda.get_device_capability()==None` 的报错；**审查后验证**：在 MindIE 的 `torchair_graph_config` 配置下走的是 AscendCompiler/AOT 路径，不触发该错误。去掉后图编译生效，vLLM 单请求 RTF 明显改善（v3：2.72→**1.81**） |
| `cosyvoice/cli/model.py` `load_vllm` | 在 `import vllm` 前把 `nvidia` triton backend 置为非活跃 | 本镜像同时上报 nvidia+ascend 两个 triton driver，vllm 会关闭 triton，导致 vllm_ascend 的 triton kernel 崩（`'function' object is not subscriptable`） |
| `cosyvoice/llm/llm.py` `inference_wrapper` | 用 `vllm.inputs.EmbedsInput(type="embeds")` 传 prompt embeddings（替代裸 `{"prompt_embeds": ...}`），并按引擎 dtype 对齐 | vLLM 0.30 的裸 dict 会走已弃用的 renderer/tokenizer 路径，在 `skip_tokenizer_init=True` 下卡住 |
| `cosyvoice/utils/file_utils.py` | 去掉 MindIE 在 `export_cosyvoice2_vllm` 里写死的 `assert False`（torchaudio≠2.9 分支） | 我们在 torchaudio 2.10 下首次导出 vLLM 模型目录时会被该断言阻断 |
| `runtime/python/grpc/server.py` | ① 在 **gRPC 线程启动前**做一次单线程预热 `self.warmup()`（**带超时保护**：预热放线程 + `--prewarm_timeout` 默认 300s；**超时或预热异常则报错退出（fail-fast）**，避免残留的预热线程与 gRPC 服务并发驱动同一 vLLM/Flow 引擎；可用 `--no_prewarm` 跳过仅当算子缓存已预热）；② `schedule_nonfirst_requests_t2w` 空列表时 `return {}`；③ 自包含：`import vllm` 前设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`、triton 修正，并注册 `CosyVoice2ForCausalLM`；④ 修正死代码 `Token2Wave.infer` 中未定义的 `self.inputs` → `self.batches_requests`（该线程默认不启动）；⑤ `--speaker_info_dir` 默认改为 `None` 并回退 `<model_dir>/speaker_info`，`warmup` 不再写死 `spk1`（选可用 speaker），`graph_mode`/无 speaker 时跳过预热；⑥ 按 `<model_dir>/vllm/config.json` 的 architectures 分流引擎：v3（Qwen2ForCausalLM）用 spawn 多进程、不做自定义注册；v2（CosyVoice2ForCausalLM）用 in-process+注册；graph_mode 改为 grpc 启动后 warmup（v3 实测跑通；注意：本 gRPC 服务仅支持 CosyVoice3，CosyVoice2 走 server 会因 speaker_info_dir/graph_mode 参数不支持而报错） | ① **CANN TBE 知识库/算子编译的首次 `multiprocessing` fork 发生在多线程(gRPC)进程里会导致子进程 IPC EOF → “main process disappeared”、Flow 推理挂起**，提前预热让此次 fork 在无 gRPC 线程时完成；预热未完成就继续服务仍会在请求期 fork 崩溃，故采用 fail-fast；② 修除零（会杀死 infer 线程）；③ 免去外部 wrapper；④ 防止有人重新启用该线程时 NameError。（详见 `docs/bugfixes/04`） |

---

## 三·补、代码审查后追加的修复

对全部修改逐项审查后，追加修复了以下问题（MindIE 同步代码中的缺陷）：

| 文件 | 修改内容 | 原因 |
| :--- | :--- | :--- |
| `cosyvoice/cli/cosyvoice.py` | `_preload_all_wave_info` 改用 **join 后的完整路径** 加载：`load_wav(prompt_wav_path, 16000)` | 原代码存在性检查用 `os.path.join(speaker_info_dir, prompt_wav)`，真正加载却传原始相对值 → 相对路径时必然 `FileNotFoundError`、说话人加载失败（此前压测是靠绝对路径绕过的） |
| `cosyvoice/flow/DiT/modules.py` | `AttnProcessor`：仅当 `attn_mask is not None` 时走 `npu_fusion_attention`；否则回退原 `F.scaled_dot_product_attention` | 原实现无条件 `torch.logical_not(attn_mask)`，`mask=None` 时直接 `TypeError`。当前调用链恒传非 None，但属于地雷（外部复用/未来改动即崩），加防护不改变现有行为（实测 v3 出音不变） |

审查中**刻意不修**、仅在此说明的两点：
1. `utils/mask.py` 的 `@torch.inference_mode()` **会破坏训练**（见第一节该行的 ⚠️ 说明）——当前只关注推理，保留原样。
2. `requirements.txt` 保持上游原版（CUDA 依赖，勿在本镜像直接安装）；安装缺失依赖请用 `requirements-npu.txt`。

---

## 四、工程脚本与文档

| 文件 | 内容 | 原因 |
| :--- | :--- | :--- |
| `run_npu.py` | NPU 推理入口：`--stream`、`--bistream`、`--bf16`、`--vllm`；顶部 `torch_npu.contrib.transfer_to_npu` 垫片 | 一行命令验证各能力；把 `torch.cuda.*` 映射到 NPU |
| `requirements-npu.txt` | 仅补缺失依赖（hyperpyyaml/conformer/inflect/pyworld/tensorboard/wetext/kaldifst/gdown/wget/lightning）；`pyarrow==18.1.0`；`wetext==0.2.1` | **不得**用官方 requirements（CUDA torch/onnxruntime-gpu/tensorrt 会破坏 NPU 栈）；pyarrow 需兼容 numpy1.26；wetext 0.2.1 内置 FST 离线 |
| `scripts/setup_npu_env.sh` | 安装缺失依赖 + pyarrow 降级（不建 venv、不动 transformers） | 可复现环境准备 |
| `docs/bugfixes/01~04` | 每个 bug 的详细报告（现象/根因/调用堆栈/修复/验证） | 便于理解与 review |
| `docs/vllm-grpc-concurrency-benchmark.md` | vLLM gRPC 并发/文本长度压测报告 | 量化性能 |

---

## 五、环境/依赖要点（部署时，非源码改动）

- **不要** `pip install -r requirements.txt`（上游锁 CUDA 版 torch/onnxruntime-gpu/tensorrt-cu12）；
  用 `requirements-npu.txt` 只补缺失依赖。
- `transformers` 用系统自带 **5.14.1** 即可（靠第二部分的 KV-cache 修复），**无需降级/隔离环境**。
- `pyarrow` 必须用兼容 numpy1.26 的版本（`18.1.0`），否则 `pyarrow requires NumPy 2.0 or newer`。
- 文本正则前端用 `wetext==0.2.1`（FST 内置、离线）；`ttsfrd` 仅 cp310，Python3.12 不可用。
- 非 vLLM 路径：`run_npu.py` 顶部 `import torch_npu.contrib.transfer_to_npu` 将 `torch.cuda.*` 映射到 NPU。

---

## 六、验证结果摘要（910B3）

| 能力 | 结果 |
| :--- | :--- |
| CosyVoice2 / CosyVoice3 普通推理 | 正常（v2 ~11.8s、v3 ~10.7s） |
| 流式 `--stream` / 双流 `--bistream --stream` | 正常（TTFB 大幅下降；分块数 3→4） |
| `--bf16` | 可跑、音频有效，但本机**不提速**（autocast 转换开销；权重多为 fp32） |
| `--vllm`（单请求） | 已打通：v2 13.16s、v3 11.16s 正常出音。去掉 `enforce_eager` 后（图编译生效）v3 RTF 2.72→**1.81** |
| vLLM gRPC 批量服务 | 已打通（需预热）：各并发 100% 成功；8 请求 wall：conc1 91.4s → conc8 **19.4s（~4.7×）**；TTFB ~1.5–1.7s，且**与文本长度基本无关**（首包固定 20 token） |

---

## 七、已知限制

- 压测为**单卡单进程**；多卡/多进程（`--num_worker`/`--use_process_pool`）未覆盖。
- vLLM 的收益主要在**并发/批处理**；单请求场景反而比 HF 后端慢（逐 token 轮询 + 冷启动）。
- bf16 仅开启 autocast，收益有限；若要更快需权重预转 bf16 / 算子融合 / NPU 版 vLLM 调优。
- gRPC 批量服务的预热会显著增加启动时间（约 40–60s，属预期；超时上限由 `--prewarm_timeout` 控制，默认 300s）。
- `runtime/python/grpc` 的 pb2 需用 `bash update_grpc_file.sh` 重新生成（proto 已扩展）。

---

## 八、快速开始

```sh
# 环境（详见 scripts/setup_npu_env.sh）
bash scripts/setup_npu_env.sh

# 下载模型（modelscope）
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
PY

# 推理
python3 run_npu.py pretrained_models/CosyVoice2-0.5B outputs/v2.wav
python3 run_npu.py pretrained_models/Fun-CosyVoice3-0.5B outputs/v3.wav
python3 run_npu.py pretrained_models/CosyVoice2-0.5B outputs/v2_bs.wav --bistream --stream
python3 run_npu.py pretrained_models/Fun-CosyVoice3-0.5B outputs/v3_vllm.wav --vllm

# gRPC 批量服务（可选）
cd runtime/python/grpc && bash update_grpc_file.sh
python3 server.py --port 50099 --model_dir <v3模型目录> --max_conc 8 --speaker_info_dir <speaker_info目录>
```
