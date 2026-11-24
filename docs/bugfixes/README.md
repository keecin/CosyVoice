# CosyVoice Bug 修复报告（索引）

本目录收录本项目在部署/调试过程中发现并修复的 **CosyVoice 代码缺陷** 的详细报告。
每份报告都面向“没读过 CosyVoice 代码”的读者，包含概念科普、现象、根因、修复、验证。

> 说明：只收录“代码 bug 修复”。昇腾 NPU 环境适配（torch_npu 兼容垫片、onnxruntime
> provider、依赖安装、pyarrow/wetext 等）不在此列，见仓库根 `README.md` 的
> “Ascend NPU 部署”章节。

| 编号 | 标题 | 严重程度 | 一句话 |
| :--- | :--- | :--- | :--- |
| 01 | [transformers 5.x 下语音被拖长、乱读](01-transformers5-kvcache-mask.md) | 高 | 增量解码时注意力掩码长度没算历史缓存，新版 transformers 下生成 40 秒乱读 |
| 02 | [双流合成 dtype 不一致](02-bistream-dtype-mismatch.md) | 中 | 纯 bf16 文本块撞上 fp32 KV-cache，SDPA 报 dtype 不一致 |
| 03 | [CosyVoice3 不支持 load_jit 参数](03-cosyvoice3-load-jit-arg.md) | 低 | AutoModel 统一入口下，CosyVoice3 构造函数多传 load_jit 会报 TypeError |
| 04 | [gRPC 服务崩溃：CANN TBE 多线程 fork 失败](04-cann-tbe-fork-multithread.md) | 高（昇腾服务侧） | 首次算子编译在多线程里 fork，TBE worker 队列 EOF → “main process disappeared”，Flow 推理挂起；用单线程预热修复 |

## 涉及的代码文件

- `cosyvoice/llm/llm.py`
  - `Qwen2Encoder._cache_len()` / `forward_one_step()`（报告 01、02）
  - `Qwen2LM.inference_wrapper()` / `inference_bistream()`（报告 01、02）
- `cosyvoice/cli/cosyvoice.py`（报告 03，`CosyVoice2` / `CosyVoice3` 构造参数）
- `run_npu.py`（报告 03，调用层兼容）
- `runtime/python/grpc/server.py`（报告 04，预热位置、`schedule_nonfirst_requests_t2w` 除零）
  + CANN TBE `repository_manager`（报告 04，只读排查）

## 修复的通用性

报告 01、02 都是**通用代码问题**（与 GPU/NPU/CPU 无关），修复对 `transformers < 5` 与 `>= 5`
以及不同后端都兼容。报告 03 是纯 Python 接口不一致。报告 04 是**昇腾服务侧**问题
（CANN TBE 在多线程里 fork），修复方式（gRPC 启动前单线程预热）对昇腾批处理服务通用。
