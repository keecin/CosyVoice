# 离线（非流式）服务化：一请求一完整 WAV

加载一次模型，`POST /synthesize` 一次性返回**完整 WAV 文件**（24kHz PCM16，单响应、无 SSE/chunk）。
按 `MODEL_DIR` 下的 `cosyvoice3.yaml`/`cosyvoice2.yaml` 自动分流，v2/v3 各用各的官方 API
（v2 不经过 v3 的 speaker_info/graph_mode/zero_shot_by_id）。

## 启动

启动全部用命令行参数（固定运行时设置已写死，无需环境变量）：

```sh
bash offline/service/run_service.sh --help

# v3（默认走 vLLM；--no_vllm 关闭）：按类型自动选目录
bash offline/service/run_service.sh --model-type v3 --device 0 --port 9500

# v2（官方 API；如需 vLLM 加 --vllm）：按类型自动选目录
bash offline/service/run_service.sh --model-type v2 --device 1 --port 9501

# 或显式指定目录（二选一，不能与 --model-type 同时给）
bash offline/service/run_service.sh \
  --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B --device 0 --port 9500
```

| 参数 | 说明 |
| :--- | :--- |
| `--model-type v2\|v3` | 按类型自动选目录（v2→`CosyVoice2-0.5B`，v3→`Fun-CosyVoice3-0.5B`）；与 `--model-dir` 互斥 |
| `--model-dir DIR` | 显式模型目录（类型由 yaml 判定）；与 `--model-type` 互斥；都不给时默认 v3 目录 |
| `--device N` | **单个** NPU 卡号（默认 0）；多卡请起多个实例（每卡一个、不同 `--port`） |
| `--port PORT` | HTTP 端口（默认 9500） |
| `--speaker-info-dir DIR` | v3 `zero_shot_by_id` 的 speaker 目录（默认 `<model-dir>/speaker_info`） |
| `--host HOST` | 绑定地址（默认 0.0.0.0） |
| `--graph-mode` / `--vllm` / `--no-vllm` | 透传（v3 图 / 强制开/关 vLLM） |

> **模型选择**：`--model-type` 与 `--model-dir` **二选一**，同时给会直接报错。给 `--model-type v2|v3`
> 时自动使用 `../cosyvoice_models/{CosyVoice2-0.5B,Fun-CosyVoice3-0.5B}`，目录不存在会报错退出；
> 都不给时默认 v3 目录。类型最终由目录里的 `cosyvoice2/3.yaml` 判定（都找不到则报错）。

模型加载在开始监听前完成（`/health` 在加载后才可用；客户端需等待/重试）。
运行时固定值（脚本内写死）：`COSY_REPO=/opt/CosyVoice`、`TORCH_NPU_DEVICE_CAPABILITY=8.0`、
`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`TASK_QUEUE_ENABLE/CPU_AFFINITY_CONF`、
`VLLM_ASCEND_ENABLE_MLP_OPTIMIZE/TOPK_TOPP`、jemalloc、ATB/CANN set_env。

## 接口

- `GET /health` → `{status, model_type, model_dir, sample_rate}`
- `POST /synthesize`（multipart/form-data）：
  | 字段 | 说明 |
  | :--- | :--- |
  | `text` | 必填 |
  | `mode` | v3：`zero_shot_by_id`(默认)/`zero_shot`（无 sft/instruct；zero_shot 的 prompt_text 需含 `<|endofprompt|>`）；v2：`zero_shot`(默认)/`sft`/`cross_lingual`/`instruct`/`instruct2` |
  | `spk_id` | v3 by_id / sft/instruct 用 |
  | `prompt_text` / `prompt_wav` | `zero_shot`（v2/v3）、`cross_lingual`（v2）、`instruct2`（v2） |
  | `instruct_text` | `instruct`/`instruct2` |
  | `speed` | 语速 |
  返回：`Content-Type: audio/wav`，响应头含 `X-Model-Type/X-Mode/X-Sample-Rate/X-Audio-Seconds/X-Rtf`。

调用示例：

```sh
# v3 预置音色
python3 offline/service/client.py --url http://127.0.0.1:9500 \
  --mode zero_shot_by_id --spk_id spk1 --text "你好，这是离线下沉服务测试。" --out /tmp/v3.wav

# v2 zero_shot（上传提示音）
python3 offline/service/client.py --url http://127.0.0.1:9501 \
  --mode zero_shot --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /path/to/prompt.wav \
  --text "你好，这是 CosyVoice2 离线服务测试。" --out /tmp/v2.wav
```

## 说明

- **串行**：单模型实例 + 全局锁，同一时刻只处理一个合成请求（模型非线程安全）。
  要并发请起**多实例**（每实例一份模型/HBM）或仅对 LLM 开 vLLM；不要把实例数盲目调大。
- 长文本按 `text_normalize(split=True)` 句块逐个合成后进程内拼接，再一次性返回。
- 与 CLI（`../cli/run_offline.sh`）共用同一套 `/opt/CosyVoice` + 权重目录（默认相对软链 `../cosyvoice_models`，`MODEL_DIR`/`SPEAKER_INFO_DIR` 可覆盖）；
  区别只是本服务常驻、以完整 WAV 响应。
- v3 `sft` 需模型带 `spk2info.pt`；v2 无 `zero_shot_by_id`，克隆请用 `zero_shot`。
