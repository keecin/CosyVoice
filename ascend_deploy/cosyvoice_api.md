# CosyVoice2/3 Ascend 服务接口使用手册

本文面向**调用服务的客户/业务侧**，说明发布包中三种服务的接口、参数、返回格式和调用示例：

| 服务 | 模型 | 协议 | 默认端口 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| 离线服务 `/synthesize` | v2 / v3 | HTTP multipart | 9500/9501 | 一次请求返回完整 WAV |
| v3 流式 gRPC | v3 | gRPC server-stream | 50000 | 低延迟流式、分段返回音频 |
| v2 流式 FastAPI | v2 | HTTP chunked | 50000 | CosyVoice2 官方流式接口 |

> 端口只是示例。同一台机器部署多个服务时请使用不同端口。
> 服务默认无鉴权，生产环境请放在网关/内网访问控制之后。

---

## 1. 通用约定

- 文本编码：UTF-8。
- 服务端输出采样率：**24000 Hz**、单声道、PCM16。
- 参考音频建议：3–10 秒、单人、清晰、无明显噪声；上传后服务会重采样到 16kHz 作为 prompt。
- 离线服务返回**完整 WAV**（含 WAV 头）。
- 两个流式服务返回**裸 PCM16LE 分块**（无 WAV 头）；调用方需要按 24kHz/单声道拼接后保存或加 WAV 头。
- 加载模型期间 `GET /health` 仍返回 200 + `status=loading`；此时调用离线服务的 `/synthesize` 会返回 503，流式服务则可能连接失败。请等待 `/health` 为 `ok` 或启动日志就绪。

### 1.1 模型与模式差异（重要）

| 能力 | v2 | v3 |
| :--- | :--- | :--- |
| 预置音色 `zero_shot_by_id` | ❌ 不支持 | ✅ 主用（需 `speaker_info`） |
| `zero_shot`（参考音频克隆） | ✅ 需 `prompt_wav` + 逐字转写 `prompt_text` | 离线服务 ✅；v3 原生 gRPC 未实现（gRPC 仅支持 `zero_shot_by_id`/`sft`）；`prompt_text` 必须含 `<|endofprompt|>` |
| `cross_lingual` | ✅ 只传 `prompt_wav` | 本发布服务未开放（仅官方 API 有能力） |
| `instruct` / `instruct2` | ✅ 需要 `spk2info.pt` / 参考音频 | v3 离线/流式 gRPC 未开放 |
| `sft` | ✅ 需要 `spk2info.pt` | v3 gRPC 支持 `sft_request`，但通常需要模型带 `spk2info.pt`；普通下载模型不可用 |

v3 的 `prompt_text` 约定：

```text
You are a helpful assistant.<|endofprompt|>参考音频的逐字转写
```

缺少 `<|endofprompt|>` 会被服务直接拒绝（400）。

---

## 2. 离线服务接口（v2 + v3）

离线服务是常驻 HTTP 服务，模型加载一次；每个请求返回一个完整 WAV（非流式、无 SSE）。
单实例内部串行处理（全局锁），要并发请启动多个实例。

### 2.1 `GET /health`

用于检查服务是否加载完成。

**响应字段**

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `status` | string | `loading` / `ok` |
| `model_type` | string | `v2` / `v3` |
| `model_dir` | string | 当前模型目录 |
| `sample_rate` | int | 输出采样率（24000） |

示例：

```sh
curl -sS http://127.0.0.1:9500/health
```

```json
{"status":"ok","model_type":"v3","model_dir":"/data/cosyvoice_models/Fun-CosyVoice3-0.5B","sample_rate":24000}
```

### 2.2 `POST /synthesize`

请求类型：`multipart/form-data`。一次性返回完整 WAV。

#### 请求参数

| 参数 | 类型 | 必填 | 适用 | 说明 |
| :--- | :--- | :---: | :--- | :--- |
| `text` | string | ✅ | v2/v3 | 待合成文本 |
| `mode` | string | 否 | v2/v3 | 不传时 v3 默认 `zero_shot_by_id`，v2 默认 `zero_shot` |
| `spk_id` | string | 按模式 | v2/v3 | v3 `zero_shot_by_id`；v2 `sft`/`instruct`；默认 `spk1` |
| `prompt_text` | string | 按模式 | v2/v3 | v2 `zero_shot` 的逐字转写；v3 `zero_shot` 必须含 `<|endofprompt|>` |
| `prompt_wav` | file | 按模式 | v2/v3 | 参考音频；v2 `zero_shot`/`cross_lingual`/`instruct2`，v3 `zero_shot` |
| `instruct_text` | string | 按模式 | v2 | v2 `instruct`/`instruct2` 的指令 |
| `speed` | float | 否 | v2/v3 | 语速倍率，默认 `1.0` |

#### 模式与必填参数

| 模型 | `mode` | 必填参数 | 说明 |
| :--- | :--- | :--- | :--- |
| v3 | `zero_shot_by_id`（默认） | `text`, `spk_id` | 使用 `speaker_info` 预置音色，适合固定音色业务 |
| v3 | `zero_shot` | `text`, `prompt_wav`, `prompt_text` | `prompt_text` 必须为 `"You are a helpful assistant.<|endofprompt|>" + 逐字转写` |
| v3 | `sft`/`instruct`/`instruct2`/`cross_lingual` | — | 本离线服务未开放，返回 400 |
| v2 | `zero_shot`（默认） | `text`, `prompt_wav`, `prompt_text` | `prompt_text` 是参考音频的逐字转写（无 endofprompt 要求） |
| v2 | `cross_lingual` | `text`, `prompt_wav` | 只借音色，不要求转写 |
| v2 | `instruct` | `text`, `spk_id`, `instruct_text` | 需要模型带 `spk2info.pt` |
| v2 | `instruct2` | `text`, `prompt_wav`, `instruct_text` | 参考音频 + 指令 |
| v2 | `sft` | `text`, `spk_id` | 需要模型带 `spk2info.pt` |
| v2 | `zero_shot_by_id` | — | v2 不支持，返回 400 |

#### 成功响应

HTTP 200，`Content-Type: audio/wav`，完整 WAV（24kHz/单声道/PCM16）。

| 响应头 | 示例 | 说明 |
| :--- | :--- | :--- |
| `X-Model-Type` | `v3` | 实际模型类型 |
| `X-Mode` | `zero_shot_by_id` | 实际使用的模式 |
| `X-Sample-Rate` | `24000` | 采样率 |
| `X-Audio-Seconds` | `3.842` | 音频时长（秒） |
| `X-Rtf` | `0.512` | 服务端耗时 / 音频时长 |

#### 错误响应

| HTTP | 场景 | 响应 |
| :---: | :--- | :--- |
| 400 | 参数缺失/模式不支持/spk 不存在/缺 endofprompt | `{"detail":"..."}` |
| 422 | 缺少必填 form 字段（FastAPI 校验） | JSON 校验错误 |
| 503 | 模型仍在加载 | `{"detail":"model not ready"}` |
| 500 | 推理异常 | `{"detail":"..."}` |

### 2.3 调用示例

#### v3 `zero_shot_by_id`（最常用）

```sh
curl -sS -D /tmp/v3.headers -o /tmp/v3.wav \
  -F text='你好，这是离线 v3 测试。' \
  -F mode=zero_shot_by_id \
  -F spk_id=spk1 \
  http://127.0.0.1:9500/synthesize
```

#### v3 `zero_shot`（上传参考音频）

```sh
curl -sS -D /tmp/v3zs.headers -o /tmp/v3_zero_shot.wav \
  -F text='你好，这是 v3 zero-shot 测试。' \
  -F mode=zero_shot \
  -F 'prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。' \
  -F prompt_wav=@/data/prompt.wav \
  http://127.0.0.1:9500/synthesize
```

#### v2 `zero_shot`

```sh
curl -sS -D /tmp/v2.headers -o /tmp/v2.wav \
  -F text='你好，这是 CosyVoice2 离线服务测试。' \
  -F mode=zero_shot \
  -F prompt_text='希望你以后能够做的比我还好呦。' \
  -F prompt_wav=@/data/prompt.wav \
  http://127.0.0.1:9501/synthesize
```

#### v2 `cross_lingual`

```sh
curl -sS -o /tmp/v2_cross.wav \
  -F text='Hello, this is a cross-lingual test.' \
  -F mode=cross_lingual \
  -F prompt_wav=@/data/prompt.wav \
  http://127.0.0.1:9501/synthesize
```

#### v2 `instruct2`

```sh
curl -sS -o /tmp/v2_instruct2.wav \
  -F text='你好，这是指令控制测试。' \
  -F mode=instruct2 \
  -F instruct_text='用四川话说这句话。' \
  -F prompt_wav=@/data/prompt.wav \
  http://127.0.0.1:9501/synthesize
```

#### 使用 Python 客户端

发布包自带 `offline/service/client.py`，会自动保存 WAV 并打印服务端 RTF：

```sh
# v3
python3 offline/service/client.py --url http://127.0.0.1:9500 \
  --mode zero_shot_by_id --spk_id spk1 \
  --text "你好，这是离线 v3 测试。" --out /tmp/offline_v3.wav

# v2
python3 offline/service/client.py --url http://127.0.0.1:9501 \
  --mode zero_shot \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /data/prompt.wav \
  --text "你好，这是离线 v2 测试。" --out /tmp/offline_v2.wav
```

客户端参数：`--url`、`--text`、`--mode`、`--spk_id`、`--prompt_text`、`--prompt_wav`、
`--instruct_text`、`--speed`、`--out`、`--timeout`。

---

## 3. v3 流式 gRPC 接口

### 3.1 连接信息

| 项 | 值 |
| :--- | :--- |
| 地址 | `<server>:50000`（示例） |
| Service | `cosyvoice.CosyVoice` |
| RPC | `Inference(Request) returns (stream Response)` |
| 传输 | gRPC 明文（无 TLS；生产建议网关/内网） |

### 3.2 请求消息

`Request` 是 `oneof RequestPayload`。当前 v3 服务实际支持：

| oneof 字段 | 消息 | 字段 | 说明 |
| :--- | :--- | :--- | :--- |
| `zero_shot_by_id_request` | `zeroshotByIdRequest` | `tts_text`=文本，`spk_id`=预置音色，`stream`=true | **主用**；需服务已加载 `speaker_info` |
| `sft_request` | `sftRequest` | `tts_text`，`spk_id`，`stream` | 模型需支持（通常需要 spk2info） |

`proto` 中还定义了 `zero_shot_request`、`cross_lingual_request`、`instruct_request`，但当前 v3
gRPC 服务端未实现这些模式；发送会得到 `NotImplementedError`/gRPC 错误。

最小 proto：

```proto
service CosyVoice {
  rpc Inference(Request) returns (stream Response) {}
}

message Request {
  oneof RequestPayload {
    sftRequest sft_request = 1;
    zeroshotRequest zero_shot_request = 2;
    crosslingualRequest cross_lingual_request = 3;
    instructRequest instruct_request = 4;
    zeroshotByIdRequest zero_shot_by_id_request = 5;
  }
}

message zeroshotByIdRequest {
  string tts_text = 1;
  string spk_id = 2;
  bool stream = 3;
}

message Response {
  bytes tts_audio = 1;
}
```

### 3.3 流式响应格式

- 每个 `Response.tts_audio` 是一段 **PCM16 little-endian、单声道、24000Hz** 原始字节；
- 多个 chunk 按顺序拼接即为完整音频；
- **没有 WAV 头**，也没有额外的完成标记；gRPC stream 正常结束即完成；
- `stream=true` 时服务按 section 分段返回；`stream=false` 时通常一次性返回。

### 3.4 调用示例

#### 统一客户端（推荐）

```sh
cd /data/cosyvoice
python3 streaming/test_streaming.py --model-type v3 --api grpc \
  --url 127.0.0.1:50000 \
  --mode zero_shot_by_id --spk-id spk1 \
  --text "你好，这是 v3 流式测试。" \
  --out /tmp/stream_v3.wav
```

输出：

```
api=grpc model_type=v3 mode=zero_shot_by_id
chunks=2 pcm=215040 sr=24000 audio=4.48s
ttfb=700ms total=1.60s rtf=0.357
out=/tmp/stream_v3.wav
```

参数说明：

| 参数 | 说明 |
| :--- | :--- |
| `--url` | gRPC 地址 `host:port` |
| `--mode` | `zero_shot_by_id` / `sft` |
| `--spk-id` | 预置音色 ID（如 `spk1`） |
| `--text` | 待合成文本 |
| `--out` | 输出 WAV 路径 |
| `--timeout` | 单请求超时秒数 |

#### 直接用 Python stub

如果业务侧自行集成，可按 `pb2` 生成 stub：

```python
import grpc
import cosyvoice_pb2 as pb2
import cosyvoice_pb2_grpc as pb2_grpc

with grpc.insecure_channel("127.0.0.1:50000") as ch:
    stub = pb2_grpc.CosyVoiceStub(ch)
    req = pb2.Request()
    z = pb2.zeroshotByIdRequest()
    z.tts_text = "你好，这是 v3 流式测试。"
    z.spk_id = "spk1"
    z.stream = True
    req.zero_shot_by_id_request.CopyFrom(z)

    chunks = []
    for resp in stub.Inference(req, timeout=120):
        chunks.append(resp.tts_audio)   # PCM16LE 24k
    open("/tmp/raw.pcm", "wb").write(b"".join(chunks))
```

#### 源码树内测试脚本

如果要在镜像 `/opt/CosyVoice/runtime/python/grpc/` 内使用自带测试脚本
（当前镜像文件名为 `test_client_npu.sh` / `batch_test_client.py`；重建源码树中为
`test_client.sh` / `batch_test_client.py`）：

```sh
cd /opt/CosyVoice/runtime/python/grpc

# 单请求：打印首包/RTF 并保存 wav
BENCH=0 MODE=zero_shot_by_id SPK_ID=spk1 \
  TTS_TEXT="你好，这是 v3 流式测试。" \
  TTS_WAV=/tmp/stream_v3_client.wav \
  bash test_client_npu.sh

# 批量/并发
MODE=zero_shot_by_id SPK_ID=spk1 STREAM=1 \
  TOTAL_REQ=12 CONCURRENCY=8 \
  SAVE_TASK_ID=-1 SAVE_DIR=/tmp/grpc_audio \
  bash test_client_npu.sh
```

### 3.5 错误与排查

| 现象 | 原因/处理 |
| :--- | :--- |
| 连接被拒绝 | 服务未启动/端口不对/尚未监听 |
| `NotImplementedError: Mode not implemented!` | 使用了当前服务未实现的模式；v3 请用 `zero_shot_by_id` |
| `speaker 'xxx' not found` | `speaker_info.json` 不含该 `spk_id` |
| 收到音频但无法播放 | `tts_audio` 是裸 PCM16；需按 24000Hz/单声道加 WAV 头 |
| 首个请求很慢 | graph 模式/首次算子编译；看服务端 `server listening` 后再发请求 |

---

## 4. v2 流式 FastAPI 接口

### 4.1 连接信息

| 项 | 值 |
| :--- | :--- |
| 地址 | `http://<server>:50000`（示例） |
| 框架 | 官方 CosyVoice FastAPI（发布包做 Ascend 适配） |
| 响应 | HTTP chunked，裸 PCM16LE 24kHz 单声道 |

### 4.2 端点与参数

所有端点同时支持 `GET` 和 `POST`，参数为 `multipart/form-data`：

| 端点 | 参数 | 说明 |
| :--- | :--- | :--- |
| `/inference_sft` | `tts_text`, `spk_id` | 内置音色；需 `spk2info.pt` |
| `/inference_zero_shot` | `tts_text`, `prompt_text`, `prompt_wav`（文件） | 参考音频克隆（v2 主用） |
| `/inference_cross_lingual` | `tts_text`, `prompt_wav`（文件） | 跨语言，仅借音色 |
| `/inference_instruct` | `tts_text`, `spk_id`, `instruct_text` | 指令控风格；需 `spk2info.pt` |
| `/inference_instruct2` | `tts_text`, `instruct_text`, `prompt_wav`（文件） | 指令 + 参考音频 |

参数说明：

| 参数 | 类型 | 说明 |
| :--- | :--- | :--- |
| `tts_text` | string | 待合成文本 |
| `spk_id` | string | v2 内置音色 ID（`sft`/`instruct`）；v2 **没有** `zero_shot_by_id` |
| `prompt_text` | string | 参考音频逐字转写（`zero_shot`） |
| `prompt_wav` | file | 参考音频，3–10s、单人、干净 |
| `instruct_text` | string | 指令文本（`instruct`/`instruct2`） |

限制：
- v2 **没有** `zero_shot_by_id` / `speaker_info`；
- `sft` / `instruct` 依赖模型自带 `spk2info.pt`，下载的 `CosyVoice2-0.5B` 通常没有，故可能不可用；
- 输出采样率为 24kHz；官方旧客户端写死 22050，请使用发布包 `client_v2.py`。

### 4.3 响应格式

- HTTP 200，`Content-Type` 为流式响应；
- body 是连续 **PCM16LE、单声道、24000Hz** 原始字节，没有 WAV 头；
- 客户端需要按 chunk 拼接并加 WAV 头，或直接按 PCM 播放。

Python 拼接示例：

```python
import wave, requests

r = requests.post("http://127.0.0.1:50000/inference_zero_shot",
                  data={"tts_text": "你好", "prompt_text": "希望你以后能够做的比我还好呦。"},
                  files={"prompt_wav": open("/data/prompt.wav", "rb")},
                  stream=True)
pcm = b"".join(r.iter_content(16000))
with wave.open("/tmp/v2.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(pcm)
```

### 4.4 调用示例

#### 客户端（推荐）

```sh
cd /data/cosyvoice
python3 streaming/v2_fastapi/client_v2.py --url http://127.0.0.1:50000 \
  --mode zero_shot \
  --tts_text "你好，这是 v2 流式测试。" \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /data/prompt.wav \
  --tts_wav /tmp/stream_v2.wav
```

#### 统一客户端

```sh
python3 streaming/test_streaming.py --model-type v2 --api fastapi \
  --url http://127.0.0.1:50000 \
  --mode zero_shot \
  --prompt-wav /data/prompt.wav \
  --prompt-text "希望你以后能够做的比我还好呦。" \
  --text "你好，这是 v2 流式统一客户端测试。" \
  --out /tmp/stream_v2_unified.wav
```

#### curl

```sh
curl -sS -o /tmp/v2_stream.pcm \
  -F tts_text='你好，这是 v2 流式测试。' \
  -F prompt_text='希望你以后能够做的比我还好呦。' \
  -F prompt_wav=@/data/prompt.wav \
  http://127.0.0.1:50000/inference_zero_shot
# /tmp/v2_stream.pcm 为裸 PCM16，需要自行加 WAV 头/按 PCM 播放
```

### 4.5 错误与排查

| 现象 | 原因/处理 |
| :--- | :--- |
| 连接失败 | 服务未启动/端口不对/模型仍在加载 |
| 422 | 缺少必填 form 字段或文件 |
| `prompt_wav` 相关错误 | 参考音频缺失/格式不支持/采样率过低 |
| 音色或内容异常 | `prompt_text` 必须是参考音频的逐字转写，且与音频一致 |
| `spk_id` 不存在 | v2 内置音色依赖 `spk2info.pt`；普通下载模型可能为空 |

---

## 5. 参数/返回速查

| 服务 | 请求 | 返回 |
| :--- | :--- | :--- |
| 离线服务 | `POST /synthesize` multipart | `audio/wav` 完整 WAV + `X-*` 头 |
| v3 gRPC | `Request.zero_shot_by_id_request` | `stream Response.tts_audio`，PCM16LE 24k |
| v2 FastAPI | `POST /inference_*` multipart | HTTP chunked PCM16LE 24k |

推荐验证顺序：

1. 离线服务：先 `GET /health`，再 `/synthesize`/client，试听 WAV；
2. v3 gRPC：用 `test_streaming.py` 看 `chunks/ttfb/rtf`，试听 WAV；
3. v2 FastAPI：用 `client_v2.py` 或 `test_streaming.py`，试听 WAV。

更完整的部署、权重和容器说明见发布包 `cosyvoice_deploy.md`。
