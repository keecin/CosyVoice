# CosyVoice2 纯 v2 在线服务（官方 FastAPI）

不经过任何 CosyVoice3 代码：直接运行**官方** `runtime/python/fastapi/server.py`
（`/opt/CosyVoice` 内，与上游 074ca6d 逐字节一致），本目录只做 Ascend 适配包装。

## 启动

```sh
ASCEND_RT_VISIBLE_DEVICES=0 PORT=50000 \
  bash streaming/v2_fastapi/run_server_v2.sh
```
说明：`serve_v2.py` 在导入 torch 前执行 `import torch_npu; transfer_to_npu`，
然后 `runpy` 原样运行官方 server（其自带的 `sys.path ../../..` 在 `/opt/CosyVoice` 下有效）。
**不支持 vLLM**（官方 FastAPI server 没有 `load_vllm` 参数；vLLM 见官方 `vllm_example.py`
或本仓 `rebuild/run_npu.py --vllm`）。

## 调用

官方端点：`/inference_sft`、`/inference_zero_shot`、`/inference_cross_lingual`、
`/inference_instruct`、`/inference_instruct2`（HTTP 流式回包）。

```sh
python3 streaming/v2_fastapi/client_v2.py --url http://127.0.0.1:50000 \
  --mode zero_shot --tts_text "你好，这是 CosyVoice2 服务测试。" \
  --prompt_text "希望你以后能够做的比我还好呦。" \
  --prompt_wav /path/to/prompt.wav \
  --tts_wav /tmp/v2.wav
```
注意：
- v2 输出 24kHz——官方 `client.py` 写死 22050，请用本目录的 `client_v2.py`（默认 24000）。
- v2 **没有** `zero_shot_by_id`/`speaker_info`：用 `zero_shot`（`prompt_wav`+`prompt_text`）、
  `cross_lingual`（仅 `prompt_wav`）、或 `sft/instruct`（需要模型的 `spk2info.pt`）。
- `instruct2` 仅官方 FastAPI 暴露（gRPC proto 里没有）。
