#!/usr/bin/env bash
# 统一的测试客户端脚本（配合 run_server_npu.sh 使用）。
#
# 角色：
#   - BENCH=1（默认）：用 batch_test_client.py 做“并发/吞吐”压测；
#   - BENCH=0：用 client.py 发“单请求”（可看首包时延/RTF，并保存单个 wav）。
#
# 与 MindIE 原脚本的关系（吸收说明）：
#   - 原 run_client.sh 只是一个“示例命令清单”（client.py 单请求 + batch_test_client.py 批量）；
#     本脚本把它吸收成参数化的“单请求 + 批量压测二合一”客户端。
#   - 原 start.sh 是“单机离线推理”（infer.py + taskset 绑核），不属于“服务+客户端”这一对；
#     该场景已由仓库里的 run_npu.py 覆盖（需要绑核可自行用 taskset 包 run_npu.py）。
#
# 用法示例（在 runtime/python/grpc 目录内执行）：
#   # 批量压测：zero_shot_by_id，12 请求 / 8 并发，保存第 9 号音频
#   MODE=zero_shot_by_id SPK_ID=spk1 TOTAL_REQ=12 CONCURRENCY=8 SAVE_TASK_ID=9 bash test_client_npu.sh
#   # 单请求：保存一个 wav
#   BENCH=0 MODE=zero_shot_by_id SPK_ID=spk1 TTS_WAV=./demo.wav bash test_client_npu.sh
#
# 环境变量：
#   HOST(127.0.0.1) PORT(50000) MODE(zero_shot_by_id) SPK_ID(spk1) STREAM(1)
#   BENCH(1|0) TOTAL_REQ(12) CONCURRENCY(8) SAVE_TASK_ID SAVE_DIR(./saved_audio) TEXT_FILE
#   TTS_TEXT TTS_WAV(./test_out.wav) PROMPT_TEXT PROMPT_WAV INSTRUCT_TEXT
set -eu
cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-50000}"
MODE="${MODE:-zero_shot_by_id}"
SPK_ID="${SPK_ID:-spk1}"
STREAM="${STREAM:-1}"
BENCH="${BENCH:-1}"

if [ ! -f cosyvoice_pb2.py ] || [ ! -f cosyvoice_pb2_grpc.py ]; then
    echo "ERROR: cosyvoice_pb2*.py missing; run 'bash update_grpc_file.sh' first" >&2
    exit 1
fi

if [ "$BENCH" = "1" ]; then
    TOTAL_REQ="${TOTAL_REQ:-12}"
    CONCURRENCY="${CONCURRENCY:-8}"
    SAVE_DIR="${SAVE_DIR:-./saved_audio}"
    args=(--host "$HOST" --port "$PORT" --mode "$MODE" --spk_id "$SPK_ID"
          --total_req "$TOTAL_REQ" --concurrency "$CONCURRENCY" --save_audio_dir "$SAVE_DIR")
    [ "$STREAM" = "1" ] && args+=(--stream)
    [ -n "${TEXT_FILE:-}" ] && args+=(--text_file "$TEXT_FILE")
    [ -n "${TTS_TEXT:-}" ] && args+=(--tts_text "$TTS_TEXT")
    [ -n "${SAVE_TASK_ID:-}" ] && args+=(--save_task_id "$SAVE_TASK_ID")
    echo "[test_client_npu] bench: mode=${MODE} spk=${SPK_ID} total=${TOTAL_REQ} conc=${CONCURRENCY} -> ${HOST}:${PORT}"
    exec python3 batch_test_client.py "${args[@]}"
else
    TTS_WAV="${TTS_WAV:-./test_out.wav}"
    args=(--host "$HOST" --port "$PORT" --mode "$MODE" --spk_id "$SPK_ID" --tts_wav "$TTS_WAV")
    [ "$STREAM" = "1" ] && args+=(--stream)
    [ -n "${TTS_TEXT:-}" ] && args+=(--tts_text "$TTS_TEXT")
    [ -n "${PROMPT_TEXT:-}" ] && args+=(--prompt_text "$PROMPT_TEXT")
    [ -n "${PROMPT_WAV:-}" ] && args+=(--prompt_wav "$PROMPT_WAV")
    [ -n "${INSTRUCT_TEXT:-}" ] && args+=(--instruct_text "$INSTRUCT_TEXT")
    echo "[test_client_npu] single: mode=${MODE} spk=${SPK_ID} -> ${HOST}:${PORT}, save=${TTS_WAV}"
    exec python3 client.py "${args[@]}"
fi
