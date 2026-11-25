#!/usr/bin/env bash
# 按 MindIE single_timbre_fm_batch 的部署方式启动 gRPC 批量/流式服务
# （沿用其 run_server.sh 的环境变量与 --graph_mode；但去掉作者机器写死的
#   模型路径/卡号，改为参数化，便于在任何 Ascend 镜像里使用）。
#
# 用法（在 runtime/python/grpc 目录内执行）：
#   MODEL_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
#   SPEAKER_INFO_DIR=/opt/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/speaker_info \
#   ASCEND_RT_VISIBLE_DEVICES=6 MAX_CONC=8 PORT=50000 \
#   bash run_server_npu.sh
#
# 可选环境变量：
#   GRAPH_MODE=1|0   MindIE 原脚本默认 --graph_mode(1)。引擎按 vLLM 导出架构分流：
#                    v3(Qwen2ForCausalLM) 用 spawn + grpc 后 warmup（实测跑通；启动较慢，
#                    含 cudagraph capture+warmup 约 5 分钟）；v2(CosyVoice2ForCausalLM) 用
#                    in-process+注册。本脚本默认 0（启动更快）；需要时设 GRAPH_MODE=1。
#   EXTRA_ARGS="..." 追加给 server.py 的参数（如 --prewarm_timeout 600）
#
# 配套测试客户端：test_client_npu.sh（BENCH=1 批量压测 / BENCH=0 单请求）
set -eu
cd "$(dirname "$0")"

# 若基础镜像没有把 ATB/Ascend 环境放进 ENV，这里按需 source（NNAL-ATB 缺失会
# 导致 vLLM 加载 libop_plugin_atb.so 失败）。
# 这些 set_env 脚本会引用未定义变量，临时关掉 set -u 再 source。
set +u
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [ -f "$_env" ]; then
        # shellcheck disable=SC1090
        . "$_env" >/dev/null 2>&1 || true
    fi
done
set -u

# ---- MindIE run_server.sh 同款环境变量（吞吐/调度/内存）----
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE="${VLLM_ASCEND_ENABLE_MLP_OPTIMIZE:-1}"
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION="${VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:-1}"
JEMALLOC=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
if [ -f "$JEMALLOC" ]; then
    export LD_PRELOAD="$JEMALLOC:${LD_PRELOAD:-}"
fi

# ---- 参数 ----
: "${MODEL_DIR:?please set MODEL_DIR=/path/to/Fun-CosyVoice3-0.5B}"
MAX_CONC="${MAX_CONC:-8}"
PORT="${PORT:-50000}"
SPEAKER_INFO_DIR="${SPEAKER_INFO_DIR:-$MODEL_DIR/speaker_info}"
GRAPH_MODE="${GRAPH_MODE:-0}"

EXTRA=()
if [ "$GRAPH_MODE" = "1" ]; then
    EXTRA+=(--graph_mode)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA+=(${EXTRA_ARGS})
fi

echo "[run_server_npu] device=${ASCEND_RT_VISIBLE_DEVICES} port=${PORT} max_conc=${MAX_CONC} graph_mode=${GRAPH_MODE}"
echo "[run_server_npu] model=${MODEL_DIR}"
echo "[run_server_npu] speaker_info_dir=${SPEAKER_INFO_DIR}"
echo "[run_server_npu] env: spawn/TASK_QUEUE_ENABLE/CPU_AFFINITY_CONF/MLP_OPTIMIZE/TOPK_TOPP/jemalloc"

exec python3 server.py \
    --port "$PORT" \
    --model_dir "$MODEL_DIR" \
    --max_conc "$MAX_CONC" \
    --speaker_info_dir "$SPEAKER_INFO_DIR" \
    "${EXTRA[@]}"
