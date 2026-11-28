#!/usr/bin/env bash
# CosyVoice2/3 offline (non-streaming) HTTP service: one POST /synthesize -> one WAV.
# All customer inputs are command-line options; fixed runtime settings are hardcoded.
# Model type (v2/v3) is auto-detected from the model dir (cosyvoice2/3.yaml).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
    cat <<'EOF'
Usage:
  bash run_service.sh [--model-type v2|v3 | --model-dir DIR] [--device N] [--port PORT] [options]

Key options (choose model by TYPE or by DIR, not both):
  --model-type v2|v3       pick model dir automatically (v2=CosyVoice2-0.5B, v3=Fun-CosyVoice3-0.5B)
  --model-dir DIR          explicit model dir; type inferred from cosyvoice2/3.yaml
                           (neither given -> default v3 dir ../cosyvoice_models/Fun-CosyVoice3-0.5B)
  --device N               single NPU card id (default 0); multi-card -> one instance per card
  --port PORT              HTTP port (default 9500)
  --speaker-info-dir DIR   v3 zero_shot_by_id speaker dir (default: <model-dir>/speaker_info)

Optional:
  --host HOST              bind host (default 0.0.0.0)
  --graph-mode             v3 torchair graph (slower start)
  --vllm | --no-vllm       force enable/disable the vLLM LLM stage
                           (default: v3 on, v2 off)
  -h, --help               show this help

Examples:
  bash run_service.sh --model-type v3 --device 0 --port 9500
  bash run_service.sh --model-type v2 --device 1 --port 9501
  bash run_service.sh --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B --device 0 --port 9500
EOF
}

MODEL_DIR=""
MODEL_TYPE=""
SPEAKER_INFO_DIR=""
DEVICE=0
PORT=9500
HOST=0.0.0.0
FORWARD=()

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir|--model)  MODEL_DIR=${2:?missing value}; shift 2 ;;
        --model-dir=*)        MODEL_DIR=${1#*=}; shift ;;
        --model-type|--version)  MODEL_TYPE=${2:?missing value}; shift 2 ;;
        --model-type=*|--version=*) MODEL_TYPE=${1#*=}; shift ;;
        --speaker-info-dir)   SPEAKER_INFO_DIR=${2:?missing value}; shift 2 ;;
        --speaker-info-dir=*) SPEAKER_INFO_DIR=${1#*=}; shift ;;
        --device)             DEVICE=${2:?missing value}; shift 2 ;;
        --device=*)           DEVICE=${1#*=}; shift ;;
        --port)               PORT=${2:?missing value}; shift 2 ;;
        --port=*)             PORT=${1#*=}; shift ;;
        --host)               HOST=${2:?missing value}; shift 2 ;;
        --host=*)             HOST=${1#*=}; shift ;;
        --graph-mode|--graph_mode) FORWARD+=(--graph_mode); shift ;;
        --vllm)               FORWARD+=(--vllm); shift ;;
        --no-vllm|--no_vllm)  FORWARD+=(--no_vllm); shift ;;
        -h|--help)            usage; exit 0 ;;
        *) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

case "$DEVICE" in
    ''|*[!0-9]*)
        echo "error: --device must be a single non-negative NPU id (got '$DEVICE')." >&2
        echo "       One service instance uses one card; for multiple cards start one instance per card" >&2
        echo "       with different --port (e.g. --device 0 --port 9500, --device 1 --port 9501)." >&2
        exit 2 ;;
esac

# Model selection: --model-type and --model-dir are mutually exclusive.
case "$MODEL_TYPE" in
    ''|v2|v3) : ;;
    *) echo "error: --model-type must be v2 or v3 (got '$MODEL_TYPE')" >&2; exit 2 ;;
esac
if [ -n "$MODEL_TYPE" ] && [ -n "$MODEL_DIR" ]; then
    echo "error: --model-type and --model-dir are mutually exclusive (give only one)" >&2
    exit 2
fi
if [ -n "$MODEL_TYPE" ]; then
    case "$MODEL_TYPE" in
        v2) MODEL_DIR="$HERE/../cosyvoice_models/CosyVoice2-0.5B" ;;
        v3) MODEL_DIR="$HERE/../cosyvoice_models/Fun-CosyVoice3-0.5B" ;;
    esac
fi
# Fixed default (no environment fallback): v3 model dir when neither was given.
[ -z "$MODEL_DIR" ] && MODEL_DIR="$HERE/../cosyvoice_models/Fun-CosyVoice3-0.5B"
[ -z "$SPEAKER_INFO_DIR" ] && SPEAKER_INFO_DIR="$MODEL_DIR/speaker_info"
if [ ! -d "$MODEL_DIR" ]; then
    if [ -n "$MODEL_TYPE" ]; then
        echo "error: model dir for --model-type $MODEL_TYPE not found: $MODEL_DIR (run download_weights.sh)" >&2
    else
        echo "error: model dir not found: $MODEL_DIR (run download_weights.sh or pass --model-dir/--model-type)" >&2
    fi
    exit 2
fi

export COSY_REPO=/opt/CosyVoice
export TORCH_NPU_DEVICE_CAPABILITY=8.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

set +u
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$_env" ] && . "$_env" >/dev/null 2>&1 || true
done
set -u

JEMALLOC=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
if [ -f "$JEMALLOC" ]; then
    export LD_PRELOAD="$JEMALLOC${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# keep image/CANN paths (e.g. six under /opt/tritonserver/... and CANN site-packages)
export PYTHONPATH="$COSY_REPO:$COSY_REPO/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

echo "[svc] model=${MODEL_DIR} speakers=${SPEAKER_INFO_DIR} device=${DEVICE} host=${HOST} port=${PORT}"
exec python3 "$HERE/server.py" \
    --model_dir "$MODEL_DIR" \
    --speaker_info_dir "$SPEAKER_INFO_DIR" \
    --cosy_repo "$COSY_REPO" \
    --host "$HOST" \
    --port "$PORT" \
    ${FORWARD[@]+"${FORWARD[@]}"}
