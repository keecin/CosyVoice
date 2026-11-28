#!/usr/bin/env bash
# Start the CosyVoice3 native gRPC streaming service (run from the cookbook's
# runtime/python/grpc directory, or from the release's streaming/grpc after
# overlaying it into /opt/CosyVoice/runtime/python/grpc).
#
# All user-facing settings are command-line options; no env vars are required.
# Internal performance settings (spawn/TASK_QUEUE/MLP/TOPK/jemalloc) are fixed.
#
# Usage:
#   bash run_server.sh --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
#     --speaker-info-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B/speaker_info \
#     --device 0 --max-conc 8 --port 50000 --graph-mode
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -f "$HERE/server.py" ]; then
    _LOCAL_REPO=$(cd "$HERE/../../.." && pwd)
else
    _LOCAL_REPO=""
fi
DEFAULT_REPO="${_LOCAL_REPO:-/opt/CosyVoice}"
cd "$HERE"

usage() {
    cat <<'USAGE'
Start the CosyVoice3 gRPC streaming service.

Usage:
  bash run_server.sh [--model-dir DIR | --model-type v3] [OPTIONS]

Model selection (like offline/service/run_service.sh):
  --model-dir DIR          explicit v3 model directory
  --model-type v3           use <weights>/Fun-CosyVoice3-0.5B
                           (v2 is not supported by this gRPC service)
                           neither given -> default v3 directory

  Default search: <script>/../cosyvoice_models/Fun-CosyVoice3-0.5B,
  then <script>/../../../cosyvoice_models/Fun-CosyVoice3-0.5B,
  then /data/cosyvoice_models/Fun-CosyVoice3-0.5B.

Options:
  --cosy-repo DIR          CosyVoice source tree (default: /opt/CosyVoice,
                           or the parent tree when server.py is next to this script)
  --speaker-info-dir DIR   v3 speaker_info directory (default: <model-dir>/speaker_info)
  --device N               single NPU card id (default: 0)
  --max-conc N             max concurrent requests / scheduling slots (default: 8)
  --port N                 gRPC port (default: 50000)
  --graph-mode             enable torchair graph + post-gRPC warmup (slower start)
  --prewarm-timeout N      prewarm timeout seconds for non-graph mode (default: server 300)
  --no-prewarm             skip pre-gRPC warmup (not recommended)
  --extra-args "ARGS"      extra server.py args, e.g. --extra-args "--enable_vllm_profile"
  -h, --help               show this help

Example:
  bash run_server.sh \
    --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
    --speaker-info-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B/speaker_info \
    --device 0 --max-conc 8 --port 50000 --graph-mode
USAGE
}

die() { echo "error: $*" >&2; exit 2; }
require_value() { [ $# -ge 2 ] || die "$1 requires a value"; }

MODEL_DIR=""
MODEL_TYPE=""
SPEAKER_INFO_DIR=""
COSY_REPO=""
DEVICE="0"
MAX_CONC="8"
PORT="50000"
GRAPH_MODE=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) require_value "$@"; MODEL_DIR=$2; shift 2 ;;
        --model-type) require_value "$@"; MODEL_TYPE=$2; shift 2 ;;
        --cosy-repo) require_value "$@"; COSY_REPO=$2; shift 2 ;;
        --speaker-info-dir) require_value "$@"; SPEAKER_INFO_DIR=$2; shift 2 ;;
        --device) require_value "$@"; DEVICE=$2; shift 2 ;;
        --max-conc) require_value "$@"; MAX_CONC=$2; shift 2 ;;
        --port) require_value "$@"; PORT=$2; shift 2 ;;
        --graph-mode) GRAPH_MODE=1; shift ;;
        --prewarm-timeout) require_value "$@"; EXTRA+=(--prewarm_timeout "$2"); shift 2 ;;
        --no-prewarm) EXTRA+=(--no_prewarm); shift ;;
        --extra-args) require_value "$@"; read -r -a _extra <<< "$2"; EXTRA+=("${_extra[@]}"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

case "$MODEL_TYPE" in
    ''|v3) : ;;
    v2) die "v2 gRPC is not supported; use the v2 FastAPI service" ;;
    *) die "--model-type must be v3 (got '$MODEL_TYPE')" ;;
esac
if [ -n "$MODEL_TYPE" ] && [ -n "$MODEL_DIR" ]; then
    die "--model-type and --model-dir are mutually exclusive (give only one)"
fi
if [ -z "$MODEL_DIR" ]; then
    for _base in "$HERE/../cosyvoice_models" "$HERE/../../../cosyvoice_models" "/data/cosyvoice_models"; do
        if [ -d "$_base/Fun-CosyVoice3-0.5B" ]; then
            MODEL_DIR="$_base/Fun-CosyVoice3-0.5B"
            break
        fi
    done
fi
[ -n "$MODEL_DIR" ] || die "v3 model dir not found; use --model-dir or --model-type v3 (run download_weights.sh first)"
[ -d "$MODEL_DIR" ] || die "model dir not found: $MODEL_DIR"
[ -n "$COSY_REPO" ] || COSY_REPO="$DEFAULT_REPO"
[ -f "$COSY_REPO/runtime/python/grpc/server.py" ] || die "server.py not found under --cosy-repo: $COSY_REPO"
case "$DEVICE" in
    ''|*[!0-9]*) die "--device must be a single non-negative NPU id (got '$DEVICE')" ;;
esac
case "$MAX_CONC" in
    ''|*[!0-9]*) die "--max-conc must be a positive integer (got '$MAX_CONC')" ;;
esac
[ "$MAX_CONC" -gt 0 ] || die "--max-conc must be > 0"
case "$PORT" in
    ''|*[!0-9]*) die "--port must be a positive integer (got '$PORT')" ;;
esac
[ "$PORT" -gt 0 ] || die "--port must be > 0"
[ -n "$SPEAKER_INFO_DIR" ] || SPEAKER_INFO_DIR="$MODEL_DIR/speaker_info"
if [ "$GRAPH_MODE" = "1" ]; then
    EXTRA+=(--graph_mode)
fi

# If the base image did not put ATB/Ascend env in ENV, source it as needed.
# These scripts reference unset variables; temporarily disable set -u.
set +u
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    if [ -f "$_env" ]; then
        . "$_env" >/dev/null 2>&1 || true
    fi
done
set -u

# Fixed internal performance settings (not user-facing).
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export VLLM_ASCEND_ENABLE_MLP_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
JEMALLOC=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
if [ -f "$JEMALLOC" ]; then
    export LD_PRELOAD="$JEMALLOC:${LD_PRELOAD:-}"
fi

echo "[run_server] device=${DEVICE} port=${PORT} max_conc=${MAX_CONC} graph_mode=${GRAPH_MODE}"
echo "[run_server] model=${MODEL_DIR}"
echo "[run_server] speaker_info_dir=${SPEAKER_INFO_DIR}"
echo "[run_server] cosy_repo=${COSY_REPO}"
export PYTHONPATH="$COSY_REPO:$COSY_REPO/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
SERVER_DIR="$COSY_REPO/runtime/python/grpc"
cd "$SERVER_DIR"
exec python3 "$SERVER_DIR/server.py" \
    --port "$PORT" \
    --model_dir "$MODEL_DIR" \
    --max_conc "$MAX_CONC" \
    --speaker_info_dir "$SPEAKER_INFO_DIR" \
    "${EXTRA[@]}"
