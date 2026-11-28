#!/usr/bin/env bash
# Pure CosyVoice2 (v2) online service via the OFFICIAL FastAPI server.
# - Official file: /opt/CosyVoice/runtime/python/fastapi/server.py (unchanged)
# - This script only sets Ascend env + starts serve_v2.py (torch_npu shim).
#   MODEL_DIR points at CosyVoice2-0.5B (has cosyvoice2.yaml).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export COSY_REPO=${COSY_REPO:-/opt/CosyVoice}
export MODEL_DIR=${MODEL_DIR:-$HERE/../cosyvoice_models/CosyVoice2-0.5B}
export PORT=${PORT:-50000}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export TORCH_NPU_DEVICE_CAPABILITY=${TORCH_NPU_DEVICE_CAPABILITY:-8.0}
export PYTHONUNBUFFERED=1

set +u
for _env in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$_env" ] && . "$_env" >/dev/null 2>&1 || true
done
set -u

echo "[v2-fastapi] model=${MODEL_DIR} device=${ASCEND_RT_VISIBLE_DEVICES} port=${PORT}"
exec python3 "$HERE/serve_v2.py" --model_dir "$MODEL_DIR" --port "$PORT" "$@"
