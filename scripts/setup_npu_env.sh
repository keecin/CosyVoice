#!/usr/bin/env bash
# Setup CosyVoice2/3 inference environment on an Ascend NPU container.
#
# This intentionally does NOT install the upstream requirements.txt (which pins
# CUDA torch / onnxruntime-gpu / tensorrt-cu12 and would break torch_npu).
# torch / torch_npu / numpy / transformers are used as provided by the base
# image; CosyVoice's KV-cache incremental decoding was patched to be compatible
# with the system transformers 5.x, so no isolated venv / transformers downgrade
# is required.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

echo "==> repo: ${REPO_DIR}"

echo "==> [1/2] install extra deps (no torch/numpy/transformers change)"
pip install -r "${REPO_DIR}/requirements-npu.txt" \
    -i "${PIP_INDEX}" --trusted-host mirrors.aliyun.com

echo "==> [2/2] pin pyarrow to a numpy-1.26 compatible build"
pip install --only-binary=:all: pyarrow==18.1.0 -i "${PIP_INDEX}" --trusted-host mirrors.aliyun.com

echo "==> sanity check"
python3 - <<'PY'
import sys
import torch
import torch_npu  # noqa: F401
import torch_npu.contrib.transfer_to_npu  # noqa: F401
import numpy
import transformers
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('npu count   ', torch_npu.npu.device_count())
print('cuda->npu   ', torch.cuda.is_available())
print('numpy       ', numpy.__version__)
print('transformers', transformers.__version__)
PY

echo "==> done. Run with: python3 run_npu.py <model_dir> <out.wav>"
