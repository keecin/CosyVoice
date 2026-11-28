#!/usr/bin/env python3
"""Ascend wrapper for the official CosyVoice FastAPI server (pure v2 path).

Runs the *official* file /opt/CosyVoice/runtime/python/fastapi/server.py
unchanged (its own ROOT_DIR sys.path logic stays valid there); this wrapper only
adds the Ascend runtime shim (torch_npu + transfer_to_npu) before the server
imports torch/AutoModel, and forwards CLI args.

No CosyVoice3-specific code (speaker_info/graph_mode/zero_shot_by_id) is used.
"""
import os
import runpy
import sys

os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "8.0")

import torch  # noqa: E402,F401
import torch_npu  # noqa: E402,F401
from torch_npu.contrib import transfer_to_npu  # noqa: E402,F401

repo = os.environ.get("COSY_REPO", "/opt/CosyVoice")
server = os.path.join(repo, "runtime/python/fastapi/server.py")
if not os.path.exists(server):
    raise SystemExit(f"official fastapi server not found: {server}")

sys.argv = [server] + sys.argv[1:]
runpy.run_path(server, run_name="__main__")
