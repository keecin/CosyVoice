#!/usr/bin/env python3
"""CosyVoice2/3 offline service: non-streaming HTTP, one request -> one WAV.

Loads the model once and exposes POST /synthesize returning a complete WAV file
(24kHz PCM16) in a single response (no SSE/chunked streaming). The model type is
auto-detected from MODEL_DIR (cosyvoice2.yaml / cosyvoice3.yaml) and each type
uses its own official API -- CosyVoice2 never goes through v3 interfaces.

Modes:
  v3: zero_shot_by_id (default, speaker_info) / zero_shot
      (v3 has no sft/instruct; zero_shot prompt_text must contain <|endofprompt|>)
  v2: zero_shot (default) / sft / cross_lingual / instruct / instruct2
Inference is serialized by a global lock (single model instance).
"""
import argparse
import io
import json
import os
import sys
import threading
import time
import wave
from typing import Optional

os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "8.0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402
from torch_npu.contrib import transfer_to_npu  # noqa: F401,E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import Response  # noqa: E402

try:
    from triton.backends import backends  # noqa: E402
    _nb = backends.get("nvidia")
    if _nb is not None and getattr(_nb, "driver", None) is not None:
        _nb.driver.is_active = lambda: False
except Exception:
    pass

app = FastAPI(title="CosyVoice offline (non-streaming) service")
STATE = {"lock": threading.Lock(), "ready": False}


def detect_type(model_dir):
    if os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
        return "v3"
    if os.path.exists(os.path.join(model_dir, "cosyvoice2.yaml")):
        return "v2"
    return None


def to_wav(speech, sr):
    pcm = (speech.squeeze(0).clamp(-1, 1).float().cpu().numpy() * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue(), len(pcm) / float(sr)


@app.get("/health")
def health():
    return {"status": "ok" if STATE["ready"] else "loading",
            "model_type": STATE.get("type"), "model_dir": STATE.get("model_dir"),
            "sample_rate": STATE.get("sample_rate")}


@app.post("/synthesize")
def synthesize(
    text: str = Form(...),
    mode: str = Form(""),
    spk_id: str = Form("spk1"),
    prompt_text: str = Form(""),
    instruct_text: str = Form(""),
    speed: float = Form(1.0),
    prompt_wav: Optional[UploadFile] = File(None),
):
    if not STATE["ready"]:
        raise HTTPException(503, "model not ready")
    model, mtype = STATE["model"], STATE["type"]

    prompt_speech = None
    if prompt_wav is not None:
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech = load_wav(prompt_wav.file, 16000)

    t0 = time.time()
    try:
        with STATE["lock"]:
            if mtype == "v3":
                mode = mode or "zero_shot_by_id"
                if mode == "zero_shot_by_id":
                    wi = getattr(model, "promote_wave_info", {}).get(spk_id, {})
                    if not wi:
                        raise HTTPException(400, f"speaker '{spk_id}' not found; available: "
                                            f"{list(getattr(model, 'promote_wave_info', {}).keys())}")
                    if '<|endofprompt|>' not in (wi.get("prompt_text") or ""):
                        raise HTTPException(400, f"speaker '{spk_id}' prompt_text must contain '<|endofprompt|>'")
                    gen = model.inference_zero_shot_by_id(text, spk_id, stream=False, speed=speed)
                elif mode == "zero_shot":
                    if prompt_speech is None:
                        raise HTTPException(400, "zero_shot needs prompt_wav")
                    if '<|endofprompt|>' not in (prompt_text or ""):
                        raise HTTPException(400, "v3 prompt_text must contain '<|endofprompt|>' "
                                            "(e.g. \"You are a helpful assistant.<|endofprompt|>\" + transcript)")
                    gen = model.inference_zero_shot(text, prompt_text, prompt_speech,
                                                    stream=False, speed=speed)
                elif mode == "sft":
                    raise HTTPException(400, "CosyVoice3 does not support sft (no spk2info.pt; "
                                        "use zero_shot_by_id or zero_shot)")
                else:
                    raise HTTPException(400, f"v3 unsupported mode '{mode}'")
            elif mtype == "v2":
                mode = mode or "zero_shot"
                if mode == "zero_shot":
                    if prompt_speech is None:
                        raise HTTPException(400, "zero_shot needs prompt_wav")
                    gen = model.inference_zero_shot(text, prompt_text, prompt_speech,
                                                    stream=False, speed=speed)
                elif mode == "cross_lingual":
                    if prompt_speech is None:
                        raise HTTPException(400, "cross_lingual needs prompt_wav")
                    gen = model.inference_cross_lingual(text, prompt_speech,
                                                        stream=False, speed=speed)
                elif mode == "sft":
                    gen = model.inference_sft(text, spk_id, stream=False, speed=speed)
                elif mode == "instruct":
                    gen = model.inference_instruct(text, spk_id, instruct_text,
                                                   stream=False, speed=speed)
                elif mode == "instruct2":
                    if prompt_speech is None:
                        raise HTTPException(400, "instruct2 needs prompt_wav")
                    gen = model.inference_instruct2(text, instruct_text, prompt_speech,
                                                    stream=False, speed=speed)
                else:
                    raise HTTPException(400, f"v2 unsupported mode '{mode}'")
            else:
                raise HTTPException(500, "unknown model type")

            chunks = [o["tts_speech"] for o in gen]
            if not chunks:
                raise HTTPException(500, "empty output")
            speech = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
            data, adur = to_wav(speech, STATE["sample_rate"])
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(500, repr(e))

    dt = time.time() - t0
    print(f"[svc] mode={mode} audio={adur:.2f}s elapsed={dt:.2f}s rtf={dt / adur:.3f}", flush=True)
    return Response(content=data, media_type="audio/wav", headers={
        "X-Model-Type": mtype, "X-Mode": mode, "X-Sample-Rate": str(STATE["sample_rate"]),
        "X-Audio-Seconds": f"{adur:.3f}", "X-Rtf": f"{dt / adur:.3f}",
    })


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.environ.get("MODEL_DIR") or os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "cosyvoice_models", "Fun-CosyVoice3-0.5B")))
    ap.add_argument("--speaker_info_dir", default=os.environ.get("SPEAKER_INFO_DIR") or "")
    ap.add_argument("--cosy_repo", default=os.environ.get("COSY_REPO", "/opt/CosyVoice"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--graph_mode", action="store_true", help="v3 torchair graph (slower load)")
    ap.add_argument("--vllm", action="store_true", help="force enable vLLM LLM stage")
    ap.add_argument("--no_vllm", action="store_true", help="force disable vLLM LLM stage")
    return ap.parse_args()


def main():
    args = parse_args()
    repo = args.cosy_repo
    for p in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
        if p not in sys.path:
            sys.path.insert(0, p)

    mtype = detect_type(args.model_dir)
    if mtype is None:
        raise SystemExit(f"cannot detect model type (no cosyvoice2/3.yaml) in {args.model_dir}")
    use_vllm = (not args.no_vllm) and (args.vllm or mtype == "v3")
    spk_dir = args.speaker_info_dir or os.path.join(args.model_dir, "speaker_info")

    if use_vllm and mtype == "v2":
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel
    print(f"[svc] model_type={mtype} model={args.model_dir} vllm={use_vllm} "
          f"graph_mode={args.graph_mode if mtype == 'v3' else False}", flush=True)
    t0 = time.time()
    if mtype == "v3":
        model = AutoModel(model_dir=args.model_dir, load_vllm=use_vllm,
                          speaker_info_dir=spk_dir, graph_mode=args.graph_mode)
    else:
        model = AutoModel(model_dir=args.model_dir, load_vllm=use_vllm)
    STATE.update(model=model, type=mtype, model_dir=args.model_dir,
                 sample_rate=model.sample_rate, ready=True)
    print(f"[svc] loaded in {time.time() - t0:.1f}s, sr={model.sample_rate}", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
