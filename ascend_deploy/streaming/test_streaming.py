#!/usr/bin/env python3
"""Unified streaming test client for CosyVoice2 and CosyVoice3.

Dispatches to the correct transport:
  v3 + grpc     -> runtime/python/grpc/server.py (cosyvoice_pb2 stub)
  v3 + triton   -> Triton python backend, gRPC ModelStreamInfer
  v2 + fastapi  -> official CosyVoice FastAPI HTTP streaming endpoints

All adapters return PCM16 chunks and report the same TTFB/chunks/total/RTF
metrics, then write the concatenated audio to --out (wav).
"""
import argparse
import os
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TEXT = "旅游是一种很好的放松方式，可以让我们暂时远离繁忙的工作，享受大自然的美好与宁静。"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Unified streaming test client (v2 FastAPI / v3 gRPC / v3 Triton).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # v3 native gRPC
  python3 streaming/test_streaming.py --model-type v3 --api grpc \\
    --url 127.0.0.1:50000 --text "你好，这是 v3 测试。" \\
    --mode zero_shot_by_id --spk-id spk1 --out /tmp/v3.wav

  # v3 Triton gRPC
  python3 streaming/test_streaming.py --model-type v3 --api triton \\
    --url 127.0.0.1:9101 --model cosyvoice --out /tmp/v3_triton.wav

  # v2 official FastAPI
  python3 streaming/test_streaming.py --model-type v2 --api fastapi \\
    --url http://127.0.0.1:50000 --mode zero_shot \\
    --prompt-wav /path/prompt.wav --prompt-text "希望你以后能够做的比我还好呦。" \\
    --out /tmp/v2.wav
""")
    ap.add_argument("--model-type", choices=["v2", "v3"], required=True)
    ap.add_argument("--api", choices=["grpc", "triton", "fastapi"],
                    help="v2 only supports fastapi; v3 supports grpc/triton "
                         "(default: v3=grpc, v2=fastapi)")
    ap.add_argument("--url", help="grpc/triton: host:port; fastapi: http://host:port")
    ap.add_argument("--model", default="cosyvoice", help="Triton model name (default: cosyvoice)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="text to synthesize")
    ap.add_argument("--mode", help="v3: zero_shot_by_id|sft; "
                                  "v2: sft|zero_shot|cross_lingual|instruct|instruct2")
    ap.add_argument("--spk-id", default="spk1", help="speaker id for by_id/sft/instruct")
    ap.add_argument("--prompt-wav", help="reference wav for v2 zero_shot/cross_lingual/instruct2")
    ap.add_argument("--prompt-text", help="prompt transcript for v2 zero_shot")
    ap.add_argument("--instruct-text", help="instruction text for v2 instruct/instruct2")
    ap.add_argument("--out", default="/tmp/test_streaming.wav", help="output wav path")
    ap.add_argument("--sample-rate", type=int, default=24000)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--pb2-dir", help="directory containing cosyvoice_pb2*.py "
                                      "(default: streaming/triton/pb2)")
    return ap


def die(msg):
    raise SystemExit(f"error: {msg}")


def save_wav(path, pcm, sr):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return path


def load_pb2(pb2_dir):
    if pb2_dir:
        candidates = [Path(pb2_dir)]
    else:
        candidates = [HERE / "triton" / "pb2", HERE / "grpc", HERE / "pb2",
                      Path("/opt/CosyVoice/runtime/python/grpc")]
    d = next((p for p in candidates if (p / "cosyvoice_pb2.py").exists()), None)
    if d is None:
        die("cosyvoice_pb2.py not found; looked in: "
            + ", ".join(str(p) for p in candidates) + "; pass --pb2-dir")
    sys.path.insert(0, str(d))
    import cosyvoice_pb2
    import cosyvoice_pb2_grpc
    return cosyvoice_pb2, cosyvoice_pb2_grpc


def run_grpc_v3(args):
    import grpc
    pb2, pb2_grpc = load_pb2(args.pb2_dir)
    request = pb2.Request()
    if args.mode == "zero_shot_by_id":
        r = pb2.zeroshotByIdRequest()
        r.tts_text = args.text
        r.spk_id = args.spk_id
        r.stream = True
        request.zero_shot_by_id_request.CopyFrom(r)
    elif args.mode == "sft":
        r = pb2.sftRequest()
        r.tts_text = args.text
        r.spk_id = args.spk_id
        r.stream = True
        request.sft_request.CopyFrom(r)
    else:
        die(f"v3 grpc does not support mode={args.mode} (use zero_shot_by_id|sft)")

    url = args.url.removeprefix("http://").removeprefix("https://")
    chunks = []
    first = None
    t0 = time.time()
    with grpc.insecure_channel(url) as channel:
        stub = pb2_grpc.CosyVoiceStub(channel)
        for r in stub.Inference(request, timeout=args.timeout):
            audio = r.tts_audio
            if not audio:
                continue
            if first is None:
                first = time.time()
            chunks.append(audio)
    return chunks, t0, first, args.sample_rate


def run_triton_v3(args):
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        die("tritonclient is not installed; use --api grpc for the native service")

    import threading
    done = threading.Event()
    chunks = []
    state = {"t0": time.time(), "first": None, "final": False, "err": None, "sr": args.sample_rate}

    def callback(result, error):
        if error is not None:
            state["err"] = str(error)
            done.set()
            return
        if result is None:
            done.set()
            return
        try:
            audio_b = result.as_numpy("audio")
            final_b = result.as_numpy("final")
            sr_b = result.as_numpy("sample_rate")
            if state["first"] is None:
                state["first"] = time.time()
            if audio_b is not None:
                import base64
                for a in audio_b:
                    if a:
                        chunks.append(base64.b64decode(a))
            if sr_b is not None:
                state["sr"] = int(sr_b[0])
            if final_b is not None and bool(final_b[0]):
                state["final"] = True
                done.set()
        except Exception as e:  # noqa: BLE001
            state["err"] = repr(e)
            done.set()

    client = grpcclient.InferenceServerClient(url=args.url.removeprefix("http://"))
    client.start_stream(callback=callback)

    def make(name, val, dtype):
        t = grpcclient.InferInput(name, [1], dtype)
        t.set_data_from_numpy(np.array([val], dtype=object if dtype == "BYTES" else np.bool_))
        return t

    inputs = [
        make("text_input", args.text.encode(), "BYTES"),
        make("spk_id", args.spk_id.encode(), "BYTES"),
        make("mode", args.mode.encode(), "BYTES"),
        make("stream", True, "BOOL"),
    ]
    outputs = [
        grpcclient.InferRequestedOutput("audio"),
        grpcclient.InferRequestedOutput("final"),
        grpcclient.InferRequestedOutput("sample_rate"),
    ]
    client.async_stream_infer(
        model_name=args.model, inputs=inputs, outputs=outputs,
        request_id=str(uuid.uuid4()), enable_empty_final_response=True,
    )
    ok = done.wait(args.timeout)
    client.stop_stream()
    if state["err"]:
        die(f"Triton: {state['err']}")
    if not ok or not state["final"]:
        die("Triton stream did not finish (timeout/final not received)")
    return chunks, state["t0"], state["first"], state["sr"]


def run_fastapi_v2(args):
    try:
        import requests
    except ImportError:
        die("requests is not installed")

    url = args.url or "http://127.0.0.1:50000"
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    endpoint = f"{url.rstrip('/')}/inference_{args.mode}"
    data = {"tts_text": args.text}
    files = None
    fh = None

    if args.mode == "sft":
        data.update(spk_id=args.spk_id)
    elif args.mode == "zero_shot":
        if not args.prompt_wav or not args.prompt_text:
            die("v2 zero_shot requires --prompt-wav and --prompt-text")
        data.update(prompt_text=args.prompt_text)
    elif args.mode == "cross_lingual":
        if not args.prompt_wav:
            die("v2 cross_lingual requires --prompt-wav")
    elif args.mode == "instruct":
        data.update(spk_id=args.spk_id, instruct_text=args.instruct_text or "")
    elif args.mode == "instruct2":
        if not args.prompt_wav:
            die("v2 instruct2 requires --prompt-wav")
        data.update(instruct_text=args.instruct_text or "")

    if args.mode in ("zero_shot", "cross_lingual", "instruct2"):
        if not os.path.isfile(args.prompt_wav):
            die(f"--prompt-wav not found: {args.prompt_wav}")
        fh = open(args.prompt_wav, "rb")
        files = [("prompt_wav", ("prompt_wav", fh, "application/octet-stream"))]

    chunks = []
    first = None
    t0 = time.time()
    try:
        with requests.request("GET", endpoint, data=data, files=files,
                              stream=True, timeout=args.timeout) as r:
            r.raise_for_status()
            for b in r.iter_content(chunk_size=16000):
                if not b:
                    continue
                if first is None:
                    first = time.time()
                chunks.append(b)
    finally:
        if fh is not None:
            fh.close()
    return chunks, t0, first, args.sample_rate


def main():
    args = build_parser().parse_args()
    if args.model_type == "v2":
        if args.api is None:
            args.api = "fastapi"
        if args.api != "fastapi":
            die("v2 only supports --api fastapi (no v2 gRPC/Triton service)")
        if args.mode is None:
            args.mode = "zero_shot"
        if args.mode not in ("sft", "zero_shot", "cross_lingual", "instruct", "instruct2"):
            die(f"invalid v2 mode: {args.mode}")
        chunks, t0, first, sr = run_fastapi_v2(args)
    else:
        if args.api is None:
            args.api = "grpc"
        if args.api not in ("grpc", "triton"):
            die("v3 supports --api grpc or --api triton")
        if args.mode is None:
            args.mode = "zero_shot_by_id"
        if args.mode not in ("zero_shot_by_id", "sft"):
            die(f"invalid v3 mode: {args.mode}")
        if args.url is None:
            args.url = "127.0.0.1:50000" if args.api == "grpc" else "127.0.0.1:9101"
        if args.api == "grpc":
            chunks, t0, first, sr = run_grpc_v3(args)
        else:
            chunks, t0, first, sr = run_triton_v3(args)

    pcm = b"".join(chunks)
    if not pcm:
        die("no audio received")
    total = time.time() - t0
    audio_dur = len(pcm) / 2.0 / sr
    ttfb = (first - t0) * 1000 if first is not None else float("nan")
    out = save_wav(args.out, pcm, sr)
    print(f"api={args.api} model_type={args.model_type} mode={args.mode}")
    print(f"chunks={len(chunks)} pcm={len(pcm)} sr={sr} audio={audio_dur:.2f}s")
    print(f"ttfb={ttfb:.0f}ms total={total:.2f}s rtf={total / audio_dur:.3f}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
