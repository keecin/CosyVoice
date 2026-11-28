#!/usr/bin/env python3
"""Thin HTTP client for the official CosyVoice FastAPI (v2) endpoints.

The upstream client.py hardcodes target_sr=22050 (v1); CosyVoice2/3 output 24kHz,
so this client takes --sample_rate (default 24000) and adds /inference_instruct2.
"""
import argparse
import wave

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--url", default="", help="e.g. http://127.0.0.1:50000")
    ap.add_argument("--mode", default="zero_shot",
                    choices=["sft", "zero_shot", "cross_lingual", "instruct", "instruct2"])
    ap.add_argument("--tts_text", default="你好，我是通义千问语音合成大模型，请问有什么可以帮您的吗？")
    ap.add_argument("--spk_id", default="中文女")
    ap.add_argument("--prompt_text", default="希望你以后能够做的比我还好呦。")
    ap.add_argument("--prompt_wav", default="/path/to/prompt.wav")
    ap.add_argument("--instruct_text", default="用四川话说这句话")
    ap.add_argument("--tts_wav", default="v2_fastapi_out.wav")
    ap.add_argument("--sample_rate", type=int, default=24000)
    args = ap.parse_args()

    url = (args.url or f"http://{args.host}:{args.port}") + f"/inference_{args.mode}"
    data = {"tts_text": args.tts_text}
    files = None
    if args.mode == "sft":
        data.update(spk_id=args.spk_id)
    elif args.mode == "zero_shot":
        data.update(prompt_text=args.prompt_text)
        files = [("prompt_wav", ("prompt_wav", open(args.prompt_wav, "rb"),
                                  "application/octet-stream"))]
    elif args.mode == "cross_lingual":
        files = [("prompt_wav", ("prompt_wav", open(args.prompt_wav, "rb"),
                                  "application/octet-stream"))]
    elif args.mode == "instruct":
        data.update(spk_id=args.spk_id, instruct_text=args.instruct_text)
    else:  # instruct2
        data.update(instruct_text=args.instruct_text)
        files = [("prompt_wav", ("prompt_wav", open(args.prompt_wav, "rb"),
                                  "application/octet-stream"))]

    r = requests.request("GET", url, data=data, files=files, stream=True, timeout=600)
    r.raise_for_status()
    pcm = b"".join(r.iter_content(chunk_size=16000))
    with wave.open(args.tts_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(args.sample_rate)
        w.writeframes(pcm)
    dur = len(pcm) / 2.0 / args.sample_rate
    print(f"mode={args.mode} pcm={len(pcm)} sr={args.sample_rate} audio={dur:.2f}s -> {args.tts_wav}")


if __name__ == "__main__":
    main()
