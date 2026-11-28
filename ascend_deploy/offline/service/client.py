#!/usr/bin/env python3
"""Client for the CosyVoice offline (non-streaming) service.

POST /synthesize (multipart) -> complete WAV in one response.
"""
import argparse

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9500")
    ap.add_argument("--text", default="你好，这是离线下沉服务测试。")
    ap.add_argument("--mode", default="", help="v3: zero_shot_by_id|zero_shot; v2: zero_shot|sft|cross_lingual|instruct|instruct2")
    ap.add_argument("--spk_id", default="spk1")
    ap.add_argument("--prompt_text", default="")
    ap.add_argument("--instruct_text", default="")
    ap.add_argument("--prompt_wav", default="")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="offline_service_out.wav")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    data = {"text": args.text, "mode": args.mode, "spk_id": args.spk_id,
            "prompt_text": args.prompt_text, "instruct_text": args.instruct_text,
            "speed": str(args.speed)}
    files = None
    if args.prompt_wav:
        files = {"prompt_wav": ("prompt_wav", open(args.prompt_wav, "rb"),
                                "application/octet-stream")}

    r = requests.post(args.url.rstrip("/") + "/synthesize", data=data, files=files,
                      timeout=args.timeout)
    if r.status_code != 200:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:500]}")
    with open(args.out, "wb") as f:
        f.write(r.content)
    h = r.headers
    print(f"ok bytes={len(r.content)} type={h.get('X-Model-Type')} mode={h.get('X-Mode')} "
          f"sr={h.get('X-Sample-Rate')} audio={h.get('X-Audio-Seconds')}s "
          f"rtf={h.get('X-Rtf')} -> {args.out}")


if __name__ == "__main__":
    main()
