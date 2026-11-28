#!/usr/bin/env bash
# Create/update v3 speaker_info for zero_shot_by_id.
# The model repository does not ship speaker_info; this script copies a reference
# wav into <model>/speaker_info and adds/updates one speaker entry in JSON.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=""
SPK_DIR=""
SPK_ID="spk1"
PROMPT_WAV=""
PROMPT_TEXT="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
FORCE=0

usage() {
    cat <<'USAGE'
Create/update v3 speaker_info for zero_shot_by_id.

Usage:
  bash create_speaker_info.sh --prompt-wav /data/prompt.wav [OPTIONS]

Options:
  --prompt-wav PATH       reference audio (required; 3-10s, single speaker, clean)
  --prompt-text TEXT      exact transcript; must contain <|endofprompt|>
                          (default: "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
  --spk-id ID             speaker id in speaker_info.json (default: spk1)
  --model-dir DIR         v3 model dir (default: first existing of
                          cosyvoice_models/Fun-CosyVoice3-0.5B,
                          streaming/cosyvoice_models/Fun-CosyVoice3-0.5B,
                          offline/cosyvoice_models/Fun-CosyVoice3-0.5B)
  --speaker-info-dir DIR  output dir (default: <model-dir>/speaker_info)
  --force                 overwrite an existing speaker id / prompt file
  -h, --help              show this help

Examples:
  bash create_speaker_info.sh --prompt-wav /data/prompt.wav
  bash create_speaker_info.sh --model-dir /data/cosyvoice_models/Fun-CosyVoice3-0.5B \
      --spk-id spk1 --prompt-wav /data/prompt.wav --force

Note: this is v3-only. v2 has no zero_shot_by_id/speaker_info; use prompt_wav +
prompt_text in the v2 zero_shot API instead.
USAGE
}

die() { echo "error: $*" >&2; exit 2; }
require_value() { [ $# -ge 2 ] || die "$1 requires a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir) require_value "$@"; MODEL_DIR=$2; shift 2 ;;
        --speaker-info-dir) require_value "$@"; SPK_DIR=$2; shift 2 ;;
        --spk-id) require_value "$@"; SPK_ID=$2; shift 2 ;;
        --prompt-wav) require_value "$@"; PROMPT_WAV=$2; shift 2 ;;
        --prompt-text) require_value "$@"; PROMPT_TEXT=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

if [ -z "$MODEL_DIR" ]; then
    for d in \
        "$HERE/cosyvoice_models/Fun-CosyVoice3-0.5B" \
        "$HERE/streaming/cosyvoice_models/Fun-CosyVoice3-0.5B" \
        "$HERE/offline/cosyvoice_models/Fun-CosyVoice3-0.5B"; do
        if [ -d "$d" ]; then MODEL_DIR="$d"; break; fi
    done
fi
[ -n "$MODEL_DIR" ] || die "v3 model dir not found; pass --model-dir (run download_weights.sh first)"
[ -d "$MODEL_DIR" ] || die "model dir not found: $MODEL_DIR"
[ -f "$MODEL_DIR/cosyvoice3.yaml" ] || die "not a v3 model dir (cosyvoice3.yaml missing): $MODEL_DIR"
[ -n "$PROMPT_WAV" ] || die "--prompt-wav is required"
[ -f "$PROMPT_WAV" ] || die "prompt wav not found: $PROMPT_WAV"
[[ "$PROMPT_TEXT" == *"<|endofprompt|>"* ]] || die "prompt text must contain <|endofprompt|> (v3 convention)"
case "$SPK_ID" in
    ''|*[!A-Za-z0-9_.-]*) die "invalid --spk-id '$SPK_ID' (use letters, digits, . _ -)" ;;
esac

if [ -z "$SPK_DIR" ]; then
    SPK_DIR="$MODEL_DIR/speaker_info"
fi
mkdir -p "$SPK_DIR"

if [ "$SPK_ID" = "spk1" ]; then
    PROMPT_NAME="prompt.wav"
else
    PROMPT_NAME="prompt_${SPK_ID}.wav"
fi
SRC=$(realpath "$PROMPT_WAV")
DST=$(realpath -m "$SPK_DIR/$PROMPT_NAME")
if [ "$SRC" != "$DST" ]; then
    cp -f "$PROMPT_WAV" "$SPK_DIR/$PROMPT_NAME"
fi

python3 - "$SPK_DIR/speaker_info.json" "$SPK_ID" "$PROMPT_NAME" "$PROMPT_TEXT" "$FORCE" <<'PY'
import json
import os
import sys

path, spk_id, prompt_name, prompt_text, force = sys.argv[1:]
data = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
if spk_id in data and force != "1":
    raise SystemExit(f"error: speaker '{spk_id}' already exists in {path}; use --force to overwrite")
data[spk_id] = {"prompt_wav": prompt_name, "prompt_text": prompt_text}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[create_speaker_info] speaker '{spk_id}' -> {path}")
print(f"[create_speaker_info] prompt  -> {os.path.join(os.path.dirname(path), prompt_name)}")
PY

echo "[create_speaker_info] done. Use: --mode zero_shot_by_id --spk-id $SPK_ID"
