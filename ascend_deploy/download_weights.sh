#!/usr/bin/env bash
# Download CosyVoice weights into a local directory (NOT under /opt/CosyVoice)
# and create the relative service symlinks. All options are CLI flags:
# run `bash download_weights.sh --help`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="$HERE/cosyvoice_models"
FORCE=0
MODELS="CosyVoice2-0.5B=iic/CosyVoice2-0.5B,Fun-CosyVoice3-0.5B=FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
LINK_DIRS=("$HERE/streaming" "$HERE/offline")

usage() {
    cat <<'EOF'
Download CosyVoice weights (default: CosyVoice2-0.5B + Fun-CosyVoice3-0.5B)
into a local directory (default: <repo>/cosyvoice_models, NOT /opt/CosyVoice),
then create relative cosyvoice_models symlinks for services/deploy dirs.

Usage:
  bash download_weights.sh [OPTIONS] [LINK_DIR ...]

Options:
  --dest DIR              weight root (default: <repo>/cosyvoice_models)
  --models SPEC           comma-separated DIR=REPO pairs
                          (default: CosyVoice2-0.5B=iic/CosyVoice2-0.5B,
                          Fun-CosyVoice3-0.5B=FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
  --force                 re-download even if a model dir already exists
  -h, --help              show this help

Positional LINK_DIR: create LINK_DIR/cosyvoice_models -> --dest as a RELATIVE
symlink, IN ADDITION to the default streaming/ and offline/ links. Existing
links are skipped.

Examples:
  bash download_weights.sh
  bash download_weights.sh --force
  bash download_weights.sh --dest /data/cosyvoice_models ../cosyvoice-src

Generate v3 preset voices separately with:
  bash create_speaker_info.sh --prompt-wav /data/prompt.wav
EOF
}

die() { echo "error: $*" >&2; exit 2; }
require_value() { [ $# -ge 2 ] || die "$1 requires a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --dest) require_value "$@"; DEST=$2; shift 2 ;;
        --models) require_value "$@"; MODELS=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1 (use --help)" ;;
        *) LINK_DIRS+=("$1"); shift ;;
    esac
done

mkdir -p "$DEST"
export DEST FORCE MODELS
echo "[download_weights] dest=$DEST force=$FORCE"
python3 - <<'PY'
import os

dest = os.environ["DEST"]
force = os.environ["FORCE"] == "1"
pairs = []
for item in os.environ["MODELS"].split(","):
    item = item.strip()
    if not item:
        continue
    if "=" in item:
        d, r = item.split("=", 1)
    else:
        d, r = item.split("/")[-1], item
    pairs.append((d.strip(), r.strip()))

for d, r in pairs:
    target = os.path.join(dest, d)
    if os.path.isdir(target) and os.listdir(target) and not force:
        print(f"[skip] {d} already exists ({target}); use --force to re-download")
        continue
    print(f"[download] {r} -> {target}")
    from modelscope import snapshot_download
    snapshot_download(r, local_dir=target)
print("[download_weights] done")
PY

# Create relative symlinks to DEST for services/deploy dirs.
for d in "${LINK_DIRS[@]}"; do
    mkdir -p "$d"
    link="$d/cosyvoice_models"
    if [ -e "$link" ] || [ -L "$link" ]; then
        echo "[skip] $link exists -> $(readlink "$link" 2>/dev/null || echo '<real dir>')"
        continue
    fi
    ln -sr "$DEST" "$link"
    echo "[download_weights] linked $link -> $(readlink "$link")"
done
