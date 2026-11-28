#!/usr/bin/env bash
# Create (and optionally enter) the CosyVoice Ascend container.
# Run this script from the extracted release root under <DATA_DIR>.
# <DATA_DIR> is mounted to /data inside the container; the release is visible at
# the same path under /data (e.g. host /data/cosyvoice -> container /data/cosyvoice).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CONTAINER_NAME="${CONTAINER_NAME:-cosyvoice}"
IMAGE="${IMAGE:-tritonserver-ascend:910b-v3}"
DATA_DIR="${DATA_DIR:-/data}"
WORKDIR="${WORKDIR:-}"
FORCE=0

usage() {
    cat <<'USAGE'
Create the CosyVoice Ascend container.

Usage:
  bash create_container.sh [--name NAME] [--image IMAGE] [--force]

Defaults:
  name  = cosyvoice
  image = tritonserver-ascend:910b-v3
          (A3 / 910C: tritonserver-ascend:910c-v3)

Mounts:
  DATA_DIR (default /data) -> /data
  /usr/local/Ascend/driver -> /usr/local/Ascend/driver
Workdir:
  /data/<release-relative-path> (default for /data/cosyvoice: /data/cosyvoice)

Extract the release under DATA_DIR first, e.g.:
  mkdir -p /data/cosyvoice
  tar -xzf cosyvoice.tar.gz -C /data/cosyvoice --strip-components=1
  cd /data/cosyvoice && bash create_container.sh

Change --name if a container with the same name already exists. For multiple
cards, edit the --device list in this script and start one service per card.
USAGE
}

die() { echo "error: $*" >&2; exit 2; }
require_value() { [ $# -ge 2 ] || die "$1 requires a value"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --name) require_value "$@"; CONTAINER_NAME=$2; shift 2 ;;
        --image) require_value "$@"; IMAGE=$2; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1 (use --help)" ;;
    esac
done

command -v docker >/dev/null 2>&1 || die "docker not found"
mkdir -p "$DATA_DIR"
if [ -z "$WORKDIR" ]; then
    case "$HERE" in
        "$DATA_DIR"/*) WORKDIR="/data/${HERE#"$DATA_DIR"/}" ;;
        *) WORKDIR="/data/$(basename "$HERE")" ;;
    esac
fi
case "$HERE" in
    "$DATA_DIR"/*) : ;;
    *) echo "[create_container] warning: release is not under DATA_DIR=$DATA_DIR;" >&2
       echo "                  the container will not see $HERE unless you copy it there." >&2 ;;
esac
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "error: image not found: $IMAGE" >&2
    echo "       get it from https://gitcode.com/Ascend-SACT/tritonserver-ascend" >&2
    echo "       (merge shards + docker load; see cosyvoice_deploy.md section 2.2)" >&2
    exit 2
}
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    if [ "$FORCE" = "1" ]; then
        echo "[create_container] removing existing container: $CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" >/dev/null
    else
        die "container already exists: $CONTAINER_NAME (use --name NAME or --force)"
    fi
fi

echo "[create_container] name=$CONTAINER_NAME image=$IMAGE"
echo "[create_container] data=$DATA_DIR -> /data"
echo "[create_container] workdir=$WORKDIR"

cid=$(docker run -itd --name "$CONTAINER_NAME" \
    --network host --ipc=host \
    --device /dev/davinci0 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v "$DATA_DIR:/data" \
    -w "$WORKDIR" \
    "$IMAGE" bash)
echo "[create_container] started: $cid"

if [ -t 0 ] && [ -t 1 ]; then
    exec docker exec -it "$CONTAINER_NAME" bash
else
    echo "[create_container] enter with: docker exec -it $CONTAINER_NAME bash"
fi
