#!/usr/bin/env bash
# Build the GPU image (once) and run the one-command pipeline inside Docker.
#
# The image carries the pinned CUDA/PyTorch/flash-attn stack, so the host needs only Docker,
# the NVIDIA driver, and the NVIDIA Container Toolkit. Nothing is installed on the host.
#
# Usage:
#   bash scripts/run_docker_experiment.sh --model /path/to/Qwen3-0.6B --max-num-seqs 8
#   bash scripts/run_docker_experiment.sh --model /path/to/Qwen3-0.6B --max-num-seqs-list 4,8,16 --profile standard
#   bash scripts/run_docker_experiment.sh --build-only
#   bash scripts/run_docker_experiment.sh --model /path/to/Qwen3-0.6B --shell
#
# Any option not listed below is passed to scripts/run_full_experiment.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nanovllm-mlfq"
GPU=0
MODEL=""
MAX_JOBS=4
REBUILD=0
BUILD_ONLY=0
SHELL_ONLY=0
PASSTHROUGH=()

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[docker] %s\n' "$*"; }

usage() {
    cat <<'EOF'
Usage: bash scripts/run_docker_experiment.sh [wrapper options] [pipeline options]

Wrapper options (consumed here):
  --model PATH       local model directory on the host (required except with --build-only)
  --gpu N            host GPU index exposed to the container (default 0)
  --image NAME       image tag (default nanovllm-mlfq)
  --rebuild          rebuild the image even if it exists
  --max-jobs N       parallel compile jobs for flash-attn during the build (default 4)
  --build-only       build the image and exit
  --shell            open an interactive shell in the container (for debugging)
  --help             show this message

Pipeline options (passed through to scripts/run_full_experiment.sh), for example:
  --max-num-seqs 8 | --max-num-seqs-list 4,8,16
  --profile smoke|standard|thorough   --repeats N   --warmup N   --seed N
  --skip-baseline-v0   --continue-on-test-failure   --dry-run   --output-dir DIR

Results appear in ./results/ on the host.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model) [ $# -ge 2 ] || die "--model needs a value"; MODEL="$2"; shift ;;
        --gpu) [ $# -ge 2 ] || die "--gpu needs a value"; GPU="$2"; shift ;;
        --image) [ $# -ge 2 ] || die "--image needs a value"; IMAGE="$2"; shift ;;
        --max-jobs) [ $# -ge 2 ] || die "--max-jobs needs a value"; MAX_JOBS="$2"; shift ;;
        --rebuild) REBUILD=1 ;;
        --build-only) BUILD_ONLY=1 ;;
        --shell) SHELL_ONLY=1 ;;
        --help|-h) usage; exit 0 ;;
        *) PASSTHROUGH+=("$1") ;;
    esac
    shift
done

case "$GPU" in ''|*[!0-9]*) die "--gpu must be a non-negative integer" ;; esac
case "$MAX_JOBS" in ''|*[!0-9]*) die "--max-jobs must be a positive integer" ;; esac

# ---- host checks ---------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker is not installed. Install Docker Engine (or Docker Desktop with WSL2) first."
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon. Start it, or add your user to the docker group."

# Pipeline options that only make sense on the host or are redundant inside the container.
for arg in "${PASSTHROUGH[@]}"; do
    case "$arg" in
        --model|--gpu|--skip-setup)
            die "$arg is handled by the wrapper or always applied inside the container; do not pass it" ;;
    esac
done

if [ "$BUILD_ONLY" -eq 0 ] && [ -z "$MODEL" ]; then
    dry_run=0
    for arg in ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; do [ "$arg" = "--dry-run" ] && dry_run=1; done
    [ "$dry_run" -eq 1 ] || die "--model PATH is required: point it at a local Qwen3-0.6B directory"
fi

# ---- build -------------------------------------------------------------------
if [ "$REBUILD" -eq 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "building image $IMAGE (flash-attn compiles from source; this can take a long time)"
    docker build --build-arg "MAX_JOBS=$MAX_JOBS" -t "$IMAGE" "$REPO_ROOT"
else
    log "using existing image $IMAGE (pass --rebuild after changing the Dockerfile or requirements)"
fi

if [ "$BUILD_ONLY" -eq 1 ]; then
    log "image ready: $IMAGE"
    exit 0
fi

# ---- GPU access --------------------------------------------------------------
if ! docker run --rm --gpus "device=$GPU" "$IMAGE" nvidia-smi -L >/dev/null 2>&1; then
    die "the container cannot see GPU $GPU. Install the NVIDIA Container Toolkit and check 'docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi'."
fi
log "GPU $GPU is visible inside the container"

# ---- mounts ------------------------------------------------------------------
mkdir -p "$REPO_ROOT/results"
MOUNTS=(-v "$REPO_ROOT/results:/workspace/results")
MODEL_IN_CONTAINER="Qwen/Qwen3-0.6B"
if [ -n "$MODEL" ]; then
    [ -d "$MODEL" ] || die "--model must be a local directory; download the model first (a Hugging Face id cannot be mounted)"
    MODEL_ABS="$(cd "$MODEL" && pwd)"
    MODEL_NAME="$(basename "$MODEL_ABS")"
    MOUNTS+=(-v "$MODEL_ABS:/models/$MODEL_NAME:ro")
    MODEL_IN_CONTAINER="/models/$MODEL_NAME"
fi

if [ "$SHELL_ONLY" -eq 1 ]; then
    exec docker run --rm -it --gpus "device=$GPU" --ipc=host --shm-size=16g "${MOUNTS[@]}" "$IMAGE" bash
fi

# ---- run ---------------------------------------------------------------------
# Inside the container the selected GPU is index 0. --skip-setup: the image already has the stack.
# Results are chowned back to the host user afterwards so they are not root-owned on the host.
log "running the pipeline in the container"
set +e
docker run --rm --gpus "device=$GPU" --ipc=host --shm-size=16g \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    "${MOUNTS[@]}" -w /workspace "$IMAGE" \
    bash -c 'bash scripts/run_full_experiment.sh --skip-setup --gpu 0 --model "$0" "${@:1}"; rc=$?; chown -R "$HOST_UID:$HOST_GID" /workspace/results 2>/dev/null || true; exit $rc' \
    "$MODEL_IN_CONTAINER" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    log "finished. Results are under $REPO_ROOT/results/ (newest experiment_* directory)."
else
    log "pipeline exited with status $rc. Logs and partial results are under $REPO_ROOT/results/."
fi
exit "$rc"
