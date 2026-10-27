#!/usr/bin/env bash
# Create or reuse the project virtual environment and validate the GPU stack.
#
# Idempotent: a stamp file records the requirements hash, so a second run only re-validates.
# Never uses sudo, never installs NVIDIA drivers or CUDA toolkits, never modifies system Python.
#
# Usage: bash scripts/setup_gpu_env.sh [--check-only] [--help]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CHECK_ONLY=0
MIN_FREE_GB=8

usage() {
    cat <<EOF
Usage: bash scripts/setup_gpu_env.sh [--check-only] [--help]

  Creates $VENV_DIR (if missing), installs requirements-gpu.txt, and validates:
  PyTorch CUDA runtime, triton, flash-attn, transformers, and free disk space.

  --check-only  Validate an existing environment without installing anything.
  --help        Show this message.

Exit status is nonzero, with an actionable message, if setup or validation fails.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check-only) CHECK_ONLY=1 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

REQ_FILE="$REPO_ROOT/requirements-gpu.txt"
STAMP_FILE="$VENV_DIR/.requirements.sha256"
[ -f "$REQ_FILE" ] || die "missing $REQ_FILE"

check_disk() {
    local free_kb free_gb
    free_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
    free_gb=$((free_kb / 1024 / 1024))
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        die "only ${free_gb} GiB free under $REPO_ROOT; need at least ${MIN_FREE_GB} GiB for the venv and model"
    fi
    log "disk: ${free_gb} GiB free"
}

create_venv() {
    local py
    py="$(find_python)" || die "no Python 3.10-3.12 found. Install one (e.g. with your system package manager) and rerun."
    log "creating venv at $VENV_DIR using $py"
    "$py" -m venv "$VENV_DIR"
}

install_requirements() {
    local py req_hash
    py="$(venv_python)"
    req_hash="$(sha256sum "$REQ_FILE" | awk '{print $1}')"
    if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$req_hash" ]; then
        log "requirements unchanged since last install; skipping pip"
        return 0
    fi
    log "installing torch from the CUDA 12.4 wheel index"
    "$py" -m pip install --upgrade pip
    "$py" -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 \
        || die "torch install failed. Check network access and that the CUDA 12.4 wheel index is reachable."
    log "installing remaining requirements"
    grep -v -E '^(#|torch==|$)' "$REQ_FILE" | sed 's/[[:space:]]*#.*$//' | grep -v '^$' > "$VENV_DIR/.reqs-noncuda.txt"
    "$py" -m pip install -r "$VENV_DIR/.reqs-noncuda.txt" \
        || die "dependency install failed; see pip output above"
    log "installing flash-attn (prebuilt wheel preferred; source build can take a long time)"
    "$py" -m pip install "flash-attn==2.7.4.post1" --no-build-isolation \
        || die "flash-attn install failed. Install a wheel matching torch 2.6 + CUDA 12.x, or see docs/GPU_TESTING.md."
    printf '%s\n' "$req_hash" > "$STAMP_FILE"
}

validate_stack() {
    local py
    py="$(venv_python)"
    "$py" - <<'PY' || die "GPU stack validation failed (see messages above). See docs/GPU_TESTING.md."
import sys
import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False: the NVIDIA driver or CUDA wheel does not match this machine")
print(f"visible GPUs: {torch.cuda.device_count()}; device 0: {torch.cuda.get_device_name(0)}")
import triton
import flash_attn
import transformers
print(f"triton {triton.__version__}, flash-attn {flash_attn.__version__}, transformers {transformers.__version__}")
PY
}

check_disk
if [ "$CHECK_ONLY" -eq 0 ]; then
    [ -x "$(venv_python)" ] || create_venv
    install_requirements
else
    require_venv
fi
validate_stack
log "GPU environment is ready: $VENV_DIR"
