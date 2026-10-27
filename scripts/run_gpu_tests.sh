#!/usr/bin/env bash
# Run only the GPU correctness tests (tests/gpu). No benchmarks.
#
# Usage: bash scripts/run_gpu_tests.sh [--gpu N] [--model PATH] [--policy fcfs|mlfq]
#                                      [--gpu-memory-utilization F] [--skip-setup] [--help]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GPU=0
MODEL="${NANOVLLM_TEST_MODEL:-$HOME/huggingface/Qwen3-0.6B}"
POLICY="mlfq"
UTIL="0.5"
SKIP_SETUP=0

usage() {
    cat <<EOF
Usage: bash scripts/run_gpu_tests.sh [options]

  --gpu N                    physical GPU index (default 0). Exported as CUDA_VISIBLE_DEVICES.
  --model PATH               local model directory (default \$NANOVLLM_TEST_MODEL or ~/huggingface/Qwen3-0.6B)
  --policy fcfs|mlfq         scheduler policy under test (default mlfq; mlfq enables preemption)
  --gpu-memory-utilization F fraction of GPU memory nano-vLLM may use, 0 < F <= 1 (default 0.5)
  --skip-setup               do not run setup_gpu_env.sh --check-only first
  --help                     show this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) [ $# -ge 2 ] || die "--gpu needs a value"; GPU="$2"; shift ;;
        --model) [ $# -ge 2 ] || die "--model needs a value"; MODEL="$2"; shift ;;
        --policy) [ $# -ge 2 ] || die "--policy needs a value"; POLICY="$2"; shift ;;
        --gpu-memory-utilization) [ $# -ge 2 ] || die "--gpu-memory-utilization needs a value"; UTIL="$2"; shift ;;
        --skip-setup) SKIP_SETUP=1 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

case "$GPU" in ''|*[!0-9]*) die "--gpu must be a non-negative integer" ;; esac
case "$POLICY" in fcfs|mlfq) ;; *) die "--policy must be fcfs or mlfq" ;; esac
awk -v u="$UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }' || die "--gpu-memory-utilization must be in (0, 1]"

if [ "$SKIP_SETUP" -eq 0 ]; then
    bash "$REPO_ROOT/scripts/setup_gpu_env.sh" --check-only
fi
require_venv

if [ -z "$(gpu_query "$GPU" || true)" ]; then
    die "GPU $GPU is not visible to nvidia-smi"
fi
log "GPU $GPU: $(gpu_query "$GPU")"

export CUDA_VISIBLE_DEVICES="$GPU"
export NANOVLLM_TEST_MODEL="$MODEL"
export NANOVLLM_TEST_POLICY="$POLICY"
export NANOVLLM_TEST_GPU_UTIL="$UTIL"

cd "$REPO_ROOT"
log "running GPU tests: policy=$POLICY model=$MODEL"
"$(venv_python)" -m pytest tests/gpu -v
