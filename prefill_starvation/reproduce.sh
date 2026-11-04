#!/usr/bin/env bash
# Reproduce the vLLM prefill-starvation study end to end.
#
# Run from anywhere inside this branch (repo root = vLLM v0.32.0 source + scheduler trace patch):
#   prefill_starvation/reproduce.sh setup      # venv: editable vLLM 0.32.0 (+ precompiled cu129 kernels), chart venv, model config
#   prefill_starvation/reproduce.sh data       # download public traces (Mooncake, Azure)
#   prefill_starvation/reproduce.sh validate   # CPU only: simulator vs REAL vLLM Scheduler, step by step (expects 28/28 OK)
#   GPU=1 prefill_starvation/reproduce.sh gpu  # one GPU: start server (priority policy), run every scenario, analyze
#   prefill_starvation/reproduce.sh charts     # PNGs in results/charts/
#   GPU=1 prefill_starvation/reproduce.sh all
#
# Requirements: NVIDIA driver >= 575 (CUDA 12.9), one GPU with >= 24 GB, uv, curl, rsync.
# Safety: every job runs under tools/safe_run.sh (RAM cap, nice, pinned CPUs, graceful timeout,
# GPU health check). Only the GPU in $GPU is used; the server is stopped with SIGTERM at the end.
set -euo pipefail
PS=$(cd "$(dirname "$0")" && pwd)          # prefill_starvation/
ROOT=$(cd "$PS/.." && pwd)                  # vLLM source root of this branch
GPU=${GPU:-0}
PORT=${PORT:-8011}
TAG=${TAG:-prio2048}
VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
QWEN=Qwen/Qwen2.5-1.5B-Instruct

setup() {
  cd "$PS"
  mkdir -p wheels models/qwen2.5-1.5b-instruct logs/gpu results/gpu results/charts
  W=wheels/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
  [ -f "$W" ] || python3 -I tools/pdl.py "$VLLM_WHEEL_URL" "$W" --conns 32
  [ -d .venv ] || uv venv -q --python 3.12 .venv
  # editable install of THIS branch's source; compiled kernels come from the official wheel
  (cd "$ROOT" && VIRTUAL_ENV="$PS/.venv" VLLM_USE_PRECOMPILED=1 \
     VLLM_PRECOMPILED_WHEEL_LOCATION="$PS/$W" uv pip install -e . --torch-backend=cu129)
  # PyPI torchcodec 0.17 is built for CUDA 13; the CPU build is enough for vLLM's text path
  VIRTUAL_ENV="$PS/.venv" uv pip install -q "torchcodec==0.17.0+cpu" \
     --index-url https://download.pytorch.org/whl/cpu --reinstall-package torchcodec
  [ -d .venv-charts ] || { uv venv -q --python 3.12 .venv-charts; \
     VIRTUAL_ENV="$PS/.venv-charts" uv pip install -q matplotlib numpy; }
  # model: config + tokenizer only (served with --load-format dummy => random weights)
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "models/qwen2.5-1.5b-instruct/$f" ] || curl -sSfL -o "models/qwen2.5-1.5b-instruct/$f" \
      "https://huggingface.co/$QWEN/resolve/main/$f"
  done
  CUDA_VISIBLE_DEVICES= .venv/bin/python -c "import vllm, vllm.v1.core.sched.scheduler as s; print('vllm', vllm.__version__, 'scheduler from', s.__file__)"
}

data() { "$PS/data/download.sh"; ls -la "$PS/data/mooncake" "$PS/data/azure"; }

validate() {
  cd "$PS"
  CUDA_VISIBLE_DEVICES= tools/safe_run.sh -m 4G -c 0-1 -t 1200 -- .venv/bin/python validate_vllm.py \
    | tee logs/validate.log | grep -E "^\[(OK|MISMATCH)"
  echo "OK scenarios: $(grep -c '^\[OK' logs/validate.log) (expect 28), mismatches: $(grep -c MISMATCH logs/validate.log || true)"
}

wait_healthy() {
  for _ in $(seq 1 200); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0
    grep -q "exit=" "$PS/logs/gpu/server_$TAG.log" 2>/dev/null && { echo "server exited"; tail -20 "$PS/logs/gpu/server_$TAG.log"; return 1; }
    sleep 3
  done
  echo "server not healthy after 10 min"; return 1
}

stop_server() {
  for p in $(pgrep -f "bin/vllm serve $PS/models" || true); do kill -TERM "$p" 2>/dev/null || true; done
  for _ in $(seq 1 60); do pgrep -f "bin/vllm serve $PS/models" >/dev/null || break; sleep 2; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU" || true
}

gpu() {
  cd "$PS"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv -i "$GPU"
  python3 -I gpu/build_micro.py gpu/workloads/micro
  CUDA_VISIBLE_DEVICES= .venv/bin/python gpu/build_workloads.py gpu/workloads/v3 --dur 450 --mc-speedup 0.2
  # server: priority policy, budget 2048, chunk cap 512 + adaptive (the code path under study)
  gpu/serve.sh "$TAG" priority 2048 512 1 "$GPU" "$PORT" > "logs/gpu/server_$TAG.log" 2>&1 &
  trap stop_server EXIT
  wait_healthy
  rm -f "results/gpu/SUMMARY_$TAG.txt"
  # micro scenarios: 20-token probes vs long 30K prompts, with 600 or 50 background decodes
  for c in decodeheavy_base decodeheavy decodeheavy_inv decodeheavy_lowprobe \
           manyprefill_base manyprefill manyprefill_inv manyprefill_lowprobe lowprio_stream; do
    gpu/run_and_analyze.sh "$c" "$TAG" --priority | tee -a "results/gpu/SUMMARY_$TAG.txt"
  done
  # real trace (Mooncake conversation), ~50% prefill load on an RTX 3090
  for c in S2base S2 S3 S3low; do
    WL_SET=v3 gpu/run_dataset.sh "$c" "$TAG" --priority | tee -a "results/gpu/SUMMARY_$TAG.txt"
  done
  gzip -kf "logs/gpu/trace_$TAG.jsonl" && mv -f "logs/gpu/trace_$TAG.jsonl.gz" "results/gpu/"
}

charts() { cd "$PS"; .venv-charts/bin/python charts.py; }

case "${1:-}" in
  setup) setup ;; data) data ;; validate) validate ;; gpu) gpu ;; charts) charts ;;
  all) setup; data; validate; gpu; charts ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
