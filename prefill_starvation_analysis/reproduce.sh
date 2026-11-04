#!/usr/bin/env bash
# Reproduce the low-priority prefill-starvation analysis.
#   ./reproduce.sh setup              venv with this branch's vLLM (editable, precompiled cu129
#                                     kernels) + the PR #65283 port, chart venv, model config
#   ./reproduce.sh cpu                CPU real-scheduler sweep        -> work/cpu_sweep.json
#   GPU=<idx> ./reproduce.sh gpu      GPU sweep, one server lifetime  -> work/analysis/
#   ./reproduce.sh process figures    -> results/processed/, figures/
#   ./reproduce.sh validate           scheduler validation (expects 28/28 OK)
# Requirements: uv, curl, git, NVIDIA driver >= 575 (CUDA 12.9), one GPU with >= 24 GB.
set -euo pipefail
A=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$A/.." && pwd); cd "$A"
WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
QWEN=Qwen/Qwen2.5-1.5B-Instruct
export WORK=${WORK:-$A/work}; mkdir -p "$WORK"

setup() {
  # The experiments ran with the PR #65283 port applied (disabled except for the pr65283 setting).
  if git -C "$ROOT" apply --check patches/pr65283_port_v0.32.diff 2>/dev/null; then
    git -C "$ROOT" apply patches/pr65283_port_v0.32.diff && echo "applied PR #65283 port"
  else echo "PR #65283 port already applied (or tree differs) - skipping"; fi
  mkdir -p wheels models/qwen2.5-1.5b-instruct
  W=wheels/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
  [ -f "$W" ] || curl -sSfL -o "$W" "$WHEEL_URL"
  [ -d .venv ] || uv venv -q --python 3.12 .venv
  (cd "$ROOT" && VIRTUAL_ENV="$A/.venv" VLLM_USE_PRECOMPILED=1 \
     VLLM_PRECOMPILED_WHEEL_LOCATION="$A/$W" uv pip install -e . --torch-backend=cu129)
  VIRTUAL_ENV="$A/.venv" uv pip install -q "torchcodec==0.17.0+cpu" aiohttp pytest \
     --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
     --reinstall-package torchcodec
  [ -d .venv-charts ] || { uv venv -q --python 3.12 .venv-charts; \
     VIRTUAL_ENV="$A/.venv-charts" uv pip install -q matplotlib numpy; }
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "models/qwen2.5-1.5b-instruct/$f" ] || curl -sSfL -o "models/qwen2.5-1.5b-instruct/$f" \
      "https://huggingface.co/$QWEN/resolve/main/$f"
  done
}
cpu() {
  CUDA_VISIBLE_DEVICES= scripts/safe_run.sh -m 4G -c 0-1 -t 3600 -- \
    .venv/bin/python scripts/cpu/cpu_sweep.py "$WORK/cpu_sweep.json"
}
gpu() { scripts/gpu/run_sweep.sh "${GPU:?set GPU=<nvidia-smi index>}"; }
process() { .venv-charts/bin/python scripts/process_results.py "$WORK/analysis" "$WORK/cpu_sweep.json"; }
figures() { .venv-charts/bin/python scripts/make_figures.py > /dev/null && ls figures; }
validate() {
  CUDA_VISIBLE_DEVICES= scripts/safe_run.sh -m 4G -c 0-1 -t 1200 -- \
    .venv/bin/python scripts/validation/validate_vllm.py | grep -E "^\[(OK|MISMATCH)"
}
[ $# -gt 0 ] || { sed -n 2,9p "$0"; exit 1; }
for step in "$@"; do "$step"; done
