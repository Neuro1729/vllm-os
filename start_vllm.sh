#!/usr/bin/env bash
set -euo pipefail
# Optional helper. If vLLM is already running, do not use this script.
# Select a CUDA device with CUDA_VISIBLE_DEVICES before invoking.
#
# Default = stock priority scheduling with a 2048-token step budget: P0 prefills can
# saturate the budget, so a P2 request waits (starvation).
# MITIGATE=1 adds the per-prefill 512-token cap + adaptive floor, which lets small
# requests slip through at moderate load (use it for the "after" comparison).
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
EXTRA=()
if [ "${MITIGATE:-0}" = "1" ]; then
  EXTRA+=(--long-prefill-token-threshold 512 --long-prefill-token-threshold-adaptive)
fi
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype float16 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --scheduling-policy priority \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --no-enable-prefix-caching \
  "${EXTRA[@]}"
