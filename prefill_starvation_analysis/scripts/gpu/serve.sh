#!/usr/bin/env bash
# Start the vLLM 0.32.0 server used by the sweep (one GPU, priority policy, RAM-capped).
# The long-prefill cap and its adaptive mode are switched at runtime via $VLLM_SCHED_CTRL.
# usage: scripts/gpu/serve.sh <gpu_index> [port]
set -u
A=$(cd "$(dirname "$0")/../.." && pwd)
GPU=$1; PORT=${2:-8011}
VENV=${VENV:-$A/.venv}; MODEL=${MODEL:-$A/models/qwen2.5-1.5b-instruct}; WORK=${WORK:-$A/work}
mkdir -p "$WORK"
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 \
       VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SCHED_TRACE="$WORK/trace_sweep.jsonl"
exec "$A/scripts/safe_run.sh" -m 20G -c 0-7 -t 14400 -a 24 -- \
  "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name m --load-format dummy --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.80 \
    --max-num-seqs 1024 --max-num-batched-tokens 2048 \
    --long-prefill-token-threshold 512 --long-prefill-token-threshold-adaptive \
    --scheduling-policy priority --no-enable-prefix-caching \
    --port "$PORT" --host 127.0.0.1
