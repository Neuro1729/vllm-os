#!/usr/bin/env bash
# Start a vLLM 0.32.0 server for the prefill-starvation experiments (one GPU, RAM-capped).
# usage: gpu/serve.sh <tag> [fcfs|priority] [budget] [threshold] [adaptive:0|1] [gpu] [port]
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
TAG=$1; POLICY=${2:-fcfs}; BUDGET=${3:-2048}; THR=${4:-512}; ADAPT=${5:-1}; GPU=${6:-1}; PORT=${7:-8011}
mkdir -p "$HERE/logs/gpu"
ADAPT_FLAG=""; [ "$ADAPT" = "1" ] && ADAPT_FLAG="--long-prefill-token-threshold-adaptive"
export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID \
       HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_SCHED_TRACE="$HERE/logs/gpu/trace_$TAG.jsonl"
rm -f "$VLLM_SCHED_TRACE"
exec "$HERE/tools/safe_run.sh" -m 20G -c 0-7 -t 14400 -a 24 -- \
  "$HERE/.venv/bin/vllm" serve "$HERE/models/qwen2.5-1.5b-instruct" \
    --served-model-name m --load-format dummy --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.85 \
    --max-num-seqs 1024 --max-num-batched-tokens "$BUDGET" \
    --long-prefill-token-threshold "$THR" $ADAPT_FLAG \
    --scheduling-policy "$POLICY" --no-enable-prefix-caching \
    --port "$PORT" --host 127.0.0.1
