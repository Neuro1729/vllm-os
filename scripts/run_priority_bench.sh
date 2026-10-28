#!/usr/bin/env bash
# Runs the policy comparison, one engine process per configuration.
#
#   GPU=3 scripts/run_priority_bench.sh [normal|batch-cap|no-priority|kv-pressure|all]
#
# Each run is wrapped in `timeout` (SIGTERM, then SIGKILL after a grace period). The benchmark
# turns SIGTERM into a normal exit so the engine tears down NCCL/CUDA cleanly. Ctrl-C stops the
# current run the same way and skips the rest.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:?set GPU to the device index to use}
PYTHON=${PYTHON:-.venv/bin/python}
SCENARIO=${1:-all}
OUT=${OUT:-results/priority_bench.jsonl}
RUN_TIMEOUT=${RUN_TIMEOUT:-900}
DURATION=${DURATION:-60}
SWAP_GB=${SWAP_GB:-4}

export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
child=
trap 'echo "interrupted, stopping current run"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait; exit 130' INT TERM

run() {
    local label=$1; shift
    echo "=== $label"
    timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" \
        "$PYTHON" -m benchmarks.priority_bench --label "$label" --out "$OUT" --duration "$DURATION" "$@" &
    child=$!
    wait "$child"
    local rc=$?
    child=
    [ $rc -ne 0 ] && echo "run $label exited with $rc"
}

# the recommended MLFQ setup (docs/SCHEDULER.md)
MLFQ=(--policy mlfq --preemption-mode auto --swap-space "$SWAP_GB" --swap-min-tokens 128
      --mlfq-level-tokens 256,512,1024 --aging-steps 200 --min-resident-steps 4
      --preempt-after-steps 8)

policies() {
    local scenario=$1; shift
    run "$scenario/fcfs" --policy fcfs "$@"
    run "$scenario/priority-recompute" --policy priority --preemption-mode recompute "$@"
    run "$scenario/priority-swap" --policy priority --preemption-mode swap --swap-space "$SWAP_GB" "$@"
    run "$scenario/mlfq" "${MLFQ[@]}" "$@"
}

want() { [[ $SCENARIO == "$1" || $SCENARIO == all ]]; }
# light load, the batch never fills: every policy should match fcfs
want normal && policies normal --max-num-seqs 32 --num-kvcache-blocks 512 --num-agents 4 --chat-rate 1
# batch slots are the bottleneck (vllm#40004): 8 slots, 12 agents, plenty of KV
want batch-cap && policies batch-cap --max-num-seqs 8 --num-kvcache-blocks 512
# same, but chat and agents share one priority: only MLFQ's demotion can tell them apart
want no-priority && policies no-priority --max-num-seqs 8 --num-kvcache-blocks 512 --uniform-priority
# KV cache is the bottleneck but slots are free: no admission preemption, should track fcfs
want kv-pressure && policies kv-pressure --max-num-seqs 64 --num-kvcache-blocks 40
exit 0
