#!/usr/bin/env bash
# fcfs, priority-swap and MLFQ v2 with mixed batching, several seeds, one engine process per
# run (docs/SCHEDULER.md, "Mixed batching"). The same configurations without mixed batching
# come from run_mlfq_v2_bench.sh (mixed_batching=False leaves the scheduler unchanged).
#
#   GPU=3 scripts/run_mixed_bench.sh [normal|all-short|batch-cap|no-priority|kv-pressure|all]
#
# Same safety rules as run_priority_bench.sh: every run under `timeout`, one at a time.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:?set GPU to the device index to use}
PYTHON=${PYTHON:-.venv/bin/python}
SCENARIO=${1:-all}
OUT=${OUT:-results/mixed_bench.jsonl}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
DURATION=${DURATION:-45}
SWAP_GB=${SWAP_GB:-4}
SEEDS=${SEEDS:-"0 1 2"}

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

V2=(--policy mlfq --preemption-mode auto --swap-space "$SWAP_GB" --swap-min-tokens 128
    --mlfq-level-tokens 256,512,1024 --aging-steps 200 --min-resident-steps 4 --preempt-after-steps 0
    --pause-victims --mlfq-preempt-max-level 0 --preempt-min-remaining-tokens 16)
PRIO=(--policy priority --preemption-mode swap --swap-space "$SWAP_GB")

configs() {
    local scenario=$1 seed=$2; shift 2
    run "$scenario/fcfs+mixed/s$seed" --seed "$seed" --policy fcfs --mixed-batching "$@"
    run "$scenario/priority-swap+mixed/s$seed" --seed "$seed" "${PRIO[@]}" --mixed-batching "$@"
    run "$scenario/mlfq-v2+mixed/s$seed" --seed "$seed" "${V2[@]}" --mixed-batching "$@"
}

want() { [[ $SCENARIO == "$1" || $SCENARIO == all ]]; }
for seed in $SEEDS; do
    want normal && configs normal "$seed" --max-num-seqs 32 --num-kvcache-blocks 512 --num-agents 4 --chat-rate 1
    want all-short && configs all-short "$seed" --max-num-seqs 8 --num-kvcache-blocks 512 --num-agents 0 --chat-rate 40 --uniform-priority
    want batch-cap && configs batch-cap "$seed" --max-num-seqs 8 --num-kvcache-blocks 512
    want no-priority && configs no-priority "$seed" --max-num-seqs 8 --num-kvcache-blocks 512 --uniform-priority
    want kv-pressure && configs kv-pressure "$seed" --max-num-seqs 64 --num-kvcache-blocks 40
done
exit 0
