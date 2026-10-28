#!/usr/bin/env bash
# MLFQ v2 ablation: adds the improvements one at a time on top of the original MLFQ setup,
# several seeds each, one engine process per run (docs/SCHEDULER.md, "MLFQ v2").
#
#   GPU=3 V1_TREE=<checkout of the original MLFQ commit> scripts/run_mlfq_v2_bench.sh [scenario|all]
#
# "mlfq-v1" runs the original scheduler code from V1_TREE (e.g. `git worktree add --detach
# /tmp/v1 6688df7`); every later stage runs this checkout's code.
#
# Same safety rules as run_priority_bench.sh: every run under `timeout`, one at a time.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:?set GPU to the device index to use}
PYTHON=$(realpath -s "${PYTHON:-.venv/bin/python}")
SCENARIO=${1:-all}
OUT=${OUT:-results/mlfq_v2_bench.jsonl}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
DURATION=${DURATION:-45}
SWAP_GB=${SWAP_GB:-4}
SEEDS=${SEEDS:-"0 1 2"}
V1_TREE=${V1_TREE:?set V1_TREE to a checkout of the original MLFQ commit (6688df7)}
HERE=$PWD

export CUDA_VISIBLE_DEVICES=$GPU CUDA_DEVICE_ORDER=PCI_BUS_ID
child=
trap 'echo "interrupted, stopping current run"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait; exit 130' INT TERM

run() {    # [--tree DIR] label args...: DIR's nanovllm is imported (python -m puts the cwd first)
    local tree=$HERE
    [ "$1" == --tree ] && { tree=$2; shift 2; }
    local label=$1; shift
    echo "=== $label"
    (cd "$tree" && exec timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT" \
        "$PYTHON" -m benchmarks.priority_bench --label "$label" --out "$HERE/$OUT" --duration "$DURATION" "$@") &
    child=$!
    wait "$child"
    local rc=$?
    child=
    [ $rc -ne 0 ] && echo "run $label exited with $rc"
}

# original recommended MLFQ (run_priority_bench.sh), then each improvement on top
V1=(--policy mlfq --preemption-mode auto --swap-space "$SWAP_GB" --swap-min-tokens 128
    --mlfq-level-tokens 256,512,1024 --aging-steps 200 --min-resident-steps 4 --preempt-after-steps 8)
PAUSE=("${V1[@]}" --pause-victims)
SHORT=("${PAUSE[@]}" --mlfq-preempt-max-level 0)
NOWAIT=("${SHORT[@]}" --preempt-after-steps 0)
V2=("${NOWAIT[@]}" --preempt-min-remaining-tokens 16)

stages() {    # full ablation
    local scenario=$1 seed=$2; shift 2
    run "$scenario/fcfs/s$seed" --seed "$seed" --policy fcfs "$@"
    run "$scenario/priority-swap/s$seed" --seed "$seed" --policy priority --preemption-mode swap --swap-space "$SWAP_GB" "$@"
    run --tree "$V1_TREE" "$scenario/mlfq-v1/s$seed" --seed "$seed" "${V1[@]}" "$@"
    run "$scenario/+low-overhead/s$seed" --seed "$seed" "${V1[@]}" "$@"
    run "$scenario/+pause/s$seed" --seed "$seed" "${PAUSE[@]}" "$@"
    run "$scenario/+short-only/s$seed" --seed "$seed" "${SHORT[@]}" "$@"
    run "$scenario/+no-wait/s$seed" --seed "$seed" "${NOWAIT[@]}" "$@"
    run "$scenario/mlfq-v2/s$seed" --seed "$seed" "${V2[@]}" "$@"
}

ends() {    # fcfs vs. first and last MLFQ (and the overhead fix): scenarios where nothing should change
    local scenario=$1 seed=$2; shift 2
    run "$scenario/fcfs/s$seed" --seed "$seed" --policy fcfs "$@"
    run --tree "$V1_TREE" "$scenario/mlfq-v1/s$seed" --seed "$seed" "${V1[@]}" "$@"
    run "$scenario/+low-overhead/s$seed" --seed "$seed" "${V1[@]}" "$@"
    run "$scenario/mlfq-v2/s$seed" --seed "$seed" "${V2[@]}" "$@"
}

want() { [[ $SCENARIO == "$1" || $SCENARIO == all ]]; }
for seed in $SEEDS; do
    want normal && ends normal "$seed" --max-num-seqs 32 --num-kvcache-blocks 512 --num-agents 4 --chat-rate 1
    # only short chat, batch saturated: everything is level 0, so MLFQ must never preempt
    want all-short && ends all-short "$seed" --max-num-seqs 8 --num-kvcache-blocks 512 --num-agents 0 --chat-rate 40 --uniform-priority
    want batch-cap && stages batch-cap "$seed" --max-num-seqs 8 --num-kvcache-blocks 512
    want no-priority && stages no-priority "$seed" --max-num-seqs 8 --num-kvcache-blocks 512 --uniform-priority
    want kv-pressure && stages kv-pressure "$seed" --max-num-seqs 64 --num-kvcache-blocks 40
done
exit 0
