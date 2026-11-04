#!/usr/bin/env bash
# Replay one workload against the running server, then join it with the server trace.
# usage: scripts/gpu/run_case.sh <workload.json> <run_name> [port]
set -u
A=$(cd "$(dirname "$0")/../.." && pwd); W=$1; N=$2; PORT=${3:-8011}
VENV=${VENV:-$A/.venv}; WORK=${WORK:-$A/work}
mkdir -p "$WORK/client" "$WORK/analysis" "$WORK/logs"
CUDA_VISIBLE_DEVICES= "$A/scripts/safe_run.sh" -m 4G -c 8-11 -t 1200 -- \
  "$VENV/bin/python" "$A/scripts/gpu/replay_client.py" "$W" "$WORK/client/$N.jsonl" \
  --port "$PORT" --priority --drain-s 600 --stop-when-cls-done short \
  > "$WORK/logs/client_$N.log" 2>&1
CUDA_VISIBLE_DEVICES= "$A/scripts/safe_run.sh" -m 6G -c 8-9 -t 600 -- "$VENV/bin/python" \
  "$A/scripts/gpu/analyze.py" "$WORK/client/$N.jsonl" "$WORK/trace_sweep.jsonl" \
  "$WORK/analysis/$N.analysis.json" | grep -v safe_run
