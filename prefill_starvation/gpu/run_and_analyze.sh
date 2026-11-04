#!/usr/bin/env bash
# usage: gpu/run_and_analyze.sh <case> <trace_tag> [client args...]
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$1; TAG=$2; shift 2; N=${TAG}_$C
echo "== $N"
"$HERE/gpu/run_case.sh" "$HERE/gpu/workloads/micro/$C.json" "$N" "$@" | grep PROBES
CUDA_VISIBLE_DEVICES= "$HERE/tools/safe_run.sh" -m 6G -c 8-9 -t 600 -- "$HERE/.venv/bin/python" \
  "$HERE/gpu/analyze.py" "$HERE/results/gpu/$N.jsonl" "$HERE/logs/gpu/trace_$TAG.jsonl" \
  "$HERE/results/gpu/$N.analysis.json" | grep -v safe_run
"$HERE/.venv/bin/python" "$HERE/gpu/probe_report.py" "$HERE/results/gpu/$N.analysis.json"
