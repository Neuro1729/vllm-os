#!/usr/bin/env bash
# usage: gpu/run_dataset.sh <workload_name> <trace_tag> [client args]   (full 300 s replay + drain)
HERE=$(cd "$(dirname "$0")/.." && pwd); C=$1; TAG=$2; shift 2; N=${TAG}_ds${WL_SET:+_$WL_SET}_$C
echo "== $N"
CUDA_VISIBLE_DEVICES= "$HERE/tools/safe_run.sh" -m 4G -c 8-11 -t 1500 -- "$HERE/.venv/bin/python" \
  "$HERE/gpu/replay_client.py" "$HERE/gpu/workloads/${WL_SET:-v2}/$C.json" "$HERE/results/gpu/$N.jsonl" \
  --drain-s 600 "$@" > "$HERE/logs/gpu/client_$N.log" 2>&1
tail -1 "$HERE/logs/gpu/client_$N.log"
CUDA_VISIBLE_DEVICES= "$HERE/tools/safe_run.sh" -m 6G -c 8-9 -t 600 -- "$HERE/.venv/bin/python" \
  "$HERE/gpu/analyze.py" "$HERE/results/gpu/$N.jsonl" "$HERE/logs/gpu/trace_$TAG.jsonl" \
  "$HERE/results/gpu/$N.analysis.json" | grep -v safe_run
"$HERE/.venv/bin/python" - "$HERE/results/gpu/$N.analysis.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
q = lambda a, p: a[min(len(a)-1, int(p*len(a)))] if a else float('nan')
for cls in ('short', 'medium', 'long'):
    g = [r for r in d['requests'] if r['cls'] == cls and r.get('ttft_s') is not None]
    tt = sorted(r['ttft_s'] for r in g); zr = sorted(r.get('max_zero_run', 0) or 0 for r in g)
    miss = sum(1 for r in d['requests'] if r['cls'] == cls and r.get('ttft_s') is None)
    if tt:
        print(f"  {cls:6s} n={len(tt)} miss={miss} TTFT p50={q(tt,.5):.2f}s p90={q(tt,.9):.2f}s p99={q(tt,.99):.2f}s max={tt[-1]:.2f}s | longest 0-token run max={zr[-1]} steps")
PY
