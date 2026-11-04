#!/usr/bin/env bash
# run one workload against the running server (client is CPU-only, RAM-capped)
# usage: gpu/run_case.sh <workload.json> <out_name> [--priority]
HERE=$(cd "$(dirname "$0")/.." && pwd); W=$1; N=$2; shift 2
CUDA_VISIBLE_DEVICES= "$HERE/tools/safe_run.sh" -m 4G -c 8-11 -t 1200 -- \
  "$HERE/.venv/bin/python" "$HERE/gpu/replay_client.py" "$W" "$HERE/results/gpu/$N.jsonl" \
  --drain-s 600 --stop-when-cls-done short "$@" > "$HERE/logs/gpu/client_$N.log" 2>&1
tail -2 "$HERE/logs/gpu/client_$N.log"
"$HERE/.venv/bin/python" - "$HERE/results/gpu/$N.jsonl" <<'PY'
import json, sys
R = [json.loads(l) for l in open(sys.argv[1])][1:]
P = [r for r in R if r['cls'] == 'short']
tt = sorted((r['t_first'] - r['t_send']) for r in P if r['t_first'])
miss = sum(1 for r in P if not r['t_first'])
q = lambda p: tt[min(len(tt)-1, int(p*len(tt)))]
print(f"PROBES(20 tok) n={len(tt)} missing={miss} TTFT p50={q(.5):.2f}s p90={q(.9):.2f}s max={tt[-1]:.2f}s"
      f" | errors={sum(1 for r in R if r['err'] and r['err']!='cancelled_at_cutoff')}")
PY
