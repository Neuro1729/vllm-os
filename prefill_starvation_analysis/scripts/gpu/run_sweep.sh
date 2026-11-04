#!/usr/bin/env bash
# Whole GPU sweep in ONE server lifetime: 5 settings x 5 burst sizes + 3 settings x 3 stream loads.
# Settings are switched between runs through the runtime control file (no restarts).
# usage: scripts/gpu/run_sweep.sh <cuda_index>     (outputs in $WORK, default ./work)
# Note: CUDA numbering can differ from nvidia-smi's (e.g. when a GPU on the host is broken);
# the script finds the GPU the server actually uses and watches that one.
set -u
A=$(cd "$(dirname "$0")/../.." && pwd); cd "$A"
CUDA_IDX=$1; PORT=${PORT:-8011}; WORK=${WORK:-$A/work}; export WORK
mkdir -p "$WORK"
CTRL=$WORK/sched_ctrl.json; ABORT=$WORK/ABORT; DONE=$WORK/DONE; rm -f "$ABORT" "$DONE"
ctrl_of() { python3 -c "import json;print(json.dumps(json.load(open('configs/variants.json'))['$1']['ctrl']))"; }
setctrl() { echo "$1" > "$CTRL.tmp" && mv "$CTRL.tmp" "$CTRL"; sleep 2; }
used() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | grep -E "^[0-9]+, [0-9]+$"; }
health() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
             --format=csv,noheader -i "$WATCH" 2>&1; }
bad() { echo "$1" | grep -qiE "error|unknown|lost|fallen"; }

before=$(used)
setctrl "$(ctrl_of adaptive)"
VLLM_SCHED_CTRL=$CTRL nohup scripts/gpu/serve.sh "$CUDA_IDX" "$PORT" > "$WORK/server.log" 2>&1 &
ok=0; for _ in $(seq 1 120); do curl -sf "localhost:$PORT/health" >/dev/null && { ok=1; break; }; sleep 5; done
[ $ok = 1 ] || { echo "[sweep] server not healthy"; touch "$ABORT"; }
# the GPU that went from idle to > 4 GB is the one the server uses
WATCH=$(join -t, <(echo "$before" | tr -d ' ' | sort) <(used | tr -d ' ' | sort) \
        | awk -F, '$2 < 1000 && $3 > 4000 {print $1}')
[ "$(echo "$WATCH" | wc -w)" = 1 ] || { echo "[sweep] cannot identify the server GPU: '$WATCH'"; touch "$ABORT"; WATCH=0; }
echo "[sweep] server on CUDA index $CUDA_IDX = nvidia-smi GPU $WATCH: $(health)"
( while [ ! -f "$DONE" ]; do h=$(health); echo "$(date +%T) $h" >> "$WORK/gpu_health.log"
    bad "$h" && touch "$ABORT"; t=$(echo "$h" | awk -F', ' '{print $4+0}'); [ "${t:-0}" -ge 88 ] && touch "$ABORT"
    sleep 5; done ) &

run() { [ -f "$ABORT" ] && return; echo "== $2"; scripts/gpu/run_case.sh "configs/workloads/$1.json" "$2" "$PORT"; sleep 3; }
for v in adaptive cap512 cap256 nocap equal; do
  setctrl "$(ctrl_of $v)"; sfx=""; [ $v = equal ] && sfx="_eq"
  for n in 1 2 4 8 16; do run "burst$n$sfx" "sw_burst_${v}_$n"; done
done
for v in adaptive cap512 equal; do
  setctrl "$(ctrl_of $v)"; sfx=""; [ $v = equal ] && sfx="_eq"
  for rho in 0.7 0.85 0.95; do run "stream$rho$sfx" "sw_stream_${v}_$rho"; done
done

# graceful stop: SIGTERM to the single API-server PID only
PID=$(pgrep -f "python3? [^ ]*/vllm serve" | head -1)
echo "[sweep] stopping server pid=$PID"; [ -n "$PID" ] && kill -TERM "$PID"
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
sleep 5; touch "$DONE"; echo "[sweep] GPU $WATCH after: $(health)"
[ -f "$ABORT" ] && echo "[sweep] ABORTED (see $WORK/gpu_health.log)"; echo "[sweep] finished"
