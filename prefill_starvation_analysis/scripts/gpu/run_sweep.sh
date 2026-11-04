#!/usr/bin/env bash
# Whole GPU sweep in ONE server lifetime: 6 settings x 5 burst sizes + 2 settings x 3 stream loads.
# Settings are switched between runs through the runtime control file (no restarts).
# usage: scripts/gpu/run_sweep.sh <gpu_index>      (outputs in $WORK, default ./work)
set -u
A=$(cd "$(dirname "$0")/../.." && pwd); cd "$A"
GPU=$1; PORT=${PORT:-8011}; WORK=${WORK:-$A/work}; export WORK
mkdir -p "$WORK"
CTRL=$WORK/sched_ctrl.json; ABORT=$WORK/ABORT; DONE=$WORK/DONE; rm -f "$ABORT" "$DONE"
setctrl() { echo "$1" > "$CTRL.tmp" && mv "$CTRL.tmp" "$CTRL"; sleep 2; }
health() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw \
             --format=csv,noheader -i "$GPU" 2>&1; }
bad() { echo "$1" | grep -qiE "error|unknown|lost|fallen"; }

h=$(health); echo "[sweep] gpu $GPU before: $h"; bad "$h" && { echo "[sweep] GPU unhealthy"; exit 3; }
setctrl "$(python3 -c "import json;print(json.dumps(json.load(open('configs/variants.json'))['adaptive']['ctrl']))")"
VLLM_SCHED_CTRL=$CTRL nohup scripts/gpu/serve.sh "$GPU" "$PORT" > "$WORK/server.log" 2>&1 &
ok=0; for _ in $(seq 1 120); do curl -sf "localhost:$PORT/health" >/dev/null && { ok=1; break; }; sleep 5; done
[ $ok = 1 ] || { echo "[sweep] server not healthy"; touch "$ABORT"; }
# CUDA numbering can differ from nvidia-smi's when a GPU on the host is broken: make sure the
# server really runs on the GPU we watch, otherwise stop.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
[ "${used:-0}" -gt 4000 ] || { echo "[sweep] server is not on nvidia-smi GPU $GPU (CUDA index mismatch)"; touch "$ABORT"; }
# watchdog: GPU error or >= 88 C -> abort after the current run
( while [ ! -f "$DONE" ]; do h=$(health); echo "$(date +%T) $h" >> "$WORK/gpu_health.log"
    bad "$h" && touch "$ABORT"; t=$(echo "$h" | awk -F', ' '{print $4+0}'); [ "${t:-0}" -ge 88 ] && touch "$ABORT"
    sleep 5; done ) &

run() { [ -f "$ABORT" ] && return; echo "== $2"; scripts/gpu/run_case.sh "configs/workloads/$1.json" "$2" "$PORT"; sleep 3; }
for v in adaptive cap512 cap256 nocap pr65283 equal; do
  setctrl "$(python3 -c "import json;print(json.dumps(json.load(open('configs/variants.json'))['$v']['ctrl']))")"
  sfx=""; [ $v = equal ] && sfx="_eq"
  for n in 1 2 4 8 16; do run "burst$n$sfx" "sw_burst_${v}_$n"; done
done
for v in adaptive pr65283; do
  setctrl "$(python3 -c "import json;print(json.dumps(json.load(open('configs/variants.json'))['$v']['ctrl']))")"
  for rho in 0.7 0.85 0.95; do run "stream$rho" "sw_stream_${v}_$rho"; done
done

# graceful stop: SIGTERM to the single API-server PID only
PID=$(pgrep -f "python3? [^ ]*/vllm serve" | head -1)
echo "[sweep] stopping server pid=$PID"; [ -n "$PID" ] && kill -TERM "$PID"
for _ in $(seq 1 60); do kill -0 "$PID" 2>/dev/null || break; sleep 2; done
sleep 5; touch "$DONE"; echo "[sweep] gpu $GPU after: $(health)"
[ -f "$ABORT" ] && echo "[sweep] ABORTED (see $WORK/gpu_health.log)"; echo "[sweep] finished"
