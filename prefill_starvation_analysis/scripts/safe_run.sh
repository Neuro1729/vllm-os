#!/usr/bin/env bash
# Run one job RAM-safe and GPU-safe.
#   * hard RAM cap via a cgroup scope (MemoryMax, no swap) -> the job is OOM-killed, never the host
#   * watchdog: if host MemAvailable drops below MIN_AVAIL_GB, the job gets SIGTERM
#   * nice 19 + pinned CPUs; single-threaded BLAS/OMP
#   * timeout sends SIGTERM first (graceful vLLM shutdown), SIGKILL only 120 s later
#   * GPU health check (nvidia-smi) before and after when a GPU is visible
# usage: safe_run.sh [-m 8G] [-c 0-3] [-t 3600] [-a 24] -- cmd args...
set -u
MEM=8G; CPUS=0-3; TMO=3600; MIN_AVAIL_GB=24
while getopts "m:c:t:a:" o; do
  case $o in m) MEM=$OPTARG;; c) CPUS=$OPTARG;; t) TMO=$OPTARG;; a) MIN_AVAIL_GB=$OPTARG;; *) exit 2;; esac
done
shift $((OPTIND-1)); [ "${1:-}" = "--" ] && shift
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

gpu_check() {  # $1 = before|after
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && return 0
  out=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" 2>&1)
  echo "[safe_run] gpu $1: $out"
  if echo "$out" | grep -qiE "error|unknown|lost|fallen"; then
    echo "[safe_run] GPU $CUDA_VISIBLE_DEVICES unhealthy ($1) - refusing/stopping"; return 1
  fi
}
gpu_check before || exit 3

CAP=(systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0)
if ! systemd-run --user --scope -q true 2>/dev/null; then
  echo "[safe_run] no user cgroups here: no hard RAM cap, host-RAM watchdog only"; CAP=()
fi
"${CAP[@]}" nice -n 19 taskset -c "$CPUS" timeout --signal=TERM --kill-after=120 "$TMO" "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$MIN_AVAIL_GB" ]; then
    echo "[safe_run] host MemAvailable=${avail}GB < ${MIN_AVAIL_GB}GB -> SIGTERM job"
    pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
    sleep 30
  fi
  sleep 3
done
wait "$pid"; rc=$?
echo "[safe_run] exit=$rc"
gpu_check after
exit $rc
