#!/usr/bin/env bash
# One-command pipeline: environment setup -> GPU preflight -> correctness tests ->
# benchmark matrix (V0, modernized FCFS, MLFQ, MLFQ + preemption) -> report.
#
# Examples:
#   bash scripts/run_full_experiment.sh --max-num-seqs 8
#   bash scripts/run_full_experiment.sh --max-num-seqs-list 4,8,16 --profile standard --repeats 3
#   bash scripts/run_full_experiment.sh --max-num-seqs 4 --profile smoke --skip-setup
#   bash scripts/run_full_experiment.sh --dry-run          # CPU-only validation of this pipeline
#
# Results are written to a NEW timestamped directory. Existing results are never deleted.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- defaults ----------------------------------------------------------------
MODEL="Qwen/Qwen3-0.6B"
GPU=0
GPU_UTIL="0.70"
MAX_NUM_SEQS=""
MAX_NUM_SEQS_LIST=""
MAX_NUM_BATCHED_TOKENS=4096
MAX_MODEL_LEN=4096
PROFILE="standard"
REPEATS=3
WARMUP=1
SEED=0
TIMEOUT_S=1800
OUTPUT_DIR=""
SKIP_SETUP=0
SETUP_ONLY=0
SKIP_BASELINE_V0=0
CONTINUE_ON_TEST_FAILURE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/run_full_experiment.sh [options]

Sequence-limit options (exactly one):
  --max-num-seqs N            run every variant with max_num_seqs = N
  --max-num-seqs-list A,B,C   run the full matrix once per limit (e.g. 4,8,16)

Model and hardware:
  --model PATH_OR_HF_ID       local model directory, or a Hugging Face repo id to download
                              into models/<name> (needs network). Default: Qwen/Qwen3-0.6B
  --gpu N                     physical GPU index (default 0)
  --gpu-memory-utilization F  0 < F <= 1 (default 0.70)
  --max-num-batched-tokens N  per-step token budget for all variants (default 4096)
  --max-model-len N           maximum sequence length (default 4096)

Benchmark shape:
  --profile smoke|standard|thorough   (default standard)
  --repeats N                 measured repetitions per configuration (default 3)
  --warmup N                  untimed warm-up runs per configuration (default 1)
  --seed N                    workload seed (default 0)
  --timeout-s S               per-trial wall-clock limit (default 1800)

Output and stages:
  --output-dir DIR            new directory to create (default results/experiment_YYYYMMDD_HHMMSS)
  --skip-setup                do not run environment setup (validate only)
  --setup-only                prepare the environment and exit
  --skip-baseline-v0          skip the original nano-vLLM (V0) variant; the report says so
  --continue-on-test-failure  run benchmarks even if correctness tests fail; results are labelled unverified
  --dry-run                   CPU-only check: argument validation, CPU tests, benchmark plan. No GPU work.
  --help                      show this message

Exit status is nonzero, with an actionable message, on any failure.
EOF
}

# ---- argument parsing --------------------------------------------------------
need_value() { [ $# -ge 2 ] || die "$1 needs a value (see --help)"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --model) need_value "$@"; MODEL="$2"; shift ;;
        --gpu) need_value "$@"; GPU="$2"; shift ;;
        --gpu-memory-utilization) need_value "$@"; GPU_UTIL="$2"; shift ;;
        --max-num-seqs) need_value "$@"; MAX_NUM_SEQS="$2"; shift ;;
        --max-num-seqs-list) need_value "$@"; MAX_NUM_SEQS_LIST="$2"; shift ;;
        --max-num-batched-tokens) need_value "$@"; MAX_NUM_BATCHED_TOKENS="$2"; shift ;;
        --max-model-len) need_value "$@"; MAX_MODEL_LEN="$2"; shift ;;
        --profile) need_value "$@"; PROFILE="$2"; shift ;;
        --repeats) need_value "$@"; REPEATS="$2"; shift ;;
        --warmup) need_value "$@"; WARMUP="$2"; shift ;;
        --seed) need_value "$@"; SEED="$2"; shift ;;
        --timeout-s) need_value "$@"; TIMEOUT_S="$2"; shift ;;
        --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift ;;
        --skip-setup) SKIP_SETUP=1 ;;
        --setup-only) SETUP_ONLY=1 ;;
        --skip-baseline-v0) SKIP_BASELINE_V0=1 ;;
        --continue-on-test-failure) CONTINUE_ON_TEST_FAILURE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

# ---- validation --------------------------------------------------------------
is_pos_int() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$1" -ge 1 ]; }
is_nonneg_int() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; }

if [ -n "$MAX_NUM_SEQS" ] && [ -n "$MAX_NUM_SEQS_LIST" ]; then
    die "use either --max-num-seqs or --max-num-seqs-list, not both"
fi
if [ -z "$MAX_NUM_SEQS" ] && [ -z "$MAX_NUM_SEQS_LIST" ] && [ "$SETUP_ONLY" -eq 0 ]; then
    die "--max-num-seqs N (or --max-num-seqs-list) is required, e.g. --max-num-seqs 8"
fi
if [ "$SKIP_SETUP" -eq 1 ] && [ "$SETUP_ONLY" -eq 1 ]; then
    die "--skip-setup and --setup-only cannot be combined"
fi
if [ "$SETUP_ONLY" -eq 1 ] && [ "$DRY_RUN" -eq 1 ]; then
    die "--setup-only and --dry-run cannot be combined"
fi

SEQ_LIMITS="${MAX_NUM_SEQS:-$MAX_NUM_SEQS_LIST}"
if [ "$SETUP_ONLY" -eq 0 ]; then
    IFS=',' read -r -a LIMIT_ARRAY <<< "$SEQ_LIMITS"
    SEEN=" "
    for limit in "${LIMIT_ARRAY[@]}"; do
        is_pos_int "$limit" || die "max_num_seqs values must be positive integers, got '$limit'"
        case "$SEEN" in *" $limit "*) die "duplicate max_num_seqs value: $limit" ;; esac
        SEEN="$SEEN$limit "
    done
fi

is_pos_int "$GPU" || is_nonneg_int "$GPU" || die "--gpu must be a non-negative integer"
awk -v u="$GPU_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }' || die "--gpu-memory-utilization must be in (0, 1]"
case "$PROFILE" in smoke|standard|thorough) ;; *) die "--profile must be smoke, standard or thorough" ;; esac
is_pos_int "$REPEATS" || die "--repeats must be a positive integer"
is_nonneg_int "$WARMUP" || die "--warmup must be a non-negative integer"
is_nonneg_int "$SEED" || die "--seed must be a non-negative integer"
is_pos_int "$TIMEOUT_S" || die "--timeout-s must be a positive integer"
is_pos_int "$MAX_NUM_BATCHED_TOKENS" || die "--max-num-batched-tokens must be a positive integer"
is_pos_int "$MAX_MODEL_LEN" || die "--max-model-len must be a positive integer"

# ---- helpers -----------------------------------------------------------------
STAGE_STATUS=""
record_stage() {  # name status detail
    STAGE_STATUS="$STAGE_STATUS$1=$2;"
    log "stage $1: $2${3:+ ($3)}"
}

run_setup() {
    if [ "$SKIP_SETUP" -eq 1 ]; then
        log "setup skipped (--skip-setup); validating existing environment"
        bash "$REPO_ROOT/scripts/setup_gpu_env.sh" --check-only || die "environment validation failed"
    else
        bash "$REPO_ROOT/scripts/setup_gpu_env.sh" || die "environment setup failed (see above)"
    fi
}

gpu_preflight() {
    local q name driver total used free budget_mib
    q="$(gpu_query "$GPU" || true)"
    [ -n "$q" ] || die "GPU $GPU is not visible to nvidia-smi. Check the index with: nvidia-smi -L"
    IFS=',' read -r name driver total used free <<< "$q"
    name="${name# }"; driver="${driver# }"; total="${total// /}"; used="${used// /}"; free="${free// /}"
    log "GPU $GPU: $name, driver $driver, total ${total} MiB, used ${used} MiB, free ${free} MiB"
    budget_mib="$(awk -v t="$total" -v u="$GPU_UTIL" 'BEGIN { printf "%d", t * u }')"
    if [ "$free" -lt 3000 ]; then
        die "only ${free} MiB free on GPU $GPU; another process is using it. Free the GPU or choose --gpu N."
    fi
    if [ "$used" -gt $(( total / 2 )) ]; then
        warn "GPU $GPU already has ${used} MiB in use by other processes; results may be affected. Unrelated processes are NOT killed."
    fi
    if [ "$free" -lt "$budget_mib" ]; then
        warn "requested budget ${budget_mib} MiB exceeds free memory ${free} MiB; nano-vLLM sizes its KV cache from free memory, so the cache will be smaller."
    fi
    local model_mb
    model_mb="$(du -sm "$MODEL_PATH" | awk '{print $1}')"
    if [ $(( model_mb * 2 )) -gt "$free" ]; then
        die "model (${model_mb} MiB on disk) may not fit in ${free} MiB free GPU memory"
    fi
}

resolve_model() {
    if [ -d "$MODEL" ]; then
        MODEL_PATH="$(cd "$MODEL" && pwd)"
        return 0
    fi
    case "$MODEL" in
        */*) ;;
        *) die "--model '$MODEL' is neither a directory nor a Hugging Face repo id (owner/name)" ;;
    esac
    MODEL_PATH="$REPO_ROOT/models/$(basename "$MODEL")"
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            log "dry run: would download $MODEL to $MODEL_PATH"
            return 0
        fi
        require_venv
        log "downloading $MODEL to $MODEL_PATH (requires network; pass a local --model PATH for offline use)"
        "$(venv_python)" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" \
            "$MODEL" "$MODEL_PATH" || die "model download failed for $MODEL"
    fi
}

write_environment() {
    local py commit
    commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    py="$(python_for_tools)"
    "$py" - "$OUT" "$commit" "$GPU" "$MODEL_PATH" <<'PY'
import json, os, platform, subprocess, sys
out, commit, gpu, model = sys.argv[1:5]
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # environment capture must never abort the run
        return f"unavailable: {exc}"
smi = run(["nvidia-smi"])
info = {
    "hostname": platform.node(),
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "gpu_index": gpu,
    "gpu_info_csv": run(["nvidia-smi", "-i", gpu, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]),
    "nvidia_smi_header": smi.splitlines()[2:3],
    "model_path": model,
    "repo_commit": commit,
    "v0_commit": os.environ.get("V0_COMMIT_FOR_RECORD", ""),
}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
except Exception as exc:
    info["torch"] = f"unavailable: {exc}"
with open(os.path.join(out, "environment.json"), "w") as fh:
    json.dump(info, fh, indent=2)
with open(os.path.join(out, "git_versions.json"), "w") as fh:
    json.dump({"repo_head": commit, "v0_commit": os.environ.get("V0_COMMIT_FOR_RECORD", "")}, fh, indent=2)
PY
}

run_correctness() {
    local cpu_xml gpu_xml status=0 py
    py="$(python_for_tools)"
    mkdir -p "$OUT/correctness"
    cpu_xml="$OUT/correctness/cpu_tests.xml"
    gpu_xml="$OUT/correctness/gpu_tests.xml"

    log "CPU correctness tests (scheduler, MLFQ, preemption, KV accounting, invariants, benchmark pipeline)"
    (cd "$REPO_ROOT" && "$py" -m pytest tests --ignore=tests/gpu -q --junitxml="$cpu_xml") || status=1
    CPU_STATUS=$([ "$status" -eq 0 ] && echo passed || echo failed)

    if [ "$DRY_RUN" -eq 1 ]; then
        GPU_STATUS="skipped-dry-run"
    else
        log "GPU correctness tests (policy=mlfq with preemption)"
        set +e
        (cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" NANOVLLM_TEST_MODEL="$MODEL_PATH" \
            NANOVLLM_TEST_POLICY=mlfq NANOVLLM_TEST_GPU_UTIL="$GPU_UTIL" \
            "$py" -m pytest tests/gpu -q --junitxml="$gpu_xml")
        local gpu_rc=$?
        set -e
        case "$gpu_rc" in
            0) GPU_STATUS=passed ;;
            5) GPU_STATUS=no-tests-collected ;;
            *) GPU_STATUS=failed; status=1 ;;
        esac
    fi
    printf '{"cpu_tests": "%s", "gpu_tests": "%s", "cpu_xml": "%s", "gpu_xml": "%s"}\n' \
        "$CPU_STATUS" "$GPU_STATUS" "correctness/cpu_tests.xml" "correctness/gpu_tests.xml" > "$OUT/correctness/summary.json"
    return "$status"
}

prepare_v0_worktree() {
    local wt="$OUT/worktrees/v0-original"
    if ! git -C "$REPO_ROOT" cat-file -e "$V0_COMMIT^{commit}" 2>/dev/null; then
        die "baseline commit $V0_COMMIT is not present in this clone. Fetch the full history before running V0."
    fi
    if [ ! -d "$wt" ]; then
        git -C "$REPO_ROOT" worktree add --detach "$wt" "$V0_COMMIT" >/dev/null \
            || die "could not create V0 worktree at $wt"
    fi
    V0_ROOT="$wt"
    log "V0 baseline worktree: $V0_ROOT ($V0_COMMIT)"
}

# ---- stages ------------------------------------------------------------------
if [ "$SETUP_ONLY" -eq 1 ]; then
    run_setup
    log "environment prepared; nothing else was run (--setup-only)"
    exit 0
fi

if [ "$DRY_RUN" -eq 1 ]; then
    log "dry run: CPU-only validation. No GPU, no model download, no benchmark trials."
else
    run_setup
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$REPO_ROOT/results/experiment_$(date +%Y%m%d_%H%M%S)"
fi
[ -e "$OUTPUT_DIR" ] && die "output directory already exists: $OUTPUT_DIR (results are never overwritten; pass a new --output-dir)"
OUT="$OUTPUT_DIR"
mkdir -p "$OUT"
log "results directory: $OUT"

V0_COMMIT_FOR_RECORD="$V0_COMMIT"
export V0_COMMIT_FOR_RECORD
resolve_model
if [ "$DRY_RUN" -eq 0 ]; then
    gpu_preflight
fi
write_environment
log "model: $MODEL_PATH"

CORRECTNESS_OK=1
if run_correctness; then
    record_stage correctness passed
else
    CORRECTNESS_OK=0
    record_stage correctness failed "see $OUT/correctness/"
    if [ "$CONTINUE_ON_TEST_FAILURE" -eq 0 ]; then
        echo "Correctness tests failed; benchmarks were NOT run. Details: $OUT/correctness/" >&2
        echo "Re-run with --continue-on-test-failure to benchmark anyway (results will be labelled unverified)." >&2
        exit 1
    fi
    warn "continuing after correctness failure: all benchmark results will be labelled UNVERIFIED"
fi

BENCH_DIR="$OUT/bench"
PY="$(python_for_tools)"
MATRIX_ARGS=(--output-dir "$BENCH_DIR" --model "$MODEL_PATH" --profile "$PROFILE" --repeats "$REPEATS"
    --warmup "$WARMUP" --seed "$SEED" --max-num-seqs-list "$SEQ_LIMITS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" --timeout-s "$TIMEOUT_S")

if [ "$DRY_RUN" -eq 1 ]; then
    MATRIX_ARGS+=(--plan-only --skip-v0)
else
    if [ "$SKIP_BASELINE_V0" -eq 0 ]; then
        prepare_v0_worktree
        MATRIX_ARGS+=(--nanovllm-root-v0 "$V0_ROOT")
    else
        MATRIX_ARGS+=(--skip-v0)
        warn "V0 baseline skipped (--skip-baseline-v0); the report will mark it as not measured"
    fi
fi

log "benchmark matrix: scenarios x max_num_seqs [$SEQ_LIMITS] x variants x $REPEATS repeats"
(cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m benchmarks.run_matrix "${MATRIX_ARGS[@]}") \
    || { record_stage benchmarks failed; die "benchmark matrix failed; partial results are in $BENCH_DIR"; }
record_stage benchmarks "$([ "$DRY_RUN" -eq 1 ] && echo planned || echo completed)"

if [ "$DRY_RUN" -eq 1 ]; then
    log "plan written to $BENCH_DIR/plan.json"
    log "dry run finished: CPU tests and benchmark planning work; no GPU results were produced."
    exit 0
fi

cp "$OUT/environment.json" "$BENCH_DIR/environment.json"
(cd "$REPO_ROOT" && "$PY" -m benchmarks.generate_report --input "$BENCH_DIR" --output "$OUT/reports") \
    || die "report generation failed; raw data is in $BENCH_DIR"
record_stage report completed

echo
echo "Experiment complete: $OUT"
echo "  correctness:  $OUT/correctness/"
echo "  raw data:     $BENCH_DIR/raw/"
echo "  summaries:    $BENCH_DIR/summaries/"
echo "  plots:        $OUT/reports/plots/"
echo "  report (MD):  $OUT/reports/REPORT.md"
echo "  report (HTML): $OUT/reports/REPORT.html"
if [ "$CORRECTNESS_OK" -eq 0 ]; then
    echo "  WARNING: correctness failed; results are UNVERIFIED."
fi
