# Shared helpers for the GPU experiment scripts. Source this file; do not execute it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V0_COMMIT="e1da1f5e2a4fc81749235492798a310a18147da4"   # original nano-vLLM (unmodified baseline)
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-gpu}"
PYTHON_CANDIDATES="python3.12 python3.11 python3.10 python3"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Print the first Python interpreter in 3.10..3.12 (pyproject requires <3.13).
find_python() {
    local cand ver
    for cand in $PYTHON_CANDIDATES; do
        if command -v "$cand" >/dev/null 2>&1; then
            ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
            case "$ver" in
                3.10|3.11|3.12) command -v "$cand"; return 0 ;;
            esac
        fi
    done
    return 1
}

# Python inside the project venv (created by setup_gpu_env.sh).
venv_python() { printf '%s\n' "$VENV_DIR/bin/python"; }

# Interpreter for helper tasks: the project venv if present, else the first Python that really runs.
# (On Windows the python3 name can be a Store stub that exits with an error, so probe each one.)
python_for_tools() {
    if [ -x "$(venv_python)" ]; then
        venv_python
        return 0
    fi
    local cand
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys' >/dev/null 2>&1; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    die "no working Python found"
}

require_venv() {
    [ -x "$(venv_python)" ] || die "virtual environment missing at $VENV_DIR; run: bash scripts/setup_gpu_env.sh"
}

# Query GPU facts with nvidia-smi. Prints: name,driver,memory.total,memory.used,memory.free (MiB)
# for the given physical GPU index. Fails if nvidia-smi or the index is unavailable.
gpu_query() {
    local index="$1"
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi -i "$index" --query-gpu=name,driver_version,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1
}
