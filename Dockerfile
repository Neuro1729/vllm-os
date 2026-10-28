# GPU image for the nano-vLLM MLFQ experiments.
# Build:  docker build -t nanovllm-mlfq .
# Run:    scripts/run_docker_experiment.sh --model /path/to/Qwen3-0.6B --max-num-seqs 8
#
# UNVERIFIED: this image has not been built or run on a GPU. The package pins match
# requirements-gpu.txt and are a best guess for torch 2.6 + CUDA 12.4.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_DIR=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Ubuntu 22.04 ships Python 3.10, which the project supports (requires-python >=3.10,<3.13).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-dev git ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --upgrade pip

# torch comes from the CUDA wheel index, matching the base image's CUDA 12.4 runtime.
RUN pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN grep -v -E '^(#|torch==|$)' /tmp/requirements-gpu.txt | sed 's/[[:space:]]*#.*$//' | grep -v '^$' > /tmp/requirements-noncuda.txt \
    && pip install -r /tmp/requirements-noncuda.txt

# Prebuilt flash-attn wheel for torch 2.6 + CUDA 12 + Python 3.10 (no source compile).
# Source builds with --no-build-isolation fail here because the venv has no `wheel` package.
RUN pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Copy the repository last so code edits do not invalidate the dependency layers.
# .git is included so the V0 baseline (commit e1da1f5) can be checked out as a worktree.
WORKDIR /workspace
COPY . /workspace

# The image already contains the environment, so the pipeline validates it and skips installation.
CMD ["bash", "scripts/run_full_experiment.sh", "--help"]
