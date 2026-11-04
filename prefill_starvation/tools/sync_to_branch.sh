#!/usr/bin/env bash
# Copy the study (code, small results, charts) into the vllm-os branch checkout as prefill_starvation/
set -e
SRC=$(cd "$(dirname "$0")/.." && pwd)
DST=${1:-$SRC/../vllm-os-prefill-starvation/prefill_starvation}
mkdir -p "$DST"
rsync -a --delete \
  --include='/README.md' --include='/.gitignore' --include='/*.py' --include='/*.sh' \
  --include='/tools/' --include='/tools/*.sh' --include='/tools/pdl.py' \
  --include='/fake_model/' --include='/fake_model/config.json' \
  --include='/data/' --include='/data/download.sh' \
  --include='/gpu/' --include='/gpu/*.py' --include='/gpu/*.sh' \
  --include='/gpu/workloads/' --include='/gpu/workloads/micro/' --include='/gpu/workloads/micro/*.json' \
  --include='/results/' --include='/results/charts/' --include='/results/charts/*.png' \
  --include='/results/gpu/' --include='/results/gpu/SUMMARY_prio2048.txt' \
  --include='/results/gpu/prio2048_*.jsonl' --include='/results/gpu/prio2048_*.analysis.json' \
  --include='/results/micro.json' --include='/results/validation.json' \
  --include='/results/gpu/trace_prio2048.jsonl.gz' \
  --exclude='*' "$SRC/" "$DST/"
du -sh "$DST"; find "$DST" -type f | wc -l
