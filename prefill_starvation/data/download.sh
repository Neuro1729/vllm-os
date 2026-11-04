#!/usr/bin/env bash
# Public traces used by workloads.py (not redistributed here).
set -e; cd "$(dirname "$0")"; mkdir -p mooncake azure
for f in conversation_trace toolagent_trace synthetic_trace; do
  curl -sSfL -o mooncake/$f.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/$f.jsonl
done
curl -sSfL -o mooncake/mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl
for f in conv code; do
  curl -sSfL -o azure/AzureLLMInferenceTrace_$f.csv https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_$f.csv
done
