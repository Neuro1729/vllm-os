"""Greedy-decode the chunked-prefill test prompts with one prefill budget, in its own process.

Each run is a separate process so the prefix cache from one run cannot serve the next.
Compare the printed token lists across runs.

  python scripts/diagnose_chunked_prefill.py --model /models/Qwen3-0.6B --budget 16384 --label unchunked-A
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--budget", type=int, required=True, help="max_num_batched_tokens for this run")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        max_num_seqs=4,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        scheduling_policy="mlfq",
        enable_mlfq_preemption=True,
    )
    llm.scheduler.max_num_batched_tokens = args.budget
    rng = random.Random(1)
    prompts = [[rng.randint(1000, 9000) for _ in range(n)] for n in (700, 500)]
    params = [SamplingParams(temperature=1e-9, max_tokens=16, ignore_eos=True)] * 2
    outputs = llm.generate(prompts, params, use_tqdm=False)
    for i, output in enumerate(outputs):
        print(f"[{args.label}] prompt {i}: {output['token_ids']}")


if __name__ == "__main__":
    sys.exit(main())
