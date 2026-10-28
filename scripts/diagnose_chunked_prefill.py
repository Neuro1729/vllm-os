"""Compare chunked and unchunked greedy decoding with the same settings as the failing GPU test.

Two unchunked runs are a control: if they already differ from each other, the test is
sensitive to numerical noise. If the two unchunked runs agree and the chunked run differs
from them, chunked prefill changes the output.

  python scripts/diagnose_chunked_prefill.py --model /models/Qwen3-0.6B
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams  # noqa: E402


def run(llm, prompts, params, budget):
    llm.scheduler.max_num_batched_tokens = budget
    return [o["token_ids"] for o in llm.generate(prompts, params, use_tqdm=False)]


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        max_num_seqs=4,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
        scheduling_policy="mlfq",
        enable_mlfq_preemption=True,
    )
    rng = random.Random(1)
    prompts = [[rng.randint(1000, 9000) for _ in range(n)] for n in (700, 500)]
    params = [SamplingParams(temperature=1e-9, max_tokens=16, ignore_eos=True)] * 2

    unchunked_a = run(llm, prompts, params, 16384)
    unchunked_b = run(llm, prompts, params, 16384)
    chunked = run(llm, prompts, params, 128)

    for i in range(len(prompts)):
        print(f"prompt {i} ({len(prompts[i])} tokens)")
        print(f"  unchunked A : {unchunked_a[i]}")
        print(f"  unchunked B : {unchunked_b[i]}")
        print(f"  chunked 128 : {chunked[i]}")
        print(f"  first mismatch A vs B: {first_mismatch(unchunked_a[i], unchunked_b[i])}")
        print(f"  first mismatch A vs chunked: {first_mismatch(unchunked_a[i], chunked[i])}")


if __name__ == "__main__":
    sys.exit(main())
