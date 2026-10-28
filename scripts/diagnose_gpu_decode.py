"""Run the GPU test-1 workload in one decode mode and report the scheduler state on failure.

Run one mode per process (a second LLM in the same process fails):
  python scripts/diagnose_gpu_decode.py --model /models/Qwen3-0.6B --eager    # no CUDA graphs
  python scripts/diagnose_gpu_decode.py --model /models/Qwen3-0.6B             # CUDA graphs
"""
import argparse
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eager", action="store_true", help="disable CUDA graphs (enforce_eager=True)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    mode = "eager" if args.eager else "cuda-graph"
    llm = LLM(
        args.model,
        enforce_eager=args.eager,
        max_num_seqs=4,
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduling_policy="mlfq",
        enable_mlfq_preemption=True,
    )
    rng = random.Random(0)
    prompts = [[rng.randint(1000, 9000) for _ in range(n)] for n in (3, 300, 20, 700, 5, 1200)]
    params = SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True)
    try:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    except Exception:
        print(f"[{mode}] FAILED")
        traceback.print_exc()
        for seq in llm.scheduler.running + llm.scheduler.waiting:
            print(f"  seq {seq.seq_id}: num_tokens={seq.num_tokens} num_blocks={seq.num_blocks} "
                  f"block_table={len(seq.block_table)} cached={seq.num_cached_tokens} "
                  f"scheduled={seq.num_scheduled_tokens}")
        return 1
    print(f"[{mode}] OK, completion lengths: {[len(o['token_ids']) for o in outputs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
