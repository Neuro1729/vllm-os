"""GPU correctness tests. They skip without CUDA and are NOT verified on this development machine.

Environment:
  NANOVLLM_TEST_MODEL   local model directory (default: ~/huggingface/Qwen3-0.6B)
  NANOVLLM_TEST_POLICY  "fcfs" or "mlfq" (default "mlfq"; "mlfq" also enables preemption)

nano-vLLM creates a process group per engine, so one LLM is shared by the module. Token-budget
experiments change `llm.scheduler.max_num_batched_tokens` in place; the policy is fixed at init.
"""
import os

import pytest

torch = pytest.importorskip("torch")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
MODEL = os.path.expanduser(os.environ.get("NANOVLLM_TEST_MODEL", "~/huggingface/Qwen3-0.6B"))
POLICY = os.environ.get("NANOVLLM_TEST_POLICY", "mlfq")
pytestmark = requires_cuda


@pytest.fixture(scope="module")
def llm():
    if not os.path.isdir(MODEL):
        pytest.skip(f"model directory not found: {MODEL} (set NANOVLLM_TEST_MODEL)")
    from nanovllm import LLM
    kwargs = dict(
        max_num_seqs=4,
        max_model_len=2048,
        gpu_memory_utilization=float(os.environ.get("NANOVLLM_TEST_GPU_UTIL", "0.5")),
        scheduling_policy=POLICY,
        enable_mlfq_preemption=(POLICY == "mlfq"),
    )
    # LLMEngine registers its own atexit cleanup, so no explicit exit() here.
    yield LLM(MODEL, **kwargs)


def sampling(max_tokens, temperature=0.6):
    from nanovllm import SamplingParams
    return SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)


def random_prompt(rng, length):
    return [rng.randint(1000, 9000) for _ in range(length)]


def test_mixed_prompt_lengths_all_complete(llm):
    import random
    rng = random.Random(0)
    prompts = [random_prompt(rng, n) for n in (3, 300, 20, 700, 5, 1200)]
    outputs = llm.generate(prompts, sampling(8), use_tqdm=False)
    assert [o["status"] for o in outputs] == ["finished"] * len(prompts)
    assert all(len(o["token_ids"]) == 8 for o in outputs)


def test_memory_integrity_after_run(llm):
    bm = llm.scheduler.block_manager
    assert llm.scheduler.is_finished()
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids


def test_chunked_prefill_matches_unchunked_prefill(llm):
    # Near-zero temperature makes sampling effectively greedy, so chunking must not change tokens
    # beyond bf16 numerical noise at near-ties. The threshold allows that noise.
    import random
    rng = random.Random(1)
    prompts = [random_prompt(rng, 700), random_prompt(rng, 500)]
    params = [sampling(16, temperature=1e-9)] * 2

    original_budget = llm.scheduler.max_num_batched_tokens
    try:
        llm.scheduler.max_num_batched_tokens = 16384
        unchunked = [o["token_ids"] for o in llm.generate(prompts, params, use_tqdm=False)]
        llm.scheduler.max_num_batched_tokens = 128
        chunked = [o["token_ids"] for o in llm.generate(prompts, params, use_tqdm=False)]
    finally:
        llm.scheduler.max_num_batched_tokens = original_budget

    matches = sum(a == b for u, c in zip(unchunked, chunked) for a, b in zip(u, c))
    total = sum(len(u) for u in unchunked)
    assert matches / total >= 0.9, f"chunked prefill diverged: {matches}/{total} tokens match"


def test_mixed_prefill_and_decode_in_one_step(llm):
    # A long prompt admitted while short requests decode must share a step with them.
    import random
    rng = random.Random(2)
    short = [llm.add_request(random_prompt(rng, 4), sampling(40)) for _ in range(2)]
    seen_mixed = False
    for _ in range(3):
        result = llm.step()
        if result.num_decode_seqs and result.num_prefill_tokens:
            seen_mixed = True
    long = llm.add_request(random_prompt(rng, 900), sampling(4))
    while not llm.is_finished():
        result = llm.step()
        if result.num_decode_seqs and result.num_prefill_tokens:
            seen_mixed = True
    assert seen_mixed
    assert all(s.is_finished for s in short + [long])


def test_shared_prefix_populates_prefix_cache(llm):
    import random
    rng = random.Random(3)
    shared = random_prompt(rng, 512)  # two full 256-token blocks
    outputs = llm.generate([shared + random_prompt(rng, 10), shared + random_prompt(rng, 12)], sampling(4), use_tqdm=False)
    assert all(o["status"] == "finished" for o in outputs)
    assert len(llm.scheduler.block_manager.hash_to_block_id) > 0
