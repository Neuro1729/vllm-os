"""GPU checks for swap-based preemption. Needs CUDA and a Qwen3 checkpoint:

    CUDA_VISIBLE_DEVICES=<gpu> NANOVLLM_MODEL=~/huggingface/Qwen3-0.6B pytest tests/gpu -s
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
MODEL = os.path.expanduser(os.environ.get("NANOVLLM_MODEL", "~/huggingface/Qwen3-0.6B/"))
if not os.path.isdir(MODEL):
    pytest.skip(f"model not found at {MODEL}", allow_module_level=True)

# tests/conftest.py stubs out nanovllm.llm for the CPU tests; this module needs the real one
sys.modules.pop("nanovllm.llm", None)
from nanovllm.llm import LLM  # noqa: E402
from nanovllm.sampling_params import SamplingParams  # noqa: E402

GREEDY = 1e-5    # SamplingParams rejects 0; the fixture swaps in a pure argmax sampler anyway


@pytest.fixture(scope="module")
def llm():
    engine = LLM(MODEL, enforce_eager=True, max_num_seqs=1, max_model_len=2048, gpu_memory_utilization=0.3,
                 num_kvcache_blocks=64, scheduling_policy="priority", preemption_mode="swap", swap_space=0.5)
    # The real sampler adds exponential noise, so even at temperature 1e-5 tied bf16 logits
    # are broken at random and two identical runs can diverge. Argmax makes runs comparable.
    engine.model_runner.sampler = lambda logits, temperatures: logits.argmax(dim=-1)
    yield engine
    engine.exit()


def test_swap_round_trip_is_bit_exact(llm):
    runner = llm.model_runner
    kv = runner.kv_cache
    kv[:, :, 0:3].normal_()
    saved = kv[:, :, 0:3].clone()
    runner.swap_blocks([(0, 2), (1, 0), (2, 1)], [])
    kv[:, :, 0:3].zero_()
    runner.swap_blocks([], [(2, 40), (0, 41), (1, 42)])
    torch.cuda.synchronize()
    assert torch.equal(kv[:, :, 40:43], saved)


def run_preempted(llm, prompt, n, preempt_after):
    low = llm.add_request(prompt, SamplingParams(temperature=GREEDY, max_tokens=n, priority=1))
    while low.num_completion_tokens < preempt_after:
        llm.step()
    llm.add_request("Say hi.", SamplingParams(temperature=GREEDY, max_tokens=8, priority=0))
    while not llm.is_finished():
        llm.step()
    return low


@pytest.mark.parametrize("mode", ["swap", "recompute", "pause"])
def test_preempted_request_matches_undisturbed_run(llm, mode):
    prompt = "Explain how a multi-level feedback queue scheduler works, step by step."
    n = 48
    ref = llm.generate([prompt], SamplingParams(temperature=GREEDY, max_tokens=n), use_tqdm=False)[0]["token_ids"]
    llm.scheduler.preemption_mode = "swap" if mode == "pause" else mode
    llm.scheduler.pause_victims = mode == "pause"
    before = dict(llm.scheduler.stats)
    low = run_preempted(llm, prompt, n, preempt_after=20)
    llm.scheduler.preemption_mode = "swap"
    llm.scheduler.pause_victims = False

    stats = llm.scheduler.stats
    assert low.num_preemptions == 1
    if mode == "pause":
        assert stats["pauses"] == before["pauses"] + 1 and stats["swap_outs"] == before["swap_outs"]
    elif mode == "swap":
        assert stats["swap_outs"] == before["swap_outs"] + 1 and stats["recomputes"] == before["recomputes"]
    else:
        assert stats["recomputes"] == before["recomputes"] + 1
    if mode in ("swap", "pause"):
        assert low.completion_token_ids == ref    # the restored / untouched KV cache is bit-exact
    else:
        # Re-prefilling rebuilds the KV with the prefill kernel instead of token-by-token decode,
        # so bf16 rounding differs slightly and greedy output may drift many tokens later.
        assert low.completion_token_ids[:20] == ref[:20] and len(low.completion_token_ids) == n
