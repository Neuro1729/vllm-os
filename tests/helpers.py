"""Shared helpers for the CPU scheduler tests. Nothing here touches CUDA."""
from collections import Counter
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
EOS = 99

# Every field the scheduler reads from Config, with the production defaults.
CONFIG_DEFAULTS = dict(
    max_num_seqs=8,
    max_num_batched_tokens=64,
    eos=EOS,
    kvcache_block_size=BLOCK_SIZE,
    num_kvcache_blocks=16,
    scheduling_policy="fcfs",
    enable_mlfq_preemption=False,
    mlfq_prefill_quanta=(512, 1024, 2048),
    mlfq_decode_quanta=(16, 48, 192),
    mlfq_aging_interval=5.0,
    mlfq_first_token_promotion=False,
    max_proactive_preemptions_per_step=1,
    preemption_cooldown_steps=0,
    max_preemptions_per_request_window=2,
    preemption_window_steps=64,
)


class FakeClock:
    """Monotonic-clock stand-in so aging tests do not sleep."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def make_config(**overrides):
    # Config.__post_init__ needs a real model directory, so tests pass a namespace
    # with the same fields the scheduler reads.
    return SimpleNamespace(**{**CONFIG_DEFAULTS, **overrides})


def make_scheduler(clock=None, **overrides) -> Scheduler:
    return Scheduler(make_config(**overrides), clock=clock or FakeClock())


def make_mlfq(clock=None, **overrides) -> Scheduler:
    return make_scheduler(clock=clock, scheduling_policy="mlfq", **overrides)


def make_seq(token_ids, **sampling) -> Sequence:
    # Small default output budget so requests fit the 16-block test cache.
    sampling.setdefault("max_tokens", 4)
    return Sequence(list(token_ids), SamplingParams(**sampling))


def step(scheduler: Scheduler, token_id: int = 7) -> list[Sequence]:
    # Mirrors LLMEngine.step, with a fixed token standing in for the model output.
    batch = scheduler.schedule()
    scheduler.postprocess(batch, [token_id] * len(batch))
    return batch


def run_to_completion(scheduler: Scheduler, max_steps: int = 500):
    for _ in range(max_steps):
        if scheduler.is_finished():
            return
        step(scheduler)
    raise AssertionError("scheduler did not drain within max_steps")


def assert_kv_invariants(scheduler: Scheduler):
    """Block accounting must be consistent after every step."""
    bm = scheduler.block_manager
    free, used = set(bm.free_block_ids), set(bm.used_block_ids)
    assert len(bm.free_block_ids) == len(free), "duplicate entry in free list (double free)"
    assert free.isdisjoint(used)
    assert len(free) + len(used) == len(bm.blocks)
    owners = Counter(bid for seq in scheduler.running for bid in seq.block_table)
    for block in bm.blocks:
        expected = owners.get(block.block_id, 0)
        assert block.ref_count == expected, f"block {block.block_id}: ref_count {block.ref_count} != owners {expected}"
        assert (block.block_id in used) == (expected > 0)
    for seq in scheduler.waiting:
        assert not seq.block_table, "waiting sequence still owns blocks"
        assert seq.num_cached_tokens == 0
    for seq in scheduler.running:
        assert seq.status is SequenceStatus.RUNNING
        # Every computed token must have a block behind it.
        assert len(seq.block_table) * BLOCK_SIZE >= seq.num_cached_tokens


def assert_kv_cache_released(scheduler: Scheduler):
    bm = scheduler.block_manager
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids
