"""Randomized workloads: scheduler invariants must hold on every step. CPU only."""
import random

import pytest

from helpers import FakeClock, assert_kv_cache_released, assert_kv_invariants, make_scheduler, make_seq
from nanovllm.engine.sequence import SequenceStatus


def run_random_workload(seed: int, policy: str, preemption: bool):
    rng = random.Random(seed)
    clock = FakeClock()
    sched = make_scheduler(
        clock=clock,
        scheduling_policy=policy,
        enable_mlfq_preemption=preemption,
        max_num_seqs=rng.choice([2, 3, 4, 6]),
        max_num_batched_tokens=rng.choice([5, 8, 16, 32]),
        num_kvcache_blocks=rng.choice([6, 8, 10, 16]),  # small: KV pressure is the common case
        mlfq_aging_interval=2.0,
        mlfq_prefill_quanta=(8, 16, 32),
        mlfq_decode_quanta=(3, 6, 12),
        preemption_cooldown_steps=rng.choice([0, 2]),
        max_preemptions_per_request_window=rng.choice([1, 3]),
        max_proactive_preemptions_per_step=rng.choice([1, 2]),
    )
    # Small vocabulary so prompts share prefixes and exercise the prefix cache.
    seqs = []
    for _ in range(rng.randint(8, 16)):
        prompt = [rng.randint(0, 3) for _ in range(rng.randint(1, 14))]
        seq = make_seq(prompt, max_tokens=rng.randint(1, 6), ignore_eos=True)
        if policy == "mlfq":
            seq.feedback.level = rng.randint(0, 3)
        seqs.append(seq)

    arrivals = list(seqs)
    completions = {seq.seq_id: [] for seq in seqs}
    budget = sched.max_num_batched_tokens
    for _ in range(3000):
        if arrivals and rng.random() < 0.6:
            sched.add(arrivals.pop(0))
        if not sched.is_finished():
            batch = sched.schedule()
            assert len({s.seq_id for s in batch}) == len(batch), "sequence scheduled twice in one step"
            assert sum(s.num_scheduled_tokens for s in batch) <= budget
            assert len(sched.running) <= sched.max_num_seqs
            sched.postprocess(batch, [rng.randint(0, 3) for _ in batch])
        clock.advance(1.0)
        assert_kv_invariants(sched)
        for seq in seqs:  # completion tokens only ever grow: no loss, no regeneration
            before = completions[seq.seq_id]
            now = seq.completion_token_ids
            assert now[:len(before)] == before
            completions[seq.seq_id] = list(now)
        if not arrivals and sched.is_finished():
            break
    return sched, seqs


@pytest.mark.parametrize("seed", range(25))
def test_fcfs_random_workload_drains_and_releases_kv(seed):
    sched, seqs = run_random_workload(seed, "fcfs", preemption=False)
    assert all(s.status in (SequenceStatus.FINISHED, SequenceStatus.REJECTED) for s in seqs)
    for seq in seqs:
        if seq.status is SequenceStatus.FINISHED:
            assert seq.num_completion_tokens == seq.max_tokens
    assert_kv_cache_released(sched)


@pytest.mark.parametrize("seed", range(25))
def test_mlfq_random_workload_drains_without_preemption(seed):
    sched, seqs = run_random_workload(seed, "mlfq", preemption=False)
    assert all(s.status in (SequenceStatus.FINISHED, SequenceStatus.REJECTED) for s in seqs)
    assert_kv_cache_released(sched)


@pytest.mark.parametrize("seed", range(40))
def test_mlfq_with_preemption_random_workload_drains(seed):
    # Liveness: anti-thrashing must not stop requests from ever finishing.
    sched, seqs = run_random_workload(seed, "mlfq", preemption=True)
    assert all(s.status in (SequenceStatus.FINISHED, SequenceStatus.REJECTED) for s in seqs)
    assert_kv_cache_released(sched)
    for seq in seqs:
        if seq.status is SequenceStatus.FINISHED:
            assert seq.num_completion_tokens == seq.max_tokens
