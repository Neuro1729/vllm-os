import random
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
EOS = 99


def make_scheduler(**overrides):
    fields = dict(
        max_num_seqs=2,
        max_num_batched_tokens=64,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=32,
        scheduling_policy="priority",
        preemption_mode="recompute",
        num_cpu_blocks=0,
    )
    fields.update(overrides)
    return Scheduler(SimpleNamespace(**fields))


def make_seq(n, start=0, **sampling):
    sampling.setdefault("ignore_eos", True)
    return Sequence(list(range(start, start + n)), SamplingParams(**sampling))


def step(sched, token_id=7):
    seqs, is_prefill = sched.schedule()
    sched.take_swap_ops()
    sched.postprocess(seqs, [token_id] * len(seqs), is_prefill)
    return seqs, is_prefill


def run_to_completion(sched, max_steps=2000):
    for _ in range(max_steps):
        if sched.is_finished():
            return
        step(sched)
        check_invariants(sched)
    raise AssertionError("scheduler did not drain within max_steps")


def check_invariants(sched):
    bm = sched.block_manager
    seqs = list(sched.waiting) + list(sched.running)
    assert len({id(s) for s in seqs}) == len(seqs), "sequence queued twice"
    refs = {}
    for seq in seqs:
        assert not (seq.block_table and seq.cpu_block_table)
        assert (seq.status == SequenceStatus.SWAPPED) == bool(seq.cpu_block_table)
        for block_id in seq.block_table:
            refs[block_id] = refs.get(block_id, 0) + 1
    for block_id, n in refs.items():
        assert bm.blocks[block_id].ref_count == n
    assert set(refs) == bm.used_block_ids
    cpu_owned = [b for s in seqs for b in s.cpu_block_table]
    assert len(cpu_owned) == len(set(cpu_owned))
    assert len(cpu_owned) + len(bm.free_cpu_block_ids) + len(bm.pending_free_cpu_block_ids) == bm.num_cpu_blocks
    # prefix-cache hash chain of every full computed block must be intact
    for seq in sched.running:
        h = -1
        for i in range(seq.num_cached_tokens // BLOCK_SIZE):
            h = BlockManager.compute_hash(seq.block(i), h)
            block = bm.blocks[seq.block_table[i]]
            assert block.hash == h and block.token_ids == seq.block(i)


def assert_all_released(sched):
    bm = sched.block_manager
    bm.release_pending_cpu_blocks()
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids
    assert len(bm.free_cpu_block_ids) == bm.num_cpu_blocks


def fill_batch(sched, n=2, priority=1, prompt_len=6, max_tokens=50):
    seqs = [make_seq(prompt_len, start=100 * i, priority=priority, max_tokens=max_tokens) for i in range(n)]
    for seq in seqs:
        sched.add(seq)
    step(sched)
    assert all(s.status == SequenceStatus.RUNNING for s in seqs)
    return seqs


@pytest.fixture(autouse=True)
def block_size_on_sequence():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


class TestFullBatch:
    """vllm-project/vllm#40004: running == max_num_seqs and a more important request arrives."""

    def test_fcfs_keeps_high_priority_waiting(self):
        sched = make_scheduler(scheduling_policy="fcfs")
        fill_batch(sched)
        urgent = make_seq(5, start=900, priority=0)
        sched.add(urgent)

        seqs, is_prefill = sched.schedule()

        assert not is_prefill and urgent not in seqs
        assert sched.stats["preemptions"] == 0

    def test_priority_preempts_lowest_ranked_runner(self):
        sched = make_scheduler()
        a, b = fill_batch(sched)
        urgent = make_seq(5, start=900, priority=0)
        sched.add(urgent)

        seqs, is_prefill = sched.schedule()

        assert is_prefill and seqs == [urgent]
        assert list(sched.running) == [a, urgent]    # b (latest arrival) was the victim
        assert b.status == SequenceStatus.WAITING and b.block_table == [] and b.num_cached_tokens == 0
        assert sched.stats["recomputed_tokens"] == 6    # prompt; the sampled token is not cached yet

    def test_equal_priority_never_preempts(self):
        sched = make_scheduler()
        fill_batch(sched, priority=0)
        sched.add(make_seq(5, start=900, priority=0))

        _, is_prefill = sched.schedule()

        assert not is_prefill
        assert sched.stats["preemptions"] == 0

    def test_no_preemption_if_newcomer_cannot_fit_anyway(self):
        sched = make_scheduler(num_kvcache_blocks=6)
        fill_batch(sched)    # 2 blocks each, 2 free
        sched.add(make_seq(40, start=900, priority=0))    # 10 blocks > 2 free + 4 freeable

        sched.schedule()

        assert sched.stats["preemptions"] == 0

    def test_evicts_several_victims_when_one_slot_is_not_enough_kv(self):
        sched = make_scheduler(num_kvcache_blocks=4)
        a, b = fill_batch(sched)    # 2 blocks each, none free
        urgent = make_seq(13, start=900, priority=0)    # 4 blocks
        sched.add(urgent)

        seqs, _ = sched.schedule()

        assert seqs == [urgent] and list(sched.running) == [urgent]
        assert sched.stats["preemptions"] == 2

    def test_patience_waits_before_preempting(self):
        sched = make_scheduler(preempt_after_steps=3)
        fill_batch(sched, max_tokens=100)
        urgent = make_seq(5, start=900, priority=0)
        sched.add(urgent)

        for _ in range(2):
            _, is_prefill = step(sched)
            assert not is_prefill and sched.stats["preemptions"] == 0
        seqs, is_prefill = step(sched)
        assert is_prefill and seqs == [urgent] and sched.stats["preemptions"] == 1

    def test_patience_lets_a_slot_free_up_on_its_own(self):
        sched = make_scheduler(preempt_after_steps=10)
        fill_batch(sched, max_tokens=3)
        urgent = make_seq(5, start=900, priority=0)
        sched.add(urgent)
        run_to_completion(sched)
        assert urgent.is_finished and sched.stats["preemptions"] == 0

    def test_victim_is_not_preempted_before_min_resident_steps(self):
        sched = make_scheduler(min_resident_steps=3)
        fill_batch(sched)
        sched.add(make_seq(5, start=900, priority=0))

        for _ in range(3):
            _, is_prefill = step(sched)
            assert not is_prefill
        _, is_prefill = step(sched)
        assert is_prefill and sched.stats["preemptions"] == 1

    def test_preempted_work_completes(self):
        sched = make_scheduler()
        low = fill_batch(sched, max_tokens=8)
        urgent = make_seq(5, start=900, priority=0, max_tokens=8)
        sched.add(urgent)
        run_to_completion(sched)

        for seq in low + [urgent]:
            assert seq.is_finished and seq.num_completion_tokens == 8
        assert urgent.finish_time is None    # timing is the engine's job, not the scheduler's
        assert_all_released(sched)


class TestKVPressure:

    def test_free_slots_mean_no_admission_preemption(self):
        # KV is full but the batch is not: the newcomer waits, exactly as in fcfs
        sched = make_scheduler(max_num_seqs=8, num_kvcache_blocks=4)
        a, b = fill_batch(sched)    # all 4 blocks used up
        urgent = make_seq(6, start=900, priority=0)
        sched.add(urgent)

        seqs, is_prefill = sched.schedule()

        assert not is_prefill and urgent not in seqs
        assert sched.stats["preemptions"] == 0

    def test_decode_pressure_evicts_worst_rank_not_tail(self):
        sched = make_scheduler(max_num_seqs=8, num_kvcache_blocks=4)
        low = make_seq(4, start=0, priority=1)
        high = make_seq(4, start=50, priority=0)
        sched.add(low)
        step(sched)
        sched.add(high)
        step(sched)    # both prefilled, all 4 blocks used? 1 + 1, then decode needs new blocks
        while sched.stats["preemptions"] == 0:
            step(sched)
        assert high.status == SequenceStatus.RUNNING
        assert low.status == SequenceStatus.WAITING


class TestSwap:

    def test_victim_is_swapped_and_resumes_without_recompute(self):
        sched = make_scheduler(preemption_mode="swap", num_cpu_blocks=8)
        a, b = fill_batch(sched, max_tokens=6)
        step(sched)    # b: 6 prompt + 2 generated, 7 cached tokens
        cached, table = b.num_cached_tokens, list(b.block_table)
        urgent = make_seq(5, start=900, priority=0, max_tokens=3)
        sched.add(urgent)

        seqs, _ = sched.schedule()
        swap_out, swap_in = sched.take_swap_ops()

        assert seqs == [urgent]
        assert b.status == SequenceStatus.SWAPPED
        assert [g for g, _ in swap_out] == table and swap_in == []
        assert b.num_cached_tokens == cached and len(b.cpu_block_table) == len(table)
        sched.postprocess(seqs, [7], True)

        while b.status == SequenceStatus.SWAPPED:
            seqs, is_prefill = sched.schedule()
            swap_out, swap_in = sched.take_swap_ops()
            sched.postprocess(seqs, [7] * len(seqs), is_prefill)
            check_invariants(sched)
        assert [c for c, _ in swap_in] and b.num_cached_tokens >= cached
        assert sched.stats["recomputes"] == 0
        run_to_completion(sched)
        assert all(s.num_completion_tokens == s.max_tokens for s in (a, b, urgent))
        assert_all_released(sched)

    def test_falls_back_to_recompute_when_host_pool_is_full(self):
        sched = make_scheduler(preemption_mode="swap", num_cpu_blocks=1)
        fill_batch(sched)    # victims need 2 blocks
        sched.add(make_seq(5, start=900, priority=0))
        sched.schedule()
        assert sched.stats["swap_outs"] == 0 and sched.stats["recomputes"] == 1

    def test_auto_mode_swaps_only_long_contexts(self):
        sched = make_scheduler(preemption_mode="auto", swap_min_tokens=10, num_cpu_blocks=16, max_num_seqs=1)
        short = make_seq(4, start=0, priority=2)
        sched.add(short)
        step(sched)
        sched.add(make_seq(4, start=50, priority=1))
        step(sched)
        assert short.status == SequenceStatus.WAITING    # 4 cached tokens: recompute

        sched = make_scheduler(preemption_mode="auto", swap_min_tokens=10, num_cpu_blocks=16, max_num_seqs=1)
        long = make_seq(12, start=0, priority=2)
        sched.add(long)
        step(sched)
        sched.add(make_seq(4, start=50, priority=1))
        step(sched)
        assert long.status == SequenceStatus.SWAPPED

    def test_cpu_blocks_read_by_swap_in_are_not_reused_in_same_step(self):
        sched = make_scheduler(preemption_mode="swap", num_cpu_blocks=2, max_num_seqs=1)
        low = make_seq(6, start=0, priority=1, max_tokens=20)
        sched.add(low)
        step(sched)
        sched.add(make_seq(3, start=50, priority=0, max_tokens=1))
        step(sched)    # low swapped out, urgent finishes after prefill
        assert low.status == SequenceStatus.SWAPPED
        cpu_blocks = list(low.cpu_block_table)
        sched.schedule()
        _, swap_in = sched.take_swap_ops()
        assert [c for c, _ in swap_in] == cpu_blocks
        assert sched.block_manager.pending_free_cpu_block_ids == cpu_blocks
        assert not set(cpu_blocks) & set(sched.block_manager.free_cpu_block_ids)


class TestMLFQ:

    def test_level_follows_total_tokens(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8, 16))
        seq = make_seq(6, max_tokens=20)
        sched.add(seq)
        levels = []
        for _ in range(14):
            step(sched)
            levels.append((seq.num_tokens, seq.level))
        assert dict(levels)[8] == 0 and dict(levels)[9] == 1
        assert dict(levels)[16] == 1 and dict(levels)[17] == 2 and dict(levels)[20] == 2

    def test_long_prompt_starts_low(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8, 16))
        seqs = [make_seq(n) for n in (4, 12, 20)]
        for seq in seqs:
            sched.add(seq)
        assert [s.level for s in seqs] == [0, 1, 2]

    def test_growing_past_a_level_does_not_preempt_with_free_slots(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8,), max_num_seqs=4)
        long = make_seq(6, max_tokens=30)
        sched.add(long)
        for _ in range(10):
            step(sched)
        assert long.level == 1
        short = make_seq(3, start=50, max_tokens=3)
        sched.add(short)
        run_to_completion(sched)
        assert sched.stats["preemptions"] == 0 and long.is_finished and short.is_finished

    def test_full_batch_evicts_the_demoted_long_runner(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8,), max_num_seqs=1)
        long = make_seq(6, max_tokens=100)
        sched.add(long)
        for _ in range(5):
            step(sched)
        assert long.level == 1
        short = make_seq(3, start=50, max_tokens=3)
        sched.add(short)

        seqs, is_prefill = sched.schedule()

        assert is_prefill and seqs == [short]
        assert long.status == SequenceStatus.WAITING

    def test_same_level_never_preempts_without_priorities(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8,), max_num_seqs=1)
        sched.add(make_seq(3, max_tokens=100))
        step(sched)
        sched.add(make_seq(3, start=50))
        _, is_prefill = sched.schedule()
        assert not is_prefill and sched.stats["preemptions"] == 0

    def test_long_newcomer_cannot_evict_short_runner(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8,), max_num_seqs=1)
        sched.add(make_seq(3, max_tokens=100))
        step(sched)
        sched.add(make_seq(12, start=50, priority=-5))    # important, but already level 1
        _, is_prefill = sched.schedule()
        assert not is_prefill

    def test_aging_brings_evicted_request_back(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(4, 8), max_num_seqs=1, aging_steps=3)
        old = make_seq(3, max_tokens=100)
        sched.add(old)
        for _ in range(7):
            step(sched)
        assert old.level == 2
        new = make_seq(3, start=50, max_tokens=100)
        sched.add(new)
        step(sched)    # newcomer at level 0 evicts old
        assert old.status == SequenceStatus.WAITING
        for _ in range(12):
            step(sched)
            if old.status == SequenceStatus.RUNNING:
                break
        assert old.status == SequenceStatus.RUNNING and new.status == SequenceStatus.WAITING
        assert sched.stats["promotions"] >= 1

    def test_priority_widens_the_bands(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(8,), mlfq_priority_scale=2.0)
        high = make_seq(3, priority=-1, max_tokens=20)    # level 0 holds 16 tokens
        low = make_seq(3, start=50, priority=1, max_tokens=20)    # level 0 holds 4 tokens
        sched.add(high)
        sched.add(low)
        demoted_at = {}
        for _ in range(16):
            step(sched)
            for name, seq in (("high", high), ("low", low)):
                if seq.level and name not in demoted_at:
                    demoted_at[name] = seq.num_tokens
        assert demoted_at == {"low": 5, "high": 17}

    def test_same_level_prefers_better_priority(self):
        sched = make_scheduler(scheduling_policy="mlfq", max_num_seqs=1)
        low = make_seq(3, priority=1, max_tokens=50)
        sched.add(low)
        step(sched)
        sched.add(make_seq(3, start=50, priority=0))
        _, is_prefill = sched.schedule()
        assert is_prefill and low.status == SequenceStatus.WAITING

    def test_max_level_keeps_a_priority_on_top(self):
        sched = make_scheduler(scheduling_policy="mlfq", mlfq_level_tokens=(4, 8), mlfq_max_levels={0: 0},
                               max_num_seqs=1)
        high = make_seq(3, priority=0, max_tokens=50)
        sched.add(high)
        for _ in range(10):
            step(sched)
        assert high.level == 0
        sched.add(make_seq(3, start=50, priority=1))
        _, is_prefill = sched.schedule()
        assert not is_prefill    # pinned at level 0, still outranks the fresh priority 1


class TestStarvation:

    def test_low_priority_is_eventually_admitted(self):
        sched = make_scheduler(max_num_seqs=1, starvation_steps=5)
        low = make_seq(3, priority=1, max_tokens=5)
        sched.add(low)
        step(sched)
        sched.add(make_seq(3, start=50, priority=0, max_tokens=1000))
        step(sched)    # low evicted
        assert low.status == SequenceStatus.WAITING
        for _ in range(30):
            step(sched)
            if low.tier_boost:
                break
        # one boost ties with priority 0; two boosts beat it and preempt
        for _ in range(30):
            step(sched)
            if low.status == SequenceStatus.RUNNING:
                break
        assert low.status == SequenceStatus.RUNNING and low.tier_boost == 0


@pytest.mark.parametrize("policy", ["fcfs", "priority", "mlfq"])
@pytest.mark.parametrize("mode", ["recompute", "swap", "auto"])
@pytest.mark.parametrize("seed", range(5))
def test_random_workload_drains_cleanly(policy, mode, seed):
    rng = random.Random(seed)
    sched = make_scheduler(
        scheduling_policy=policy, preemption_mode=mode, num_cpu_blocks=12, swap_min_tokens=8,
        max_num_seqs=4, max_num_batched_tokens=24, num_kvcache_blocks=20,
        mlfq_level_tokens=(12, 24, 36), aging_steps=7, starvation_steps=25,
        mlfq_priority_scale=rng.choice([1.0, 2.0]), mlfq_max_levels=rng.choice([{}, {0: 1}]),
        min_resident_steps=rng.choice([0, 2]), preempt_after_steps=rng.choice([0, 3]),
    )
    seqs = []
    pending = [(rng.randrange(60), make_seq(rng.randint(1, 30), start=rng.randrange(5) * 40,
                                            priority=rng.randrange(3), max_tokens=rng.randint(1, 25)))
               for _ in range(25)]
    pending.sort(key=lambda x: x[0])
    for t in range(5000):
        while pending and pending[0][0] <= t:
            seqs.append(pending.pop(0)[1])
            sched.add(seqs[-1])
        if sched.is_finished():
            if not pending:
                break
            continue
        step(sched, token_id=rng.choice([7, 8, EOS]))
        check_invariants(sched)
        assert len(sched.running) <= sched.max_num_seqs
    assert not pending and sched.is_finished()
    for seq in seqs:
        assert seq.is_finished and 1 <= seq.num_completion_tokens <= seq.max_tokens
    assert_all_released(sched)
