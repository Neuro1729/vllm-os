from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

BLOCK_SIZE = 4
EOS = 99


def make_config(**overrides):
    # Config.__post_init__ needs a real model directory, so pass a plain namespace.
    # The scheduler only reads the fields below.
    fields = dict(
        max_num_seqs=8,
        max_num_batched_tokens=64,
        eos=EOS,
        kvcache_block_size=BLOCK_SIZE,
        num_kvcache_blocks=16,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_scheduler(**overrides):
    return Scheduler(make_config(**overrides))


def make_seq(token_ids, **sampling):
    return Sequence(list(token_ids), SamplingParams(**sampling))


def step(scheduler, token_id=7):
    # Mirrors LLMEngine.step, with a fixed token id standing in for the model.
    seqs, is_prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [token_id] * len(seqs), is_prefill)
    return seqs, is_prefill


def run_to_completion(scheduler, max_steps=100):
    for _ in range(max_steps):
        if scheduler.is_finished():
            return
        step(scheduler)
    raise AssertionError("scheduler did not drain within max_steps")


def assert_kv_cache_released(scheduler):
    bm = scheduler.block_manager
    assert len(bm.free_block_ids) == len(bm.blocks)
    assert not bm.used_block_ids


@pytest.fixture(autouse=True)
def block_size_on_sequence():
    # LLMEngine sets Sequence.block_size from the config; do the same here.
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


class TestAdmission:

    def test_single_prompt_is_prefilled_in_one_step(self):
        sched = make_scheduler()
        seq = make_seq(range(6))
        sched.add(seq)

        seqs, is_prefill = sched.schedule()

        assert seqs == [seq]
        assert is_prefill
        assert seq.num_scheduled_tokens == 6
        assert len(seq.block_table) == 2  # 6 tokens / block size 4

    def test_admission_stops_at_max_num_seqs(self):
        sched = make_scheduler(max_num_seqs=2)
        a, b, c = make_seq([0, 1, 2]), make_seq([10, 11, 12]), make_seq([20, 21, 22])
        for seq in (a, b, c):
            sched.add(seq)

        seqs, is_prefill = sched.schedule()

        assert seqs == [a, b]
        assert is_prefill
        assert list(sched.waiting) == [c]

    def test_admission_stops_at_max_num_batched_tokens(self):
        sched = make_scheduler(max_num_batched_tokens=10)
        a = make_seq([1, 2, 3, 4])
        b = make_seq([5, 6, 7, 8])
        c = make_seq([9, 10, 11, 12])  # 4 + 4 + 4 > 10
        for seq in (a, b, c):
            sched.add(seq)

        seqs, _ = sched.schedule()

        assert seqs == [a, b]
        assert list(sched.waiting) == [c]

    def test_only_first_sequence_may_be_chunked(self):
        sched = make_scheduler(max_num_batched_tokens=6)
        short = make_seq([1, 2, 3])
        long = make_seq(range(10, 20))
        sched.add(short)
        sched.add(long)

        seqs, _ = sched.schedule()

        assert seqs == [short]
        assert list(sched.waiting) == [long]

    def test_long_prompt_is_chunked_across_steps(self):
        sched = make_scheduler(max_num_batched_tokens=6)
        seq = make_seq(range(10))
        sched.add(seq)

        seqs, is_prefill = step(sched)
        assert seqs == [seq] and is_prefill
        assert seq.num_cached_tokens == 6
        assert seq.num_completion_tokens == 0  # no token is kept for a partial chunk
        assert seq.status == SequenceStatus.WAITING
        assert list(sched.waiting) == [seq]

        seqs, _ = step(sched)
        assert seqs == [seq]
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_completion_tokens == 1
        assert list(sched.waiting) == []
        assert list(sched.running) == [seq]


class TestDecode:

    def test_running_sequences_decode_one_token_each(self):
        sched = make_scheduler()
        a = make_seq([1, 2, 3])
        b = make_seq([4, 5, 6, 7, 8])
        sched.add(a)
        sched.add(b)
        step(sched)  # prefill both

        seqs, is_prefill = sched.schedule()

        assert seqs == [a, b]
        assert not is_prefill
        assert [s.num_scheduled_tokens for s in seqs] == [1, 1]

    def test_sampled_tokens_are_appended(self):
        sched = make_scheduler()
        seq = make_seq([1, 2, 3])
        sched.add(seq)

        step(sched, token_id=7)  # prefill samples 7
        step(sched, token_id=8)  # decode samples 8

        assert seq.completion_token_ids == [7, 8]

    def test_preempted_sequence_is_requeued_then_finishes(self):
        # Two blocks total: each prompt takes one, and the first decode step
        # needs a second block for the first sequence.
        sched = make_scheduler(num_kvcache_blocks=2)
        a = make_seq([1, 2, 3, 4], max_tokens=3)
        b = make_seq([5, 6, 7, 8], max_tokens=3)
        sched.add(a)
        sched.add(b)
        step(sched)  # prefill both, cache is now full
        step(sched)  # decode: b gets preempted to make room for a

        assert b.status == SequenceStatus.WAITING
        assert b.is_prefill
        assert b.block_table == []
        assert b.num_cached_tokens == 0
        assert list(sched.waiting) == [b]

        run_to_completion(sched)
        assert a.is_finished
        assert b.is_finished
        assert_kv_cache_released(sched)


class TestFinishing:

    def test_eos_finishes_sequence_and_releases_blocks(self):
        sched = make_scheduler()
        seq = make_seq([1, 2, 3])
        sched.add(seq)

        step(sched, token_id=7)
        step(sched, token_id=EOS)

        assert seq.status == SequenceStatus.FINISHED
        assert seq.completion_token_ids == [7, EOS]
        assert not sched.running
        assert_kv_cache_released(sched)

    def test_ignore_eos_keeps_generating(self):
        sched = make_scheduler()
        seq = make_seq([1, 2, 3], ignore_eos=True, max_tokens=3)
        sched.add(seq)

        step(sched, token_id=EOS)
        step(sched, token_id=EOS)
        assert seq.status == SequenceStatus.RUNNING

        step(sched, token_id=EOS)
        assert seq.status == SequenceStatus.FINISHED

    def test_max_tokens_finishes_sequence(self):
        sched = make_scheduler()
        seq = make_seq([1, 2, 3], max_tokens=2)
        sched.add(seq)

        step(sched)
        assert seq.status == SequenceStatus.RUNNING
        step(sched)
        assert seq.status == SequenceStatus.FINISHED
        assert len(seq.completion_token_ids) == 2


class TestPrefixCache:

    def test_shared_full_block_prefix_is_reused(self):
        sched = make_scheduler()
        first = make_seq([1, 2, 3, 4, 5, 6, 7, 8], max_tokens=2)
        sched.add(first)
        step(sched)  # prefill hashes both full blocks

        second = make_seq([1, 2, 3, 4, 9, 9, 9, 9, 9], max_tokens=2)
        sched.add(second)
        seqs, is_prefill = sched.schedule()

        assert seqs == [second]
        assert is_prefill
        assert second.num_cached_tokens == BLOCK_SIZE  # first block came from cache
        assert second.num_scheduled_tokens == 5  # only the uncached tail is computed
        assert second.block_table[0] == first.block_table[0]
        assert sched.block_manager.blocks[first.block_table[0]].ref_count == 2

        sched.postprocess(seqs, [7], is_prefill)
        run_to_completion(sched)
        assert_kv_cache_released(sched)


class TestLifecycle:

    def test_is_finished_tracks_queues(self):
        sched = make_scheduler()
        assert sched.is_finished()

        sched.add(make_seq([1, 2], max_tokens=1))
        assert not sched.is_finished()

        run_to_completion(sched)
        assert sched.is_finished()

    def test_mixed_workload_releases_all_blocks(self):
        sched = make_scheduler(num_kvcache_blocks=32, max_num_batched_tokens=16, max_num_seqs=3)
        for i, n in enumerate([3, 9, 5, 12, 2, 7]):
            sched.add(make_seq(range(100 * i, 100 * i + n), max_tokens=4))

        run_to_completion(sched)

        assert_kv_cache_released(sched)
