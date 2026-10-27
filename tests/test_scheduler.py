"""FCFS unified token-budget scheduler (V1). Runs on CPU; no model is loaded."""
import pytest

from helpers import (
    BLOCK_SIZE, EOS, assert_kv_cache_released, assert_kv_invariants,
    make_scheduler, make_seq, run_to_completion, step,
)
from nanovllm.engine.sequence import SequenceStatus


class TestAdmission:

    def test_single_prompt_is_prefilled_in_one_step(self):
        sched = make_scheduler()
        seq = make_seq(range(6))
        sched.add(seq)

        batch = sched.schedule()

        assert batch == [seq]
        assert seq.num_scheduled_tokens == 6
        assert len(seq.block_table) == 2  # 6 tokens / block size 4

    def test_admission_stops_at_max_num_seqs(self):
        sched = make_scheduler(max_num_seqs=2)
        a, b, c = make_seq([0, 1, 2]), make_seq([10, 11, 12]), make_seq([20, 21, 22])
        for seq in (a, b, c):
            sched.add(seq)

        batch = sched.schedule()

        assert batch == [a, b]
        assert list(sched.waiting) == [c]
        assert len(sched.running) == 2

    def test_admission_fills_remaining_budget_with_a_chunk(self):
        # 4 + 4 + 4 > 10: the third prompt is admitted with the 2 tokens that remain.
        sched = make_scheduler(max_num_batched_tokens=10)
        a, b, c = make_seq([1, 2, 3, 4]), make_seq([5, 6, 7, 8]), make_seq([9, 10, 11, 12])
        for seq in (a, b, c):
            sched.add(seq)

        batch = sched.schedule()

        assert batch == [a, b, c]
        assert [s.num_scheduled_tokens for s in batch] == [4, 4, 2]
        assert sum(s.num_scheduled_tokens for s in batch) <= 10
        assert c.status == SequenceStatus.RUNNING
        assert not sched.waiting

    def test_batch_never_exceeds_token_budget(self):
        sched = make_scheduler(max_num_batched_tokens=6, num_kvcache_blocks=64)
        seqs = [make_seq(range(100 * i, 100 * i + 10), max_tokens=3) for i in range(4)]
        for seq in seqs:
            sched.add(seq)
        for _ in range(100):
            if sched.is_finished():
                break
            batch = sched.schedule()
            assert sum(s.num_scheduled_tokens for s in batch) <= 6
            sched.postprocess(batch, [1] * len(batch))
        assert all(s.is_finished for s in seqs)


class TestChunkedPrefill:

    def test_long_prompt_is_chunked_across_steps(self):
        sched = make_scheduler(max_num_batched_tokens=6)
        seq = make_seq(range(10))
        sched.add(seq)

        batch = step(sched)
        assert batch == [seq]
        assert seq.num_cached_tokens == 6
        assert seq.num_completion_tokens == 0  # a partial chunk produces no token
        assert seq.status == SequenceStatus.RUNNING  # admitted: its KV progress is kept

        batch = step(sched)
        assert batch == [seq]
        assert seq.num_cached_tokens == 10
        assert seq.num_completion_tokens == 1
        assert list(sched.running) == [seq]

    def test_multiple_partial_prefills_share_the_budget(self):
        sched = make_scheduler(max_num_batched_tokens=6, num_kvcache_blocks=32)
        a = make_seq(range(10), max_tokens=3)
        b = make_seq(range(50, 60), max_tokens=3)
        sched.add(a)
        sched.add(b)

        batch = step(sched)
        assert batch == [a]
        assert a.num_cached_tokens == 6 and b.num_cached_tokens == 0

        batch = step(sched)  # a finishes its prompt (4), b gets the remaining 2
        assert batch == [a, b]
        assert a.num_cached_tokens == 10 and a.num_completion_tokens == 1
        assert b.num_cached_tokens == 2 and b.num_completion_tokens == 0

        run_to_completion(sched)
        assert a.num_completion_tokens == 3 and b.num_completion_tokens == 3
        assert_kv_cache_released(sched)

    def test_long_prefill_does_not_starve_running_decodes(self):
        sched = make_scheduler(max_num_batched_tokens=8, num_kvcache_blocks=64)
        short = make_seq([1, 2, 3], max_tokens=50, ignore_eos=True)
        sched.add(short)
        step(sched)  # short is now decoding
        long = make_seq(range(40), max_tokens=2)
        sched.add(long)

        for _ in range(4):
            batch = sched.schedule()
            assert short in batch  # decode is always served first
            assert short.num_scheduled_tokens == 1
            sched.postprocess(batch, [1] * len(batch))
        assert long.num_cached_tokens > 0

    def test_decode_and_prefill_share_one_step(self):
        sched = make_scheduler(max_num_batched_tokens=16)
        a = make_seq([1, 2, 3])
        sched.add(a)
        step(sched)  # a is decoding
        b = make_seq(range(20, 25))
        sched.add(b)

        batch = sched.schedule()

        assert batch == [a, b]
        assert a.num_scheduled_tokens == 1 and a.in_decode
        assert b.num_scheduled_tokens == 5 and not b.in_decode


class TestDecode:

    def test_running_sequences_decode_one_token_each(self):
        sched = make_scheduler()
        a = make_seq([1, 2, 3])
        b = make_seq([4, 5, 6, 7, 8])
        sched.add(a)
        sched.add(b)
        step(sched)  # prefill both

        batch = sched.schedule()

        assert batch == [a, b]
        assert [s.num_scheduled_tokens for s in batch] == [1, 1]

    def test_sampled_tokens_are_appended(self):
        sched = make_scheduler()
        seq = make_seq([1, 2, 3])
        sched.add(seq)

        step(sched, token_id=7)  # prefill samples 7
        step(sched, token_id=8)  # decode samples 8

        assert seq.completion_token_ids == [7, 8]

    def test_kv_preemption_requeues_victim_then_everything_finishes(self):
        # Two blocks total: each prompt takes one, and the first decode step
        # needs a second block for the first sequence.
        sched = make_scheduler(num_kvcache_blocks=2)
        a = make_seq([1, 2, 3, 4], max_tokens=3)
        b = make_seq([5, 6, 7, 8], max_tokens=3)
        sched.add(a)
        sched.add(b)
        step(sched)  # prefill both, cache is now full
        step(sched)  # decode: b (the later admission) is preempted to make room for a

        assert b.status == SequenceStatus.WAITING
        assert b.block_table == []
        assert b.num_cached_tokens == 0
        assert list(sched.waiting) == [b]
        assert_kv_invariants(sched)

        run_to_completion(sched)
        assert a.is_finished and b.is_finished
        assert len(b.completion_token_ids) == 3
        assert_kv_cache_released(sched)

    def test_resumed_sequence_recomputes_prompt_and_generated_tokens(self):
        sched = make_scheduler(num_kvcache_blocks=2)
        a = make_seq([1, 2, 3, 4], max_tokens=4)
        b = make_seq([5, 6, 7, 8], max_tokens=4)
        sched.add(a)
        sched.add(b)
        step(sched, token_id=11)
        step(sched, token_id=12)  # b preempted after one generated token
        assert b.completion_token_ids == [11]

        batch = sched.schedule()  # a is still running; b waits for room
        sched.postprocess(batch, [13] * len(batch))
        run_to_completion(sched)

        assert b.completion_token_ids[0] == 11  # earlier output is preserved, not regenerated
        assert len(b.completion_token_ids) == 4
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
        batch = sched.schedule()

        assert batch == [first, second]  # first is still decoding its 2nd token
        assert first.num_scheduled_tokens == 1
        assert second.num_cached_tokens == BLOCK_SIZE  # first block came from cache
        assert second.num_scheduled_tokens == 5  # only the uncached tail is computed
        assert second.block_table[0] == first.block_table[0]
        assert sched.block_manager.blocks[first.block_table[0]].ref_count == 2
        assert_kv_invariants(sched)

        sched.postprocess(batch, [7] * len(batch))
        run_to_completion(sched)
        assert_kv_cache_released(sched)

    def test_freed_prefix_block_is_reused_by_a_later_request(self):
        sched = make_scheduler()
        first = make_seq([1, 2, 3, 4, 5], max_tokens=1)
        sched.add(first)
        step(sched)  # first finishes immediately and frees its own blocks
        second = make_seq([1, 2, 3, 4, 6], max_tokens=5, ignore_eos=True)
        sched.add(second)
        batch = sched.schedule()
        assert second.num_cached_tokens == BLOCK_SIZE  # the freed, hashed block was reused
        assert second.num_scheduled_tokens == 1
        sched.postprocess(batch, [7])
        assert_kv_invariants(sched)
        run_to_completion(sched)
        assert_kv_cache_released(sched)


class TestCancelAndReject:

    def test_cancel_waiting_sequence(self):
        sched = make_scheduler(max_num_seqs=1)
        a, b = make_seq([1, 2]), make_seq([3, 4])
        sched.add(a)
        sched.add(b)
        step(sched)

        assert sched.cancel(b)
        assert b.status == SequenceStatus.CANCELLED
        assert not sched.waiting
        assert not sched.cancel(b)  # already gone

    def test_cancel_running_sequence_releases_blocks(self):
        sched = make_scheduler()
        seq = make_seq(range(8), max_tokens=10)
        sched.add(seq)
        step(sched)

        assert sched.cancel(seq)
        assert seq.status == SequenceStatus.CANCELLED
        assert_kv_cache_released(sched)
        assert sched.is_finished()

    def test_impossible_request_is_rejected_immediately(self):
        sched = make_scheduler(num_kvcache_blocks=2)
        too_big = make_seq(range(20), max_tokens=4)  # needs 6 blocks, only 2 exist

        assert not sched.add(too_big)
        assert too_big.status == SequenceStatus.REJECTED
        assert sched.is_finished()
        assert sched.stats.rejected == 1

    def test_unschedulable_state_raises_instead_of_looping(self):
        # Nothing is running, yet the only waiter needs more blocks than are free.
        # An empty batch here would make generate() spin forever, so it must raise.
        sched = make_scheduler(num_kvcache_blocks=3)
        for _ in range(2):  # two blocks held by sequences that are not in the scheduler
            sched.block_manager.allocate(make_seq(range(4)), 0)
        waiter = make_seq(range(8), max_tokens=1)  # needs 2 blocks, 1 is free
        sched.add(waiter)

        with pytest.raises(RuntimeError, match="scheduler stalled"):
            sched.schedule()


class TestDeterminism:

    def test_same_inputs_produce_same_batches(self):
        def trace():
            sched = make_scheduler(max_num_batched_tokens=7, num_kvcache_blocks=24, max_num_seqs=3)
            seqs = [make_seq(range(100 * i, 100 * i + 3 + 4 * i), max_tokens=3 + i) for i in range(6)]
            for seq in seqs:
                sched.add(seq)
            index = {seq.seq_id: i for i, seq in enumerate(seqs)}
            events = []
            for _ in range(200):
                if sched.is_finished():
                    break
                batch = sched.schedule()
                events.append([(index[s.seq_id], s.num_scheduled_tokens) for s in batch])
                sched.postprocess(batch, [5] * len(batch))
            return events

        assert trace() == trace()


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

        for _ in range(500):
            if sched.is_finished():
                break
            step(sched)
            assert_kv_invariants(sched)

        assert sched.is_finished()
        assert_kv_cache_released(sched)
