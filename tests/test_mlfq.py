"""MLFQ feedback policy: levels, quanta, demotion, aging, first-token promotion. CPU only."""
from nanovllm.engine import mlfq
from nanovllm.engine.mlfq import FeedbackState, MLFQConfig
from nanovllm.engine.sequence import SequenceStatus
from helpers import FakeClock, make_mlfq, make_seq, step

CFG = MLFQConfig(prefill_quanta=(512, 1024, 2048), decode_quanta=(16, 48, 192), aging_interval=5.0)


class TestPolicyUnits:

    def test_new_request_enters_q0(self):
        assert FeedbackState().level == 0

    def test_prefill_quantum_demotes_once_and_resets_level_counter(self):
        state = FeedbackState()
        assert not mlfq.account_service(state, CFG, decode=False, tokens=300, step=1, queue_seq=0)
        assert state.level == 0
        assert mlfq.account_service(state, CFG, decode=False, tokens=300, step=2, queue_seq=0)
        assert state.level == 1
        assert state.level_prefill_service == 0  # counter reset on demotion
        assert state.prefill_service == 600      # lifetime total preserved
        assert state.demotion_history == [(2, 0, 1)]

    def test_decode_quantum_is_separate_from_prefill_quantum(self):
        state = FeedbackState()
        mlfq.account_service(state, CFG, decode=False, tokens=500, step=1, queue_seq=0)  # prefill at Q0: 500 < 512
        assert state.level == 0
        for _ in range(16):
            demoted = mlfq.account_service(state, CFG, decode=True, tokens=1, step=2, queue_seq=0)
        assert demoted and state.level == 1
        assert state.level_decode_service == 0

    def test_second_level_uses_its_own_decode_quantum(self):
        state = FeedbackState(level=1)
        for _ in range(47):
            assert not mlfq.account_service(state, CFG, decode=True, tokens=1, step=1, queue_seq=0)
        assert mlfq.account_service(state, CFG, decode=True, tokens=1, step=1, queue_seq=0)
        assert state.level == 2

    def test_bottom_level_never_demotes(self):
        state = FeedbackState(level=3)
        for _ in range(10_000):
            assert not mlfq.account_service(state, CFG, decode=True, tokens=1, step=1, queue_seq=0)
        assert state.level == 3 and state.demotion_history == []

    def test_aging_waits_full_interval_then_promotes(self):
        state = FeedbackState(level=3)
        assert not mlfq.age(state, CFG, now=0.0, step=1, queue_seq=0)  # starts the wait clock
        assert not mlfq.age(state, CFG, now=4.99, step=2, queue_seq=0)
        assert mlfq.age(state, CFG, now=5.0, step=3, queue_seq=0)
        assert state.level == 2
        assert state.aging_history == [(3, 3, 2)]

    def test_aging_clock_restarts_so_levels_are_not_skipped(self):
        state = FeedbackState(level=3)
        mlfq.age(state, CFG, now=0.0, step=1, queue_seq=0)
        assert mlfq.age(state, CFG, now=5.0, step=2, queue_seq=0)
        assert not mlfq.age(state, CFG, now=5.0, step=3, queue_seq=0)  # same instant: no second jump
        assert not mlfq.age(state, CFG, now=9.9, step=4, queue_seq=0)
        assert mlfq.age(state, CFG, now=10.0, step=5, queue_seq=0)
        assert state.level == 1

    def test_aging_never_goes_above_q0(self):
        state = FeedbackState(level=0)
        mlfq.age(state, CFG, now=0.0, step=1, queue_seq=0)
        assert not mlfq.age(state, CFG, now=100.0, step=2, queue_seq=0)
        assert state.level == 0

    def test_first_token_promotion_is_off_by_default(self):
        state = FeedbackState(level=2)
        assert not mlfq.first_token(state, CFG, step=1, queue_seq=0)
        assert state.level == 2

    def test_first_token_promotion_when_enabled(self):
        cfg = MLFQConfig(first_token_promotion=True)
        state = FeedbackState(level=2)
        assert mlfq.first_token(state, cfg, step=1, queue_seq=0)
        assert state.level == 0


class TestQueueOrdering:

    def test_fifo_within_a_level(self):
        sched = make_mlfq(max_num_seqs=1)
        a, b, c = make_seq([1, 2]), make_seq([3, 4]), make_seq([5, 6])
        for seq in (a, b, c):
            sched.add(seq)

        assert sched.schedule() == [a]
        sched.postprocess([a], [7])
        sched.cancel(a)
        assert sched.schedule() == [b]

    def test_higher_level_is_admitted_before_earlier_arrivals(self):
        sched = make_mlfq(max_num_seqs=1)
        early_low, late_high = make_seq([1, 2]), make_seq([3, 4])
        early_low.feedback.level = 2
        sched.add(early_low)
        sched.add(late_high)

        assert sched.schedule() == [late_high]

    def test_preempted_request_keeps_its_level(self):
        sched = make_mlfq(num_kvcache_blocks=2)
        a = make_seq([1, 2, 3, 4], max_tokens=3)
        b = make_seq([5, 6, 7, 8], max_tokens=3)
        b.feedback.level = 2
        sched.add(a)
        sched.add(b)
        step(sched)
        step(sched)  # b is preempted by the KV pressure from a's decode

        assert b.status == SequenceStatus.WAITING
        assert b.feedback.level == 2
        assert b.feedback.preemptions == 1


class TestServiceAccounting:

    def test_prompt_tokens_are_charged_once(self):
        sched = make_mlfq()
        seq = make_seq([1, 2, 3], max_tokens=5)
        sched.add(seq)

        step(sched)  # prefill: 3 prompt tokens computed, one token sampled
        assert seq.feedback.prefill_service == 3
        assert seq.feedback.decode_service == 0

        step(sched)  # decode: one token computed
        assert seq.feedback.prefill_service == 3
        assert seq.feedback.decode_service == 1

    def test_chunked_prefill_demotes_after_its_quantum(self):
        sched = make_mlfq(max_num_batched_tokens=10, mlfq_prefill_quanta=(10, 20, 30))
        seq = make_seq(range(30), max_tokens=2)
        sched.add(seq)

        step(sched)  # first chunk: 10 tokens reaches Q0's allowance
        assert seq.feedback.level == 1
        assert seq.feedback.level_prefill_service == 0
        assert sched.stats.quantum_demotions == 1

    def test_demotion_does_not_preempt_the_running_request(self):
        sched = make_mlfq(mlfq_decode_quanta=(2, 2, 2))
        seq = make_seq([1, 2, 3], max_tokens=50, ignore_eos=True)
        sched.add(seq)
        for _ in range(6):
            step(sched, token_id=5)

        assert seq.feedback.level >= 1
        assert seq.status == SequenceStatus.RUNNING
        assert seq in sched.running

    def test_decode_start_keeps_prefill_feedback_history(self):
        sched = make_mlfq(max_num_batched_tokens=8, mlfq_prefill_quanta=(4, 100, 100))
        seq = make_seq(range(8), max_tokens=3)
        sched.add(seq)
        step(sched)  # 8 prompt tokens demote Q0 -> Q1
        assert seq.feedback.level == 1

        step(sched)  # decode begins; the level must not reset to Q0
        assert seq.feedback.level == 1
        assert seq.feedback.prefill_service == 8

    def test_bottom_level_request_keeps_running(self):
        sched = make_mlfq(mlfq_decode_quanta=(1, 1, 1))
        seq = make_seq([1, 2, 3], max_tokens=20, ignore_eos=True)
        seq.feedback.level = 3
        sched.add(seq)
        for _ in range(10):
            step(sched, token_id=5)
        assert seq.feedback.level == 3
        assert sched.stats.quantum_demotions == 0


class TestAging:

    def test_waiting_request_is_promoted_one_level_per_interval(self):
        clock = FakeClock()
        sched = make_mlfq(clock=clock, max_num_seqs=1, mlfq_aging_interval=5.0)
        blocker = make_seq([1, 2], max_tokens=50, ignore_eos=True)
        waiter = make_seq([3, 4])
        waiter.feedback.level = 3
        sched.add(blocker)
        sched.add(waiter)
        step(sched)  # blocker takes the only slot

        clock.advance(5.0)
        sched.schedule()
        assert waiter.feedback.level == 2
        clock.advance(5.0)
        sched.schedule()
        assert waiter.feedback.level == 1
        assert sched.stats.aging_promotions == 2

    def test_preempted_request_is_aged_while_it_waits(self):
        clock = FakeClock()
        sched = make_mlfq(clock=clock, num_kvcache_blocks=2, mlfq_aging_interval=5.0)
        a = make_seq([1, 2, 3, 4], max_tokens=3)
        b = make_seq([5, 6, 7, 8], max_tokens=3)
        b.feedback.level = 2
        sched.add(a)
        sched.add(b)
        step(sched)
        step(sched)  # b preempted at clock 0
        assert b.status == SequenceStatus.WAITING

        clock.advance(5.0)
        sched.schedule()
        assert b.feedback.level == 1  # promoted while waiting, even though it was preempted

    def test_aging_prevents_starvation_under_repeated_short_arrivals(self):
        # One slot. Each step a fresh Q0 request arrives and finishes in its admission step,
        # so Q0 never drains. Without aging the long request would never be admitted.
        clock = FakeClock()
        sched = make_mlfq(clock=clock, max_num_seqs=1, mlfq_aging_interval=3.0)
        long_request = make_seq([7, 8, 9, 10], max_tokens=4)
        long_request.feedback.level = 3
        sched.add(long_request)

        admitted_at = None
        for i in range(40):
            clock.advance(1.0)
            sched.add(make_seq([100 + i, 200 + i], max_tokens=1))
            batch = sched.schedule()
            if long_request in batch and admitted_at is None:
                admitted_at = i
            sched.postprocess(batch, [5] * len(batch))
            if admitted_at is not None:
                break
        assert admitted_at is not None and admitted_at <= 15

    def test_without_aging_the_long_request_is_starved(self):
        clock = FakeClock()
        sched = make_mlfq(clock=clock, max_num_seqs=1, mlfq_aging_interval=1e9)
        long_request = make_seq([7, 8, 9, 10], max_tokens=4)
        long_request.feedback.level = 3
        sched.add(long_request)

        for i in range(40):
            clock.advance(1.0)
            sched.add(make_seq([100 + i, 200 + i], max_tokens=1))
            batch = sched.schedule()
            assert long_request not in batch
            sched.postprocess(batch, [5] * len(batch))
        assert long_request.status == SequenceStatus.WAITING
