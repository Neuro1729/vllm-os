"""MLFQ proactive preemption: victim selection, feasibility planning, anti-thrashing. CPU only."""
from helpers import assert_kv_invariants, make_mlfq, make_scheduler, make_seq, step
from nanovllm.engine.sequence import SequenceStatus


def set_level(seq, level):
    seq.feedback.level = level
    return seq


class TestSequenceSlotPreemption:

    def test_spec_scenario_evicts_q3_prefill_before_q3_decode(self):
        # Running: A Q3 decode, B Q2 decode, C Q3 prefill (mid-prompt), D Q1 decode. Waiting: E Q0.
        sched = make_mlfq(max_num_seqs=4, max_num_batched_tokens=8, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        a = make_seq([1, 2, 3], max_tokens=30, ignore_eos=True)
        b = make_seq([4, 5, 6], max_tokens=30, ignore_eos=True)
        d = make_seq([7, 8, 9], max_tokens=30, ignore_eos=True)
        for seq in (a, d, b):
            sched.add(seq)
        step(sched)  # a, d, b prefilled (budget 8 leaves 2 tokens for the next prompt)
        c = make_seq(range(100, 120), max_tokens=30, ignore_eos=True)
        sched.add(c)
        step(sched)  # c's first chunk (budget 8: decodes take 3, leaving 5)
        assert c.status == SequenceStatus.RUNNING and not c.in_decode

        set_level(a, 3)
        set_level(c, 3)
        set_level(b, 2)
        set_level(d, 1)
        e = make_seq([200, 201], max_tokens=4)
        sched.add(e)
        batch = sched.schedule()

        assert c.status == SequenceStatus.WAITING
        assert c.feedback.preemptions == 1
        assert e in batch and e.status == SequenceStatus.RUNNING
        assert a.status == SequenceStatus.RUNNING and b.status == SequenceStatus.RUNNING
        assert d.status == SequenceStatus.RUNNING
        assert sched.stats.proactive_victim_phases == {"prefill": 1}
        assert sched.stats.proactive_victim_levels == {3: 1}
        assert_kv_invariants(sched)

    def test_prefill_victim_preferred_over_decode_at_same_level(self):
        sched = make_mlfq(max_num_seqs=2, max_num_batched_tokens=4, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        decode_victim = make_seq([1], max_tokens=30, ignore_eos=True)
        sched.add(decode_victim)
        step(sched)  # now decoding, admitted first
        prefill_victim = make_seq(range(50, 58), max_tokens=30, ignore_eos=True)
        sched.add(prefill_victim)
        step(sched)  # 3 tokens for the decode + 1 chunk token -> still prefilling
        set_level(decode_victim, 3)
        set_level(prefill_victim, 3)
        urgent = make_seq([9], max_tokens=2)
        urgent.feedback.level = 0
        sched.add(urgent)

        sched.schedule()

        assert prefill_victim.status == SequenceStatus.WAITING
        assert decode_victim.status == SequenceStatus.RUNNING

    def test_equal_priority_is_never_proactively_preempted(self):
        # A Q0 request waiting behind four Q0 requests must not evict any of them.
        sched = make_mlfq(max_num_seqs=4, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        running = [make_seq([i], max_tokens=30, ignore_eos=True) for i in range(4)]
        for seq in running:
            sched.add(seq)
        step(sched)
        late = make_seq([50], max_tokens=2)
        sched.add(late)

        for _ in range(5):
            sched.schedule()
        assert late.status == SequenceStatus.WAITING
        assert all(seq.status == SequenceStatus.RUNNING for seq in running)
        assert sched.stats.preemptions_proactive == 0

    def test_latest_admission_breaks_ties_among_same_level_same_phase(self):
        sched = make_mlfq(max_num_seqs=2, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        older = make_seq([1], max_tokens=30, ignore_eos=True)
        newer = make_seq([2], max_tokens=30, ignore_eos=True)
        sched.add(older)
        step(sched)
        sched.add(newer)
        step(sched)
        set_level(older, 3)
        set_level(newer, 3)
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        sched.schedule()

        assert newer.status == SequenceStatus.WAITING
        assert older.status == SequenceStatus.RUNNING

    def test_no_eviction_when_slots_are_free(self):
        sched = make_mlfq(max_num_seqs=8, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        a = set_level(make_seq([1], max_tokens=30, ignore_eos=True), 3)
        sched.add(a)
        step(sched)
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        batch = sched.schedule()

        assert urgent in batch and a in batch
        assert sched.stats.preemptions_proactive == 0


class TestFeasibilityAndLimits:

    def test_fcfs_never_proactively_preempts(self):
        sched = make_scheduler(max_num_seqs=1, scheduling_policy="fcfs", num_kvcache_blocks=64)
        a = make_seq([1], max_tokens=30, ignore_eos=True)
        sched.add(a)
        step(sched)
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        sched.schedule()
        assert a.status == SequenceStatus.RUNNING
        assert urgent.status == SequenceStatus.WAITING

    def test_kv_pressure_evicts_a_lower_priority_victim(self):
        # 6 blocks total. Victim holds 2, a Q0 request holds 2. The waiting Q0 needs 3.
        sched = make_mlfq(max_num_seqs=8, enable_mlfq_preemption=True, num_kvcache_blocks=6)
        victim = set_level(make_seq(range(0, 8), max_tokens=4), 3)
        keeper = make_seq(range(20, 28), max_tokens=4)
        sched.add(victim)
        sched.add(keeper)
        step(sched)
        urgent = make_seq(range(40, 52), max_tokens=4)
        sched.add(urgent)

        batch = sched.schedule()

        assert victim.status == SequenceStatus.WAITING
        assert urgent in batch
        assert keeper.status == SequenceStatus.RUNNING
        assert_kv_invariants(sched)

    def test_infeasible_plan_leaves_state_untouched(self):
        # Total 3 blocks: victim holds 1, keeper holds 2 (prompt lengths chosen so that
        # decoding needs no new block). The Q0 candidate needs 2. The only eligible
        # victim frees 1, so no eviction happens.
        sched = make_mlfq(max_num_seqs=8, enable_mlfq_preemption=True, num_kvcache_blocks=3)
        victim = set_level(make_seq([1, 2, 3], max_tokens=4), 3)
        keeper = make_seq(range(20, 27), max_tokens=4)
        sched.add(victim)
        sched.add(keeper)
        step(sched)
        urgent = make_seq(range(40, 48), max_tokens=4)
        sched.add(urgent)

        sched.schedule()

        assert victim.status == SequenceStatus.RUNNING
        assert keeper.status == SequenceStatus.RUNNING
        assert urgent.status == SequenceStatus.WAITING
        assert sched.stats.preemptions_proactive == 0
        assert_kv_invariants(sched)

    def test_multi_victim_plan_is_used_when_one_victim_is_not_enough(self):
        # 6 blocks: v1 and v2 hold 1 each, keeper holds 2, 2 are free.
        # The Q0 candidate needs 4: one victim gives 3 (too few), two give 4.
        sched = make_mlfq(max_num_seqs=8, max_proactive_preemptions_per_step=2,
                          enable_mlfq_preemption=True, num_kvcache_blocks=6)
        v1 = set_level(make_seq([1, 2, 3], max_tokens=4), 3)
        v2 = set_level(make_seq([5, 6, 7], max_tokens=4), 3)
        keeper = make_seq(range(20, 27), max_tokens=4)
        for seq in (v1, v2, keeper):
            sched.add(seq)
        step(sched)
        r = make_seq(range(40, 56), max_tokens=1)  # 4 blocks
        sched.add(r)

        batch = sched.schedule()

        assert sched.stats.preemptions_proactive == 2
        assert r in batch
        assert v1.status == SequenceStatus.WAITING and v2.status == SequenceStatus.WAITING
        assert keeper.status == SequenceStatus.RUNNING
        assert_kv_invariants(sched)

    def test_per_step_limit_blocks_a_plan_needing_more_victims(self):
        sched = make_mlfq(max_num_seqs=8, max_proactive_preemptions_per_step=1,
                          enable_mlfq_preemption=True, num_kvcache_blocks=6)
        v1 = set_level(make_seq([1, 2, 3], max_tokens=4), 3)
        v2 = set_level(make_seq([5, 6, 7], max_tokens=4), 3)
        keeper = make_seq(range(20, 27), max_tokens=4)
        for seq in (v1, v2, keeper):
            sched.add(seq)
        step(sched)
        r = make_seq(range(40, 56), max_tokens=1)
        sched.add(r)

        sched.schedule()

        assert sched.stats.preemptions_proactive == 0
        assert r.status == SequenceStatus.WAITING

    def test_shared_prefix_block_is_not_counted_as_freed(self):
        # V and R share block 0. Evicting V releases only its private block, so a
        # candidate needing two fresh blocks stays infeasible.
        # 4 blocks. Victim and sharer hold 2 each with block 0 shared: 3 used, 1 free.
        # The candidate needs 3 fresh blocks. Evicting the victim frees only its
        # private block (2 free in total), so the plan is infeasible.
        sched = make_mlfq(max_num_seqs=8, enable_mlfq_preemption=True, num_kvcache_blocks=4)
        victim = set_level(make_seq([1, 2, 3, 4, 5, 6, 7], max_tokens=4), 3)
        sched.add(victim)
        step(sched)
        sharer = make_seq([1, 2, 3, 4, 9, 9, 9], max_tokens=4)
        sched.add(sharer)
        step(sched)
        shared_block = sched.block_manager.blocks[victim.block_table[0]]
        assert shared_block.ref_count == 2 and sharer.block_table[0] == victim.block_table[0]
        candidate = make_seq(range(50, 61), max_tokens=1)
        sched.add(candidate)

        sched.schedule()

        assert sched.stats.preemptions_proactive == 0
        assert victim.status == SequenceStatus.RUNNING
        assert candidate.status == SequenceStatus.WAITING

    def test_token_budget_pressure_alone_never_preempts(self):
        sched = make_mlfq(max_num_seqs=8, max_num_batched_tokens=4, enable_mlfq_preemption=True, num_kvcache_blocks=64)
        runners = [set_level(make_seq([i], max_tokens=30, ignore_eos=True), 3) for i in range(4)]
        for seq in runners:
            sched.add(seq)
        step(sched)  # four one-token prompts exactly fill the 4-token budget
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        sched.schedule()

        assert sched.stats.preemptions_proactive == 0
        assert urgent.status == SequenceStatus.WAITING
        assert sched.stats.token_budget_hits == 1


class TestAntiThrashing:

    def test_recently_admitted_victim_is_protected_until_cooldown_ends(self):
        sched = make_mlfq(max_num_seqs=1, enable_mlfq_preemption=True,
                          preemption_cooldown_steps=3, num_kvcache_blocks=64)
        victim = set_level(make_seq([1], max_tokens=30, ignore_eos=True), 3)
        sched.add(victim)
        sched.schedule()  # admitted at step 1, protected through step 3
        sched.postprocess([victim], [5])
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        batch = sched.schedule()  # step 2
        assert victim.status == SequenceStatus.RUNNING and urgent not in batch
        sched.postprocess(batch, [5] * len(batch))
        batch = sched.schedule()  # step 3
        assert victim.status == SequenceStatus.RUNNING and urgent not in batch
        sched.postprocess(batch, [5] * len(batch))

        batch = sched.schedule()  # step 4: cooldown over
        assert victim.status == SequenceStatus.WAITING
        assert urgent in batch

    def test_request_over_its_window_quota_is_not_evicted_again(self):
        sched = make_mlfq(max_num_seqs=1, enable_mlfq_preemption=True,
                          max_preemptions_per_request_window=1, preemption_window_steps=10,
                          num_kvcache_blocks=64)
        victim = set_level(make_seq([1], max_tokens=30, ignore_eos=True), 3)
        sched.add(victim)
        step(sched)
        victim.feedback.proactive_preempt_steps = [sched.step_id]  # already evicted once this window
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        sched.schedule()
        assert victim.status == SequenceStatus.RUNNING
        assert urgent.status == SequenceStatus.WAITING

    def test_window_quota_resets_after_the_window(self):
        sched = make_mlfq(max_num_seqs=1, enable_mlfq_preemption=True,
                          max_preemptions_per_request_window=1, preemption_window_steps=2,
                          num_kvcache_blocks=64)
        victim = set_level(make_seq([1], max_tokens=30, ignore_eos=True), 3)
        sched.add(victim)
        step(sched)
        victim.feedback.proactive_preempt_steps = [sched.step_id - 5]  # outside the 2-step window
        urgent = set_level(make_seq([9], max_tokens=2), 0)
        sched.add(urgent)

        sched.schedule()
        assert victim.status == SequenceStatus.WAITING
