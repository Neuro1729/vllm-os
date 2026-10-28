from collections import defaultdict, deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """Continuous-batching scheduler with saturation-triggered preemption.

    fcfs:     nano-vllm's original policy.
    priority: rank = priority. Waiting requests are admitted best rank first.
    mlfq:     rank = (MLFQ level, priority). The level follows the sequence's size, prompt
              plus generated tokens: with mlfq_level_tokens=(1024, 4096, 8192) level 0 holds
              up to 1024 tokens, level 1 up to 4096, level 2 up to 8192, level 3 the rest.
              Levels only matter once the batch is full; until then a sequence that grows past
              1024 tokens keeps running like anywhere else. Waiting sequences age up one level
              every aging_steps, and a promoted sequence gets that level's whole band again.
              mlfq_priority_scale widens the bands of important requests, mlfq_max_levels can
              pin a priority above some level.

    Preemption is only for the saturated case of vllm-project/vllm#40004: max_num_seqs
    sequences are running and a waiting request of strictly better rank has waited
    preempt_after_steps steps. It then evicts the worst-ranked running sequences, just
    enough to free a slot and the KV blocks it needs (all or nothing). With free slots
    nothing is evicted: a request that does not fit in the KV cache waits, as in fcfs.
    Running out of KV blocks during decode evicts as in fcfs, but the worst-ranked victim.
    Victims are recomputed or swapped to host memory (preemption_mode).

    With pause_victims, a victim evicted only for its batch slot keeps its KV blocks on the
    GPU (PAUSED) and resumes without any copy. Its blocks are reclaimed, i.e. swapped or
    recomputed as above, only once a newcomer or a decode step needs them.
    mlfq_preempt_max_level and preempt_min_remaining_tokens limit who may evict and who may
    be evicted, so that preemption only happens where it buys latency for short requests.
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.policy = getattr(config, "scheduling_policy", "fcfs")
        level_tokens = tuple(getattr(config, "mlfq_level_tokens", (1024, 4096, 8192)))
        self.mlfq_bands = tuple(b - a for a, b in zip((0,) + level_tokens, level_tokens))    # tokens per level
        self.mlfq_priority_scale = getattr(config, "mlfq_priority_scale", 1.0)
        self.mlfq_max_levels = dict(getattr(config, "mlfq_max_levels", {}))
        self.preempt_after_steps = getattr(config, "preempt_after_steps", 0)
        self.aging_steps = getattr(config, "aging_steps", 0)
        self.starvation_steps = getattr(config, "starvation_steps", 0)
        self.min_resident_steps = getattr(config, "min_resident_steps", 0)
        self.preemption_mode = getattr(config, "preemption_mode", "recompute")
        self.swap_min_tokens = getattr(config, "swap_min_tokens", 0)
        self.pause_victims = getattr(config, "pause_victims", False)
        self.mlfq_preempt_max_level = getattr(config, "mlfq_preempt_max_level", -1)
        self.preempt_min_remaining_tokens = getattr(config, "preempt_min_remaining_tokens", 0)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          getattr(config, "num_cpu_blocks", 0))
        self.waiting: deque[Sequence] = deque()
        self.waiting_sorted = True    # re-sorting a long queue every step costs more than a decode step's worth of CPU
        # Aging without touching every waiting sequence each step: a sequence that started
        # waiting at step w is due at the steps congruent to w modulo the period.
        self.num_steps = 0
        self.aging_due = defaultdict(dict)    # step % aging_steps -> {id: seq}
        self.starvation_due = defaultdict(dict)    # step % starvation_steps -> {id: seq}
        self.running: deque[Sequence] = deque()
        self.blocks_to_swap_out: list[tuple[int, int]] = []
        self.blocks_to_swap_in: list[tuple[int, int]] = []
        self.stats = dict(preemptions=0, recomputes=0, recomputed_tokens=0, swap_outs=0, swap_ins=0,
                          swapped_out_blocks=0, swapped_in_blocks=0, demotions=0, promotions=0,
                          pauses=0, resumes=0, reclaims=0)

    @property
    def preemptive(self):
        return self.policy != "fcfs"

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if self.policy == "mlfq":
            self._update_level(seq)    # a long prompt starts in a lower level
        if self.waiting and self._waiting_key(seq) < self._waiting_key(self.waiting[-1]):
            self.waiting_sorted = False
        self._start_waiting(seq)
        self.waiting.append(seq)

    def rank(self, seq: Sequence):
        if self.policy == "mlfq":
            return (seq.level, seq.effective_priority)
        return (seq.effective_priority,)

    def _max_level(self, seq: Sequence) -> int:
        return min(len(self.mlfq_bands), self.mlfq_max_levels.get(seq.priority, len(self.mlfq_bands)))

    def band(self, seq: Sequence) -> int:
        """Tokens seq may grow by in its current level before it drops to the next one."""
        return max(1, round(self.mlfq_bands[seq.level] * self.mlfq_priority_scale ** -seq.priority))

    def _update_level(self, seq: Sequence):
        while seq.level < self._max_level(seq) and seq.num_tokens - seq.level_start > self.band(seq):
            seq.level_start += self.band(seq)
            seq.level += 1
            self.stats["demotions"] += 1

    def take_swap_ops(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Block copies the model runner must perform before running this step's batch."""
        ops = self.blocks_to_swap_out, self.blocks_to_swap_in
        self.blocks_to_swap_out, self.blocks_to_swap_in = [], []
        return ops

    def schedule(self) -> tuple[list[Sequence], bool]:
        self.block_manager.release_pending_cpu_blocks()
        if self.preemptive:
            self._age_waiting()
            if not self.waiting_sorted:
                self.waiting = deque(sorted(self.waiting, key=self._waiting_key))
                self.waiting_sorted = True
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if len(self.running) >= self.max_num_seqs:
                # batch is full: only a strictly better newcomer gets in, by evicting someone
                if not self.preemptive or not self._preempt_for(seq, scheduled_seqs):
                    break
            if seq.status == SequenceStatus.PAUSED:
                self.stats["resumes"] += 1
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self._start_running(seq)
                continue
            if seq.status == SequenceStatus.SWAPPED:
                if not self._swap_in(seq):
                    break
                continue
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1 and self._reclaim_paused(self._num_blocks_needed(seq), seq):
                    num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self._start_running(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        if self.preemptive:
            self.running = deque(sorted(self.running, key=lambda s: (self.rank(s), s.seq_id)))
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self._reclaim_paused(1, seq):
                    continue
                if self.running:
                    victim = max(self.running, key=self._victim_key) if self.preemptive else self.running[-1]
                    self.running.remove(victim)
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                seq.resident_steps += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _waiting_key(self, seq: Sequence):
        # A chunked prefill in progress already holds its blocks; it must stay at the head
        # (as in fcfs), or a newcomer sorted ahead of it could wait forever for those blocks.
        # Paused sequences hold blocks too, but can give them back, so they queue by rank.
        return not (seq.block_table and seq.status == SequenceStatus.WAITING), self.rank(seq), seq.seq_id

    def _start_waiting(self, seq: Sequence):
        seq.wait_since = self.num_steps
        if self.aging_steps:
            self.aging_due[seq.wait_since % self.aging_steps][id(seq)] = seq
        if self.starvation_steps:
            self.starvation_due[seq.wait_since % self.starvation_steps][id(seq)] = seq

    def wait_steps(self, seq: Sequence) -> int:
        """Steps seq has spent waiting/swapped/paused since it was last served."""
        return self.num_steps - seq.wait_since

    def _due(self, buckets, period: int) -> list[Sequence]:
        """Waiting sequences that have waited a positive multiple of period steps now."""
        phase = self.num_steps % period
        bucket = buckets.get(phase, {})
        due = []
        for key, seq in list(bucket.items()):
            # entries are dropped lazily once the sequence was served or re-queued with another phase
            if seq.status in (SequenceStatus.RUNNING, SequenceStatus.FINISHED) or seq.wait_since % period != phase:
                del bucket[key]
            elif self.wait_steps(seq) > 0:
                due.append(seq)
        return due

    def _start_running(self, seq: Sequence):
        seq.resident_steps = 0
        seq.tier_boost = 0    # boost only lasts until the sequence is served again
        self.running.append(seq)

    def _age_waiting(self):
        self.num_steps += 1
        if self.aging_steps:
            for seq in self._due(self.aging_due, self.aging_steps):
                if seq.level > 0:
                    seq.level -= 1
                    seq.level_start = seq.num_tokens
                    self.stats["promotions"] += 1
                    self.waiting_sorted = False
        if self.starvation_steps:
            for seq in self._due(self.starvation_due, self.starvation_steps):
                seq.tier_boost += 1
                self.stats["promotions"] += 1
                self.waiting_sorted = False

    def _victim_key(self, seq: Sequence):
        return (self.rank(seq), seq.seq_id)    # larger = evicted first

    def _victims(self, seq: Sequence, scheduled_seqs: list[Sequence]) -> list[Sequence]:
        """Running sequences seq may evict, worst first.

        Strictly worse rank only (equal ranks would ping-pong), never one already in this
        step's batch, and not one that has not yet had min_resident_steps of service.
        """
        scheduled = set(map(id, scheduled_seqs))
        rank = self.rank(seq)
        victims = [s for s in self.running
                   if id(s) not in scheduled and self.rank(s) > rank
                   and s.resident_steps >= self.min_resident_steps
                   and s.max_tokens - s.num_completion_tokens >= self.preempt_min_remaining_tokens]
        return sorted(victims, key=self._victim_key, reverse=True)

    def _may_preempt(self, seq: Sequence) -> bool:
        """Whether seq may evict runners at all. With mlfq_preempt_max_level only fresh, short
        requests may: re-admitting earlier victims or aged-up long requests only evicts one
        long request for another and buys no latency."""
        if self.policy != "mlfq" or self.mlfq_preempt_max_level < 0:
            return True
        return seq.num_preemptions == 0 and seq.level <= self.mlfq_preempt_max_level

    def _paused(self, exclude: Sequence) -> list[Sequence]:
        """Paused sequences whose blocks may be reclaimed, worst rank first."""
        paused = [s for s in self.waiting if s.status == SequenceStatus.PAUSED and s is not exclude]
        return sorted(paused, key=self._victim_key, reverse=True)

    def _reclaim_paused(self, num_blocks: int, seq: Sequence) -> bool:
        """Swap out / drop paused sequences until num_blocks GPU blocks are free (all or nothing).

        Every paused sequence other than seq is queued behind it, so taking its blocks keeps
        the queue order.
        """
        shortfall = num_blocks - len(self.block_manager.free_block_ids)
        paused = self._paused(seq)
        if shortfall <= 0 or not paused:
            return False
        if sum(map(self.block_manager.num_freeable_blocks, paused)) < shortfall:
            return False
        for victim in paused:
            if shortfall <= 0:
                break
            shortfall -= self.block_manager.num_freeable_blocks(victim)
            self.stats["reclaims"] += 1
            self._release_kv(victim)
        return True

    def _num_blocks_needed(self, seq: Sequence) -> int:
        if seq.status == SequenceStatus.SWAPPED:
            return len(seq.cpu_block_table)
        if seq.block_table:    # chunked prefill in progress, blocks already allocated
            return 0
        return self.block_manager.num_blocks_to_allocate(seq)[1]

    def _preempt_for(self, seq: Sequence, scheduled_seqs: list[Sequence]) -> bool:
        """The batch is full: evict worst-ranked runners to free a slot and seq's blocks.

        All or nothing, so nobody is evicted for a newcomer that would not fit anyway.
        """
        if self.wait_steps(seq) < self.preempt_after_steps or not self._may_preempt(seq):
            return False    # be patient, a slot often frees up on its own
        victims = self._victims(seq, scheduled_seqs)
        if not victims:
            return False
        needed = self._num_blocks_needed(seq)
        shortfall = needed - len(self.block_manager.free_block_ids)
        if self.pause_victims:
            # idle paused KV goes first; if that covers the newcomer, the victim keeps its KV
            shortfall -= sum(map(self.block_manager.num_freeable_blocks, self._paused(seq)))
            if shortfall <= 0:
                self._reclaim_paused(needed, seq)
                self.running.remove(victims[0])
                self.preempt(victims[0], keep_kv=True)
                return True
        plan = victims[:1]
        shortfall -= self.block_manager.num_freeable_blocks(victims[0])
        for victim in victims[1:]:
            if shortfall <= 0:
                break
            plan.append(victim)
            shortfall -= self.block_manager.num_freeable_blocks(victim)
        if shortfall > 0:
            return False
        for victim in plan:
            self.running.remove(victim)
            self.preempt(victim)
        if self.pause_victims:
            self._reclaim_paused(needed, seq)    # the rest, counted above
        return True

    def _swap_in(self, seq: Sequence) -> bool:
        if not self.block_manager.can_swap_in(seq) and not self._reclaim_paused(len(seq.cpu_block_table), seq):
            return False
        self.blocks_to_swap_in.extend(self.block_manager.swap_in(seq))
        self.stats["swap_ins"] += 1
        self.stats["swapped_in_blocks"] += len(seq.block_table)
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self._start_running(seq)
        return True

    def _should_swap(self, seq: Sequence) -> bool:
        if self.preemption_mode == "recompute" or not self.block_manager.can_swap_out(seq):
            return False
        # auto: short contexts are about as cheap to recompute as to copy, keep host memory for long ones
        return self.preemption_mode == "swap" or seq.num_cached_tokens >= self.swap_min_tokens

    def preempt(self, seq: Sequence, keep_kv: bool = False):
        seq.num_preemptions += 1
        self.stats["preemptions"] += 1
        if keep_kv:
            self.stats["pauses"] += 1
            seq.status = SequenceStatus.PAUSED
        else:
            self._release_kv(seq)
        self._start_waiting(seq)
        if self.preemptive:
            self.waiting.append(seq)    # re-sorted next step; the head may be the seq being admitted
            self.waiting_sorted = False
        else:
            self.waiting.appendleft(seq)

    def _release_kv(self, seq: Sequence):
        """Free seq's GPU blocks: swap them to host memory or drop them for recompute."""
        if self._should_swap(seq):
            mapping = self.block_manager.swap_out(seq)
            self.blocks_to_swap_out.extend(mapping)
            self.stats["swap_outs"] += 1
            self.stats["swapped_out_blocks"] += len(mapping)
            seq.status = SequenceStatus.SWAPPED
        else:
            self.stats["recomputes"] += 1
            self.stats["recomputed_tokens"] += seq.num_cached_tokens
            seq.status = SequenceStatus.WAITING
            seq.is_prefill = True
            self.block_manager.deallocate(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if self.policy == "mlfq":
                self._update_level(seq)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
