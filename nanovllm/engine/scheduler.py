"""Unified token-budget scheduler with FCFS and MLFQ policies.

Every step builds one batch under `max_num_batched_tokens`:

  Phase 1. Admitted sequences (decode steps and continuing prefill chunks),
           least recently served first (MLFQ: lowest-numbered level first).
  Phase 2. Waiting sequences in policy order, admitted while the batch has
           budget, a sequence slot is free, and KV blocks are available. A new
           prompt may be split into chunks.

Victims are chosen by a single rank: worst level first (MLFQ), then prefill
before decode, then latest admission. Reactive KV preemption (a running
sequence needs a block) uses this rank among unprotected sequences. Proactive
preemption (MLFQ only) additionally requires a strictly lower level, a plan
that is feasible before any eviction, and passing the anti-thrashing limits.
"""
import time
from collections import Counter
from dataclasses import dataclass, asdict
from itertools import count
from typing import Callable

from nanovllm.config import Config
from nanovllm.engine import mlfq
from nanovllm.engine.mlfq import MLFQConfig
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass
class SchedulerStats:
    steps: int = 0
    admissions: int = 0
    resumptions: int = 0
    preemptions_kv: int = 0
    preemptions_proactive: int = 0
    proactive_victim_levels: dict = None
    proactive_victim_phases: dict = None
    aging_promotions: int = 0
    quantum_demotions: int = 0
    first_token_promotions: int = 0
    slot_limit_hits: int = 0
    kv_limit_hits: int = 0
    token_budget_hits: int = 0
    rejected: int = 0
    cancelled: int = 0

    def __post_init__(self):
        self.proactive_victim_levels = Counter()
        self.proactive_victim_phases = Counter()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["proactive_victim_levels"] = dict(self.proactive_victim_levels)
        d["proactive_victim_phases"] = dict(self.proactive_victim_phases)
        return d


class Scheduler:

    def __init__(self, config: Config, clock: Callable[[], float] = time.monotonic, event_sink: Callable[[dict], None] | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.policy = config.scheduling_policy
        self.mlfq_enabled = self.policy == "mlfq"
        self.preemption_enabled = self.mlfq_enabled and config.enable_mlfq_preemption
        self.max_proactive_per_step = config.max_proactive_preemptions_per_step
        self.cooldown_steps = config.preemption_cooldown_steps
        self.max_preempts_per_window = config.max_preemptions_per_request_window
        self.preempt_window = config.preemption_window_steps
        self.mlfq = MLFQConfig.from_config(config)
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.clock = clock
        self.event_sink = event_sink
        self.step_id = 0
        self.stats = SchedulerStats()
        self._queue_counter = count()
        self._admit_counter = count()

    def is_finished(self):
        return not self.waiting and not self.running

    # ------------------------------------------------------------------ admission queue

    def add(self, seq: Sequence) -> bool:
        lifetime_blocks = (seq.num_prompt_tokens + seq.max_tokens + self.block_size - 1) // self.block_size
        if lifetime_blocks > len(self.block_manager.blocks):
            # Could never finish even with the whole cache free. Reject now rather than
            # admitting it, running it out of KV space, and preempting it forever.
            seq.status = SequenceStatus.REJECTED
            self.stats.rejected += 1
            self._emit("reject", seq)
            return False
        seq.feedback.waiting_since = self.clock()
        self._enqueue_waiting(seq)
        self._emit("arrive", seq)
        return True

    def _enqueue_waiting(self, seq: Sequence, front: bool = False):
        seq.status = SequenceStatus.WAITING
        seq.feedback.queue_seq = next(self._queue_counter)
        if front and not self.mlfq_enabled:
            self.waiting.insert(0, seq)
        else:
            self.waiting.append(seq)

    def _waiting_key(self, seq: Sequence):
        return seq.feedback.level, seq.feedback.queue_seq

    def _victim_key(self, seq: Sequence):
        # Larger is worse, i.e. a better victim.
        level = seq.feedback.level if self.mlfq_enabled else 0
        prefill_first = 0 if seq.in_decode else 1
        return level, prefill_first, seq.feedback.admit_order

    def _service_key(self, seq: Sequence):
        fb = seq.feedback
        if self.mlfq_enabled:
            return fb.level, fb.last_scheduled_step, fb.admit_order
        return fb.last_scheduled_step, fb.admit_order

    # ------------------------------------------------------------------ scheduling step

    def schedule(self) -> list[Sequence]:
        self.step_id += 1
        self.stats.steps += 1
        if self.mlfq_enabled:
            self._age_waiting(self.clock())
            self.waiting.sort(key=self._waiting_key)

        budget = self.max_num_batched_tokens
        batch: list[Sequence] = []
        protected: set[int] = set()

        # Phase 0 (MLFQ with preemption only): a waiting request blocked by slots or KV
        # may evict strictly lower-priority sequences. Victims are chosen before this step's
        # batch exists, so the freed resources go to the candidate and cannot be taken by
        # decode growth of other sequences. The candidate is admitted right away.
        if self.preemption_enabled:
            while self.waiting and budget > 0:
                cand = self.waiting[0]
                blocked = len(self.running) >= self.max_num_seqs or self.block_manager.can_allocate(cand) == -1
                if not blocked or not self._try_proactive(cand):
                    break
                num_cached_blocks = self.block_manager.can_allocate(cand)
                n = min(cand.num_tokens - num_cached_blocks * self.block_size, budget)
                budget -= self._admit(cand, num_cached_blocks, n, batch, protected)

        # Phase 1: admitted work.
        for seq in sorted(self.running, key=self._service_key):
            if budget == 0:
                break
            if seq.seq_id in protected or seq.status is not SequenceStatus.RUNNING:
                continue  # admitted in phase 0, or preempted earlier in this step
            n = min(seq.num_pending_tokens, budget)
            if not self._ensure_growth(seq, protected):
                continue
            seq.num_scheduled_tokens = n
            seq.feedback.last_scheduled_step = self.step_id
            budget -= n
            batch.append(seq)
            protected.add(seq.seq_id)

        # Phase 2: admission from the waiting queue, head of line first.
        slot_blocked = kv_blocked = False
        while self.waiting and budget > 0:
            cand = self.waiting[0]
            if len(self.running) >= self.max_num_seqs:
                slot_blocked = True
                break
            num_cached_blocks = self.block_manager.can_allocate(cand)
            if num_cached_blocks == -1:
                kv_blocked = True
                break
            n = min(cand.num_tokens - num_cached_blocks * self.block_size, budget)
            budget -= self._admit(cand, num_cached_blocks, n, batch, protected)

        if slot_blocked:
            self.stats.slot_limit_hits += 1
        if kv_blocked:
            self.stats.kv_limit_hits += 1
        if self.waiting and budget == 0:
            self.stats.token_budget_hits += 1
        if not batch and not self.is_finished():
            raise RuntimeError(
                f"scheduler stalled: {len(self.waiting)} waiting, {len(self.running)} running, "
                f"{self.block_manager.num_free_blocks} free blocks"
            )
        return batch

    def _admit(self, seq: Sequence, num_cached_blocks: int, n: int, batch: list[Sequence], protected: set[int]) -> int:
        self.waiting.pop(0)
        self.block_manager.allocate(seq, num_cached_blocks)
        seq.status = SequenceStatus.RUNNING
        seq.num_scheduled_tokens = n
        fb = seq.feedback
        fb.waiting_since = None
        fb.admit_order = next(self._admit_counter)
        fb.admitted_step = self.step_id
        fb.last_scheduled_step = self.step_id
        fb.cooldown_until_step = self.step_id + self.cooldown_steps
        self.running.append(seq)
        batch.append(seq)
        protected.add(seq.seq_id)
        self.stats.admissions += 1
        if fb.preemptions:
            self.stats.resumptions += 1
        self._emit("admit", seq, tokens=n, cached_blocks=num_cached_blocks)
        return n

    # ------------------------------------------------------------------ KV growth and reactive preemption

    def _ensure_growth(self, seq: Sequence, protected: set[int]) -> bool:
        need = seq.num_blocks - len(seq.block_table)
        while need > self.block_manager.num_free_blocks:
            victim = self._worst_unprotected(seq, protected)
            if victim is seq:
                self._preempt(seq, "kv")
                return False
            self._preempt(victim, "kv")
        if need > 0:
            self.block_manager.grow(seq, need)
        return True

    def _worst_unprotected(self, seq: Sequence, protected: set[int]) -> Sequence:
        pool = [s for s in self.running if s is not seq and s.seq_id not in protected]
        worst = max(pool, key=self._victim_key, default=None)
        if worst is None or self._victim_key(worst) < self._victim_key(seq):
            return seq  # seq is itself the lowest-priority candidate
        return worst

    def _preempt(self, seq: Sequence, reason: str):
        self.running.remove(seq)
        self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        seq.feedback.preemptions += 1
        seq.feedback.waiting_since = self.clock()
        if reason == "kv":
            self.stats.preemptions_kv += 1
        self._enqueue_waiting(seq, front=True)
        self._emit("preempt", seq, reason=reason)

    # ------------------------------------------------------------------ proactive (MLFQ) preemption

    def _try_proactive(self, cand: Sequence) -> bool:
        """Evict strictly lower-priority sequences so `cand` can be admitted.

        Called before the step's batch is built, so every running sequence is a
        candidate. The plan is checked first; nothing is evicted unless the whole
        plan makes `cand` admissible. Returns True if victims were preempted.
        """
        level = cand.feedback.level
        eligible = [
            s for s in self.running
            if s.feedback.level > level
            and s.feedback.cooldown_until_step <= self.step_id
            and self._within_preempt_budget(s)
        ]
        eligible.sort(key=self._victim_key, reverse=True)

        plan: list[Sequence] = []
        for victim in eligible:
            if len(plan) >= self.max_proactive_per_step:
                break
            plan.append(victim)
            if self._plan_admits(cand, plan):
                break
        if not plan or not self._plan_admits(cand, plan):
            return False

        for victim in plan:
            victim.feedback.proactive_preempt_steps.append(self.step_id)
            self.stats.proactive_victim_levels[victim.feedback.level] += 1
            self.stats.proactive_victim_phases["decode" if victim.in_decode else "prefill"] += 1
            self._preempt(victim, "proactive")
            self.stats.preemptions_proactive += 1
        return True

    def _plan_admits(self, cand: Sequence, plan: list[Sequence]) -> bool:
        slots_ok = len(self.running) - len(plan) < self.max_num_seqs
        released = self.block_manager.blocks_released_by(plan)
        free_after = self.block_manager.num_free_blocks + len(released)
        return slots_ok and free_after >= self.block_manager.admission_needs(cand, released)

    def _within_preempt_budget(self, seq: Sequence) -> bool:
        fb = seq.feedback
        window_start = self.step_id - self.preempt_window
        fb.proactive_preempt_steps = [t for t in fb.proactive_preempt_steps if t > window_start]
        return len(fb.proactive_preempt_steps) < self.max_preempts_per_window

    # ------------------------------------------------------------------ MLFQ feedback

    def _age_waiting(self, now: float):
        for seq in self.waiting:
            if mlfq.age(seq.feedback, self.mlfq, now=now, step=self.step_id, queue_seq=next(self._queue_counter)):
                self.stats.aging_promotions += 1
                self._emit("promote", seq)

    def _charge(self, seq: Sequence, decode: bool, tokens: int):
        if not self.mlfq_enabled:
            return
        demoted = mlfq.account_service(
            seq.feedback, self.mlfq, decode=decode, tokens=tokens,
            step=self.step_id, queue_seq=next(self._queue_counter),
        )
        if demoted:
            self.stats.quantum_demotions += 1
            self._emit("demote", seq)

    # ------------------------------------------------------------------ results

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            n = seq.num_scheduled_tokens
            pure_decode = n == 1 and seq.in_decode
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += n
            seq.num_scheduled_tokens = 0
            self._charge(seq, decode=pure_decode, tokens=n)
            if seq.num_cached_tokens < seq.num_tokens:
                continue  # partial prefill chunk: this step produced no token
            seq.append_token(token_id)
            if seq.num_completion_tokens == 1 and self.mlfq_enabled and mlfq.first_token(
                seq.feedback, self.mlfq, step=self.step_id, queue_seq=next(self._queue_counter),
            ):
                self.stats.first_token_promotions += 1
                self._emit("promote", seq, reason="first_token")
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                self._emit("finish", seq)

    def cancel(self, seq: Sequence) -> bool:
        """Remove a sequence between steps. Returns False if it is not queued or running."""
        if seq in self.waiting:
            self.waiting.remove(seq)
        elif seq in self.running:
            self.running.remove(seq)
            self.block_manager.deallocate(seq)
        else:
            return False
        seq.status = SequenceStatus.CANCELLED
        seq.num_scheduled_tokens = 0
        self.stats.cancelled += 1
        self._emit("cancel", seq)
        return True

    def _emit(self, event: str, seq: Sequence, **extra):
        if self.event_sink is None:
            return
        fb = seq.feedback
        self.event_sink({
            "step": self.step_id,
            "event": event,
            "seq_id": seq.seq_id,
            "level": fb.level,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "free_blocks": self.block_manager.num_free_blocks,
            **extra,
        })
