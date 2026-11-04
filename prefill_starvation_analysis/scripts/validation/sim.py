"""CPU-only discrete-event replica of the vLLM V1 (0.32.0) scheduler's token-budget logic.

Faithfully mirrors vllm/v1/core/sched/scheduler.py::Scheduler.schedule() for the parts that
decide *who gets how many tokens each step*:
  * one shared token budget (max_num_batched_tokens) for prefill chunks AND decodes
  * RUNNING requests served first, in admission order; loop stops when budget hits 0
    (`while req_index < len(self.running) and token_budget > 0`)
  * long_prefill_token_threshold cap; disabled when only 1 eligible request
  * long_prefill_token_threshold_adaptive: cap = max(T, max_num_batched_tokens // (running+waiting))
  * KV allocation failure -> preempt running[-1] (FCFS) or max(running, key=(priority, arrival))
    (PRIORITY); preempted request loses KV (recompute) and is re-queued at the front
  * WAITING loop skipped if anything was preempted this step; waiting queue = FIFO deque (FCFS)
    or heap on (priority, arrival) (PRIORITY); admission needs the full prompt to fit in KV
    (scheduler_reserve_full_isl=True) and len(running) < max_num_seqs
Not modelled: prefix caching, spec decode, async scheduling, encoder inputs, PP.

Extra (non-vLLM) policies for comparison:
  * adaptive_mode="prefill_only": cap = max(T, (B - decodes) // num_prefills)  (user's variant)
  * order="srpt": decodes first, then all prefill candidates (running partial prefills + waiting)
    by remaining tokens (with aging), i.e. a candidate fix.
"""
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------- cost model
@dataclass
class CostModel:
    """Rough Llama-3.1-8B bf16 on one H100-80GB. Only used to turn steps into milliseconds."""
    base_ms: float = 6.0            # weight read / launch floor per step
    per_tok_ms: float = 0.032       # dense GEMM cost per batched token (~31K tok/s prefill)
    attn_prefill_ms: float = 1.31e-6  # per (chunk_tok * attended_ctx) pair  (4*L*H flops @400TF)
    attn_decode_ms: float = 4.4e-5    # per context token read by a decode (131KB/tok @3TB/s)
    overhead_ms: float = 1.0        # scheduler + sampling CPU overhead

    def step_ms(self, prefill_chunks, decode_ctx_sum, n_tokens):
        t = max(self.base_ms, n_tokens * self.per_tok_ms) + self.overhead_ms
        for c, n in prefill_chunks:  # c = tokens already computed, n = chunk size
            t += self.attn_prefill_ms * n * (c + n / 2)
        t += self.attn_decode_ms * decode_ctx_sum
        return t

    def alone_ttft_ms(self, prompt, budget):
        """TTFT if the request had the GPU to itself (cap disabled when alone, as in vLLM)."""
        t, c = 0.0, 0
        while c < prompt:
            n = min(budget, prompt - c)
            t += self.step_ms([(c, n)], 0, n)
            c += n
        return t


# ----------------------------------------------------------------------------- request
class Req:
    __slots__ = ("rid", "arrival", "prompt", "out_len", "prio", "tag", "computed", "gen",
                 "blocks", "t_first", "t_finish", "last_tok", "max_itl", "starved",
                 "cur_starve", "max_starve_run", "decode_skips", "preempts", "admitted_t",
                 "chunks")

    def __init__(self, rid, arrival, prompt, out_len, prio=0, tag=""):
        self.rid, self.arrival, self.prompt, self.out_len = rid, arrival, prompt, max(1, out_len)
        self.prio, self.tag = prio, tag
        self.computed = self.gen = self.blocks = 0
        self.t_first = self.t_finish = self.last_tok = self.admitted_t = None
        self.max_itl = 0.0
        self.starved = self.cur_starve = self.max_starve_run = 0
        self.decode_skips = self.preempts = 0
        self.chunks = []  # (step, n) for the first few prefill chunks (debug / tracing)

    @property
    def need(self):
        return self.prompt + self.gen - self.computed

    @property
    def in_prefill(self):
        return self.need > 1 or self.gen == 0


# ----------------------------------------------------------------------------- scheduler
@dataclass
class SchedConfig:
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    long_prefill_token_threshold: int = 0
    adaptive: bool = False
    adaptive_mode: str = "upstream"  # upstream | prefill_only
    policy: str = "fcfs"             # fcfs | priority
    order: str = "vllm"              # vllm | srpt
    srpt_aging_s: float = 10.0
    kv_tokens: int = 400_000
    block_size: int = 16


class Sim:
    def __init__(self, cfg: SchedConfig, cost: CostModel | None = None, trace_rids=()):
        self.cfg, self.cost = cfg, cost or CostModel()
        self.free_blocks = cfg.kv_tokens // cfg.block_size
        self.running: list[Req] = []
        self._fifo: deque[Req] = deque()
        self._heap: list = []
        self.t = 0.0  # ms
        self.step = 0
        self.trace_rids = set(trace_rids)
        self.trace: list[dict] = []
        self.n_preempt = 0
        self.decode_skip_steps = 0
        self.tokens_done = 0

    # --- waiting queue (FCFS deque or PRIORITY heap), mirroring vllm request_queue.py
    def w_len(self):
        return len(self._heap) if self.cfg.policy == "priority" else len(self._fifo)

    def w_add(self, r):
        if self.cfg.policy == "priority":
            heapq.heappush(self._heap, (r.prio, r.arrival, r.rid, r))
        else:
            self._fifo.append(r)

    def w_prepend(self, r):
        if self.cfg.policy == "priority":
            heapq.heappush(self._heap, (r.prio, r.arrival, r.rid, r))
        else:
            self._fifo.appendleft(r)

    def w_peek(self):
        return self._heap[0][3] if self.cfg.policy == "priority" else self._fifo[0]

    def w_pop(self):
        return heapq.heappop(self._heap)[3] if self.cfg.policy == "priority" else self._fifo.popleft()

    def w_items(self):
        return [x[3] for x in self._heap] if self.cfg.policy == "priority" else list(self._fifo)

    def w_remove(self, r):
        if self.cfg.policy == "priority":
            self._heap = [x for x in self._heap if x[3] is not r]
            heapq.heapify(self._heap)
        else:
            self._fifo.remove(r)

    # --- KV
    def _alloc(self, r, n):
        need = -(-(r.computed + n) // self.cfg.block_size) - r.blocks
        if need <= self.free_blocks:
            self.free_blocks -= max(0, need)
            r.blocks += max(0, need)
            return True
        return False

    def _free(self, r):
        self.free_blocks += r.blocks
        r.blocks = 0

    def _preempt(self, r):
        self._free(r)
        r.computed = 0
        r.preempts += 1
        self.n_preempt += 1
        self.w_prepend(r)

    def _cap(self):
        c = self.cfg
        eligible = len(self.running) + self.w_len()
        T = c.long_prefill_token_threshold if eligible > 1 else 0
        if T > 0 and c.adaptive:
            if c.adaptive_mode == "upstream":
                T = max(T, c.max_num_batched_tokens // eligible)
            else:  # prefill_only: (budget - decode tokens) / number of prefills
                dec = sum(1 for r in self.running if not r.in_prefill)
                npf = max(1, eligible - dec)
                T = max(T, (c.max_num_batched_tokens - dec) // npf)
        return T

    # --- one scheduling step, vLLM order
    def _schedule_vllm(self):
        c = self.cfg
        budget = c.max_num_batched_tokens
        cap = self._cap()
        sched: dict[int, tuple[Req, int]] = {}
        preempted = False
        i = 0
        while i < len(self.running) and budget > 0:
            r = self.running[i]
            n = r.need
            if 0 < cap < n:
                n = cap
            n = min(n, budget)
            ok = True
            while not self._alloc(r, n):
                if c.policy == "priority":
                    victim = max(self.running, key=lambda x: (x.prio, x.arrival))
                    idx = self.running.index(victim)
                    del self.running[idx]
                    if idx < i:
                        i -= 1
                    if victim.rid in sched:
                        budget += sched.pop(victim.rid)[1]
                else:
                    victim = self.running.pop()
                self._preempt(victim)
                preempted = True
                if victim is r:
                    ok = False
                    break
            if not ok:
                break
            sched[r.rid] = (r, n)
            budget -= n
            i += 1

        if not preempted:
            while budget > 0 and self.w_len() and len(self.running) < c.max_num_seqs:
                r = self.w_peek()
                n = r.need
                if 0 < cap < n:
                    n = cap
                n = min(n, budget)
                full = -(-(r.prompt + r.gen) // c.block_size)
                if full > self.free_blocks:
                    break
                self._alloc(r, n)
                self.w_pop()
                if r.admitted_t is None:
                    r.admitted_t = self.t
                self.running.append(r)
                sched[r.rid] = (r, n)
                budget -= n
        return sched

    # --- candidate fix: decodes first, then SRPT (with aging) over all prefill candidates
    def _schedule_srpt(self):
        c = self.cfg
        budget = c.max_num_batched_tokens
        cap = self._cap()
        sched = {}
        for r in list(self.running):
            if budget <= 0:
                break
            if not r.in_prefill:
                if not self._alloc(r, 1):
                    victim = self.running.pop()
                    if victim.rid in sched:
                        budget += sched.pop(victim.rid)[1]
                    self._preempt(victim)
                    if victim is r:
                        continue
                    if not self._alloc(r, 1):
                        continue
                sched[r.rid] = (r, 1)
                budget -= 1

        def key(r):
            waited_s = (self.t - r.arrival) / 1000.0
            return (r.prio, r.need / (1.0 + waited_s / c.srpt_aging_s))

        run_ids = {r.rid for r in self.running}
        cands = [r for r in self.running if r.in_prefill] + self.w_items()
        cands.sort(key=key)
        for r in cands:
            if budget <= 0:
                break
            is_waiting = r.rid not in run_ids
            if is_waiting:
                if len(self.running) >= c.max_num_seqs:
                    continue
                if -(-(r.prompt + r.gen) // c.block_size) > self.free_blocks:
                    continue
            n = r.need
            if 0 < cap < n:
                n = cap
            n = min(n, budget)
            if not self._alloc(r, n):
                continue
            if is_waiting:
                self.w_remove(r)
                if r.admitted_t is None:
                    r.admitted_t = self.t
                self.running.append(r)
                run_ids.add(r.rid)
            sched[r.rid] = (r, n)
            budget -= n
        return sched

    # --- main loop
    def run(self, reqs: list[Req], stop=None, max_steps=10_000_000):
        pending = deque(sorted(reqs, key=lambda r: (r.arrival, r.rid)))
        alive: dict[int, Req] = {}
        done = []
        while (pending or alive) and self.step < max_steps:
            while pending and pending[0].arrival <= self.t:
                r = pending.popleft()
                alive[r.rid] = r
                self.w_add(r)
            if not self.running and not self.w_len():
                self.t = max(self.t, pending[0].arrival)
                continue
            self.step += 1
            sched = self._schedule_srpt() if self.cfg.order == "srpt" else self._schedule_vllm()
            if not sched:  # blocked on KV with nothing runnable: jump to next arrival
                if pending:
                    self.t = max(self.t + 1.0, pending[0].arrival)
                    continue
                raise RuntimeError("deadlock: nothing schedulable")
            chunks, dec_ctx, ntok = [], 0, 0
            for r, n in sched.values():
                ntok += n
                if n == 1 and not r.in_prefill:
                    dec_ctx += r.computed
                else:
                    chunks.append((r.computed, n))
            dt = self.cost.step_ms(chunks, dec_ctx, ntok)
            t0 = self.t
            self.t += dt
            self.tokens_done += ntok
            if self.trace_rids:
                rec = {"step": self.step, "t_ms": round(t0, 2), "dt_ms": round(dt, 2),
                       "cap": self._cap(), "n_running": len(self.running),
                       "n_waiting": self.w_len(),
                       "alloc": {alive[k].tag or k: v[1] for k, v in sched.items()
                                 if k in self.trace_rids},
                       "decode_tokens": sum(1 for r, n in sched.values()
                                            if n == 1 and not r.in_prefill),
                       "budget_used": ntok}
                self.trace.append(rec)
            # starvation / decode-skip accounting (before state update)
            skipped_dec = 0
            for r in alive.values():
                if r.rid in sched:
                    r.cur_starve = 0
                    continue
                if r.t_first is None:
                    r.starved += 1
                    r.cur_starve += 1
                    r.max_starve_run = max(r.max_starve_run, r.cur_starve)
                elif r.computed > 0:
                    r.decode_skips += 1
                    skipped_dec += 1
            if skipped_dec:
                self.decode_skip_steps += 1
            finished = []
            for r, n in sched.values():
                r.computed += n
                if len(r.chunks) < 64 and r.t_first is None:
                    r.chunks.append((self.step, n))
                if r.computed == r.prompt + r.gen:
                    r.gen += 1
                    if r.t_first is None:
                        r.t_first = self.t
                    else:
                        r.max_itl = max(r.max_itl, self.t - r.last_tok)
                    r.last_tok = self.t
                    if r.gen >= r.out_len:
                        r.t_finish = self.t
                        finished.append(r)
            if finished:
                fs = set(id(r) for r in finished)
                self.running = [r for r in self.running if id(r) not in fs]
                for r in finished:
                    self._free(r)
                    alive.pop(r.rid)
                    done.append(r)
            if stop is not None and stop(self, alive, done):
                break
        return done, list(alive.values())


def summarize(reqs, cost: CostModel, budget, groups):
    """groups: list of (name, lo, hi) prompt-length buckets."""
    out = {}
    for name, lo, hi in groups:
        g = [r for r in reqs if lo <= r.prompt < hi and r.t_first is not None]
        if not g:
            continue
        ttft = sorted(r.t_first - r.arrival for r in g)
        sd = sorted((r.t_first - r.arrival) / cost.alone_ttft_ms(r.prompt, budget) for r in g)
        st = sorted(r.starved for r in g)
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
        out[name] = {
            "n": len(g),
            "ttft_ms_p50": round(q(ttft, .5), 1), "ttft_ms_p99": round(q(ttft, .99), 1),
            "ttft_ms_max": round(ttft[-1], 1),
            "slowdown_p50": round(q(sd, .5), 2), "slowdown_p99": round(q(sd, .99), 2),
            "slowdown_max": round(sd[-1], 2),
            "starved_steps_p50": q(st, .5), "starved_steps_p99": q(st, .99),
            "starved_steps_max": st[-1],
            "frac_slowdown_ge_10x": round(sum(x >= 10 for x in sd) / len(sd), 4),
            "frac_slowdown_ge_100x": round(sum(x >= 100 for x in sd) / len(sd), 4),
        }
    return out


if __name__ == "__main__":
    print(json.dumps(SchedConfig().__dict__, indent=1))
