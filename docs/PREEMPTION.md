# Preemption

Two kinds of preemption exist. Both recompute the victim from scratch when it resumes, because the CPU KV swap backend is **not implemented** (see the last section).

## 1. Reactive (KV necessity), both policies

A running request needs a KV block for its next decode token, and none is free. The scheduler must evict someone or evict the request itself.

Victim rank (worst first): for MLFQ, the highest level, then prefill before decode, then the latest admission. For FCFS, the latest admission. The candidate set excludes requests already scheduled in this step, so the batch never loses a member. If the worst candidate ranks below the needing request, the needing request preempts **itself** and waits. This keeps priority inversion from happening under memory pressure.

Victims go back to the waiting queue. Under FCFS they go to the front, which matches V0.

## 2. Proactive (MLFQ only, `enable_mlfq_preemption=True`)

A waiting request is blocked by **sequence slots** (`len(running) == max_num_seqs`) or by **KV blocks**. Strictly lower-priority running requests may be evicted so the waiter can be admitted.

Decision procedure (`Scheduler._try_proactive`):

1. Eligible victims: running, level **strictly greater** than the waiter's, not in cooldown, and under the per-request window quota.
2. Sort eligible victims worst first (level, then prefill before decode, then latest admission). Prefill-before-decode means a Q3 prefill is evicted before a Q3 decode.
3. Add victims one at a time. After each addition, check **feasibility before committing anything**:
   - a free slot exists after the plan, and
   - free blocks after the plan (current free plus blocks that would reach ref count 0) cover the waiter's requirement. Prefix-cache hits that are still in use by another request need no new block, so they are subtracted from the requirement.
4. Stop when the plan is feasible or the per-step limit is reached. If no feasible plan exists, evict **nothing** and leave the waiter queued.
5. A committed plan preempts its victims, then admits the waiter immediately, in the same step, **before** the batch is built. Freed blocks therefore cannot be taken by other requests' decode growth first.

Token-budget pressure alone never triggers proactive preemption. Deferring or chunking the work is enough then, and evicting would waste computed work.

Equal priority is never evicted. A Q0 request does not repeatedly evict another Q0 request because it arrived later.

### Anti-thrashing (configurable, all in `Config`)

| Option | Default | Effect |
|---|---:|---|
| `max_proactive_preemptions_per_step` | 1 | upper bound on victims in one plan |
| `preemption_cooldown_steps` | 4 | a request admitted (or resumed) is protected for this many steps |
| `max_preemptions_per_request_window` | 2 | proactive evictions allowed per request within the window |
| `preemption_window_steps` | 64 | the window length, in scheduler steps |

Limits are expressed in steps, not in seconds or kernel time, so they do not assume any GPU speed.

## Recompute on resume

A preempted request has its KV blocks released and `num_cached_tokens` reset to 0. On re-admission it recomputes its prompt **and** the tokens it already generated, then continues sampling. Tests confirm that generated tokens are preserved and not regenerated (`test_resumed_sequence_recomputes_prompt_and_generated_tokens`). Service counters are kept, so recomputation counts as prefill service.

## Shared prefix blocks

A block shared with another request is not counted as freed when a victim is evicted. Only blocks whose reference count reaches zero are released. `test_shared_prefix_block_is_not_counted_as_freed` covers this.

## CPU KV swapping (not implemented)

Not implemented. Preserving KV tensors in CPU RAM would need: a per-layer copy of blocks, synchronization with the CUDA stream, accounting of CPU RAM, and correct behaviour for shared prefix blocks. None of that can be verified without a GPU, so it is listed as future work rather than shipped as an unverified path. Recomputation is the only preemption backend.
