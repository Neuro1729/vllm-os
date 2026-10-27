# Scheduler modernization (V1: unified token-budget scheduling)

## What the original nano-vLLM did (V0, commit `e1da1f5`)

`Scheduler.schedule()` returned **either** a prefill batch **or** a decode batch:

- Any waiting prompt that could be admitted forced a prefill-only step. Running sequences did not decode while prefill work existed.
- Only the first waiting prompt could be chunked. A chunk was allowed only when no other sequence was in the batch.
- Decode-time preemption popped the tail of `running`, which could include sequences already scheduled in this step.
- Admission was FIFO with no priority concept.
- `ModelRunner.run(seqs, is_prefill)` took one flag for the whole batch.

## What V1 does

Every step builds **one batch** under `max_num_batched_tokens`, in two phases.

1. **Admitted work.** Running sequences in service order. Each contributes its pending tokens: one token for a decode step, or up to the remaining budget for a continuing prefill chunk. Service order is least-recently-served first, with ties broken by admission order (MLFQ: lowest level first). The rotation prevents the tail of `running` from starving when the budget is smaller than the running set.
2. **Admission.** Waiting sequences in policy order (FCFS: arrival order, with preempted sequences at the front; MLFQ: level, then FIFO within the level). A sequence is admitted when a slot is free and KV blocks are available. It may be chunked into the budget that remains.

Invariants enforced by the scheduler and checked by tests after every step:

| Invariant | Where it is enforced |
|---|---|
| Sum of scheduled tokens ≤ `max_num_batched_tokens` | `schedule()` budget accounting; property tests |
| Active sequences ≤ `max_num_seqs` | admission checks `len(running)` |
| Each computed token has a KV block behind it | `tests/helpers.assert_kv_invariants` |
| Block ref counts equal the number of owners | same; catches double free and leaks |
| A sequence appears at most once per batch | property tests |
| Completion tokens only grow (no loss, no regeneration after preemption) | property tests |
| Impossible requests are rejected at arrival, not retried forever | `Scheduler.add` checks lifetime blocks against the whole cache |
| An empty batch with work remaining raises | `schedule()` raises `RuntimeError("scheduler stalled")` |

### Chunked prefill and KV progress

`Sequence.num_cached_tokens` is the number of tokens whose KV entries exist. A partial prefill chunk advances it without producing a token. The sequence stays admitted (`RUNNING`), so its blocks are kept between chunks. A token is sampled only when `num_cached_tokens` reaches `num_tokens`.

Blocks for the whole prompt are allocated at admission. Decode growth needs at most one new block per step.

### Model execution (`ModelRunner.run`)

- **Pure decode batches** (every sequence has one pending token and a block table) keep the original CUDA-graph path (`prepare_decode`).
- **Any other batch** (contains a prefill chunk, or mixes phases) runs `prepare_unified`: one varlen batch, with each sequence's new tokens as queries and its full cached context as keys and values through the block table. It runs in eager mode. This is the same attention branch the original code used for prefix-cache hits, so `attention.py` is unchanged.

The unified path is **CPU-unverified and GPU-unverified** in this repository: it has been checked by reading, not executed. `tests/gpu/test_gpu_correctness.py` exercises it on a GPU machine.

## Files

- `nanovllm/engine/scheduler.py`: the scheduler (both policies, preemption, cancellation, rejection).
- `nanovllm/engine/mlfq.py`: pure feedback-policy state (no torch import).
- `nanovllm/engine/sequence.py`: request state, including `feedback` and the tensor-parallel serialization fix.
- `nanovllm/engine/block_manager.py`: KV allocation, prefix cache, and planning helpers (`blocks_released_by`, `admission_needs`).
- `nanovllm/engine/model_runner.py`: `prepare_unified` and the policy-independent `run`.
- `nanovllm/engine/llm_engine.py`: `step()` returns a `StepResult` (per-token events for TTFT and ITL measurement).
