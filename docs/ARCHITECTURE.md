# Architecture

## Original nano-vLLM (V0)

```mermaid
flowchart LR
    U[LLM.generate] --> E[LLMEngine]
    E -->|schedule / postprocess| S[Scheduler]
    S --> B[BlockManager]
    E -->|run seqs, is_prefill| M[ModelRunner]
    M --> Q[Qwen3 model]
    Q --> A[Attention: flash-attn + Triton KV store]
    M --> K[(paged KV cache)]
    B -.block ids.-> M
```

The scheduler returns a prefill batch or a decode batch, never both. See [SCHEDULER_MODERNIZATION.md](SCHEDULER_MODERNIZATION.md) for what changed.

## Modernized engine (V1 and V2)

```mermaid
flowchart TD
    A[add_request] --> W[waiting queue]
    W --> P{policy}
    P -- fcfs --> F[FIFO order]
    P -- mlfq --> L[level, then FIFO; aging]
    F --> SCH[Scheduler.schedule]
    L --> SCH
    SCH -->|batch of sequences with num_scheduled_tokens| R[ModelRunner.run]
    R -->|pure decode| G[prepare_decode + CUDA graph]
    R -->|prefill chunk or mixed| U[prepare_unified: varlen with block table, eager]
    G --> T[sampled tokens]
    U --> T
    T --> PP[Scheduler.postprocess]
    PP -->|quantum exhausted| D[demote one level]
    PP -->|finished| FIN[release KV, return outputs]
```

| Module | Responsibility |
|---|---|
| `engine/scheduler.py` | token-budget batching, admission, reactive and proactive preemption, stats, events |
| `engine/mlfq.py` | feedback state, quanta, demotion, aging, first-token promotion (no torch) |
| `engine/block_manager.py` | KV blocks, prefix cache, release-planning helpers |
| `engine/sequence.py` | request state, feedback record, tensor-parallel serialization |
| `engine/model_runner.py` | batch preparation (decode, unified) and model execution |
| `engine/llm_engine.py` | step loop, `StepResult`, `generate`, `cancel_request` |
| `config.py` | all scheduler and MLFQ knobs; validated at construction |

## Scheduler decision flow

```mermaid
flowchart TD
    S0[step starts] --> AG{MLFQ?}
    AG -- yes --> AGE[promote waiters whose wait exceeded the interval]
    AG -- no --> P0
    AGE --> P0{preemption enabled?}
    P0 -- yes --> HEAD[head waiter blocked by slots or KV?]
    HEAD -- yes --> PLAN[plan victims: strictly lower level, cooldown, window quota]
    PLAN --> FEAS{feasible plan?}
    FEAS -- yes --> DO[preempt victims, admit waiter]
    FEAS -- no --> P1
    HEAD -- no --> P1
    DO --> P1[Phase 1: running sequences, budget-limited]
    P0 -- no --> P1
    P1 --> GROW{needs KV block?}
    GROW -- yes, none free --> VIC[reactive victim: worst rank, or self]
    VIC --> P1
    GROW -- ok --> P2[Phase 2: admit waiting sequences while slots, KV and budget allow]
    P2 --> OUT[batch; empty with work left raises]
```

## Invariants and where they are tested

- Token budget and slot limits: scheduler, property tests.
- KV reference counts equal owners; no double free: `tests/helpers.assert_kv_invariants`, called after each step.
- Completion tokens never lost or regenerated after preemption: property tests.
- Aging bounded and non-skipping: `tests/test_mlfq.py`.

## Benchmark pipeline

```mermaid
flowchart LR
    SH[scripts/run_full_experiment.sh] --> PRE[preflight and environment]
    PRE --> CT[CPU tests]
    CT --> GT[GPU tests]
    GT --> RM[benchmarks.run_matrix]
    RM -->|subprocess per trial| RT[benchmarks.run_trial]
    RT -->|trial.json, requests.csv, events| RAW[(bench/raw)]
    RM --> AG[benchmarks.aggregate]
    AG --> SUM[(bench/summaries)]
    SUM --> GR[benchmarks.generate_report]
    GR --> REP[reports: REPORT.md, REPORT.html, plots]
```

See [BENCHMARKING.md](BENCHMARKING.md).
