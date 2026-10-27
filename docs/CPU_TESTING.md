# CPU testing

CPU tests need no GPU, no triton and no flash-attn. They cover the scheduler, MLFQ policy, preemption, KV accounting, randomized invariants, the benchmark pipeline, and the shell entry points.

## Run

From the repository root:

```bash
python -m pip install pytest transformers xxhash numpy tqdm matplotlib
python -m pytest tests --ignore=tests/gpu -q
```

GPU tests under `tests/gpu` skip automatically without CUDA, so `python -m pytest tests -q` is also safe on a CPU machine.

## What is covered

| File | Coverage |
|---|---|
| `tests/test_scheduler.py` | FCFS admission, sequence and token limits, chunked prefill, decode and prefill in one step, KV preemption and resumption, EOS and `max_tokens`, prefix cache, cancellation, rejection, stall detection, determinism |
| `tests/test_mlfq.py` | policy units (quanta, demotion, reset, bottom level, aging clock, first-token promotion), queue order, service accounting (no double count), aging with a fake clock, anti-starvation with and without aging |
| `tests/test_preemption.py` | the Part 6 scenario (Q3 prefill evicted before Q3 decode), equal priority, latest-admission ties, no eviction when slots are free, FCFS never preempts proactively, KV-pressure eviction, infeasible plans leave state untouched, multi-victim plans, the per-step limit, shared-prefix ownership, token-budget-only pressure, cooldown, window quota |
| `tests/test_invariants.py` | 90 randomized workloads (FCFS, MLFQ, MLFQ with preemption) checking block accounting, token budget, no duplicate scheduling, completion-token prefix stability, liveness, and full KV release |
| `tests/bench/` | workload determinism, metric definitions and percentiles, the replay loop with a fake engine, the orchestrator plan, aggregation, and report generation from **synthetic fixtures** |
| `tests/scripts/` | shell syntax, `--help`, and argument validation of the pipeline scripts |

## Helpers

`tests/helpers.py` builds schedulers from a namespace with the same fields as `Config` and provides `FakeClock`. Its `assert_kv_invariants` checks the block accounting after each step.

The randomized tests use small KV budgets so that KV pressure is the normal case. Before relying on them, the coverage was checked: across the seeds, the runs include reactive KV preemption, proactive preemption, aging, demotion, and resumption.

## What CPU tests cannot prove

- Anything about the model runner's tensor code (`prepare_unified`, attention with a block table, CUDA graphs). Those paths run only on a GPU; see [GPU_TESTING.md](GPU_TESTING.md).
- Real timings. The fake clock makes aging deterministic; wall-clock behaviour is measured only by the GPU benchmarks.
