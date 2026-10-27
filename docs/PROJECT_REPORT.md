# Project report: contention-aware MLFQ scheduling for continuous-batched LLM inference

## Status at a glance

| Area | Status |
|---|---|
| Unified token-budget scheduler (V1) | implemented; CPU-tested |
| Four-level MLFQ with aging (V2) | implemented; CPU-tested |
| Proactive MLFQ preemption | implemented; CPU-tested |
| Mixed-batch model execution (unified varlen path) | implemented; **not executed on a GPU** |
| CPU KV swapping | **not implemented** (future work) |
| GPU correctness tests | written; skip without CUDA; **not run** |
| One-command GPU pipeline | written; CPU dry run verified; **not run on a GPU** |
| Benchmark results | **none** (no GPU was available) |

## Checkpoints

| Label | Commit | Content |
|---|---|---|
| V0: original nano-vLLM | `e1da1f5` (tag `v0-original`, on `main`) | unmodified baseline |
| V1 and V2: modernized engine | `042bae9` (tag `v2-mlfq`, on `modernized-mlfq`) | unified scheduler, MLFQ policy, preemption, benchmark pipeline, LF line-ending policy, docs |

**V1 is not a separate commit.** The implementation was written as one feature-flagged codebase. V1 (modernized FCFS) is the same code run with `scheduling_policy="fcfs"`. V2 (MLFQ) is the same code with `scheduling_policy="mlfq"`. Because both use identical execution, the comparison the project needs is valid. A separate V1 commit could be created from an intermediate state if the course requires it, but that state was not preserved.

Nothing has been pushed. Tags exist only locally. Nothing was force-pushed.

## What was implemented

**Scheduler** (`nanovllm/engine/scheduler.py`). Each step builds one batch under `max_num_batched_tokens`. Running work comes first, then admissions. Any prompt may be chunked. Batches can mix decode and prefill. `max_num_seqs`, KV-block limits, prefix caching, EOS and `max_tokens` semantics, cancellation, and rejection of impossible requests are enforced.

**MLFQ policy** (`nanovllm/engine/mlfq.py`). Four levels, level-specific prefill and decode quanta (512/1024/2048 and 16/48/192 by default), demotion without forced preemption, FIFO within a level, aging on a monotonic clock, and optional first-token promotion (off by default).

**Preemption.** Reactive preemption (KV needed by a running decode) ranks victims by level, prefill before decode, then latest admission, and can preempt the requester itself. Proactive preemption (MLFQ only) plans victims against slots and KV blocks before committing; it never evicts equal priority and never evicts for token-budget pressure alone; it respects cooldown and per-request window limits. The plan and feasibility rules are described in [PREEMPTION.md](PREEMPTION.md).

**Model runner.** Pure decode batches keep the CUDA-graph path. Any batch containing a prefill chunk or mixing phases uses a unified varlen path in eager mode. This path is **unverified on GPU**.

**Benchmarking.** Eight scenarios, three profiles, four variants (V0, FCFS, MLFQ, MLFQ + preemption), open-loop traces, per-request metrics, repeat aggregation, and an HTML/Markdown/PNG report generator. The report never plots missing data as zero.

**Pipeline.** `scripts/run_full_experiment.sh` runs setup, GPU preflight, CPU and GPU correctness tests, the benchmark matrix, and report generation. It stops on correctness failure unless `--continue-on-test-failure` is given, and then labels results unverified. It never deletes results, never uses `sudo`, and never installs drivers.

## Verification performed (CPU only)

- `python -m pytest tests -q`: 310 passed, 5 skipped (the GPU tests), including the 140 benchmark-package tests.
- Randomized invariant tests: 90 workloads, checking block accounting, token budget, completion-token stability, liveness, and full KV release. Coverage was checked: the runs include reactive KV preemption, proactive preemption, aging, demotion, and resumption.
- `bash scripts/run_full_experiment.sh --dry-run --max-num-seqs-list 4,8 --profile smoke`: CPU tests pass; the plan contains 48 trials (8 scenarios × 2 limits × 3 variants × 1 repeat, with V0 skipped in dry run).
- Argument validation of all three shell scripts, and the benchmark CLIs (`--help`; invalid variant exits 2).

## Not verified

- Every GPU code path (model runner, CUDA graphs, unified attention with a block table, `LLM` construction with the new kwargs).
- The version pins in `requirements-gpu.txt` (torch 2.6.0, triton 3.2.0, flash-attn 2.7.4.post1, and others). They are a best guess for a torch 2.6 + CUDA 12.x stack.
- Whether `setup_gpu_env.sh` completes on a real machine (network, wheel availability).
- The benchmark pipeline end to end on a GPU. Report generation was exercised only on synthetic fixtures, which are labelled as such.

## Deviations from the specification

- **No separate V1 commit** (see Checkpoints above).
- **CPU KV swapping is not implemented.** Recomputation is the only preemption backend.
- **Mixed-batch execution is unified varlen, eager.** The spec's preferred CUDA-graph support for mixed batches is not present. Pure decode batches keep the CUDA-graph path.
- **Thorough profile:** a second load factor (2.0) is defined but unused by the matrix runner.
- **Queue wait by MLFQ level** cannot be charted per level: `requests.csv` does not record a per-request level. The report says so.
- **Original nano-vLLM (V0)** has no per-token events or statistics. Its TTFT and ITL come from polling sequence state after each step, so its measurements are coarser than the modernized variants'. This is a comparability limit, and the report should say so.
- **Model:** nano-vLLM requires a local model directory (`Config` asserts it), so `--model` accepts a local path or a Hugging Face id that the script downloads. The default `Qwen/Qwen3-0.6B` is the model family the repository's own examples use; it was not run.
- **Tensor parallelism** is not tested. The sequence serialization was changed so workers receive the tokens they need for chunks, but this is unverified.

## How to run on the GPU machine

```bash
git clone <repo> && cd nano-vllm && git checkout modernized-mlfq
bash scripts/run_full_experiment.sh --max-num-seqs 8
```

Interpret results with the caveats in [BENCHMARKING.md](BENCHMARKING.md#known-limitations). Do not treat a single-model, single-GPU run as a general result.
