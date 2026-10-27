# vllm-os: nano-vLLM with a contention-aware MLFQ scheduler

A compact vLLM-style inference engine (Qwen3) extended with a unified token-budget scheduler and a selectable four-level feedback queue (MLFQ) policy, plus a reproducible GPU benchmark pipeline. Built as an operating-systems course project.

**Status:** CPU-tested (scheduler, MLFQ, preemption, invariants, benchmark pipeline). **GPU code is not yet verified** on hardware. No benchmark results are included.

## Quick start on a GPU machine

Clone, then run one command. Change only `--max-num-seqs` to vary contention.

```bash
bash scripts/run_full_experiment.sh --max-num-seqs 8
```

Other forms:

```bash
bash scripts/run_full_experiment.sh --max-num-seqs-list 4,8,16 --profile standard --repeats 3
bash scripts/run_full_experiment.sh --max-num-seqs 4 --profile smoke --skip-setup
bash scripts/run_full_experiment.sh --model /path/to/Qwen3-0.6B --gpu 0 --gpu-memory-utilization 0.70 --max-num-seqs 8
bash scripts/run_full_experiment.sh --help
```

Results go to a new `results/experiment_YYYYMMDD_HHMMSS/` directory. The final report is `reports/REPORT.md` and `reports/REPORT.html` in that directory.

Only run the GPU correctness tests:

```bash
bash scripts/run_gpu_tests.sh --gpu 0 --policy mlfq
```

## Validate on a CPU-only machine

```bash
python -m pip install pytest transformers xxhash numpy tqdm matplotlib
python -m pytest tests --ignore=tests/gpu -q
bash scripts/run_full_experiment.sh --dry-run --max-num-seqs-list 4,8 --profile smoke
```

The dry run executes the CPU tests and writes a benchmark plan. It does not touch a GPU or download a model.

## Schedulers

| Policy | Config | Behaviour |
|---|---|---|
| `fcfs` | `scheduling_policy="fcfs"` | unified token-budget batching, admission in arrival order |
| `mlfq` | `scheduling_policy="mlfq"` | four levels, demotion by service quanta, aging |
| `mlfq` + preemption | `enable_mlfq_preemption=True` | additionally evicts strictly lower-priority requests when slots or KV blocks block a waiter |

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B", max_num_seqs=8, scheduling_policy="mlfq", enable_mlfq_preemption=True)
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): modules and flowcharts
- [docs/SCHEDULER_MODERNIZATION.md](docs/SCHEDULER_MODERNIZATION.md): what changed from the original scheduler
- [docs/MLFQ_ALGORITHM.md](docs/MLFQ_ALGORITHM.md): levels, quanta, demotion, aging
- [docs/PREEMPTION.md](docs/PREEMPTION.md): reactive and proactive preemption, anti-thrashing
- [docs/CPU_TESTING.md](docs/CPU_TESTING.md), [docs/GPU_TESTING.md](docs/GPU_TESTING.md)
- [docs/BENCHMARKING.md](docs/BENCHMARKING.md): methodology, metrics, limitations
- [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md): project summary, checkpoints, status

## Repository layout

```
nanovllm/          engine (scheduler, mlfq policy, block manager, model runner, Qwen3)
benchmarks/        workload traces, trial replay, matrix runner, aggregation, report generation
scripts/           one-command pipeline and environment setup
tests/             CPU tests; tests/gpu GPU tests (skipped without CUDA); tests/bench; tests/scripts
docs/              design and methodology documents
```

## License and attribution

MIT, as in the original project (see `LICENSE`). The scheduler design is inspired by vLLM V1; no vLLM code is copied.
