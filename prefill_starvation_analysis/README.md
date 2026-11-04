# Low-priority prefill starvation in vLLM (chunked prefill + priority scheduling)

**Question.** Under `--scheduling-policy priority` with chunked prefill, can a *short*,
*low-priority* request be kept from starting for a long time by high-priority long prompts, even
though it needs only ~1% of one scheduler step?

**Answer (vLLM v0.32.0).** Yes.

* **Bursts.** A 20-token request got 0 tokens for 60–123 consecutive steps when 8–16
  high-priority 16K prompts arrived together. On an RTX 3090 that meant waits of up to 8.9–19 s.
  The per-request caps (512, 256, adaptive) stop helping once enough prompts share the step.
* **Steady traffic.** Nothing in the scheduler bounds the wait. With high-priority demand kept
  saturated, a low-priority 20-token request got **0 tokens for all 19,888 steps** of a run,
  while the high-priority work completed normally. The same request at equal priority was served
  after 49 steps. This result comes from the real vLLM scheduler code run on CPU; see *Evidence*.
* **PR #65283** (open, unmerged) makes the low-priority wait longer, not shorter. In our 0.32 port,
  the worst wait went from 0.35 s to 8–14 s at a measured high-priority load of only 0.42–0.55 on
  the GPU, because the PR holds lower-priority prefills back on purpose.

These results are a baseline only. **No fix is implemented on this branch**, and the scheduler
behaves exactly like vLLM v0.32.0; see *What was changed in vLLM*.

---

## Why this happens (mechanism)

1. **One budget per step.** Every scheduler step has a single token budget
   (`--max-num-batched-tokens`, 2048 here). Prefill chunks and decodes share it.
2. **Running requests go first.** The scheduler hands out budget to requests that are already
   running before it looks at the waiting queue. Long prompts that are mid-prefill are running
   requests.
3. **The per-request cap limits each prompt, not the total.** `--long-prefill-token-threshold 512`
   caps each prompt at 512 tokens per step. Four capped prompts plus a few decodes still fill 2048.
   The adaptive cap is `max(512, 2048 / (running + waiting))`. Running decodes count in that
   denominator, so here it stays at 512.
4. **Priority only reorders the waiting queue.** It never takes budget away from a running
   prefill, and nothing ages. A low-priority request waits until no higher-priority prefill
   wants the budget.

## Setup

| | |
|---|---|
| vLLM | **v0.32.0** (upstream `5df63a7`) + opt-in trace instrumentation on this branch |
| GPU | 1× RTX 3090 24 GB, driver 575.57, torch 2.13.0+cu129. Async scheduling is on (vLLM default). |
| Model | Qwen2.5-1.5B-Instruct architecture, `--load-format dummy` (random weights, real compute), bf16, max-model-len 32768 |
| Scheduler | `--scheduling-policy priority`, chunked prefill, `--max-num-batched-tokens 2048`, `--max-num-seqs 1024`, prefix caching off |
| Background | 32 decode streams (200-token prompts, long outputs), priority 0 |
| High-priority work | 16K-token prompts, priority 0, 1 output token |
| Probe | **20-token prompt, priority 2 (low), 1 output token**. TTFT is measured at the client. |
| Burst workload | N ∈ {1, 2, 4, 8, 16} high-priority prompts arrive together; a probe every 250 ms until they drain |
| Stream workload | Poisson high-priority prompts for 60 s; a probe every 1 s |

**Measured GPU capacity:** 12,959 prefill tokens/s while the 16-prompt burst drains. All GPU
stream loads below are measured offered load divided by this capacity, not the planned load.
In every one of the 36 GPU runs, KV-cache usage stayed at or below 28% and there were 0 preemptions, so the
waits are not caused by memory. The trace also records why each zero-token step happened (budget
exhausted, priority hold, slots, KV). Across all runs, every zero-token step of a probe was either
"budget exhausted" or, with PR #65283, "priority hold". GPU temperature peaked at 84 °C.

### Two kinds of evidence

| | GPU measurement | CPU, real scheduler |
|---|---|---|
| What runs | full vLLM server on the GPU, HTTP clients | the **same** `vllm.v1.core.sched.Scheduler` code, driven step by step with a fake model runner |
| Unit | seconds (client TTFT) plus steps (server trace) | **steps**, exact; no GPU timing |
| Clock | real | constant 180 ms per step (≈ measured 3090 full-budget step), used only to space arrivals |
| Load | measured | exact by construction: (2048 − 32) prefill tokens per step |
| Used for | Figs 1–3, Fig 4a | Fig 2b, Fig 4b, unbounded test |

## Scheduler settings compared

| Key | Setting |
|---|---|
| `adaptive` | cap 512 + `--long-prefill-token-threshold-adaptive` (the server's default config here) |
| `cap512` / `cap256` | fixed per-request cap |
| `nocap` | no per-request cap |
| `pr65283` | `adaptive` + PR #65283 `priority_prefill_budget_cap` (port, see below) |
| `equal` | `adaptive`, but every request has priority 0: an FCFS control |

All settings ran in **one server lifetime**. They were switched between runs through a runtime
control file (`VLLM_SCHED_CTRL`), as listed in [`configs/variants.json`](configs/variants.json).

## Results

### Burst: worst wait of a low-priority 20-token request (Fig 2)

GPU worst TTFT in seconds; in brackets, the longest run of 0-token steps from the CPU scheduler.

| High-priority 16K prompts | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| cap 512 + adaptive | 0.18 (0) | 0.31 (0) | 4.4 (30) | 8.9 (61) | 19.1 (123) |
| cap 256 | 0.15 (0) | 0.20 (0) | 0.34 (0) | 8.9 (61) | 19.2 (123) |
| no cap | 1.1 (6) | 2.3 (14) | 4.6 (30) | 9.3 (62) | 19.5 (125) |
| + PR #65283 | 1.6 (31) | 2.7 (31) | 4.6 (33) | 9.4 (66) | 19.6 (131) |
| equal priority (FCFS) | 0.20 (0) | 0.33 (0) | 4.5 (30) | 9.0 (61) | 18.7 (122) |

*Cap 512 without adaptive is within 0.3 s / 1 step of cap 512 + adaptive.* The cap delays the
problem: the wait starts once the capped prompts fill the step, at N ≥ 4 (N ≥ 8 for cap 256).
**In bursts, equal priority starves the same way.** There the cause is budget plus arrival order,
not priority. Priority matters in the next two results.

### Steady high-priority traffic (Fig 4)

| | median / worst TTFT (s) or longest 0-token run (steps) |
|---|---|
| GPU, cap 512 + adaptive, measured load 0.43 / 0.49 / 0.52 | 0.16 / 0.35 · 0.18 / 0.35 · 0.21 / 0.81 s |
| GPU, + PR #65283, measured load 0.42 / 0.49 / 0.55 | 0.74 / 8.0 · 1.8 / 13.8 · 1.9 / 13.3 s |
| CPU, cap 512 + adaptive, load 0.5 / 0.7 / 0.85 / 0.95 / 0.99 | 6 / 17 / 56 / 107 / 116 steps |
| CPU, + PR #65283, same loads | 140 / 203 / 173 / 179 / 345 steps |
| CPU, equal priority, same loads | 6 / 11 / 18 / 20 / 24 steps |

The GPU stream runs reached only 0.42–0.55 measured load. At that load stock vLLM serves the
probe in under 1 s. Higher loads are covered by the CPU runs only.

### Unbounded wait (CPU, real scheduler, 20,000 steps)

Setup: 32 decodes, plus 8 high-priority 16K prompts kept in flight (each is replaced as soon as
it finishes), plus one low-priority 20-token request.

| Setting | low-priority request | high-priority prompts finished (TTFT p50 / max) |
|---|---|---|
| cap 512, cap 512 + adaptive, + PR #65283 | **never served: 0 tokens for 19,888 steps** | 2,206 (11.5 s / 46.6 s) |
| cap 256 | served after 19 steps | 2,084 |
| equal priority | served after 49 steps | 2,206 |

With cap 256, 8 × 256 tokens no longer fill what is left of the budget, so slack appears. This is
the same threshold effect seen in the burst table.

## Figures

| Figure | What it shows |
|---|---|
| [**Fig 1**](figures/fig1_step_heatmap.png) ([pdf](figures/fig1_step_heatmap.pdf)) | **The mechanism, step by step** (GPU trace, burst of 8, cap 512 + adaptive). Each row is a request, each column a scheduler step. High-priority prompts take 512 tokens per step; each low-priority 20-token request is present but gets 0 tokens (red) for 33–60 steps, until the burst is almost drained. |
| [**Fig 2**](figures/fig2_wait_vs_n.png) ([pdf](figures/fig2_wait_vs_n.pdf)) | **Caps do not bound the wait.** Worst wait vs. number of high-priority prompts: (a) GPU seconds, (b) CPU steps. Every setting converges once the step is full; PR #65283 adds ~30 steps even with a single high-priority prompt. |
| [**Fig 3**](figures/fig3_unserved_ccdf.png) ([pdf](figures/fig3_unserved_ccdf.pdf)) | **Tail distribution** (GPU, burst of 8): the share of low-priority 20-token requests that got 0 tokens for at least k steps. About 44% waited at least one step, and the tail reaches about 60 steps (about 70 with PR #65283). No-cap and cap 512 are omitted; they lie within 4 and 1.3 percentage points of cap 512 + adaptive. |
| [**Fig 4**](figures/fig4_wait_vs_load.png) ([pdf](figures/fig4_wait_vs_load.pdf)) | **Steady traffic.** (a) GPU at *measured* load: PR #65283 raises the worst wait to 8–14 s. (b) CPU at exact load: the low-priority wait grows with load (116 steps at 0.99), while equal priority stays at 24 steps or fewer. |

Values plotted in each figure are dumped to [`figures/figure_data.json`](figures/figure_data.json).

## PR #65283 comparison

[vllm-project/vllm#65283](https://github.com/vllm-project/vllm/pull/65283) (open, head
`4960575`) adds `priority_prefill_budget_cap`. While a higher-priority request awaits its first
16 output tokens, it gives lower-priority prefills 0 tokens. That helps *high*-priority requests;
we did not measure that side here. For *low*-priority short requests it adds waiting even when
budget is left over. The trace labels these steps `prio_hold`: 89–2,206 probe-steps per run.

The PR targets upstream `main`, so we **ported** it to v0.32.0
([`patches/pr65283_port_v0.32.diff`](patches/pr65283_port_v0.32.diff)). Port adaptations:

* an empty `async_kv_waiting` queue (0.32 has no async KV-load queue);
* the 0.32 prefix-cache-hit bound (`num_tokens - 1`);
* `KVCacheBlock.pool` is a direct reference in 0.32, not a weakref.

Validation of the port:

* the PR's own test file passes **81/90** ([log](results/logs/pr65283_own_tests.txt)); 7
  failures need upstream-only APIs (async KV-connector loads, test-helper arguments) and 2 are
  prefix-cache cases of its KV make-room path;
* we ran only the budget cap: make-room preemption was off (`max_priority_preemptions_per_step=0`)
  and prefix caching was off;
* with the PR disabled, the patched scheduler matches stock behaviour on 28/28 validation
  scenarios ([log](results/logs/validation_pr65283_port_disabled.txt)).

## What was changed in vLLM (this branch)

| Commit | Change | Behaviour change |
|---|---|---|
| `5edbef9` | vLLM v0.32.0 source import (upstream `5df63a7`) | – |
| `cf8178b` | `VLLM_SCHED_TRACE=<file>`: per-step JSONL trace | none, opt-in |
| this branch | trace: budget left, hold, slots and 0-token requests per step; `VLLM_SCHED_CTRL=<file>` runtime control of cap/adaptive | none unless set |
| this branch | `platforms/cuda.py`: skip GPUs whose NVML handle errors (a broken GPU on the host crashed startup) | startup robustness only |

The scheduler on this branch reproduces stock v0.32.0 schedules on 28/28 CPU validation scenarios,
step by step against an independent reimplementation
([log](results/logs/validation_branch_scheduler.txt)).

## Limitations

* One GPU type, one small model with random weights, one budget (2048). Absolute seconds will
  differ elsewhere; step counts are the portable result.
* Each GPU configuration ran once (24–160 probes per run). Worst values are single observations
  with no confidence intervals.
* The GPU stream runs reached only 0.42–0.55 measured load. The workloads assumed 9K tokens/s;
  the GPU did 13K. Load ≥ 0.7 comes from the CPU scheduler runs, which have exact steps but a
  modeled clock.
* The unbounded result is a constructed worst case: high-priority demand held at saturation. It
  shows that no bound exists, not how often this happens in production.
* PR #65283 is our port of an unmerged PR. Upstream behaviour may differ, and only its
  budget-cap mode was evaluated, for low-priority requests.
* An earlier study on branch `vllm-prefill-starvation` covers the general head-of-line blocking
  of short requests with real traces (Mooncake); it is not repeated here.

## Reproduce

```bash
cd prefill_starvation_analysis
./reproduce.sh setup            # venv: this branch's vLLM (editable, precompiled kernels) + PR port
./reproduce.sh cpu              # CPU real-scheduler sweep (~15 min, no GPU)
GPU=0 ./reproduce.sh gpu        # GPU sweep, one server lifetime (~25 min, 1 GPU >= 24 GB)
./reproduce.sh process figures  # results/processed + figures/
./reproduce.sh validate         # optional: 28/28 scheduler validation
```

`reproduce.sh process` reads new outputs from `work/`. The committed `results/` hold the data
behind every number above.

## Layout

```
prefill_starvation_analysis/
├── README.md, reproduce.sh
├── configs/
│   ├── variants.json          scheduler settings (runtime control values)
│   └── workloads/*.json       exact GPU workloads (burst N, burst N equal-priority, stream)
├── scripts/
│   ├── sweep_workloads.py     workload definitions shared by CPU and GPU
│   ├── build_workloads.py     regenerates configs/workloads
│   ├── cpu/                   real-scheduler harness + CPU sweep
│   ├── gpu/                   server, client, sweep runner, trace analysis
│   ├── validation/            step-exact scheduler validation (simulator vs vLLM)
│   ├── process_results.py     raw -> processed + summary CSVs (measured load)
│   ├── make_figures.py        figures (PNG 300 dpi + PDF)
│   └── safe_run.sh            RAM cap / timeout / GPU health wrapper
├── patches/pr65283_port_v0.32.diff
├── results/
│   ├── raw/                   client records per run + server per-step trace (gz)
│   ├── processed/             per-run JSON, summary_gpu.csv, summary_cpu.csv, cpu_sweep.json
│   └── logs/                  validation, PR tests, GPU health
└── figures/                   fig1–fig4 (.png, .pdf), figure_data.json
```
