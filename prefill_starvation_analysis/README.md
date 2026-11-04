# Tiny low-priority requests can starve in vLLM's priority scheduler

> Chunked prefill limits how much work one long request receives per step, but it does not
> ensure that a tiny, lower-priority request ever receives service under sustained
> high-priority traffic.

| Longest CPU starvation | Worst GPU wait, steady traffic | Worst GPU wait, burst | Equal-priority control (CPU) |
|---|---|---|---|
| **19,888 steps**: zero tokens for a 20-token request; never served | **46.7 s** at offered load 0.94 (equal priority: 13.0 s) | **19.1 s** with 16 long requests and cap 256 | **49 steps**: same request is served |

The two GPU columns are measured on an RTX 3090. The CPU columns run the real vLLM scheduler code
step by step. No fix is included in this branch.

## 1. Problem

With `--scheduling-policy priority` and chunked prefill, a 20-token low-priority request can get
**0 prefill tokens in step after step** while higher-priority long prompts are prefilling. It
needs about 1% of one step's token budget, yet:

* it waits for the whole burst of long prompts to drain;
* under steady high-priority traffic its wait grows with load;
* if high-priority demand never stops, it is never served.

The per-request chunk caps (`--long-prefill-token-threshold`, fixed or adaptive) do not prevent
this.

## 2. Setup

| | |
|---|---|
| vLLM | **v0.32.0** (upstream `5df63a7`) + opt-in tracing (this branch, no scheduling change) |
| GPU | 1× RTX 3090 24 GB, driver 575.57, torch 2.13.0+cu129. Async scheduling is on (vLLM default). |
| Model | Qwen2.5-1.5B-Instruct architecture, random weights (`--load-format dummy`), bf16 |
| Token budget | `--max-num-batched-tokens 2048`, `--max-num-seqs 1024`, prefix caching off |
| Chunk caps | none · fixed 512 · **512 + adaptive** (the server default here) · fixed 256 |
| Priorities | long prompts priority 0 (high), tiny probe priority 2 (low). Control: everything priority 0, i.e. FCFS. |
| Background | 32 running decode streams |
| Long request | 16K-token prompt, 1 output token |
| Tiny request | 20-token prompt, 1 output token. TTFT is measured at the client. |
| Burst | N ∈ {1, 2, 4, 8, 16} long prompts arrive together; a tiny request every 250 ms |
| Steady traffic | random long-prompt arrivals for 60 s at a target offered load; a tiny request every 1 s |

**Two kinds of evidence.**

* **GPU measurement.** The full vLLM server, HTTP clients, and a per-step server trace.
  * Measured prefill capacity is 13,080 tokens/s.
  * Load is the *offered* high-priority load: prompt tokens actually sent per second, divided
    by that capacity. The steady-traffic runs reached 0.69 / 0.84 / 0.94, with measured
    token-budget utilization of 0.57 / 0.71 / 0.77.
* **CPU, real scheduler.** The same `vllm.v1.core.sched.Scheduler` code, driven step by step
  with a fake model runner.
  * Step counts are exact; time uses a constant 180 ms per step.
  * Load is exact by construction.

## 3. Evidence

| Figure | What it shows | What it proves |
|---|---|---|
| [**1. Step-by-step heatmap**](figures/fig1_step_heatmap.png) ([pdf](figures/fig1_step_heatmap.pdf)) | GPU trace, burst of 8: tokens per step for each long request and each tiny request | Each tiny request is present but gets **0 tokens for 33–60 consecutive steps** (red) while the long prompts take the budget |
| [**2. Wait vs. number of long requests**](figures/fig2_wait_vs_n.png) ([pdf](figures/fig2_wait_vs_n.pdf)) | Worst wait for N = 1–16, no cap / fixed 512 / adaptive / fixed 256; (a) GPU seconds, (b) CPU steps | Caps only move the threshold: from 4 long requests (8 for cap 256) every setting starves the tiny request, up to 19.5 s / 125 steps |
| [**3. Starvation duration (CCDF)**](figures/fig3_unserved_ccdf.png) ([pdf](figures/fig3_unserved_ccdf.pdf)) | GPU, burst of 8: % of tiny requests with 0 tokens for at least k steps | The tail, not the average: ~44% waited, 35–38% waited ≥ 10 steps, 6–9% ≥ 50 steps, the longest 60–61 |
| [**4. Starvation vs. load**](figures/fig4_wait_vs_load.png) ([pdf](figures/fig4_wait_vs_load.pdf)) | Steady traffic: longest 0-token run of the low-priority request vs. the equal-priority control; (a) GPU, (b) CPU | Strict priority, not capacity, drives it: at load ≈ 0.95 the low-priority request starves about 4× longer than at equal priority (GPU 267 vs. 73 steps; CPU 248 vs. 59) |

**Burst.** Worst TTFT on the GPU (s) / longest run of 0-token steps on the CPU.

| Long requests (N) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| no cap | 1.1 / 6 | 2.3 / 14 | 4.6 / 30 | 9.3 / 62 | 19.5 / 125 |
| fixed 512 | 0.15 / 0 | 0.31 / 0 | 4.5 / 30 | 9.0 / 61 | 19.2 / 123 |
| 512 + adaptive | 0.19 / 0 | 0.32 / 0 | 4.4 / 30 | 8.9 / 61 | 18.9 / 123 |
| fixed 256 | 0.15 / 0 | 0.19 / 0 | 0.33 / 0 | 9.0 / 61 | 19.1 / 123 |
| equal priority (control) | 0.15 / 0 | 0.33 / 0 | 4.5 / 30 | 9.0 / 61 | 18.8 / 122 |

The tiny request arrives *after* the burst, so equal priority waits as long here. In a burst the
cause is the full budget plus arrival order. Priority makes the difference under steady
traffic.

**Steady traffic.** Low-priority request with 512 + adaptive, compared with the equal-priority
control. GPU columns are median / worst TTFT and the longest 0-token run; CPU columns are the
longest 0-token run.

| Offered load | GPU low priority | GPU equal priority | CPU low priority | CPU equal priority |
|---|---|---|---|---|
| 0.5 | – | – | 14 steps | 12 steps |
| ≈ 0.7 (GPU 0.69) | 0.35 / 9.0 s, 49 steps | 0.37 / 5.1 s, 26 steps | 34 | 24 |
| ≈ 0.85 (GPU 0.84) | 1.2 / 21.1 s, 120 steps | 1.0 / 8.0 s, 44 steps | 104 | 38 |
| ≈ 0.95 (GPU 0.94) | **17.9 / 46.7 s, 267 steps** | 3.0 / 13.0 s, 73 steps | 248 | 59 |
| 0.99 | – | – | 254 | 65 |

Fixed 512 behaves the same as 512 + adaptive: within 12% on the GPU and identical on the CPU. With
32 decodes running, the adaptive cap stays at 512.

**Unbounded wait** (CPU, 20,000 steps). Eight high-priority 16K prompts are kept in flight: each
is replaced when it finishes. One low-priority 20-token request arrives.

| Setting | Tiny low-priority request | High-priority prompts finished |
|---|---|---|
| fixed 512 · 512 + adaptive · no cap | **never served: 0 tokens for 19,888 steps** | 2,206 / 2,206 / 2,272 |
| fixed 256 | served after 19 steps: 8 × 256 no longer fills the budget | 2,084 |
| equal priority (control) | served after 49 steps | 2,206 |

**Sanity checks** (all 34 GPU runs)

* KV cache ≤ 28% and 0 preemptions, so memory is not the cause.
* The trace records a reason for every zero-token step of a tiny request. All of them were
  "token budget exhausted"; none were "no free slot" or "KV cache".
* Every value plotted is dumped in [`figures/figure_data.json`](figures/figure_data.json).

## 4. Root cause

* **One budget per step.** Prefill chunks and decodes share 2048 tokens per step.
* **Running requests first.** The scheduler fills the budget from requests that are already
  running, which includes long prompts mid-prefill, before it looks at the waiting queue.
* **The cap limits each prompt, not the total.** Four prompts at 512 tokens, plus decodes,
  already fill 2048. The adaptive cap is `max(512, 2048 / (running + waiting))`; running decodes
  count in the denominator, so here it stays at 512.
* **Priority has no time limit.** Priority orders the waiting queue, so a newly arriving
  high-priority prompt always goes ahead of the waiting tiny request. Nothing ages and no budget
  is reserved, so the tiny request waits as long as high-priority prefill work exists.

## 5. Conclusion

vLLM's priority scheduling has no protection against starvation. Fixed and adaptive chunk caps
bound the work of *one* long request per step, not the wait of *another* request. Under
sustained high-priority prefill demand, a tiny low-priority request can wait without limit
(19,888 steps in our CPU run). On the GPU we measured worst waits of 46.7 s at offered load 0.94.

Worth investigating next: **aging**, or a **bounded service reservation** for short waiting
prefills. This branch establishes the problem only.

**Limitations**

* One GPU, one small model with random weights, one budget (2048). Step counts carry over to
  other hardware better than seconds do.
* Each configuration ran once, with 24–160 tiny requests per run. Worst values are single
  observations.
* CPU time uses a modeled 180 ms step.
* The unbounded case is a constructed worst case: demand is held at saturation.
* One tiny request (1 of 1,692) failed with a client-side connection reset and is excluded.

*Optional, not part of this analysis:* a backport of the unmerged
[PR #65283](https://github.com/vllm-project/vllm/pull/65283) and its exploratory results are in
[`optional/pr65283_backport/`](optional/pr65283_backport/).

---

### Reproduce

```bash
cd prefill_starvation_analysis
./reproduce.sh setup               # venv: this branch's vLLM (editable, precompiled kernels), charts venv
./reproduce.sh validate            # optional: scheduler matches stock v0.32.0 (expects 28/28 OK)
./reproduce.sh cpu                 # CPU real-scheduler sweep (~15 min)
CUDA_IDX=0 ./reproduce.sh gpu      # GPU sweep, one server lifetime (~35 min, 1 GPU >= 24 GB)
./reproduce.sh process figures     # -> results/processed/, figures/
```

New outputs go to `work/`. The committed `results/` hold the data behind every number above.

### Layout

```
prefill_starvation_analysis/
├── README.md · reproduce.sh
├── configs/       variants.json (cap settings) · workloads/*.json (exact GPU workloads)
├── scripts/       sweep_workloads.py · build_workloads.py · process_results.py · make_figures.py
│   ├── cpu/       real-scheduler harness + CPU sweep
│   ├── gpu/       server, client, sweep runner, trace analysis
│   └── validation/  step-exact check of the scheduler against an independent reimplementation
├── results/       raw/ (client records, per-step server trace) · processed/ (per-run JSON,
│                  summary_gpu.csv, summary_cpu.csv) · logs/ (validation, GPU health, CPU sweep)
├── figures/       fig1–fig4 (.png 300 dpi, .pdf) · figure_data.json
└── optional/pr65283_backport/   separate, experimental (see its README)
```
