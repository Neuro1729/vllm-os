"""Per-scenario bar charts (the docs/img/<scenario>.png layout) from the 3-seed results,
with MLFQ v2 and mixed batching added. Bars are means over seeds, whiskers min-max.

    python -m benchmarks.plot_scenarios [results/mlfq_v2_bench.jsonl] [results/mixed_bench.jsonl] [docs/img] [paper/figures]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.plot_ablation import INK, INK_2, GRID, SURFACE, load

CONFIGS = [    # key, label, color: same colors as benchmarks/plot_bars.py
    ("fcfs", "fcfs", "#8c8b86"),
    ("priority-swap", "priority\n+ swap", "#1baf7a"),
    ("mlfq-v1", "MLFQ v1", "#eda100"),
    ("mlfq-v2", "MLFQ v2", "#eb6834"),
    ("fcfs+mixed", "fcfs\n+ mixed", "#b5b4ae"),
    ("mlfq-v2+mixed", "MLFQ v2\n+ mixed", "#2a78d6"),
]
SCENARIOS = {
    "batch-cap": "Batch full: 12 agents, 8 slots, chat has priority",
    "no-priority": "Batch full, no priorities set",
    "kv-pressure": "KV cache full, batch slots free",
    "normal": "Normal load (batch never full)",
    "all-short": "Only short requests (chat only, overloaded queue)",
}
METRICS = [    # title, getter, unit, lower is better
    ("chat TTFT p50", lambda r: r["chat"]["ttft_p50"] * 1e3, "ms", True),
    ("chat TTFT p99", lambda r: r["chat"]["ttft_p99"] * 1e3, "ms", True),
    ("chat end-to-end p99", lambda r: r["chat"]["e2e_p99"], "s", True),
    ("agent turn end-to-end p50", lambda r: r["agent"]["e2e_p50"], "s", True),
    ("output throughput", lambda r: r["output_tokens_per_s"], "tok/s", False),
]


def stats(runs, key, get):
    vals = [v for v in map(get, runs.get(key, [])) if v is not None]
    return (np.mean(vals), min(vals), max(vals)) if vals else None


def plot_scenario(runs, scenario, out_dir, configs=CONFIGS, prefix="scenario-"):
    configs = [c for c in configs if (scenario, c[0]) in runs]
    metrics = [m for m in METRICS if stats(runs, (scenario, "fcfs"), m[1])]    # all-short has no agents
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.9 * len(metrics), 3.9), facecolor=SURFACE)
    for ax, (title, get, unit, lower) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        base = stats(runs, (scenario, "fcfs"), get)[0]
        for x, (key, label, color) in enumerate(configs):
            s = stats(runs, (scenario, key), get)
            if s is None:
                continue
            mean, lo, hi = s
            ax.bar(x, mean, 0.72, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], color=INK_2, lw=1, capsize=0, zorder=4)
            note = f"{mean:.0f}" if mean >= 10 else f"{mean:.2f}"
            if key != "fcfs" and base:
                note += f"\n{mean / base:.2f}x"
            ax.annotate(note, (x, hi), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=INK)
        ax.set_xticks(range(len(configs)), [c[1] for c in configs], fontsize=7)
        ax.set_title(f"{title} ({unit}, {'lower' if lower else 'higher'} is better)", fontsize=8.5,
                     color=INK, loc="left")
        ax.margins(y=0.28)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=7, length=0)
    fig.suptitle(f"{SCENARIOS[scenario]}  (mean of 3 seeds, whisker = min-max, x = relative to fcfs)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, f"{prefix}{scenario}.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


# the paper shows only the final scheduler against the two baselines
PAPER_CONFIGS = [CONFIGS[0], CONFIGS[1], ("mlfq-v2+mixed", "ours", "#2a78d6")]


def main(base_path="results/mlfq_v2_bench.jsonl", mixed_path="results/mixed_bench.jsonl", out_dir="docs/img",
         paper_dir=None):
    runs = load(base_path)
    runs.update(load(mixed_path))
    os.makedirs(out_dir, exist_ok=True)
    for scenario in SCENARIOS:
        print(plot_scenario(runs, scenario, out_dir))
        if paper_dir:
            print(plot_scenario(runs, scenario, paper_dir, PAPER_CONFIGS, prefix="results-"))


if __name__ == "__main__":
    main(*sys.argv[1:])
