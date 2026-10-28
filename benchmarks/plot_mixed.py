"""Slope charts: each policy without and with mixed batching (scripts/run_mixed_bench.sh).

Lines are the mean over seeds, whiskers span the min and max seed.

    python -m benchmarks.plot_mixed [results/mlfq_v2_bench.jsonl] [results/mixed_bench.jsonl] [docs/img]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.plot_ablation import INK, INK_2, GRID, SURFACE, load

POLICIES = {    # name -> (label, color, marker); reference palette slots 1-3, fixed order
    "fcfs": ("fcfs", "#2a78d6", "o"),
    "mlfq-v2": ("MLFQ v2", "#eb6834", "s"),
    "priority-swap": ("priority + swap", "#1baf7a", "D"),
}
SCENARIOS = {
    "batch-cap": "batch full, chat has priority",
    "no-priority": "batch full, no priorities",
    "kv-pressure": "KV cache full",
    "normal": "normal load",
    "all-short": "only short requests",
}
METRICS = [
    ("chat TTFT p99", "ms, log", lambda r: r["chat"]["ttft_p99"] * 1e3, True),
    ("agent turn e2e p50", "s", lambda r: r["agent"]["e2e_p50"], False),
    ("output throughput", "tok/s", lambda r: r["output_tokens_per_s"], False),
]


def stats(runs, key, get):
    vals = [v for v in map(get, runs.get(key, [])) if v is not None]
    return (np.mean(vals), min(vals), max(vals)) if vals else None


def main(base_path="results/mlfq_v2_bench.jsonl", mixed_path="results/mixed_bench.jsonl", out_dir="docs/img"):
    runs = load(base_path)
    runs.update(load(mixed_path))
    scenarios = [s for s in SCENARIOS if any((s, f"{p}+mixed") in runs for p in POLICIES)]
    fig, axes = plt.subplots(len(scenarios), len(METRICS), figsize=(13, 3.1 * len(scenarios)),
                             facecolor=SURFACE, squeeze=False)
    for row, scenario in zip(axes, scenarios):
        for ax, (title, unit, get, log) in zip(row, METRICS):
            ax.set_facecolor(SURFACE)
            for policy, (name, color, marker) in POLICIES.items():
                pts = [stats(runs, (scenario, policy), get), stats(runs, (scenario, f"{policy}+mixed"), get)]
                if None in pts:
                    continue
                mean = [p[0] for p in pts]
                ax.errorbar([0, 1], mean, yerr=[[p[0] - p[1] for p in pts], [p[2] - p[0] for p in pts]],
                            color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2, capsize=0,
                            elinewidth=1, label=name, zorder=3)
                change = 100 * (mean[1] / mean[0] - 1)
                ax.annotate(f"{change:+.0f}%", (1, mean[1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"{SCENARIOS[scenario]}: {title} ({unit})", fontsize=9, color=INK, loc="left")
            ax.set_xticks([0, 1], ["separate prefill steps", "mixed batching"], fontsize=8)
            ax.set_xlim(-0.25, 1.45)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=8, length=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Mixed batching: decodes share the step with new prefills (mean of seeds, bars = min-max)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(scenarios)))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mixed.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:])
