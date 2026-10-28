"""Headline bar charts: each scheduler side by side per scenario (mean of seeds).

    python -m benchmarks.plot_bars [results/mlfq_v2_bench.jsonl] [results/mixed_bench.jsonl] [docs/img]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks.plot_ablation import INK, INK_2, GRID, SURFACE, load

CONFIGS = [    # label, color; fcfs neutral, the rest reference palette slots in fixed order
    ("fcfs", "fcfs (original)", "#8c8b86"),
    ("priority-swap", "priority + swap", "#1baf7a"),
    ("mlfq-v1", "MLFQ v1", "#eda100"),
    ("mlfq-v2", "MLFQ v2", "#eb6834"),
    ("mlfq-v2+mixed", "MLFQ v2 + mixed", "#2a78d6"),
]
SCENARIOS = [
    ("batch-cap", "batch full,\nchat has priority"),
    ("no-priority", "batch full,\nno priorities"),
    ("kv-pressure", "KV cache full"),
]


def mean(runs, key, get):
    vals = [v for v in map(get, runs.get(key, [])) if v is not None]
    return np.mean(vals) if vals else None


def bars(ax, runs, get, fmt, log=False, skip=()):
    width = 0.8 / len(CONFIGS)
    for i, (cfg, label, color) in enumerate(CONFIGS):
        if cfg in skip:
            continue
        for j, (scenario, _) in enumerate(SCENARIOS):
            v = mean(runs, (scenario, cfg), get)
            if v is None:
                continue
            x = j + (i - (len(CONFIGS) - 1) / 2) * width
            # 2px surface gap between adjacent bars comes from the edge line
            ax.bar(x, v, width, color=color, edgecolor=SURFACE, linewidth=2, label=label if j == 0 else None,
                   zorder=3)
            ax.annotate(fmt(v), (x, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(SCENARIOS)), [name for _, name in SCENARIOS], fontsize=9)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_facecolor(SURFACE)


def main(base_path="results/mlfq_v2_bench.jsonl", mixed_path="results/mixed_bench.jsonl", out_dir="docs/img"):
    runs = load(base_path)
    runs.update(load(mixed_path))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURFACE)

    bars(axes[0], runs, lambda r: r["chat"]["ttft_p99"] * 1e3, lambda v: f"{v:.0f}", log=True)
    axes[0].set_title("chat time to first token, p99 (ms, log scale, lower is better)", fontsize=10,
                      color=INK, loc="left")
    axes[0].margins(y=0.15)

    def rel(scenario_get):
        def get(r):
            base = mean(runs, (r["label"].split("/")[0], "fcfs"), lambda x: x["output_tokens_per_s"])
            return 100 * (r["output_tokens_per_s"] / base - 1)
        return get
    bars(axes[1], runs, rel(None), lambda v: f"{v:+.1f}", skip=("fcfs",))    # 0 by definition
    axes[1].axhline(0, color=INK_2, lw=1, zorder=4)
    axes[1].set_title("output throughput vs. fcfs (%, 0 = no loss)", fontsize=10, color=INK, loc="left")
    axes[1].set_ylim(top=1.2)

    bars(axes[2], runs, lambda r: r["stats"]["swapped_out_blocks"] + r["stats"]["swapped_in_blocks"],
         lambda v: f"{v:.0f}")
    axes[2].set_title("KV blocks copied to/from host per run (lower is better)", fontsize=10, color=INK, loc="left")
    axes[2].margins(y=0.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.92), ncol=len(CONFIGS),
               frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Schedulers side by side, mean of 3 seeds", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bars.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:])
