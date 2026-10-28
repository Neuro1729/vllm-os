"""Line charts for the MLFQ v2 ablation (scripts/run_mlfq_v2_bench.sh).

Each improvement is added on top of the previous one, left to right. Lines are the mean
over seeds, the shaded band spans the min and max seed.

    python -m benchmarks.plot_ablation [results/mlfq_v2_bench.jsonl] [docs/img]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGES = ["fcfs", "mlfq-v1", "+low-overhead", "+pause", "+short-only", "+no-wait", "mlfq-v2"]
STAGE_LABELS = ["fcfs", "MLFQ v1", "+ low\noverhead", "+ pause\nvictims", "+ short\nevict only",
                "+ no\nwait", "+ spare done\n= MLFQ v2"]
SCENARIOS = {    # name -> (label, color, marker); reference palette slots 1-5, fixed order
    "batch-cap": ("batch full, chat has priority", "#2a78d6", "o"),
    "no-priority": ("batch full, no priorities", "#eb6834", "s"),
    "kv-pressure": ("KV cache full", "#1baf7a", "D"),
    "normal": ("normal load", "#eda100", "^"),
    "all-short": ("only short requests", "#e87ba4", "v"),
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path):
    runs = defaultdict(list)    # (scenario, stage) -> [result per seed]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            scenario, stage, _seed = r["label"].split("/")
            runs[scenario, stage].append(r)
    return runs


def series(runs, scenario, get):
    xs, mean, lo, hi = [], [], [], []
    for i, stage in enumerate(STAGES):
        vals = [v for v in map(get, runs.get((scenario, stage), [])) if v is not None]
        if vals:
            xs.append(i)
            mean.append(np.mean(vals))
            lo.append(min(vals))
            hi.append(max(vals))
    return np.array(xs), np.array(mean), np.array(lo), np.array(hi)


def style(ax, title, unit):
    ax.set_title(f"{title} ({unit})", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_xticks(range(len(STAGES)), STAGE_LABELS, fontsize=7.5)
    ax.set_xlim(-0.3, len(STAGES) - 0.3)


def label_ends(ax, ends, fmt):
    """Value at each line's end, unless two would overlap (the legend names the lines then)."""
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    pos = sorted(np.log10(v) if log else v for _, v in ends)
    span = (np.log10(hi) - np.log10(lo)) if log else hi - lo
    if any(b - a < 0.06 * span for a, b in zip(pos, pos[1:])):
        return
    for x, v in ends:
        ax.annotate(fmt(v), (x, v), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def fmt_value(v):
    return f"{v:.0f}" if v >= 10 or v == 0 else f"{v:.2f}"


def line(ax, runs, scenario, get):
    xs, mean, lo, hi = series(runs, scenario, get)
    if not len(xs):
        return None
    name, color, marker = SCENARIOS[scenario]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
    ax.plot(xs, mean, color=color, lw=2, solid_joinstyle="round", solid_capstyle="round",
            marker=marker, ms=8, mec=SURFACE, mew=2, label=name, zorder=3)
    return xs[-1], mean[-1]


def plot_ablation(runs, out_dir):
    """Absolute values, the three scenarios where the batch or the KV cache is full."""
    scenarios = ["batch-cap", "no-priority", "kv-pressure"]
    metrics = [
        ("chat time to first token, p50", "ms, log scale, lower is better", lambda r: r["chat"]["ttft_p50"] * 1e3, True),
        ("chat time to first token, p99", "ms, log scale, lower is better", lambda r: r["chat"]["ttft_p99"] * 1e3, True),
        ("agent turn end-to-end, p50", "s, lower is better", lambda r: r["agent"]["e2e_p50"], False),
        ("output throughput", "tok/s, higher is better", lambda r: r["output_tokens_per_s"], False),
        ("preemptions", "count", lambda r: r["stats"]["preemptions"], False),
        ("KV blocks copied (swap out + in)", "blocks", lambda r: r["stats"]["swapped_out_blocks"] + r["stats"]["swapped_in_blocks"], False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor=SURFACE)
    for ax, (title, unit, get, log) in zip(axes.flat, metrics):
        ax.set_facecolor(SURFACE)
        ends = [e for e in (line(ax, runs, s, get) for s in scenarios) if e]
        if log:
            ax.set_yscale("log")
        style(ax, title, unit)
        label_ends(ax, ends, fmt_value)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("MLFQ v2: each improvement added on top of the previous one (mean of seeds, band = min-max)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, "ablation.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_cost(runs, out_dir):
    """What MLFQ costs the rest of the system, relative to fcfs in the same scenario and seed."""
    def rel(get):
        def f(scenario, stage):
            base = {r["args"]["seed"]: get(r) for r in runs.get((scenario, "fcfs"), []) if get(r)}
            return [100 * (get(r) / base[r["args"]["seed"]] - 1)
                    for r in runs.get((scenario, stage), []) if r["args"]["seed"] in base and get(r) is not None]
        return f
    metrics = [
        ("output throughput vs. fcfs", "% change, 0 = no damage", rel(lambda r: r["output_tokens_per_s"])),
        ("agent turn end-to-end p50 vs. fcfs", "% change, 0 = no damage", rel(lambda r: r["agent"]["e2e_p50"])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURFACE)
    for ax, (title, unit, f) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK_2, lw=1)
        ends = []
        for scenario, (name, color, marker) in SCENARIOS.items():
            xs, mean, lo, hi = [], [], [], []
            for i, stage in enumerate(STAGES[1:], start=1):
                vals = f(scenario, stage)
                if vals:
                    xs.append(i), mean.append(np.mean(vals)), lo.append(min(vals)), hi.append(max(vals))
            if not xs:
                continue
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(xs, mean, color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2, label=name, zorder=3)
            ends.append((xs[-1], mean[-1]))
        style(ax, title, unit)
        ax.set_xlim(0.7, len(STAGES) - 0.3)
        label_ends(ax, ends, lambda v: f"{v:+.1f}%")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.93), ncol=5, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.suptitle("Damage to the rest of the system (normal load and only-short skip the middle stages)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = os.path.join(out_dir, "ablation_cost.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def table(runs):
    """Markdown table, mean over seeds."""
    cols = [("chat TTFT p50", lambda r: r["chat"]["ttft_p50"] * 1e3, "{:.0f} ms"),
            ("chat TTFT p99", lambda r: r["chat"]["ttft_p99"] * 1e3, "{:.0f} ms"),
            ("chat e2e p99", lambda r: r["chat"]["e2e_p99"], "{:.2f} s"),
            ("agent e2e p50", lambda r: r["agent"]["e2e_p50"], "{:.2f} s"),
            ("tok/s", lambda r: r["output_tokens_per_s"], "{:.0f}"),
            ("preempt", lambda r: r["stats"]["preemptions"], "{:.0f}"),
            ("paused", lambda r: r["stats"].get("pauses", 0), "{:.0f}"),
            ("swapped out", lambda r: r["stats"]["swap_outs"], "{:.0f}")]
    lines = ["| scenario | config | " + " | ".join(c for c, _, _ in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for scenario in SCENARIOS:
        for stage in ["fcfs", "priority-swap"] + STAGES[1:]:
            rs = runs.get((scenario, stage))
            if rs:
                cells = [fmt.format(np.mean(vals)) if (vals := [v for v in map(get, rs) if v is not None]) else "-"
                         for _, get, fmt in cols]
                lines.append(f"| {scenario} | {stage} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(path="results/mlfq_v2_bench.jsonl", out_dir="docs/img"):
    os.makedirs(out_dir, exist_ok=True)
    runs = load(path)
    print(plot_ablation(runs, out_dir))
    print(plot_cost(runs, out_dir))
    print(table(runs))


if __name__ == "__main__":
    main(*sys.argv[1:])
