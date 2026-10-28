"""Bar charts from priority_bench JSONL results (last run per label wins).

    python -m benchmarks.plot [results/priority_bench.jsonl] [docs/img]
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

POLICIES = ["fcfs", "priority-recompute", "priority-swap", "mlfq"]
COLORS = {"fcfs": "#8c8c8c", "priority-recompute": "#e3a33b", "priority-swap": "#3b82c4", "mlfq": "#2a9d5c"}
SCENARIOS = {
    "normal": "Normal load (batch never full)",
    "batch-cap": "Batch full: 12 agents, 8 slots, chat has priority",
    "no-priority": "Batch full, no priorities set",
    "kv-pressure": "KV cache full, batch slots free",
}
METRICS = [    # (title, getter, unit, lower is better)
    ("chat TTFT p50", lambda r: r["chat"]["ttft_p50"] * 1e3, "ms", True),
    ("chat TTFT p99", lambda r: r["chat"]["ttft_p99"] * 1e3, "ms", True),
    ("chat end-to-end p99", lambda r: r["chat"]["e2e_p99"], "s", True),
    ("agent turn end-to-end p50", lambda r: r["agent"]["e2e_p50"], "s", True),
    ("output throughput", lambda r: r["output_tokens_per_s"], "tok/s", False),
]


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["label"]] = r
    return rows


def bars(ax, names, values, title, unit, lower_better):
    xs = range(len(names))
    ax.bar(xs, values, color=[COLORS[n] for n in names])
    base = values[0]
    for x, v in zip(xs, values):
        note = f"{v:.0f}" if v >= 10 else f"{v:.2f}"
        if x and base:
            note += f"\n{v / base:.2f}x"
        ax.text(x, v, note, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(xs), names, rotation=20, ha="right", fontsize=8)
    ax.set_title(f"{title} ({unit}, {'lower' if lower_better else 'higher'} is better)", fontsize=9)
    ax.margins(y=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def plot_scenario(rows, scenario, out_dir):
    names = [p for p in POLICIES if f"{scenario}/{p}" in rows]
    if not names:
        return None
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.6))
    for ax, (title, get, unit, lower) in zip(axes, METRICS):
        bars(ax, names, [get(rows[f"{scenario}/{n}"]) for n in names], title, unit, lower)
    fig.suptitle(SCENARIOS[scenario], fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{scenario}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_overview(rows, out_dir):
    """Per scenario: MLFQ relative to fcfs (1.0 = no change)."""
    scenarios = [s for s in SCENARIOS if f"{s}/fcfs" in rows and f"{s}/mlfq" in rows]
    if not scenarios:
        return None
    picks = [METRICS[1], METRICS[3], METRICS[4]]    # chat TTFT p99, agent e2e p50, throughput
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(picks)
    for i, (title, get, _, lower) in enumerate(picks):
        ratios = [get(rows[f"{s}/mlfq"]) / get(rows[f"{s}/fcfs"]) for s in scenarios]
        xs = [j + i * width for j in range(len(scenarios))]
        ax.bar(xs, ratios, width, label=f"{title} ({'lower' if lower else 'higher'} is better)")
        for x, v in zip(xs, ratios):
            ax.text(x, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_xticks([j + width for j in range(len(scenarios))],
                  [textwrap.fill(SCENARIOS[s], 24) for s in scenarios], fontsize=8)
    ax.set_ylabel("MLFQ / fcfs")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("MLFQ vs. nano-vllm fcfs", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(path="results/priority_bench.jsonl", out_dir="docs/img"):
    os.makedirs(out_dir, exist_ok=True)
    rows = load(path)
    for p in [plot_overview(rows, out_dir)] + [plot_scenario(rows, s, out_dir) for s in SCENARIOS]:
        if p:
            print(p)


if __name__ == "__main__":
    main(*sys.argv[1:])
