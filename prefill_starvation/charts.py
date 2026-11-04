"""Charts for the prefill-starvation report (RTX 3090 GPU runs, vLLM 0.32.0, priority policy).

Reads results/gpu/prio2048_*.analysis.json and writes PNGs to results/charts/.
usage: .venv-charts/bin/python charts.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results/gpu")
OUT = os.path.join(HERE, "results/charts")
os.makedirs(OUT, exist_ok=True)

# Validated reference palette (light), fixed order; text never wears series color.
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # blue, orange, aqua, yellow

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
    "ytick.color": TEXT2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
})


def load(name):
    p = os.path.join(RES, f"prio2048_{name}.analysis.json")
    return json.load(open(p)) if os.path.exists(p) else None


def probes(d):
    return sorted([r for r in d["requests"] if r["cls"] == "short" and r.get("ttft_s") is not None],
                  key=lambda r: r["arrival_ms"])


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))]


def secs(v):
    return f"{v:.2f} s" if v < 1 else f"{v:.1f} s"


def bar_chart():
    """Hero: worst-case TTFT of a 20-token request, per scenario and priority setup."""
    groups = [
        ("Decode-heavy\n600 decodes + 4×30K prompts",
         ["decodeheavy_base", "decodeheavy", "decodeheavy_inv", "decodeheavy_lowprobe"]),
        ("Normal decode, many prefills\n50 decodes + 8×30K prompts",
         ["manyprefill_base", "manyprefill", "manyprefill_inv", "manyprefill_lowprobe"]),
    ]
    series = [("No long prompts (baseline)", S1), ("Long prompts, same priority", S2),
              ("Long prompts, short = HIGH priority", S3),
              ("Long prompts, short = LOW priority", S4)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, (title, names) in zip(axes, groups):
        vals_max, vals_p50 = [], []
        for n in names:
            d = load(n)
            tt = [r["ttft_s"] for r in probes(d)] if d else [float("nan")]
            vals_max.append(max(tt))
            vals_p50.append(q(tt, .5))
        xs = range(len(names))
        bars = ax.bar(xs, vals_max, width=0.72, color=[c for _, c in series],
                      edgecolor=SURFACE, linewidth=2)
        for b, vm, vp in zip(bars, vals_max, vals_p50):
            ax.text(b.get_x() + b.get_width() / 2, vm + 0.6, secs(vm), ha="center",
                    va="bottom", fontsize=12, fontweight="bold", color=TEXT)
            ax.text(b.get_x() + b.get_width() / 2, vm + 2.4, f"median {secs(vp)}",
                    ha="center", va="bottom", fontsize=9, color=TEXT2)
        ax.set_xticks([])
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Worst time-to-first-token of a 20-token request (s)")
    axes[0].set_ylim(0, 31)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in series]
    fig.legend(handles, [s for s, _ in series], loc="lower center", ncol=4, frameon=False,
               fontsize=10)
    fig.suptitle("A 20-token request needs ONE scheduler step, yet waits up to 26 s in vLLM "
                 "0.32 (priority scheduling)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.905, "RTX 3090, Qwen2.5-1.5B, max_num_batched_tokens=2048, "
             "long_prefill_token_threshold=512 (adaptive). 40 probes, one every 0.5 s. "
             "0 preemptions, KV cache ≤ 77% full.", fontsize=9.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(os.path.join(OUT, "01_worst_ttft_bars.png"), dpi=150)
    plt.close(fig)


def zero_run_chart():
    """Longest run of consecutive scheduler steps in which a 20-token request got 0 tokens."""
    rows = [("Decode-heavy, no long prompts", "decodeheavy_base", S1),
            ("Decode-heavy, same priority", "decodeheavy", S2),
            ("Decode-heavy, short HIGH priority", "decodeheavy_inv", S3),
            ("Decode-heavy, short LOW priority", "decodeheavy_lowprobe", S4),
            ("Many prefills, no long prompts", "manyprefill_base", S1),
            ("Many prefills, same priority", "manyprefill", S2),
            ("Many prefills, short HIGH priority", "manyprefill_inv", S3),
            ("Many prefills, short LOW priority", "manyprefill_lowprobe", S4),
            ("Steady HIGH-prio stream, short LOW priority", "lowprio_stream", S4)]
    labels, vals, cols = [], [], []
    for lab, n, c in rows:
        d = load(n)
        if not d:
            continue
        labels.append(lab)
        vals.append(max((r.get("max_zero_run") or 0) for r in probes(d)))
        cols.append(c)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ys = list(range(len(labels)))[::-1]
    ax.barh(ys, vals, height=0.66, color=cols, edgecolor=SURFACE, linewidth=2)
    for y, v in zip(ys, vals):
        ax.text(v + 3, y, f"{v} steps", va="center", fontsize=10.5, color=TEXT)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Longest run of consecutive scheduler steps with 0 tokens "
                  "(server-side VLLM_SCHED_TRACE)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("A 20-token request gets 0 tokens for hundreds of steps in a row\n"
                 "(each step has a 2048-token budget)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_zero_token_runs.png"), dpi=150)
    plt.close(fig)


def timeline_chart(name, fname, title, sub):
    """Per-step view: prefill tokens of long prompts vs. probe TTFT over time."""
    d = load(name)
    if not d:
        return
    tl = [x for x in d["timeline"] if x["t"] >= 0]
    P = probes(d)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    ax1.plot([x["t"] for x in tl], [x["starved"] for x in tl], color=S2, linewidth=2)
    ax1.set_ylabel("Requests waiting for\ntheir first token")
    ax1.set_title(title)
    ax2.plot([r["arrival_ms"] / 1000 for r in P], [r["ttft_s"] for r in P], color=S4,
             linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    worst = max(P, key=lambda r: r["ttft_s"])
    ax2.annotate(f"{worst['ttft_s']:.1f} s for a 20-token prompt",
                 (worst["arrival_ms"] / 1000, worst["ttft_s"]),
                 xytext=(12, -4), textcoords="offset points", fontsize=10.5, color=TEXT)
    ax2.set_ylabel("Time-to-first-token of the\n20-token probe sent at time t (s)")
    ax2.set_xlabel("Time since start of run (s)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    fig.text(0.01, 0.005, sub, fontsize=9.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def dataset_chart():
    """Mooncake real trace: TTFT CDF of short requests (<=2K tokens), base vs. with long ones."""
    runs = [("ds_v3_S2base", "Trace without its long (≥16K) requests", S1),
            ("ds_v3_S2", "Full trace, same priority", S2),
            ("ds_v3_S3", "Full trace, short = HIGH priority", S3),
            ("ds_v3_S3low", "Full trace, short = LOW priority", S4)]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    any_ = False
    for n, lab, c in runs:
        d = load(n)
        if not d:
            continue
        tt = sorted(r["ttft_s"] for r in d["requests"]
                    if r["cls"] == "short" and r.get("ttft_s") is not None)
        if not tt:
            continue
        any_ = True
        ys = [(i + 1) / len(tt) for i in range(len(tt))]
        ax.step(tt, ys, where="post", color=c, linewidth=2,
                label=f"{lab}  (p50 {secs(q(tt, .5))}, max {secs(tt[-1])})")
    if not any_:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("Time-to-first-token of short requests, prompt ≤ 2K tokens (s, log scale)")
    ax.set_ylabel("Fraction of short requests")
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    ax.set_title("Real Mooncake trace (7.5 min, ~50% prefill load): time-to-first-token "
                 "of short requests", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_mooncake_trace_cdf.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    bar_chart()
    zero_run_chart()
    timeline_chart("manyprefill_lowprobe", "03_timeline_many_prefills.png",
                   "Normal decode load: while 8 long HIGH-priority prompts prefill, "
                   "a LOW-priority 20-token request waits",
                   "50 decodes, eight 30K-token prompts (high priority) arrive at t≈20 s; a 20-token "
                   "low-priority probe every 0.5 s from t=21 s. Every probe gets its first token only "
                   "when the long prefills finish (t≈47 s).")
    timeline_chart("lowprio_stream", "04_timeline_no_aging.png",
                   "No aging: once a high-priority backlog forms, a LOW-priority 20-token "
                   "request waits for all of it",
                   "Steady stream of 16K-token high-priority prompts (every 1.2 s from t=5 s); "
                   "a 20-token low-priority probe every 1 s.")
    dataset_chart()
    print(sorted(os.listdir(OUT)))
