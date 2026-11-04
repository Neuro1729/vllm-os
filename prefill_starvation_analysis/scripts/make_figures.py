"""Figures for the low-priority prefill-starvation analysis (PNG 300 dpi + PDF).

Inputs : results/processed/{gpu_runs/*.json, cpu_sweep.json, gpu_capacity.json}
         results/raw/gpu_sched_trace.jsonl.gz (Figure 1 only)
Outputs: figures/fig1_step_heatmap, fig2_wait_vs_n, fig3_unserved_ccdf, fig4_wait_vs_load
usage  : python scripts/make_figures.py      (needs matplotlib + numpy)
"""
import gzip
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(A, "results", "processed")
TRACE = os.path.join(A, "results", "raw", "gpu_sched_trace.jsonl.gz")
FIG = os.path.join(A, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
# one fixed color + marker per scheduler setting, used in every figure
VARIANTS = {
    "adaptive": ("Cap 512 + adaptive", "#2a78d6", "o", "-"),
    "cap512":   ("Cap 512", "#eb6834", "s", "--"),
    "cap256":   ("Cap 256", "#1baf7a", "^", "-"),
    "nocap":    ("No per-request cap", "#eda100", "D", "-"),
    "pr65283":  ("Cap 512 + adaptive + PR #65283", "#e87ba4", "v", "-"),
    "equal":    ("Equal priority (FCFS control)", "#008300", "x", "-"),
}
NS = [1, 2, 4, 8, 16]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.labelcolor": INK2, "axes.edgecolor": "#bdbcb6",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.fontsize": 8.8, "legend.frameon": False, "lines.linewidth": 1.8,
    "lines.markersize": 5.5, "figure.dpi": 100, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


def run(name):
    return json.load(open(os.path.join(PROC, "gpu_runs", f"{name}.json")))


def probes(d):
    return [r for r in d["requests"] if r["cls"] == "short"]


def line(ax, x, y, v):
    lab, c, m, ls = VARIANTS[v]
    hollow = ls == "--"
    if v == "adaptive":  # wide band underneath: settings that coincide with it stay visible
        ax.plot(x, y, color=c, linewidth=5.0, alpha=0.45, solid_capstyle="round", zorder=1)
        ax.plot(x, y, color=c, marker=m, markersize=7, linewidth=0, zorder=1.5, alpha=0.6)
        ax.plot([], [], color=c, linewidth=5.0, alpha=0.45, label=lab)
        return
    ax.plot(x, y, color=c, marker=m, linestyle=ls, label=lab, zorder=2,
            linewidth=1.5 if v in ("cap512", "equal") else 1.8,
            markerfacecolor=SURF if hollow else c,
            markeredgecolor=c, markeredgewidth=1.3 if (hollow or m == "x") else 0.8)


# ------------------------------------------------------------------ Figure 1: step heatmap
def fig1(name="sw_burst_adaptive_8", n_probe_rows=16):
    d = run(name)
    t_lo = d["meta"]["t0"] - 3.0
    t_hi = max((r["t_done"] or r["t_send"]) for r in d["requests"]) + 2.0
    cls = {r["id"]: r["cls"] for r in d["requests"]}

    def bid(rid):
        i = rid.find("r0")
        return int(rid[i + 1:i + 10]) if i >= 0 else None

    steps, add_t = [], {}
    with gzip.open(TRACE, "rt") as f:
        for line_ in f:
            x = json.loads(line_)
            if not (t_lo <= x["t"] <= t_hi):
                continue
            if x.get("ev") == "add":
                add_t[bid(x["rid"])] = x["t"]
            elif "step" in x:
                steps.append(x)
    longs = sorted(b for b in add_t if cls.get(b) == "long")
    prb = sorted(b for b in add_t if cls.get(b) == "short")[:n_probe_rows]
    rows = longs + prb
    ridx = {b: i for i, b in enumerate(rows)}
    k0 = next(k for k, s in enumerate(steps) if any(cls.get(bid(r)) == "long" for r in s["pf"]))
    k_last = max(k for k, s in enumerate(steps) if any(bid(r) in ridx for r in s["pf"]))
    S = steps[k0:k_last + 2]
    # codes: 0 absent / finished, 1 present but 0 tokens, 2: 1-255 tokens, 3: 256-511, 4: 512
    M = np.zeros((len(rows), len(S)), dtype=int)
    done = {}
    for j, s in enumerate(S):
        for r, (n, before, total) in s["pf"].items():
            b = bid(r)
            if b in ridx:
                M[ridx[b], j] = 2 if n < 256 else (3 if n < 512 else 4)
                if before + n >= total:
                    done[b] = j
    for b, i in ridx.items():
        for j, s in enumerate(S):
            if M[i, j] == 0 and add_t[b] <= s["t"] and j <= done.get(b, -1):
                M[i, j] = 1
    cmap = ListedColormap([SURF, "#e34948", "#b7d3f6", "#5598e7", "#184f95"])
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.imshow(M, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"High-priority 16K #{i + 1}" for i in range(len(longs))] +
                       [f"Low-priority 20-tok #{i + 1}" for i in range(len(prb))], fontsize=7.5)
    ax.axhline(len(longs) - 0.5, color=INK2, linewidth=0.8)
    ax.set_xlabel("Scheduler step (2048-token budget; 32 running decodes not shown)")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors[1:]]
    ax.legend(handles, ["waiting, 0 tokens", "1–255 tokens", "256–511 tokens", "512 tokens (cap)"],
              loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=4, handlelength=1.2,
              columnspacing=1.2)
    ax.set_title("GPU trace: 8 high-priority 16K prompts use every step's budget;\n"
                 "low-priority 20-token requests wait until the burst is almost drained")
    save(fig, "fig1_step_heatmap")
    zero = (M[len(longs):] == 1).sum(axis=1)
    return {"steps_shown": len(S), "probe_zero_steps_shown": zero.tolist()}


# ------------------------------------------------------------------ Figure 2: wait vs N
def fig2(cpu):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    out = {}
    for v in VARIANTS:
        gy = [max(r["ttft_s"] for r in probes(run(f"sw_burst_{v}_{n}"))) for n in NS]
        cy = [max(p["max_zero_run"] for p in cpu["burst"][f"{v}/{n}"]["probes"]) for n in NS]
        line(a, NS, gy, v)
        line(b, NS, cy, v)
        out[v] = {"gpu_max_ttft_s": gy, "cpu_max_zero_run": cy}
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.minorticks_off()
        ax.set_xlabel("High-priority 16K prompts arriving together")
        ax.set_ylim(bottom=0)
    a.set_ylabel("Worst time to first token (s)")
    a.set_title("(a) GPU measurement")
    b.set_ylabel("Longest run of 0-token steps")
    b.set_title("(b) CPU, real vLLM scheduler")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.suptitle("Low-priority 20-token request: worst wait vs. high-priority burst size",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "fig2_wait_vs_n")
    return out


# ------------------------------------------------------------------ Figure 3: CCDF
def fig3(n=8, shown=("adaptive", "cap256", "pr65283", "equal")):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    curves = {}
    for v in VARIANTS:
        z = np.array([r.get("starved_steps") or 0 for r in probes(run(f"sw_burst_{v}_{n}"))])
        xs = np.arange(1, 80)
        curves[v] = np.array([100.0 * (z >= k).mean() for k in xs])
        if v in shown:
            lab, c, _, ls = VARIANTS[v]
            wide = v == "adaptive"  # band underneath: the equal-priority curve coincides
            ax.step(xs, curves[v], where="post", color=c, label=lab,
                    linewidth=5.0 if wide else 1.6, alpha=0.45 if wide else 1.0,
                    zorder=1 if wide else 2)
    ax.set_xlim(1, 75)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(0, 60)
    ax.set_xlabel("Scheduler steps with 0 tokens (k)")
    ax.set_ylabel("% of requests that got 0 tokens\nfor at least k steps")
    ax.legend(loc="upper right")
    ax.set_title(f"GPU, burst of {n} high-priority 16K prompts: how long low-priority\n"
                 "20-token requests got 0 tokens")
    save(fig, "fig3_unserved_ccdf")
    omitted = {v: float(np.abs(curves[v] - curves["adaptive"]).max())
               for v in VARIANTS if v not in shown}
    return {"max_abs_gap_pct_points_vs_adaptive_for_omitted": omitted,
            "pct_waiting_ge_1_step": {v: float(curves[v][0]) for v in VARIANTS}}


# ------------------------------------------------------------------ Figure 4: wait vs load
def fig4(cpu):
    cap = json.load(open(os.path.join(PROC, "gpu_capacity.json")))["prefill_capacity_tok_s"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    out = {"gpu": {}, "cpu": {}}
    for v in ("adaptive", "pr65283"):
        pts = []
        for rho in ("0.7", "0.85", "0.95"):
            m = run(f"sw_stream_{v}_{rho}")["metrics"]
            pts.append((m["measured_load"], m["probe_ttft_max_s"]))
        pts.sort()
        line(a, [p[0] for p in pts], [p[1] for p in pts], v)
        out["gpu"][v] = pts
    for v in ("adaptive", "pr65283", "equal"):
        keys = sorted((float(k.split("/")[1]), k) for k in cpu["stream"] if k.startswith(v + "/"))
        xs = [x for x, _ in keys]
        ys = [max(p["max_zero_run"] for p in cpu["stream"][k]["probes"]) for _, k in keys]
        line(b, xs, ys, v)
        out["cpu"][v] = list(zip(xs, ys))
    a.set_xlabel("Measured high-priority prefill load")
    a.set_ylabel("Worst time to first token (s)")
    a.set_title("(a) GPU measurement")
    a.set_xlim(0.38, 0.6)
    b.set_xlabel("High-priority prefill load (exact)")
    b.set_ylabel("Longest run of 0-token steps")
    b.set_title("(b) CPU, real vLLM scheduler")
    b.set_xlim(0.45, 1.0)
    for ax in (a, b):
        ax.set_ylim(bottom=0)
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    fig.suptitle("Low-priority 20-token request under steady high-priority traffic",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "fig4_wait_vs_load")
    out["gpu_capacity_tok_s"] = cap
    return out


if __name__ == "__main__":
    cpu = json.load(open(os.path.join(PROC, "cpu_sweep.json")))
    check = {"fig1": fig1(), "fig2": fig2(cpu), "fig3": fig3(), "fig4": fig4(cpu)}
    json.dump(check, open(os.path.join(FIG, "figure_data.json"), "w"), indent=1)
    print(json.dumps(check, indent=1)[:3000])
