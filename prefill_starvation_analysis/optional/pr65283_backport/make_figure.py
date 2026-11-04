"""Optional comparison figure: cap 512 + adaptive vs. the same + PR #65283 backport.

Both settings ran in the same GPU session (backport applied, switched at runtime).
usage: python optional/pr65283_backport/make_figure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, "results", "gpu_runs")
NS = [1, 2, 4, 8, 16]
S = {"adaptive": ("Cap 512 + adaptive", "#2a78d6", "o"),
     "pr65283": ("Cap 512 + adaptive + PR #65283 backport", "#e87ba4", "v")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "pdf.fonttype": 42})


def m(name):
    return json.load(open(os.path.join(R, f"{name}.json")))["metrics"]


def offered(name, cap=12959.0, window_s=60.0):
    """High-priority prompt tokens sent per second / prefill capacity measured in session 1."""
    d = json.load(open(os.path.join(R, f"{name}.json")))
    return sum(r["prompt"] for r in d["requests"] if r["cls"] == "long") / window_s / cap


fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
for v, (lab, c, mk) in S.items():
    a.plot(NS, [m(f"sw_burst_{v}_{n}")["probe_ttft_max_s"] for n in NS], color=c, marker=mk,
           label=lab)
    pts = sorted((offered(f"sw_stream_{v}_{r}"), m(f"sw_stream_{v}_{r}")["probe_ttft_max_s"])
                 for r in ("0.7", "0.85", "0.95"))
    b.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=mk, label=lab)
a.set_xscale("log", base=2)
a.set_xticks(NS)
a.set_xticklabels([str(n) for n in NS])
a.minorticks_off()
a.set_xlabel("High-priority 16K prompts arriving together")
a.set_title("(a) Burst")
b.set_xlabel("Offered high-priority load (measured)")
b.set_title("(b) Steady traffic")
for ax in (a, b):
    ax.set_ylabel("Worst TTFT (s)")
    ax.set_ylim(bottom=0)
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=2)
fig.suptitle("Optional: PR #65283 backport vs. baseline, low-priority 20-token requests (GPU)",
             x=0.01, ha="left", fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
    fig.savefig(os.path.join(D, "figures", f"pr65283_comparison.{ext}"), bbox_inches="tight", **kw)
