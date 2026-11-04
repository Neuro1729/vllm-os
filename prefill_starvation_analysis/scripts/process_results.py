"""Turn raw sweep outputs into the compact processed results used by the figures and README.

GPU: <analysis_dir>/sw_*.analysis.json (from scripts/gpu/analyze.py)
       -> results/processed/gpu_runs/<run>.json  (per-request rows + run metrics, no step timeline)
       -> results/processed/summary_gpu.csv
CPU: <cpu_sweep.json> (from scripts/cpu/cpu_sweep.py)
       -> results/processed/cpu_sweep.json, results/processed/summary_cpu.csv

All loads reported for GPU runs are MEASURED: offered long-prompt tokens per second divided by
the measured prefill capacity (tokens/s while the 16-prompt burst drains with a full budget).

usage: python scripts/process_results.py <analysis_dir> <cpu_sweep.json>
"""
import csv
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(A, "results", "processed")
BUDGET = 2048


def q(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))] if a else None


def capacity(runs):
    """Prefill tokens/s while 16 x 16K high-priority prompts drain (budget always full)."""
    d = runs["sw_burst_adaptive_16"]
    lt = [r for r in d["requests"] if r["cls"] == "long" and r.get("t_first")]
    return sum(r["prompt"] for r in lt) / (max(r["t_first"] for r in lt) -
                                         min(r["t_send"] for r in lt))


def budget_use(tl):
    num = den = 0.0
    for a, b in zip(tl, tl[1:]):
        dt = b["t"] - a["t"]
        if 0 < dt < 2.0:  # skip idle gaps between requests
            num += a["tok"] * dt
            den += BUDGET * dt
    return num / den if den else None


def run_metrics(name, d, cap):
    kind, variant, x = name[3:].split("_", 2)  # sw_<kind>_<variant>_<x>
    P = [r for r in d["requests"] if r["cls"] == "short"]
    L = [r for r in d["requests"] if r["cls"] == "long"]
    tt = [r["ttft_s"] for r in P if r.get("ttft_s") is not None]
    reasons = defaultdict(int)
    for r in P:
        for k, v in (r.get("reasons") or {}).items():
            reasons[k] += v
    tl = d["timeline"]
    lt_done = [r for r in L if r.get("t_first")]
    m = {
        "run": name, "kind": kind, "variant": variant,
        "n_long" if kind == "burst" else "nominal_load": x,
        "probes": len(P), "probes_served": len(tt),
        "probe_ttft_p50_s": q(tt, .5), "probe_ttft_p90_s": q(tt, .9),
        "probe_ttft_max_s": max(tt) if tt else None,
        "probe_max_zero_run_steps": max((r.get("max_zero_run") or 0) for r in P),
        "zero_steps_budget": reasons.get("budget", 0),
        "zero_steps_prio_hold": reasons.get("prio_hold", 0),
        "zero_steps_slots": reasons.get("slots", 0),
        "zero_steps_other_kv": reasons.get("other_kv", 0),
        "long_ttft_p50_s": q([r["ttft_s"] for r in lt_done], .5),
        "steps": len(tl),
        # token-budget use weighted by step duration (decode-only steps are short)
        "budget_use_time_weighted": budget_use(tl),
        "kv_usage_max": max(s["kv_usage"] for s in tl) if tl else None,
        "preemptions": sum(s.get("preempted", 0) for s in tl),
    }
    if kind == "stream" and lt_done:
        a = min(r["t_send"] for r in lt_done)
        b = max(r["t_first"] for r in lt_done)
        m["measured_load"] = sum(r["prompt"] for r in lt_done) / (b - a) / cap
    return m


def main():
    adir, cpu_path = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(OUT, "gpu_runs"), exist_ok=True)
    runs = {os.path.basename(p)[:-len(".analysis.json")]: json.load(open(p))
            for p in sorted(glob.glob(os.path.join(adir, "sw_*.analysis.json")))}
    cap = capacity(runs)
    rows = []
    for name, d in runs.items():
        m = run_metrics(name, d, cap)
        rows.append(m)
        keep = ["id", "cls", "prompt", "output", "prio", "arrival_ms", "t_send", "t_first",
                "t_done", "err", "ttft_s", "starved_steps", "max_zero_run", "wait_steps",
                "prefill_steps", "reasons"]
        json.dump({"meta": d["meta"], "metrics": m,
                   "requests": [{k: r.get(k) for k in keep} for r in d["requests"]]},
                  open(os.path.join(OUT, "gpu_runs", name + ".json"), "w"))
    cols = ["run", "kind", "variant", "n_long", "nominal_load", "measured_load", "probes",
            "probes_served", "probe_ttft_p50_s", "probe_ttft_p90_s", "probe_ttft_max_s",
            "probe_max_zero_run_steps", "zero_steps_budget", "zero_steps_prio_hold",
            "zero_steps_slots", "zero_steps_other_kv", "long_ttft_p50_s", "steps",
            "budget_use_time_weighted", "kv_usage_max", "preemptions"]
    with open(os.path.join(OUT, "summary_gpu.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for m in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in m.items() if k in cols})
    json.dump({"prefill_capacity_tok_s": cap, "budget": BUDGET},
              open(os.path.join(OUT, "gpu_capacity.json"), "w"), indent=1)

    shutil.copy(cpu_path, os.path.join(OUT, "cpu_sweep.json"))
    cpu = json.load(open(cpu_path))
    with open(os.path.join(OUT, "summary_cpu.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "variant", "x", "probes", "probes_served",
                    "probe_max_zero_run_steps", "probe_zero_steps_p50"])
        for exp in ("burst", "stream"):
            for k, v in cpu[exp].items():
                var, x = k.split("/")
                P = v["probes"]
                w.writerow([exp, var, x, len(P), sum(p["served"] for p in P),
                            max(p["max_zero_run"] for p in P),
                            q([p["zero_steps"] for p in P], .5)])
        for var, r in cpu["unbounded"].items():
            w.writerow(["unbounded", var, "8 in flight", 1, int(r["probe_served"]),
                        r["probe_max_zero_run"], r["probe_zero_steps"]])
    print(f"GPU runs: {len(rows)}  capacity {cap:.0f} prefill tok/s  -> {OUT}")


if __name__ == "__main__":
    main()
