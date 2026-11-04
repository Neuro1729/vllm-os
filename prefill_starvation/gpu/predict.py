"""Predict a GPU scenario on CPU: real vLLM Scheduler + step cost fitted to the RTX 3090 probe.

usage: predict.py WORKLOAD.json --budget 2048 [--policy fcfs] [--threshold 512] [--no-adaptive]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vllm_harness as H  # noqa: E402
from sim import CostModel  # noqa: E402

# Qwen2.5-1.5B bf16 on RTX 3090, budget 2048 (fit to gpu/workloads/probe.json measurements):
#   30K prompt alone 2.79 s, 16K alone 1.16 s, 400 concurrent decodes ~84 ms/step
COST_3090 = CostModel(base_ms=15.0, per_tok_ms=0.05, attn_prefill_ms=2.9e-6,
                      attn_decode_ms=1.06e-4, overhead_ms=0.0)


def q(a, p):
    return a[min(len(a) - 1, int(p * len(a)))] if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--threshold", type=int, default=512)
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--policy", default="fcfs")
    ap.add_argument("--kv", type=int, default=577_216)
    a = ap.parse_args()
    wl = json.load(open(a.workload))
    dur = max(w["arrival_ms"] for w in wl) / 1000
    recs, tl, st = H.run(wl, budget=a.budget, threshold=a.threshold,
                         adaptive=not a.no_adaptive, policy=a.policy, kv_tokens=a.kv,
                         cost=COST_3090, t_end_ms=(dur + 900) * 1000, verbose=False)
    out = {"workload": os.path.basename(a.workload), "budget": a.budget, "policy": a.policy,
           "util": round(st["busy_s"] / st["sim_time_s"], 3),
           "drain_s": round(st["sim_time_s"] - dur, 1), "unfinished": st["unfinished"],
           "avg_decodes": round(sum(x["decodes"] for x in tl) / max(1, len(tl)))}
    for cls in ("short", "medium", "long"):
        g = [r for r in recs.values() if r["cls"] == cls and r["t_first"] is not None]
        tt = sorted((r["t_first"] - r["arrival_ms"]) / 1000 for r in g)
        stv = sorted(r["starved"] for r in g)
        if tt:
            out[cls] = {"n": len(tt), "p50": round(q(tt, .5), 2), "p99": round(q(tt, .99), 2),
                        "max": round(tt[-1], 2), "starved_p99": q(stv, .99),
                        "starved_max": stv[-1], "frac_gt_5s": round(sum(x > 5 for x in tt) /
                                                                   len(tt), 3)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
