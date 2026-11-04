"""CPU sweep with the REAL vLLM 0.32 Scheduler driven step by step (no model, no GPU).

Exact step counts for LOW-priority 20-token probes:
  A) burst: N high-priority 16K prompts arrive together (N = 1..16) x 6 scheduler settings
  B) stream: Poisson high-priority 16K prompts at load rho x settings
  C) unbounded: 8 high-priority 16K prompts kept in flight (closed loop) for 20,000 steps
usage: python scripts/cpu/cpu_sweep.py OUT.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
from cost_model import CostModel  # noqa: E402
from sweep_workloads import NS, RHOS, VARIANTS, burst, stream, LONG  # noqa: E402
import vllm_harness as H  # noqa: E402

# RTX 3090 fit (results/gpu/costfit_3090.json); only places arrivals relative to steps.
# Constant step time = measured RTX 3090 full-budget step (~180 ms). The stream load rho is
# defined against the scheduler's own prefill capacity: (2048 - 32 decodes) tokens per step.
STEP_MS = 180.0
COST = CostModel(base_ms=STEP_MS - 1.0, per_tok_ms=0.0, attn_prefill_ms=0.0, attn_decode_ms=0.0)
CAP_TOK_S = (2048 - 32) / STEP_MS * 1000.0


def run_variant(wl, v, **kw):
    recs, _, stats = H.run(wl, budget=2048, threshold=v["thr"], adaptive=v["adaptive"],
                           policy="priority", kv_tokens=600_000, cost=COST, verbose=False, **kw)
    return recs, stats


def probe_rows(recs):
    out = []
    for r in recs.values():
        if r["cls"] != "short":
            continue
        out.append({"arr_ms": r["arrival_ms"],
                    "served": r["t_first"] is not None,
                    "ttft_ms": None if r["t_first"] is None else r["t_first"] - r["arrival_ms"],
                    "zero_steps": r["starved"], "max_zero_run": r["max_starve_run"]})
    return out


def long_ttft(recs):
    return sorted(r["t_first"] - r["arrival_ms"] for r in recs.values()
                  if r["cls"] == "long" and r["t_first"] is not None)


def main():
    res = {"burst": {}, "stream": {}, "unbounded": {}}
    for vn, v in VARIANTS.items():
        for n in NS:
            wl = burst(n, equal=v["equal"], cap_tok_s=CAP_TOK_S)
            recs, st = run_variant(wl, v, t_end_ms=max(w["arrival_ms"] for w in wl) + 120000)
            res["burst"][f"{vn}/{n}"] = {"probes": probe_rows(recs), "long_ttft_ms":
                                        long_ttft(recs), "steps": st["steps"]}
            p = res["burst"][f"{vn}/{n}"]["probes"]
            print(f"burst {vn:9s} N={n:2d} probes={len(p)} max_zero_run="
                  f"{max(x['max_zero_run'] for x in p):4d} unserved={sum(not x['served'] for x in p)}",
                  flush=True)
    for vn in ("adaptive", "cap512", "equal"):
        v = VARIANTS[vn]
        for rho in RHOS:
            wl = stream(rho, equal=v["equal"], cap_tok_s=CAP_TOK_S)
            recs, st = run_variant(wl, v, t_end_ms=max(w["arrival_ms"] for w in wl) + 120000)
            res["stream"][f"{vn}/{rho}"] = {"probes": probe_rows(recs), "long_ttft_ms":
                                           long_ttft(recs), "steps": st["steps"]}
            p = res["stream"][f"{vn}/{rho}"]["probes"]
            print(f"stream {vn:9s} rho={rho} max_zero_run={max(x['max_zero_run'] for x in p):5d}"
                  f" unserved={sum(not x['served'] for x in p)}", flush=True)
    # C) unbounded: closed loop of 8 high-priority 16K prompts + one LOW-priority probe
    for vn in ("cap512", "adaptive", "cap256", "nocap", "equal"):
        v = VARIANTS[vn]
        pp = 0 if v["equal"] else 2
        wl = [dict(id=i, arrival_ms=i * 20.0, prompt=200, output=100000, prio=0, cls="bg",
                   src="decoder") for i in range(32)]
        wl += [dict(id=100 + i, arrival_ms=1000.0 + i, prompt=LONG, output=1, prio=0,
                    cls="long", src="long", replenish=True) for i in range(8)]
        wl += [dict(id=900, arrival_ms=20000.0, prompt=20, output=1, prio=pp, cls="short",
                    src="probe")]
        recs, st = run_variant(wl, v, max_steps=20000)
        pr = recs[900]
        lt = long_ttft(recs)
        res["unbounded"][vn] = {"steps": st["steps"], "probe_served": pr["t_first"] is not None,
                                "probe_zero_steps": pr["starved"],
                                "probe_max_zero_run": pr["max_starve_run"],
                                "long_done": len(lt),
                                "long_ttft_p50_ms": lt[len(lt) // 2] if lt else None,
                                "long_ttft_max_ms": lt[-1] if lt else None}
        print("unbounded", vn, res["unbounded"][vn], flush=True)
    json.dump(res, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
