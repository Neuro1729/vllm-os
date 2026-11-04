"""Find load levels where the system is NOT overloaded (queue drains, short requests fine without
long prompts), so any starvation we show is a scheduling artefact, not lack of capacity."""
import argparse
import json
import sys

import vllm_harness as H
import workloads as W


def q(a, p):
    return a[min(len(a) - 1, int(p * len(a)))] if a else float("nan")


def summarize(recs, stats, dur_s):
    out = {}
    for cls in ("short", "medium", "long"):
        tt = sorted((r["t_first"] - r["arrival_ms"]) / 1000 for r in recs.values()
                    if r["cls"] == cls and r["t_first"] is not None)
        st = sorted(r["starved"] for r in recs.values() if r["cls"] == cls)
        out[cls] = dict(n=len(tt), p50=round(q(tt, .5), 2), p99=round(q(tt, .99), 2),
                        max=round(tt[-1], 2) if tt else None, starved_max=st[-1] if st else None)
    out["drain_s"] = round(stats["sim_time_s"] - dur_s, 1)
    out["util"] = round(stats["busy_s"] / stats["sim_time_s"], 3)
    out["unfinished"] = stats["unfinished"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--chat-speedups", default="1,2,3,4")
    ap.add_argument("--long-every", default="0,8,4")  # 0 = no long requests; k = every k-th
    ap.add_argument("--dur", type=float, default=300)
    ap.add_argument("--mooncake-only", action="store_true")
    a = ap.parse_args()
    az, mc = W.load_azure(), W.load_mooncake()
    for sp in map(float, a.chat_speedups.split(",")):
        for le in map(int, a.long_every.split(",")):
            if a.mooncake_only:
                rows = W.window(mc, 0, a.dur, sp)
                wl = W.to_workload((rows, "mooncake"))
            else:
                chat = W.window(az, 600, a.dur, sp)
                longs = [] if le == 0 else \
                    [r for r in W.window(mc, 0, a.dur, 1.0) if r[1] >= 16384][::le]
                wl = W.to_workload((chat, "azure_chat"), (longs, "mooncake_long"))
            recs, tl, st = H.run(wl, budget=a.budget, threshold=512, adaptive=True,
                                 t_end_ms=(a.dur + 600) * 1000, verbose=False)
            dec = sum(x["decodes"] for x in tl) / max(1, len(tl))
            s = summarize(recs, st, a.dur)
            print(json.dumps({"speedup": sp, "long_every": le, **W.describe(wl),
                              "avg_decodes": round(dec), **s}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
