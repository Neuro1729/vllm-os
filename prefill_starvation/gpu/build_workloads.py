"""Build GPU scenario workloads from the real traces (all requests <= 32K total tokens).

Scenarios (each = real arrivals + real lengths from public traces, time-compressed):
  S1 decode-heavy  : Azure chat (outputs x6: long, reasoning-style generations -> hundreds of
                     concurrent decodes) + Mooncake long-context requests (16K-31K prompts)
  S1base           : same chat stream, no long-context requests
  S2 many-prefills : Mooncake conversation trace (real mix of short/medium/long), normal outputs
  S2base           : same trace with the long (>=16K) requests removed
  S3 priority      : S2 workload, short (<=2K) requests high priority (0), others low (1)
  S3low            : S2 workload, short requests LOW priority (1), others high (0)
usage: build_workloads.py OUTDIR [--dur 300]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import workloads as W  # noqa: E402

MAXTOT = 32000  # stay inside Qwen2.5's native 32K context (prompt + output)


def clip_ok(rows):
    return [(t, i, o) for t, i, o in rows if i + o <= MAXTOT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--dur", type=float, default=300)
    ap.add_argument("--chat-speedup", type=float, default=1.0)
    ap.add_argument("--chat-out-mult", type=float, default=6.0)
    ap.add_argument("--long-every", type=int, default=6)
    ap.add_argument("--mc-speedup", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    az, mc = W.load_azure(), W.load_mooncake()

    chat = W.window(az, 600, a.dur, a.chat_speedup)
    chat = [(t, i, min(4000, int(o * a.chat_out_mult))) for t, i, o in chat]
    longs = [r for r in clip_ok(W.window(mc, 0, a.dur, 1.0)) if r[1] >= 16384][::a.long_every]
    s1 = W.to_workload((chat, "azure_chat"), (longs, "mooncake_long"))
    s1b = W.to_workload((chat, "azure_chat"))

    mcw = clip_ok(W.window(mc, 0, a.dur, a.mc_speedup))
    s2 = W.to_workload((mcw, "mooncake"))
    s2b = W.to_workload(([r for r in mcw if r[1] < 16384], "mooncake"))
    s3 = W.to_workload((mcw, "mooncake"),
                       prio_fn=lambda w, rnd: 0 if w["cls"] == "short" else 1)
    s3l = W.to_workload((mcw, "mooncake"),
                        prio_fn=lambda w, rnd: 1 if w["cls"] == "short" else 0)
    for name, wl in [("S1", s1), ("S1base", s1b), ("S2", s2), ("S2base", s2b), ("S3", s3),
                     ("S3low", s3l)]:
        json.dump(wl, open(os.path.join(a.outdir, f"{name}.json"), "w"))
        print(name, json.dumps(W.describe(wl)))


if __name__ == "__main__":
    main()
