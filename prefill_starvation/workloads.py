"""Build request workloads from real public traces.

* Mooncake (Kimi, FAST'25) conversation trace: long-context chat, real arrivals & lengths
  data/mooncake/conversation_trace.jsonl  {timestamp(ms), input_length, output_length}
* Azure LLM inference trace 2023 (conversation): production chat, real arrivals & lengths
  data/azure/AzureLLMInferenceTrace_conv.csv  {TIMESTAMP, ContextTokens, GeneratedTokens}

Time scaling ("speedup" k) compresses inter-arrival gaps by k, i.e. replays the trace as if the
deployment were k times busier; request lengths are never changed.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT_MAX = 2048  # prompts <= this are "short" (interactive) requests


def load_mooncake(name="conversation_trace.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data/mooncake", name))]
    t0 = rows[0]["timestamp"]
    return [(r["timestamp"] - t0, r["input_length"], r["output_length"]) for r in rows]


def load_azure(name="AzureLLMInferenceTrace_conv.csv"):
    rows = list(csv.DictReader(open(os.path.join(HERE, "data/azure", name))))
    ts = [dt.datetime.fromisoformat(r["TIMESTAMP"][:26]) for r in rows]
    t0 = ts[0]
    return [((t - t0).total_seconds() * 1000.0, int(r["ContextTokens"]),
             int(r["GeneratedTokens"])) for t, r in zip(ts, rows)]


def window(rows, start_s, dur_s, speedup):
    """Take the trace slice whose *replayed* time falls in [0, dur_s) after compressing by speedup
    and starting at original time start_s."""
    out = []
    for ts, i, o in rows:
        rt = (ts - start_s * 1000.0) / speedup
        if 0 <= rt < dur_s * 1000.0:
            out.append((rt, i, o))
    return out


def cls_of(prompt):
    if prompt <= SHORT_MAX:
        return "short"
    if prompt >= 16384:
        return "long"
    return "medium"


def to_workload(*streams, prio_fn=None, seed=0):
    """streams: (rows, source_name). Returns list of request dicts with ids."""
    rnd = random.Random(seed)
    wl = []
    for rows, src in streams:
        for ts, i, o in rows:
            wl.append({"arrival_ms": ts, "prompt": max(1, i), "output": max(1, o), "src": src,
                       "cls": cls_of(i)})
    wl.sort(key=lambda w: w["arrival_ms"])
    for k, w in enumerate(wl):
        w["id"] = k
        w["prio"] = prio_fn(w, rnd) if prio_fn else 0
    return wl


def describe(wl):
    from collections import Counter
    c = Counter((w["src"], w["cls"]) for w in wl)
    dur = (wl[-1]["arrival_ms"] - wl[0]["arrival_ms"]) / 1000.0 if wl else 0
    ptok = sum(w["prompt"] for w in wl)
    return {"n": len(wl), "duration_s": round(dur, 1), "rps": round(len(wl) / max(dur, 1e-9), 2),
            "prefill_tok_per_s": round(ptok / max(dur, 1e-9)),
            "by_src_cls": {f"{a}/{b}": n for (a, b), n in sorted(c.items())}}
