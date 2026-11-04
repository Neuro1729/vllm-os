"""Join client results (TTFT) with the server's VLLM_SCHED_TRACE (per-step allocations).

Per request: TTFT (client), and from the server trace: arrival step, number of scheduler steps
between arrival and its last prefill chunk in which it got ZERO tokens ("starved steps"), and the
longest consecutive zero run.

usage: analyze.py RESULTS.jsonl TRACE.jsonl OUT.json
"""
import json
import sys
from collections import defaultdict


def load_results(path):
    meta, recs = None, []
    for l in open(path):
        d = json.loads(l)
        if d.get("meta"):
            meta = d
        else:
            recs.append(d)
    return meta, recs


def load_trace(path):
    adds, steps = {}, []
    for l in open(path):
        d = json.loads(l)
        if d.get("ev") == "ctrl":
            continue
        if d.get("ev") == "add":
            adds[d["rid"]] = d
        else:
            steps.append(d)
    return adds, steps


def base_id(rid):
    # vLLM may append a suffix to the client request_id; keep the "r#########" part
    if rid.startswith("r") and len(rid) >= 10 and rid[1:10].isdigit():
        return int(rid[1:10])
    i = rid.find("r0")
    return int(rid[i + 1:i + 10]) if i >= 0 and rid[i + 1:i + 10].isdigit() else None


def per_request_starvation(adds, steps):
    """For each request: steps from its add to its final prefill chunk with 0 tokens."""
    add_t = {base_id(r): d["t"] for r, d in adds.items() if base_id(r) is not None}
    chunks = defaultdict(list)  # id -> [(step_idx, n, before, total)]
    for k, s in enumerate(steps):
        for rid, (n, before, total) in s["pf"].items():
            b = base_id(rid)
            if b is not None:
                chunks[b].append((k, n, before, total))
    step_t = [s["t"] for s in steps]
    import bisect
    out = {}
    for b, t_add in add_t.items():
        ch = chunks.get(b)
        if not ch:
            continue
        first_k = bisect.bisect_left(step_t, t_add)
        last_k = max(k for k, n, before, total in ch if before + n >= total) \
            if any(before + n >= total for k, n, before, total in ch) else ch[-1][0]
        got = {k for k, *_ in ch}
        reasons = defaultdict(int)  # why it got 0 tokens while eligible (small requests only)
        for k in range(first_k, last_k + 1):
            if k in got:
                continue
            sw = steps[k].get("sw") or {}
            hit = [v for r, v in sw.items() if base_id(r) == b]
            if not hit:
                continue
            state = hit[0][1]
            s = steps[k]
            if s.get("left", 1) <= 0:
                reasons["budget"] += 1
            elif state == "W" and s.get("slots_full"):
                reasons["slots"] += 1
            else:
                reasons["other_kv"] += 1
        starved, run, best = 0, 0, 0
        for k in range(first_k, last_k + 1):
            if k in got:
                run = 0
            else:
                starved += 1
                run += 1
                best = max(best, run)
        out[b] = {"starved_steps": starved, "max_zero_run": best,
                  "prefill_steps": len(got), "wait_steps": last_k - first_k + 1,
                  "reasons": dict(reasons)}
    return out


def main():
    res_path, trace_path, out_path = sys.argv[1:4]
    meta, recs = load_results(res_path)
    adds, steps = load_trace(trace_path)
    # the server trace spans several runs and request ids repeat: keep only this run's window
    t_lo = meta["t0"] - 3.0
    t_hi = max(r["t_done"] or r["t_send"] for r in recs) + 2.0
    adds = {k: v for k, v in adds.items() if t_lo <= v["t"] <= t_hi}
    steps = [s for s in steps if t_lo <= s["t"] <= t_hi]
    st = per_request_starvation(adds, steps)
    rows = []
    for r in recs:
        ttft = (r["t_first"] - r["t_send"]) if r["t_first"] else None
        rows.append({**r, "ttft_s": ttft, **st.get(r["id"], {})})
    # step-level timeline
    tl = [{"t": s["t"] - meta["t0"], "tok": s["tok"], "n_dec": s["n_dec"], "run": s["run"],
           "wait": s["wait"], "starved": s["starved"], "n_pf": len(s["pf"]), "cap": s["cap"],
           "preempted": s.get("preempted", 0), "kv_usage": s.get("kv_usage", 0),
           "pf_tok": sum(v[0] for v in s["pf"].values())}
          for s in steps]
    dts = [steps[i + 1]["t"] - steps[i]["t"] for i in range(len(steps) - 1)]
    json.dump({"meta": meta, "requests": rows, "timeline": tl,
               "step_dt_s": dts}, open(out_path, "w"))
    print(f"requests={len(rows)} steps={len(steps)} matched_trace={len(st)}")


if __name__ == "__main__":
    main()
