"""Micro scenario: N background decodes, then long prefills A and B, then a tiny request C.

Measures how many scheduler steps C (and B) get 0 tokens, and C's TTFT slowdown vs. running alone.
Usage: python3 micro.py [--out results/micro.json]
"""
import argparse
import json

from sim import CostModel, Req, SchedConfig, Sim

POLICIES = {
    "nocap":          dict(long_prefill_token_threshold=0),
    "static512":      dict(long_prefill_token_threshold=512),
    "adaptive512":    dict(long_prefill_token_threshold=512, adaptive=True),  # vLLM 0.32 flag
    "adaptive512_pf": dict(long_prefill_token_threshold=512, adaptive=True,
                           adaptive_mode="prefill_only"),                   # (B-decodes)/prefills
    "srpt_fix":       dict(long_prefill_token_threshold=512, adaptive=True, order="srpt"),
}


def run_case(budget, n_dec, a_len, b_len, c_len, pol, policy="fcfs", c_prio=0, other_prio=0,
             cost=None):
    cost = cost or CostModel()
    cfg = SchedConfig(max_num_batched_tokens=budget, max_num_seqs=1024, kv_tokens=4_000_000,
                      policy=policy, **POLICIES[pol])
    t_ab = 60_000.0  # long prefills arrive after all decoders are in steady decode
    reqs = [Req(i, 0.0, 200, 10**9, prio=other_prio, tag=f"d{i}") for i in range(n_dec)]
    A = Req(10**6, t_ab, a_len, 50, prio=other_prio, tag="A")
    B = Req(10**6 + 1, t_ab, b_len, 50, prio=other_prio, tag="B")
    C = Req(10**6 + 2, t_ab + 200.0, c_len, 20, prio=c_prio, tag="C")
    reqs += [A, B, C]
    sim = Sim(cfg, cost, trace_rids={A.rid, B.rid, C.rid})
    sim.run(reqs, stop=lambda s, alive, done: all(x.t_first is not None for x in (A, B, C)))
    alone = cost.alone_ttft_ms(c_len, budget)
    res = {}
    for x in (A, B, C):
        res[x.tag] = {
            "ttft_ms": round(x.t_first - x.arrival, 1),
            "alone_ttft_ms": round(cost.alone_ttft_ms(x.prompt, budget), 1),
            "slowdown": round((x.t_first - x.arrival) / cost.alone_ttft_ms(x.prompt, budget), 1),
            "starved_steps": x.starved,
            "first_chunks": x.chunks[:3],
        }
    # decode stalls among background decoders during the A/B/C window
    res["decoder_skips_max"] = max((r.decode_skips for r in reqs if r.tag.startswith("d")),
                                   default=0)
    res["trace_head"] = [t for t in sim.trace if t["alloc"]][:4]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/micro.json")
    args = ap.parse_args()
    out = []
    scenarios = [
        # (label, budget, A, B, C)
        ("user_example", 1024, 10_000, 20_000, 20),
        ("extreme_100k", 1024, 100_000, 100_000, 20),
        ("h100_default_budget", 8192, 100_000, 100_000, 20),
    ]
    for label, budget, a, b, c in scenarios:
        for n_dec in (0, 256, 512, 768):
            if n_dec >= budget:
                continue
            for pol in POLICIES:
                for policy, cp, op in (("fcfs", 0, 0), ("priority", 0, 1)):
                    r = run_case(budget, n_dec, a, b, c, pol, policy, cp, op)
                    row = dict(scenario=label, budget=budget, n_decodes=n_dec, cap_policy=pol,
                               queue=policy, **{f"C_{k}": v for k, v in r["C"].items()
                                               if k != "first_chunks"},
                               B_starved_steps=r["B"]["starved_steps"],
                               B_slowdown=r["B"]["slowdown"],
                               A_slowdown=r["A"]["slowdown"],
                               decoder_skips_max=r["decoder_skips_max"],
                               trace_head=r["trace_head"])
                    out.append(row)
                    print(f"{label:20s} B={budget:5d} dec={n_dec:3d} {pol:15s} {policy:8s} "
                          f"C: ttft={row['C_ttft_ms']:9.1f}ms x{row['C_slowdown']:7.1f} "
                          f"starved={row['C_starved_steps']:4d} | B starved={row['B_starved_steps']:4d}"
                          f" | dec_skips={row['decoder_skips_max']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
