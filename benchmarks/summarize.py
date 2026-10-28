"""Print a markdown table from priority_bench JSONL results (last run per label wins)."""
import json
import sys


def main(path="results/priority_bench.jsonl"):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["label"]] = r
    print("| config | chat TTFT p50 | chat TTFT p99 | chat e2e p99 | agent e2e p50 | agent turns | tok/s "
          "| preempt | swap out/in | recomputed tok |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        c, a, s = r["chat"], r["agent"], r["stats"]
        print(f"| {label} | {c['ttft_p50']*1e3:.0f} ms | {c['ttft_p99']*1e3:.0f} ms | {c['e2e_p99']:.2f} s "
              f"| {a['e2e_p50']:.2f} s | {r['agent_turns']} | {r['output_tokens_per_s']:.0f} | {s['preemptions']} "
              f"| {s['swap_outs']}/{s['swap_ins']} | {s['recomputed_tokens']} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
