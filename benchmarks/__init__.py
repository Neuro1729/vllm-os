"""Benchmark package for the nano-vLLM scheduler comparison.

Modules:
    workloads     deterministic traces (no GPU, no tokenizer)
    metrics       pure latency / throughput statistics
    replay        open-loop trace replay against an engine (no GPU imports)
    run_trial     one trial per process; writes trial.json, requests.csv, events
    run_matrix    orchestrates many run_trial subprocesses
    aggregate     raw trial outputs -> CSV summaries
    generate_report  recorded results -> REPORT.md / REPORT.html / plots
"""
