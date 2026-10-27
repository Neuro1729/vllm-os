"""Aggregate raw trial outputs into CSV summaries. Reads files only; no GPU.

    python -m benchmarks.aggregate RESULTS_DIR

Inputs: every raw/**/trial.json and raw/**/requests.csv (skipping raw/warmup/).
A directory with failed.json but no trial.json counts as a failed trial.

Only trials with status "ok" feed the latency and throughput statistics.
Timeouts and failures are still counted (completed, unfinished, failed_trials).

Outputs:
    raw/trial_results.json              one entry per trial, sorted
    raw/requests.csv                    union of all request rows plus scenario, max_num_seqs, variant, repeat
    summaries/metrics_by_scenario.csv   scenario x max_num_seqs x variant, aggregated over repeats
    summaries/metrics_by_variant.csv    variant x max_num_seqs, pooled over scenarios (CI over trials)
    summaries/improvements_vs_fcfs.csv  each other variant against fcfs, percent improvement
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from benchmarks import metrics
from benchmarks.run_trial import REQUEST_COLUMNS
from benchmarks.workloads import SCENARIOS

VARIANT_ORDER = ("v0", "fcfs", "mlfq", "mlfq_preempt")
BASELINE_VARIANT = "fcfs"
IMPROVEMENT_VARIANTS = ("v0", "mlfq", "mlfq_preempt")
IMPROVEMENT_METRICS = (("ttft_p95_s", False), ("itl_p99_s", False),
                       ("e2e_p95_s", False), ("output_tok_s", True))
TRIAL_METRICS = ("ttft_p50_s", "ttft_p95_s", "ttft_p99_s", "itl_p99_s", "e2e_p95_s",
                 "output_tok_s", "completed", "unfinished", "failed_trials",
                 "preemptions", "peak_vram_bytes")
KEY_COLUMNS = ("scenario", "max_num_seqs", "variant", "repeat")


def _order(value, sequence):
    return sequence.index(value) if value in sequence else len(sequence)


def discover_trial_dirs(results_dir):
    raw = Path(results_dir) / "raw"
    found = set()
    for name in ("trial.json", "failed.json"):
        for path in raw.rglob(name):
            rel = path.relative_to(raw).parts
            if rel[0] == "warmup":
                continue
            found.add(path.parent)
    return sorted(found)


def _read_requests(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _trial_metric_values(trial, records):
    """Metric values for one trial. Non-ok trials give None for every latency metric."""
    values = {m: None for m in TRIAL_METRICS}
    ok = trial["status"] == "ok"
    values["failed_trials"] = 0 if ok else 1
    counts = [r["status"] for r in records]
    values["completed"] = counts.count("finished") if records else None
    values["unfinished"] = counts.count("unfinished") if records else None
    device = trial.get("device") or {}
    values["peak_vram_bytes"] = device.get("peak_allocated_bytes") if ok else None
    if not ok or not records:
        return values

    parsed = [metrics.parse_csv_record(row) for row in records]
    per_request = [metrics.request_metrics(r) for r in parsed]
    ttft = [m["ttft"] for m in per_request]
    finished_e2e = [m["e2e"] for m, r in zip(per_request, parsed) if r["status"] == "finished"]
    itl = [gap for m in per_request for gap in m["itl"]]
    values.update(
        ttft_p50_s=metrics.percentile(ttft, 50),
        ttft_p95_s=metrics.percentile(ttft, 95),
        ttft_p99_s=metrics.percentile(ttft, 99),
        itl_p99_s=metrics.percentile(itl, 99),
        e2e_p95_s=metrics.percentile(finished_e2e, 95),
    )
    total_tokens = sum(r["n_tokens"] or 0 for r in parsed)
    wall = trial.get("wall_time_s")
    values["output_tok_s"] = total_tokens / wall if wall else None
    preemptions = [r["preemptions"] for r in parsed if r["preemptions"] is not None]
    values["preemptions"] = sum(preemptions) if preemptions else None
    return values


def _load_trial(trial_dir, results_dir):
    trial_path = trial_dir / "trial.json"
    if trial_path.exists():
        with open(trial_path, encoding="utf-8") as f:
            trial = json.load(f)
        head = {
            "scenario": trial["scenario"],
            "max_num_seqs": trial["max_num_seqs"],
            "variant": trial["variant"],
            "repeat": trial["repeat_index"],
            "status": trial["status"],
        }
        rows = _read_requests(trial_dir / "requests.csv")
    else:
        with open(trial_dir / "failed.json", encoding="utf-8") as f:
            failed = json.load(f)
        head = {
            "scenario": failed["scenario"],
            "max_num_seqs": failed["max_num_seqs"],
            "variant": failed["variant"],
            "repeat": failed["repeat"],
            "status": failed["status"],
        }
        trial = {"status": failed["status"], "device": None, "wall_time_s": None}
        rows = []
    entry = dict(head)
    entry["path"] = trial_dir.relative_to(results_dir).as_posix()
    entry["trace_id"] = trial.get("trace_id")
    entry["saturation"] = trial.get("saturation")
    entry["kv_blocks_total"] = trial.get("kv_blocks_total")
    entry["events_truncated"] = trial.get("events_truncated")
    entry["error"] = trial.get("error") or None
    entry.update(_trial_metric_values(trial, rows))
    return entry, rows


def _key(entry):
    return (_order(entry["scenario"], SCENARIOS), entry["max_num_seqs"],
            _order(entry["variant"], VARIANT_ORDER), entry["repeat"])


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def _stat_columns(prefix_fields):
    columns = list(prefix_fields)
    for metric in TRIAL_METRICS:
        for stat in ("n", "mean", "std", "ci95"):
            columns.append(f"{metric}_{stat}")
    return columns


def _stat_values(values_by_metric):
    row = {}
    for metric in TRIAL_METRICS:
        stats = metrics.aggregate_repeats(values_by_metric[metric])
        for stat in ("n", "mean", "std", "ci95"):
            row[f"{metric}_{stat}"] = stats[stat]
    return row


def _group(entries, key_fn):
    groups = {}
    for entry in entries:
        groups.setdefault(key_fn(entry), []).append(entry)
    return groups


def _metrics_by_scenario(entries):
    rows = []
    for (scenario, msn, variant), members in sorted(
            _group(entries, lambda e: (e["scenario"], e["max_num_seqs"], e["variant"])).items(),
            key=lambda item: (_order(item[0][0], SCENARIOS), item[0][1], _order(item[0][2], VARIANT_ORDER))):
        row = {"scenario": scenario, "max_num_seqs": msn, "variant": variant, "trials": len(members)}
        row.update(_stat_values({m: [e[m] for e in members] for m in TRIAL_METRICS}))
        rows.append(row)
    return rows


def _metrics_by_variant(entries):
    rows = []
    groups = _group(entries, lambda e: (e["variant"], e["max_num_seqs"]))
    for (variant, msn) in sorted(groups, key=lambda k: (_order(k[0], VARIANT_ORDER), k[1])):
        members = groups[(variant, msn)]
        row = {
            "variant": variant,
            "max_num_seqs": msn,
            "scenarios": len({e["scenario"] for e in members}),
            "trials": len(members),
        }
        row.update(_stat_values({m: [e[m] for e in members] for m in TRIAL_METRICS}))
        rows.append(row)
    return rows


def _improvements(entries):
    groups = _group(entries, lambda e: (e["scenario"], e["max_num_seqs"], e["variant"]))
    rows = []
    for (scenario, msn) in sorted({(e["scenario"], e["max_num_seqs"]) for e in entries},
                                  key=lambda k: (_order(k[0], SCENARIOS), k[1])):
        base_members = groups.get((scenario, msn, BASELINE_VARIANT), [])
        for variant in IMPROVEMENT_VARIANTS:
            cand_members = groups.get((scenario, msn, variant), [])
            for metric, higher in IMPROVEMENT_METRICS:
                base = metrics.aggregate_repeats([e[metric] for e in base_members])
                cand = metrics.aggregate_repeats([e[metric] for e in cand_members])
                rows.append({
                    "scenario": scenario,
                    "max_num_seqs": msn,
                    "variant": variant,
                    "baseline_variant": BASELINE_VARIANT,
                    "metric": metric,
                    "higher_is_better": higher,
                    "baseline_mean": base["mean"],
                    "baseline_n": base["n"],
                    "candidate_mean": cand["mean"],
                    "candidate_n": cand["n"],
                    "improvement_pct": metrics.improvement_pct(base["mean"], cand["mean"], higher),
                })
    return rows


def _requests_union(loaded):
    """loaded: list of (trial entry, request rows). Returns (columns, rows with key columns added)."""
    columns = list(KEY_COLUMNS) + list(REQUEST_COLUMNS)
    for _, rows in loaded:
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    out = []
    for entry, rows in loaded:
        for row in rows:
            merged = dict(row)
            merged.update(scenario=entry["scenario"], max_num_seqs=entry["max_num_seqs"],
                          variant=entry["variant"], repeat=entry["repeat"])
            out.append(merged)
    return columns, out


def aggregate(results_dir):
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    loaded = [_load_trial(trial_dir, results_dir) for trial_dir in discover_trial_dirs(results_dir)]
    loaded.sort(key=lambda item: _key(item[0]))
    entries = [entry for entry, _ in loaded]

    raw.mkdir(parents=True, exist_ok=True)
    with open(raw / "trial_results.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, sort_keys=True)
        f.write("\n")

    columns, request_rows = _requests_union(loaded)
    _write_csv(raw / "requests.csv", columns, request_rows)

    summaries = results_dir / "summaries"
    written = {
        "trial_results": raw / "trial_results.json",
        "requests": raw / "requests.csv",
    }
    scenario_rows = _metrics_by_scenario(entries)
    _write_csv(summaries / "metrics_by_scenario.csv",
               _stat_columns(("scenario", "max_num_seqs", "variant", "trials")), scenario_rows)
    variant_rows = _metrics_by_variant(entries)
    _write_csv(summaries / "metrics_by_variant.csv",
               _stat_columns(("variant", "max_num_seqs", "scenarios", "trials")), variant_rows)
    improvement_rows = _improvements(entries)
    _write_csv(summaries / "improvements_vs_fcfs.csv",
               ["scenario", "max_num_seqs", "variant", "baseline_variant", "metric", "higher_is_better",
                "baseline_mean", "baseline_n", "candidate_mean", "candidate_n", "improvement_pct"],
               improvement_rows)
    written.update(
        metrics_by_scenario=summaries / "metrics_by_scenario.csv",
        metrics_by_variant=summaries / "metrics_by_variant.csv",
        improvements=summaries / "improvements_vs_fcfs.csv",
    )
    return written


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m benchmarks.aggregate",
                                     description="Aggregate raw trial outputs into CSV summaries.")
    parser.add_argument("results_dir", help="Run directory containing raw/ (the run_matrix output dir).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not Path(args.results_dir).is_dir():
        print(f"not a directory: {args.results_dir}", file=sys.stderr)
        return 2
    written = aggregate(args.results_dir)
    for name in sorted(written):
        print(f"{name}: {written[name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
