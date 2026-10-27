"""Turn recorded benchmark results into REPORT.md, REPORT.html, SUMMARY.json and plots/*.png.

    python -m benchmarks.generate_report --input RESULTS_DIR --output REPORT_DIR

Reads only. Needs the output of benchmarks.aggregate (summaries/ and raw/trial_results.json).
Refuses to write inside the input directory. No GPU is used. Plots use matplotlib's Agg backend.
Every number in the text is computed from the recorded files. Missing values are
printed as "n/a (not measured)" and never plotted as zero.
"""
import argparse
import base64
import csv
import html
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmarks import metrics  # noqa: E402
from benchmarks.aggregate import VARIANT_ORDER  # noqa: E402
from benchmarks.run_trial import EVENTS_CSV  # noqa: E402
from benchmarks.workloads import SCENARIOS  # noqa: E402

FIXTURE_BANNER = ("TEST FIXTURE DATA. The numbers in this report are synthetic and were "
                  "generated for testing the pipeline. They are NOT a benchmark result.")
NOT_MEASURED = "n/a (not measured)"
SHORT_GROUPS = ("short", "burst")
LONG_GROUPS = ("long", "background")
VARIANT_LABELS = {
    "v0": "Original V0",
    "fcfs": "Modernized FCFS",
    "mlfq": "MLFQ",
    "mlfq_preempt": "MLFQ + Preemption",
}
COMPARISON_ROWS = (
    ("P95 TTFT (s)", "ttft_p95_s", 1.0, "{:.3f}"),
    ("P99 ITL (s)", "itl_p99_s", 1.0, "{:.4f}"),
    ("Output tok/s", "output_tok_s", 1.0, "{:.1f}"),
    ("P95 latency, E2E (s)", "e2e_p95_s", 1.0, "{:.3f}"),
    ("Preemptions", "preemptions", 1.0, "{:.1f}"),
    ("Peak VRAM (GiB)", "peak_vram_bytes", 1.0 / 2 ** 30, "{:.2f}"),
)
SATURATION_NAMES = {
    "slot_limit_hits": "sequence slots (max_num_seqs)",
    "kv_limit_hits": "KV cache blocks",
    "token_budget_hits": "per-step token budget",
}
COMPARE_METRICS = ("ttft_p95_s", "itl_p99_s", "e2e_p95_s", "output_tok_s")
LIMITATIONS = (
    "Single machine and single model. Results describe this hardware and this model only.",
    "Arrivals are open-loop Poisson at one load factor per profile. Other loads can change the ranking.",
    "Latency percentiles are over repeats; each repeat uses a different seeded trace, so CIs with n = 3 are wide.",
    "The original V0 engine has no scheduler counters, per-token events, or preemption counts. Those cells are not measured.",
    "Queue wait by MLFQ level is not charted: requests.csv does not record the per-request MLFQ level.",
    "KV usage is derived from scheduler_events.csv. Trials whose event log was truncated give a lower bound.",
    "Output tok/s divides all generated tokens by the replay wall time, which includes idle gaps between arrivals.",
)


# ---------------------------------------------------------------- loading

def _float(value):
    if value in (None, ""):
        return None
    return float(value)


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path, default):
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _numeric_summary_row(row):
    out = {}
    for key, value in row.items():
        try:
            out[key] = _float(value)
        except ValueError:
            out[key] = value  # text columns such as scenario, variant, metric, higher_is_better
    out["max_num_seqs"] = int(out["max_num_seqs"])
    return out


@dataclass
class ReportData:
    config: dict
    trials: list
    index: dict = field(default_factory=dict)          # (scenario, msn, variant) -> summary row
    improvements: list = field(default_factory=list)
    group_stats: dict = field(default_factory=dict)    # (scenario, msn, variant) -> {"short_ttft_p95": stat, ...}
    kv_peak: dict = field(default_factory=dict)        # (scenario, msn, variant) -> stat

    @property
    def fixture(self):
        return bool(self.config.get("synthetic_fixture"))

    @property
    def points(self):
        keys = {(s, m) for (s, m, _) in self.index}
        return sorted(keys, key=lambda k: (_scenario_order(k[0]), k[1]))

    @property
    def headline_msn(self):
        msns = [m for (_, m) in self.points]
        return max(msns) if msns else None

    def stat(self, s, m, variant, metric):
        row = self.index.get((s, m, variant))
        if row is None:
            return {"mean": None, "ci95": None, "n": 0}
        return {"mean": row.get(f"{metric}_mean"), "ci95": row.get(f"{metric}_ci95"),
                "n": int(row.get(f"{metric}_n") or 0)}


def _scenario_order(name):
    return SCENARIOS.index(name) if name in SCENARIOS else len(SCENARIOS)


def _ok_keys(trials):
    return {(t["scenario"], t["max_num_seqs"], t["variant"], t["repeat"]) for t in trials if t["status"] == "ok"}


def _group_stats(request_rows, ok_keys):
    per_repeat = {}
    for row in request_rows:
        key = (row["scenario"], int(row["max_num_seqs"]), row["variant"], int(row["repeat"]))
        if key not in ok_keys:
            continue
        parsed = metrics.parse_csv_record(row)
        derived = metrics.request_metrics(parsed)
        bucket = per_repeat.setdefault(key, {"short_ttft": [], "long_e2e": []})
        if row["group"] in SHORT_GROUPS:
            bucket["short_ttft"].append(derived["ttft"])
        if row["group"] in LONG_GROUPS and parsed["status"] == "finished":
            bucket["long_e2e"].append(derived["e2e"])
    stats = {}
    groups = {}
    for (s, m, v, _), bucket in per_repeat.items():
        groups.setdefault((s, m, v), []).append(bucket)
    for key, buckets in groups.items():
        stats[key] = {
            "short_ttft_p95": metrics.aggregate_repeats(
                [metrics.percentile(b["short_ttft"], 95) for b in buckets]),
            "long_e2e_p95": metrics.aggregate_repeats(
                [metrics.percentile(b["long_e2e"], 95) for b in buckets]),
        }
    return stats


def _kv_peak_from_events(path, kv_total):
    if not path.exists() or not kv_total:
        return None
    min_free = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            value = row.get("free_blocks")
            if value not in (None, ""):
                free = int(value)
                min_free = free if min_free is None else min(min_free, free)
    if min_free is None:
        return None
    return (kv_total - min_free) / kv_total


def _kv_peaks(trials, ok_keys, results_dir):
    per_point = {}
    for trial in trials:
        key = (trial["scenario"], trial["max_num_seqs"], trial["variant"], trial["repeat"])
        if key not in ok_keys or trial.get("kv_blocks_total") is None:
            continue
        peak = _kv_peak_from_events(results_dir / trial["path"] / EVENTS_CSV, trial["kv_blocks_total"])
        per_point.setdefault(key[:3], []).append(peak)
    return {k: metrics.aggregate_repeats(v) for k, v in per_point.items()}


def load(input_dir):
    root = Path(input_dir)
    config = _read_json(root / "configuration.json", {})
    trials = _read_json(root / "raw" / "trial_results.json", [])
    data = ReportData(config=config, trials=trials)
    for row in _read_csv(root / "summaries" / "metrics_by_scenario.csv"):
        numeric = _numeric_summary_row(row)
        data.index[(numeric["scenario"], numeric["max_num_seqs"], numeric["variant"])] = numeric
    data.improvements = [_numeric_summary_row(r) for r in _read_csv(root / "summaries" / "improvements_vs_fcfs.csv")]
    ok = _ok_keys(trials)
    data.group_stats = _group_stats(_read_csv(root / "raw" / "requests.csv"), ok)
    data.kv_peak = _kv_peaks(trials, ok, root)
    return data


# ---------------------------------------------------------------- analysis helpers

def _fmt(value, pattern):
    return NOT_MEASURED if value is None else pattern.format(value)


def _cell(stat, scale, pattern):
    if stat["mean"] is None:
        return NOT_MEASURED
    text = pattern.format(stat["mean"] * scale)
    if stat["n"] >= 2 and stat["ci95"] is not None:
        text += " ± " + pattern.format(stat["ci95"] * scale)
    return text


def _variant_average(data, variant, metric):
    values = [data.index[(s, m, v)][f"{metric}_mean"] for (s, m, v) in data.index
              if v == variant and data.index[(s, m, v)].get(f"{metric}_mean") is not None]
    if not values:
        return None, 0
    return sum(values) / len(values), len(values)


def _best_variant(data, metric, pick):
    candidates = []
    for variant in VARIANT_ORDER:
        avg, count = _variant_average(data, variant, metric)
        if avg is not None:
            candidates.append((avg, variant, count))
    if not candidates:
        return None
    return pick(candidates, key=lambda c: c[0])


def _direction_counts(data, metric, higher_is_better, candidate, baseline="fcfs"):
    """Returns (better, worse, compared) over the points where both variants have a measured mean."""
    better = worse = compared = 0
    for (s, m) in data.points:
        base = data.stat(s, m, baseline, metric)["mean"]
        cand = data.stat(s, m, candidate, metric)["mean"]
        pct = metrics.improvement_pct(base, cand, higher_is_better)
        if pct is None:
            continue
        compared += 1
        better += pct > 0
        worse += pct < 0
    return better, worse, compared


def _short_ttft_improvement(data):
    improved = compared = 0
    for (s, m) in data.points:
        base = data.group_stats.get((s, m, "fcfs"), {}).get("short_ttft_p95", {}).get("mean")
        cand = data.group_stats.get((s, m, "mlfq"), {}).get("short_ttft_p95", {}).get("mean")
        pct = metrics.improvement_pct(base, cand, False)
        if pct is None:
            continue
        compared += 1
        improved += pct > 0
    return improved, compared


def _preemption_totals(data):
    """Returns (total preemptions measured for mlfq_preempt, number of scenarios that fired, measured)."""
    fired = set()
    total = 0.0
    measured = False
    for (s, m, v), row in data.index.items():
        if v != "mlfq_preempt" or row.get("preemptions_mean") is None:
            continue
        measured = True
        total += row["preemptions_mean"] * (row.get("preemptions_n") or 0)
        if row["preemptions_mean"] > 0:
            fired.add(s)
    return total, fired, measured


def _dominant_bottleneck(data):
    totals = {name: 0 for name in SATURATION_NAMES}
    seen = False
    for trial in data.trials:
        sat = trial.get("saturation")
        if trial["variant"] == "v0" or not sat or trial["status"] != "ok":
            continue
        seen = True
        for name in SATURATION_NAMES:
            totals[name] += int(sat.get(name) or 0)
    if not seen:
        return None, totals
    top = max(totals, key=lambda k: totals[k])
    return (top if totals[top] > 0 else None), totals


# ---------------------------------------------------------------- text blocks

def _executive_lines(data):
    lines = []
    for metric, label, pick in (("ttft_p95_s", "Lowest mean P95 TTFT", min),
                                ("output_tok_s", "Best mean output throughput", max),
                                ("itl_p99_s", "Lowest mean P99 ITL", min)):
        best = _best_variant(data, metric, pick)
        if best is None:
            lines.append(f"{label}: {NOT_MEASURED}.")
        else:
            value, variant, count = best[0], best[1], best[2]
            unit = " tok/s" if metric == "output_tok_s" else " s"
            lines.append(f"{label}: {VARIANT_LABELS[variant]} ({value:.4g}{unit}, averaged over {count} scenario/max_num_seqs points).")
    improved, compared = _short_ttft_improvement(data)
    if compared:
        lines.append(f"MLFQ lowered the mean P95 TTFT of short requests versus FCFS at {improved} of {compared} points.")
    else:
        lines.append("MLFQ versus FCFS for short requests: no point has both measured.")
    _, itl_worse, compared_itl = _direction_counts(data, "itl_p99_s", False, "mlfq")
    if compared_itl:
        lines.append(f"MLFQ harmed streaming ITL (P99 ITL higher than FCFS) at {itl_worse} of {compared_itl} points.")
    else:
        lines.append("MLFQ versus FCFS for streaming ITL: no point has both measured.")
    total, fired, measured = _preemption_totals(data)
    if not measured:
        lines.append(f"Preemption: {NOT_MEASURED}.")
    elif total == 0:
        lines.append("MLFQ preemption was not exercised in this run; this run cannot establish its effect.")
    else:
        lines.append(f"MLFQ preemption fired {total:.0f} times across {len(fired)} scenario(s).")
    bottleneck, totals = _dominant_bottleneck(data)
    if bottleneck is None and not any(totals.values()):
        lines.append("Dominant bottleneck: no saturation hits were recorded by V1/V2 trials.")
    elif bottleneck is None:
        lines.append("Dominant bottleneck: no single counter dominated.")
    else:
        lines.append(f"Dominant bottleneck from saturation counters: {SATURATION_NAMES[bottleneck]} "
                     f"({totals[bottleneck]} hits across V1/V2 trials).")
    return lines


def _interpretation(data):
    """Returns a list of (point label, [lines]). Every line is generated from measured numbers."""
    out = []
    for (s, m) in data.points:
        lines = []
        for metric, label, higher in (("ttft_p95_s", "P95 TTFT", False),
                                      ("itl_p99_s", "P99 ITL", False),
                                      ("e2e_p95_s", "P95 E2E latency", False),
                                      ("output_tok_s", "output tok/s", True)):
            base = data.stat(s, m, "fcfs", metric)
            cand = data.stat(s, m, "mlfq", metric)
            pct = metrics.improvement_pct(base["mean"], cand["mean"], higher)
            if pct is not None:
                lines.append(f"{label}: MLFQ {cand['mean']:.4g} versus FCFS {base['mean']:.4g} "
                             f"({pct:+.1f}% vs FCFS, positive is better).")
        short_base = data.group_stats.get((s, m, "fcfs"), {}).get("short_ttft_p95", {}).get("mean")
        short_cand = data.group_stats.get((s, m, "mlfq"), {}).get("short_ttft_p95", {}).get("mean")
        short_pct = metrics.improvement_pct(short_base, short_cand, False)
        if short_pct is not None:
            lines.append(f"Short-request P95 TTFT: MLFQ {short_cand:.4g} s versus FCFS {short_base:.4g} s "
                         f"({short_pct:+.1f}% vs FCFS, positive is better).")
        preempt = data.stat(s, m, "mlfq_preempt", "preemptions")
        if preempt["mean"] is not None and preempt["mean"] == 0:
            lines.append("MLFQ preemption was not exercised; this run cannot establish its effect.")
        elif preempt["mean"] is not None:
            lines.append(f"MLFQ with preemption averaged {preempt['mean']:.1f} preemptions per trial.")
        if lines:
            out.append((f"{s}, max_num_seqs={m}", lines))
    return out


def _conclusion_lines(data):
    lines = _executive_lines(data)
    # Latency and throughput leaders, then the preemption and bottleneck findings.
    picked = lines[:3] + lines[-2:]
    picked.append("These conclusions cover only the measured scenarios, load factor, and hardware. "
                  "They do not generalize beyond them.")
    return picked


def _run_counts(data):
    counts = {"total": len(data.trials), "ok": 0, "timeout": 0, "failed": 0}
    for trial in data.trials:
        status = trial["status"]
        if status in counts:
            counts[status] += 1
    unfinished = sum(int(t.get("unfinished") or 0) for t in data.trials if t["status"] == "ok")
    unfinished_all = sum(int(t.get("unfinished") or 0) for t in data.trials)
    truncated = sum(1 for t in data.trials if t.get("events_truncated"))
    return counts, unfinished, unfinished_all, truncated


# ---------------------------------------------------------------- charts

def _draw_grouped(ax, categories, series, ylabel, title):
    """series: list of (label, values, errors). Returns True if anything was drawn."""
    drawn = any(v is not None for _, values, _ in series for v in values)
    if not categories or not drawn:
        return False
    width = 0.8 / len(series)
    for i, (label, values, errors) in enumerate(series):
        xs = [x + (i - (len(series) - 1) / 2) * width for x in range(len(categories))]
        heights = [math.nan if v is None else v for v in values]
        err = [0.0 if e is None else e for e in errors]
        ax.bar(xs, heights, width, label=label, yerr=err if any(err) else None, capsize=3)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return True


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110, format="png")
    plt.close(fig)


def _bar_chart(path, title, ylabel, categories, series):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not _draw_grouped(ax, categories, series, ylabel, title):
        plt.close(fig)
        return False
    _save(fig, path)
    return True


def _variant_series(data, s_list, m, metric, scale=1.0, variants=VARIANT_ORDER):
    out = []
    for variant in variants:
        values, errors = [], []
        for s in s_list:
            stat = data.stat(s, m, variant, metric)
            values.append(None if stat["mean"] is None else stat["mean"] * scale)
            errors.append(stat["ci95"] * scale if stat["n"] >= 2 and stat["ci95"] is not None else None)
        if any(v is not None for v in values):
            out.append((VARIANT_LABELS[variant], values, errors))
    return out


def _scenarios_at(data, m):
    present = set(data.points)
    return [s for s in SCENARIOS if (s, m) in present]


def _chart_plan(data, plots_dir):
    """Returns (charts made, charts skipped). Each chart is a dict with file, title, caption.

    Bar charts use the headline max_num_seqs (the largest measured). With no data the
    scenario list is empty and every chart is skipped.
    """
    made, skipped = [], []
    m = data.headline_msn
    scenarios = _scenarios_at(data, m) if m is not None else []

    def attempt(name, title, caption, builder):
        if builder(plots_dir / name):
            made.append({"file": f"plots/{name}", "title": title, "caption": caption})
        else:
            skipped.append({"title": title, "reason": "no measured data"})

    bars = [
        ("p95_ttft_by_policy.png", "P95 TTFT by policy", "P95 TTFT (s)", "ttft_p95_s",
         "Bars are means over repeats; error bars are 95% CIs (drawn only when n >= 2)."),
        ("p99_itl_by_policy.png", "P99 ITL by policy", "P99 ITL (s)", "itl_p99_s",
         "Inter-token latency over the streamed tokens; error bars as above."),
        ("output_tok_s_by_policy.png", "Output tok/s by policy", "tokens per second", "output_tok_s",
         "Throughput is generated tokens divided by replay wall time."),
        ("e2e_latency_by_policy.png", "E2E latency by policy", "P95 E2E latency (s)", "e2e_p95_s",
         "E2E is measured over finished requests only."),
    ]
    for name, title, ylabel, metric, caption in bars:
        full_title = f"{title} (max_num_seqs={m})"
        attempt(name, full_title, caption, lambda p, t=full_title, y=ylabel, k=metric: _bar_chart(
            p, t, y, scenarios, _variant_series(data, scenarios, m, k)))

    def improvement(path):
        series = []
        for variant in ("mlfq", "mlfq_preempt"):
            for metric, label in (("ttft_p95_s", "P95 TTFT"), ("itl_p99_s", "P99 ITL")):
                values = []
                for s in scenarios:
                    base = data.stat(s, m, "fcfs", metric)["mean"]
                    cand = data.stat(s, m, variant, metric)["mean"]
                    values.append(metrics.improvement_pct(base, cand, False))
                if any(v is not None for v in values):
                    series.append((f"{VARIANT_LABELS[variant]}: {label}", values, [None] * len(values)))
        return _bar_chart(path, f"Improvement over FCFS (max_num_seqs={m}, positive is better)",
                          "% vs FCFS", scenarios, series)

    attempt("improvement_vs_fcfs.png", "Improvement over FCFS",
            "Percent improvement of each policy over FCFS, computed from the mean over repeats.",
            improvement)

    def perf_vs_msn(path):
        panels = [s for s in SCENARIOS
                  if any(data.stat(s, msn, v, "ttft_p95_s")["mean"] is not None
                         for (s_, msn) in data.points if s_ == s for v in VARIANT_ORDER)]
        if not panels:
            return False
        cols = 2
        rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), squeeze=False)
        for idx, s in enumerate(panels):
            ax = axes[idx // cols][idx % cols]
            for variant in VARIANT_ORDER:
                xs, ys, es = [], [], []
                for msn in sorted({m_ for (s_, m_) in data.points if s_ == s}):
                    stat = data.stat(s, msn, variant, "ttft_p95_s")
                    if stat["mean"] is None:
                        continue
                    xs.append(msn)
                    ys.append(stat["mean"])
                    es.append(stat["ci95"] if stat["n"] >= 2 and stat["ci95"] is not None else 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", label=VARIANT_LABELS[variant], capsize=3)
            ax.set_title(s, fontsize=9)
            ax.set_xlabel("max_num_seqs", fontsize=8)
            ax.set_ylabel("P95 TTFT (s)", fontsize=8)
        for idx in range(len(panels), rows * cols):
            axes[idx // cols][idx % cols].set_visible(False)
        axes[0][0].legend(fontsize=7)
        _save(fig, path)
        return True

    attempt("performance_vs_max_num_seqs.png", "Performance vs max_num_seqs",
            "P95 TTFT as max_num_seqs grows, one panel per scenario. Error bars: 95% CI where n >= 2.",
            perf_vs_msn)

    def preemption(path):
        series = _variant_series(data, scenarios, m, "preemptions", 1.0, variants=("mlfq", "mlfq_preempt"))
        return _bar_chart(path, f"Preemptions by workload (max_num_seqs={m})", "preemptions per trial",
                          scenarios, series)

    attempt("preemptions_by_workload.png", "Preemption count by workload",
            "Mean preemptions per trial. Zero bars are measured zeros; missing bars were not measured.",
            preemption)

    def kv_usage(path):
        series = []
        for variant in ("fcfs", "mlfq", "mlfq_preempt"):
            values, errors = [], []
            for s in scenarios:
                stat = data.kv_peak.get((s, m, variant))
                values.append(stat["mean"] if stat else None)
                errors.append(stat["ci95"] if stat and stat["n"] >= 2 else None)
            if any(v is not None for v in values):
                series.append((VARIANT_LABELS[variant], values, errors))
        return _bar_chart(path, f"Peak KV block usage (max_num_seqs={m})", "fraction of KV blocks",
                          scenarios, series)

    attempt("kv_usage.png", "KV usage", "Peak fraction of KV blocks in use, from scheduler_events.csv. "
            "Trials with truncated event logs give a lower bound.",
            kv_usage)

    def short_vs_long(path):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
        short_series, long_series = [], []
        for variant in VARIANT_ORDER:
            short_vals, short_errs, long_vals, long_errs = [], [], [], []
            for s in scenarios:
                st = data.group_stats.get((s, m, variant), {}).get("short_ttft_p95", {})
                lt = data.group_stats.get((s, m, variant), {}).get("long_e2e_p95", {})
                short_vals.append(st.get("mean"))
                short_errs.append(st.get("ci95") if st.get("n", 0) >= 2 else None)
                long_vals.append(lt.get("mean"))
                long_errs.append(lt.get("ci95") if lt.get("n", 0) >= 2 else None)
            if any(v is not None for v in short_vals):
                short_series.append((VARIANT_LABELS[variant], short_vals, short_errs))
            if any(v is not None for v in long_vals):
                long_series.append((VARIANT_LABELS[variant], long_vals, long_errs))
        a = _draw_grouped(left, scenarios, short_series, "P95 TTFT (s)", "Short requests: P95 TTFT")
        b = _draw_grouped(right, scenarios, long_series, "P95 E2E (s)", "Long requests: P95 E2E latency")
        if not (a or b):
            plt.close(fig)
            return False
        _save(fig, path)
        return True

    attempt("short_vs_long_latency.png", "Short vs long request latency",
            "Short = short and burst groups. Long = long and background groups (finished requests only for E2E).",
            short_vs_long)
    return made, skipped


def _make_plots(data, plots_dir):
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old in plots_dir.glob("*.png"):
        old.unlink()
    return _chart_plan(data, plots_dir)


# ---------------------------------------------------------------- document model

def _table(headers, rows):
    return ("table", headers, rows)


def _comparison_blocks(data):
    blocks = []
    for (s, m) in data.points:
        blocks.append(("h3", f"{s}, max_num_seqs={m}"))
        headers = ["Metric"] + [VARIANT_LABELS[v] for v in ("v0", "fcfs", "mlfq", "mlfq_preempt")]
        rows = []
        for label, metric, scale, pattern in COMPARISON_ROWS:
            row = [label]
            for variant in ("v0", "fcfs", "mlfq", "mlfq_preempt"):
                stat = data.stat(s, m, variant, metric)
                row.append(_cell(stat, scale, pattern))
            rows.append(row)
        blocks.append(_table(headers, rows))
    return blocks


def _improvement_blocks(data):
    rows = []
    for row in data.improvements:
        if row.get("improvement_pct") is None or row["variant"] == "fcfs":
            continue
        rows.append([row["scenario"], str(row["max_num_seqs"]), VARIANT_LABELS[row["variant"]], row["metric"],
                     _fmt(row["baseline_mean"], "{:.4g}"), _fmt(row["candidate_mean"], "{:.4g}"),
                     f"{row['improvement_pct']:+.1f}%"])
    if not rows:
        return [("p", NOT_MEASURED + " (no variant pair has a measured percent improvement).")]
    return [_table(["Scenario", "max_num_seqs", "Policy", "Metric", "FCFS mean", "Policy mean", "Improvement"], rows)]


def _environment_blocks(data, counts, unfinished):
    blocks = []
    if data.fixture:
        blocks.append(("banner", FIXTURE_BANNER))
    rows = []
    for key in sorted(data.config):
        value = data.config[key]
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value) if value else "(none)"
        elif value is None:
            value = "null (not recorded)"
        rows.append([key, str(value)])
    blocks.append(("p", "Configuration recorded by run_matrix (configuration.json):"))
    blocks.append(_table(["Key", "Value"], rows or [["configuration.json", "missing"]]))
    blocks.append(("p", f"Trials: {counts['total']} total, {counts['ok']} ok, "
                        f"{counts['timeout']} timed out, {counts['failed']} failed."))
    return blocks


def _chart_blocks(made, skipped):
    blocks = []
    for chart in made:
        blocks.append(("img", chart["file"], chart["title"], chart["caption"]))
    for chart in skipped:
        blocks.append(("p", f"Chart skipped: {chart['title']} ({chart['reason']})."))
    blocks.append(("p", "Queue wait by MLFQ level: skipped. requests.csv does not record the per-request MLFQ level."))
    return blocks


def build_document(data, made, skipped):
    counts, unfinished_ok, unfinished_all, truncated = _run_counts(data)
    doc = [("title", "nano-vLLM scheduler benchmark report")]
    if data.fixture:
        doc.append(("banner", FIXTURE_BANNER))
    doc.append(("h2", "1. Environment and configuration"))
    doc.extend(_environment_blocks(data, counts, unfinished_ok))
    doc.append(("h2", "2. Executive summary"))
    doc.append(("ul", _executive_lines(data)))
    doc.append(("h2", "3. Comparison"))
    doc.append(("p", "Mean over repeats, with 95% CI where n >= 2. Original V0 has no scheduler counters, "
                     "so its preemption cell is not measured."))
    doc.extend(_comparison_blocks(data) or [("p", NOT_MEASURED + ".")])
    doc.append(("h2", "4. Improvement vs FCFS"))
    doc.extend(_improvement_blocks(data))
    doc.append(("h2", "5. Charts"))
    doc.extend(_chart_blocks(made, skipped))
    doc.append(("h2", "6. Interpretation"))
    interpretation = _interpretation(data)
    if not interpretation:
        doc.append(("p", "No point has measured data for both FCFS and MLFQ."))
    for label, lines in interpretation:
        doc.append(("h3", label))
        doc.append(("ul", lines))
    doc.append(("h2", "7. Overall conclusion"))
    doc.append(("ul", _conclusion_lines(data)))
    doc.append(("h2", "8. Limitations"))
    doc.append(("ul", list(LIMITATIONS) + [
        f"Unfinished requests in ok trials: {unfinished_ok}. Across all trials, including timeouts: {unfinished_all}.",
        f"Failed or timed-out trials: {counts['failed'] + counts['timeout']} "
        f"({counts['failed']} failed, {counts['timeout']} timed out).",
        f"Trials with a truncated scheduler event log: {truncated}.",
    ]))
    return doc


# ---------------------------------------------------------------- renderers

def _md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(doc):
    lines = []
    for block in doc:
        kind = block[0]
        if kind == "title":
            lines += [f"# {block[1]}", ""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "h3":
            lines += [f"### {block[1]}", ""]
        elif kind == "banner":
            lines += [f"> **{block[1]}**", ""]
        elif kind == "p":
            lines += [block[1], ""]
        elif kind == "ul":
            lines += [f"- {item}" for item in block[1]] + [""]
        elif kind == "table":
            headers, rows = block[1], block[2]
            lines.append("| " + " | ".join(_md_cell(h) for h in headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            for row in rows:
                lines.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
            lines.append("")
        elif kind == "img":
            lines += [f"![{block[2]}]({block[1]})", f"*{block[3]}*", ""]
    return "\n".join(lines) + "\n"


_HTML_STYLE = """
:root { --fg: #1d1f23; --bg: #ffffff; --muted: #5b6270; --line: #d7dbe3; --banner: #fff1c2; --banner-line: #c99a00; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --fg: #e8eaf0; --bg: #14161b; --muted: #9aa3b2; --line: #3a404c; --banner: #3d3412; --banner-line: #d7a600; }
}
:root[data-theme="dark"] { --fg: #e8eaf0; --bg: #14161b; --muted: #9aa3b2; --line: #3a404c; --banner: #3d3412; --banner-line: #d7a600; }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font-family: system-ui, sans-serif; line-height: 1.45; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
h3 { font-size: 1.05rem; margin-top: 1.4rem; }
.banner { background: var(--banner); border: 2px solid var(--banner-line); padding: 12px; border-radius: 6px; font-weight: 600; margin: 12px 0; }
.table-wrap { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; min-width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; vertical-align: top; }
td:nth-child(n+2) { font-variant-numeric: tabular-nums; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0; }
figcaption, .caption { color: var(--muted); font-size: 0.9rem; }
"""


def _png_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def render_html(doc, output_dir):
    out = []
    for block in doc:
        kind = block[0]
        if kind == "title":
            out.append(f"<h1>{html.escape(block[1])}</h1>")
        elif kind == "h2":
            out.append(f"<h2>{html.escape(block[1])}</h2>")
        elif kind == "h3":
            out.append(f"<h3>{html.escape(block[1])}</h3>")
        elif kind == "banner":
            out.append(f'<div class="banner" role="alert">{html.escape(block[1])}</div>')
        elif kind == "p":
            out.append(f"<p>{html.escape(block[1])}</p>")
        elif kind == "ul":
            items = "".join(f"<li>{html.escape(item)}</li>" for item in block[1])
            out.append(f"<ul>{items}</ul>")
        elif kind == "table":
            head = "".join(f"<th>{html.escape(str(h))}</th>" for h in block[1])
            body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                           for row in block[2])
            out.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
                       f"<tbody>{body}</tbody></table></div>")
        elif kind == "img":
            src = _png_data_uri(Path(output_dir) / block[1])
            out.append(f'<figure><img src="{src}" alt="{html.escape(block[2])}">'
                       f'<figcaption class="caption">{html.escape(block[2])}. {html.escape(block[3])}</figcaption></figure>')
    title = html.escape(next((b[1] for b in doc if b[0] == "title"), "benchmark report"))
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{title}</title>\n<style>{_HTML_STYLE}</style>\n</head>\n<body>\n<main>\n"
            + "\n".join(out) + "\n</main>\n</body>\n</html>\n")


# ---------------------------------------------------------------- driver

def _summary_json(data, made, skipped):
    counts, unfinished_ok, unfinished_all, truncated = _run_counts(data)
    comparison = {}
    for (s, m) in data.points:
        per_variant = {}
        for variant in ("v0", "fcfs", "mlfq", "mlfq_preempt"):
            per_variant[variant] = {metric: data.stat(s, m, variant, metric) for _, metric, _, _ in COMPARISON_ROWS}
        comparison[f"{s}|{m}"] = per_variant
    total, fired, measured = _preemption_totals(data)
    return {
        "synthetic_fixture": data.fixture,
        "headline_max_num_seqs": data.headline_msn,
        "executive_summary": _executive_lines(data),
        "comparison": comparison,
        "improvements": data.improvements,
        "kv_peak_usage": {f"{s}|{m}|{v}": stat for (s, m, v), stat in sorted(data.kv_peak.items())},
        "preemption": {"measured": measured, "total_mlfq_preempt": total, "scenarios_fired": sorted(fired)},
        "trials": counts,
        "unfinished_requests_ok_trials": unfinished_ok,
        "unfinished_requests_all_trials": unfinished_all,
        "trials_with_truncated_events": truncated,
        "charts": made,
        "charts_skipped": skipped,
    }


def _check_paths(input_dir, output_dir):
    inp = Path(input_dir).resolve()
    out = Path(output_dir).resolve()
    if out == inp or inp in out.parents:
        raise ValueError(f"--output must not be inside the input directory: {out}")
    if not (inp / "summaries").is_dir() and not (inp / "raw").is_dir():
        raise ValueError(f"no aggregated results in {inp}; run benchmarks.aggregate first")


def generate(input_dir, output_dir):
    _check_paths(input_dir, output_dir)
    data = load(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = _make_plots(data, out / "plots")
    doc = build_document(data, made, skipped)
    with open(out / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(render_markdown(doc))
    with open(out / "REPORT.html", "w", encoding="utf-8") as f:
        f.write(render_html(doc, out))
    with open(out / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(_summary_json(data, made, skipped), f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return out


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m benchmarks.generate_report",
                                     description="Generate a report from recorded benchmark results.")
    parser.add_argument("--input", required=True, help="Output directory of benchmarks.run_matrix / aggregate.")
    parser.add_argument("--output", required=True, help="Directory for REPORT.md, REPORT.html and plots/.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        out = generate(args.input, args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {out / 'REPORT.md'} and {out / 'REPORT.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
