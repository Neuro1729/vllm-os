"""Pure statistics for benchmark results. No numpy, no GPU, no I/O.

Missing values are None throughout. A missing value is never turned into 0.
"""
import json
import math

# Two-sided 95% Student t critical values for df = 1..10. Beyond that, 1.96.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

TIMESTAMP_FIELDS = ("arrival_time", "admit_time", "first_token_time", "finish_time")


def _clean(values):
    return [float(v) for v in values if v is not None]


def percentile(values, p):
    """Linear-interpolation percentile (the same convention as numpy's default)."""
    if not 0 <= p <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    data = sorted(_clean(values))
    if not data:
        return None
    rank = (len(data) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return data[lo] + (data[hi] - data[lo]) * (rank - lo)


def summarize(values):
    data = _clean(values)
    if not data:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None, "min": None}
    return {
        "n": len(data),
        "mean": sum(data) / len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data),
        "min": min(data),
    }


def improvement_pct(baseline, candidate, higher_is_better):
    """Positive means the candidate is better than the baseline."""
    if baseline is None or candidate is None or baseline == 0:
        return None
    if higher_is_better:
        return (candidate - baseline) / baseline * 100.0
    return (baseline - candidate) / baseline * 100.0


def _delta(later, earlier):
    if later is None or earlier is None:
        return None
    return later - earlier


def request_metrics(record):
    """Latency metrics for one request record. Missing timestamps give None."""
    arrival = record.get("arrival_time")
    admit = record.get("admit_time")
    first = record.get("first_token_time")
    finish = record.get("finish_time")
    n_tokens = record.get("n_tokens")
    token_times = [t for t in (record.get("token_times") or []) if t is not None]

    tpot = None
    if n_tokens is not None and n_tokens > 1:
        tpot = _delta(finish, first)
        if tpot is not None:
            tpot = tpot / (n_tokens - 1)
    return {
        "ttft": _delta(first, arrival),
        "e2e": _delta(finish, arrival),
        "queue_wait": _delta(admit, arrival),
        "tpot": tpot,
        "itl": [b - a for a, b in zip(token_times, token_times[1:])],
    }


def aggregate_repeats(values):
    """Mean, sample std, n, min, max and 95% CI half-width over repeats.

    std and ci95 are None when fewer than two values are present.
    """
    data = _clean(values)
    n = len(data)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "min": None, "max": None}
    mean = sum(data) / n
    if n < 2:
        return {"n": n, "mean": mean, "std": None, "ci95": None, "min": min(data), "max": max(data)}
    variance = sum((v - mean) ** 2 for v in data) / (n - 1)
    std = math.sqrt(variance)
    df = n - 1
    crit = T_CRIT_95[df] if df <= 10 else 1.96
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": crit * std / math.sqrt(n),
        "min": min(data),
        "max": max(data),
    }


def parse_csv_record(row):
    """Convert one requests.csv row (all strings) into a typed record."""
    def number(key, cast):
        value = row.get(key)
        return None if value in (None, "") else cast(value)

    token_text = row.get("token_times") or ""
    record = {
        "request_id": row.get("request_id"),
        "group": row.get("group") or None,
        "arrival_s": number("arrival_s", float),
        "n_tokens": number("n_tokens", int),
        "status": row.get("status") or None,
        "preemptions": number("preemptions", int),
        "token_times": [float(t) for t in json.loads(token_text)] if token_text else [],
    }
    for key in TIMESTAMP_FIELDS:
        record[key] = number(key, float)
    return record


def summarize_requests(records, wall_time_s=None):
    """Request-level summary written into trial.json."""
    metrics = [request_metrics(r) for r in records]
    statuses = [r.get("status") for r in records]
    finished = [m for m, r in zip(metrics, records) if r.get("status") == "finished"]
    groups = sorted({r.get("group") for r in records if r.get("group")})
    by_group = {}
    for group in groups:
        members = [m for m, r in zip(metrics, records) if r.get("group") == group]
        by_group[group] = {
            "ttft_s": summarize([m["ttft"] for m in members]),
            "e2e_s": summarize([m["e2e"] for m in members]),
        }
    total_tokens = sum(r.get("n_tokens") or 0 for r in records)
    return {
        "counts": {
            "total": len(records),
            "finished": statuses.count("finished"),
            "unfinished": statuses.count("unfinished"),
            "rejected": statuses.count("rejected"),
        },
        "total_output_tokens": total_tokens,
        "output_tok_s": (total_tokens / wall_time_s) if wall_time_s else None,
        "ttft_s": summarize([m["ttft"] for m in metrics]),
        "e2e_s": summarize([m["e2e"] for m in finished]),
        "queue_wait_s": summarize([m["queue_wait"] for m in metrics]),
        "tpot_s": summarize([m["tpot"] for m in metrics]),
        "itl_s": summarize([gap for m in metrics for gap in m["itl"]]),
        "by_group": by_group,
    }
