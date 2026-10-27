import math

import pytest

from benchmarks.metrics import (aggregate_repeats, improvement_pct, parse_csv_record, percentile,
                                request_metrics, summarize, summarize_requests)


def test_percentile_empty_is_none_not_zero():
    assert percentile([], 95) is None
    assert percentile([None, None], 50) is None


def test_percentile_single_value():
    assert percentile([3.5], 0) == 3.5
    assert percentile([3.5], 99) == 3.5


def test_percentile_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4], 25) == pytest.approx(1.75)
    assert percentile([4, 1, 3, 2], 100) == 4
    assert percentile([0, 10], 95) == pytest.approx(9.5)


def test_percentile_p99_on_hundred_values():
    values = list(range(1, 101))
    assert percentile(values, 99) == pytest.approx(99.01)
    assert percentile(values, 95) == pytest.approx(95.05)


def test_percentile_ignores_none_and_rejects_bad_p():
    assert percentile([None, 1, 3], 50) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_summarize_keys_and_empty():
    stats = summarize([1, 2, 3, 4])
    assert set(stats) == {"n", "mean", "p50", "p95", "p99", "max", "min"}
    assert stats["n"] == 4 and stats["mean"] == pytest.approx(2.5) and stats["max"] == 4
    empty = summarize([])
    assert empty["n"] == 0
    assert all(empty[k] is None for k in ("mean", "p50", "p95", "p99", "max", "min"))


def test_improvement_pct_lower_is_better_signs():
    assert improvement_pct(10, 8, higher_is_better=False) == pytest.approx(20.0)
    assert improvement_pct(10, 12, higher_is_better=False) == pytest.approx(-20.0)


def test_improvement_pct_higher_is_better_signs():
    assert improvement_pct(10, 12, higher_is_better=True) == pytest.approx(20.0)
    assert improvement_pct(10, 8, higher_is_better=True) == pytest.approx(-20.0)


def test_improvement_pct_none_and_zero_baseline():
    assert improvement_pct(0, 5, higher_is_better=True) is None
    assert improvement_pct(0, 5, higher_is_better=False) is None
    assert improvement_pct(None, 5, higher_is_better=False) is None
    assert improvement_pct(5, None, higher_is_better=True) is None


def test_request_metrics_full_record():
    record = {"arrival_time": 1.0, "admit_time": 1.5, "first_token_time": 2.0,
              "finish_time": 4.0, "n_tokens": 5, "token_times": [2.0, 2.5, 3.0, 3.5, 4.0]}
    m = request_metrics(record)
    assert m["ttft"] == pytest.approx(1.0)
    assert m["e2e"] == pytest.approx(3.0)
    assert m["queue_wait"] == pytest.approx(0.5)
    assert m["tpot"] == pytest.approx(0.5)
    assert m["itl"] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_request_metrics_missing_timestamps_give_none():
    record = {"arrival_time": 1.0, "admit_time": None, "first_token_time": None,
              "finish_time": None, "n_tokens": 0, "token_times": []}
    m = request_metrics(record)
    assert m["ttft"] is None and m["e2e"] is None and m["queue_wait"] is None
    assert m["tpot"] is None and m["itl"] == []


def test_request_metrics_single_token_has_no_tpot():
    record = {"arrival_time": 0.0, "admit_time": 0.1, "first_token_time": 0.2,
              "finish_time": 0.2, "n_tokens": 1, "token_times": [0.2]}
    m = request_metrics(record)
    assert m["tpot"] is None
    assert m["ttft"] == pytest.approx(0.2)
    assert m["itl"] == []


def test_aggregate_repeats_single_value_has_no_std_or_ci():
    stats = aggregate_repeats([2.0])
    assert stats["n"] == 1 and stats["mean"] == 2.0
    assert stats["std"] is None and stats["ci95"] is None


def test_aggregate_repeats_empty_and_none_values():
    stats = aggregate_repeats([None])
    assert stats["n"] == 0 and stats["mean"] is None and stats["std"] is None


def test_aggregate_repeats_three_values_uses_t_table():
    stats = aggregate_repeats([1.0, 2.0, 3.0])
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["std"] == pytest.approx(1.0)
    assert stats["ci95"] == pytest.approx(4.303 / math.sqrt(3))
    assert stats["min"] == 1.0 and stats["max"] == 3.0


def test_aggregate_repeats_large_n_uses_1_96():
    values = [float(i) for i in range(12)]
    stats = aggregate_repeats(values)
    mean = sum(values) / 12
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / 11)
    assert stats["ci95"] == pytest.approx(1.96 * std / math.sqrt(12))


def test_parse_csv_record_converts_types():
    row = {"request_id": "r1", "group": "short", "arrival_s": "0.5", "arrival_time": "0.6",
           "admit_time": "", "first_token_time": "0.9", "finish_time": "1.2", "n_tokens": "3",
           "status": "finished", "preemptions": "", "token_times": "[0.9, 1.0, 1.2]"}
    rec = parse_csv_record(row)
    assert rec["arrival_time"] == pytest.approx(0.6)
    assert rec["admit_time"] is None
    assert rec["n_tokens"] == 3
    assert rec["preemptions"] is None
    assert rec["token_times"] == [0.9, 1.0, 1.2]


def test_summarize_requests_counts_and_groups():
    records = [
        {"arrival_time": 0.0, "admit_time": 0.0, "first_token_time": 0.1, "finish_time": 0.3,
         "n_tokens": 2, "token_times": [0.1, 0.3], "status": "finished", "group": "short"},
        {"arrival_time": 0.0, "admit_time": None, "first_token_time": None, "finish_time": None,
         "n_tokens": 0, "token_times": [], "status": "unfinished", "group": "long"},
        {"arrival_time": None, "admit_time": None, "first_token_time": None, "finish_time": None,
         "n_tokens": 0, "token_times": [], "status": "rejected", "group": "long"},
    ]
    summary = summarize_requests(records, wall_time_s=2.0)
    assert summary["counts"] == {"total": 3, "finished": 1, "unfinished": 1, "rejected": 1}
    assert summary["total_output_tokens"] == 2
    assert summary["output_tok_s"] == pytest.approx(1.0)
    assert summary["by_group"]["short"]["ttft_s"]["n"] == 1
    assert summary["ttft_s"]["n"] == 1
