import pytest

from benchmarks.workloads import (DEFAULT_MAX_NUM_BATCHED_TOKENS, GROUPS, PROFILES, SCENARIOS,
                                  build_trace, load_trace, save_trace)


def _requests(scenario, profile="standard", msn=8, seed=0, **kw):
    return build_trace(scenario, profile, msn, seed, **kw)["requests"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_same_trace(scenario):
    a = build_trace(scenario, "standard", 8, seed=7)
    b = build_trace(scenario, "standard", 8, seed=7)
    assert a["trace_id"] == b["trace_id"]
    assert a["requests"] == b["requests"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_different_seed_differs(scenario):
    a = build_trace(scenario, "standard", 8, seed=1)
    b = build_trace(scenario, "standard", 8, seed=2)
    assert a["trace_id"] != b["trace_id"]
    assert a["requests"] != b["requests"]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_request_shape_and_ordering(scenario, profile):
    trace = build_trace(scenario, profile, 6, seed=3)
    reqs = trace["requests"]
    ids = [r["request_id"] for r in reqs]
    assert len(ids) == len(set(ids))
    arrivals = [r["arrival_s"] for r in reqs]
    assert arrivals == sorted(arrivals)
    for r in reqs:
        assert r["group"] in GROUPS
        assert r["max_tokens"] >= 1
        assert len(r["prompt_token_ids"]) >= 1
        assert all(1000 <= t < 10000 for t in r["prompt_token_ids"])
        assert isinstance(r["arrival_s"], float)


@pytest.mark.parametrize("msn", [1, 3, 8, 40])
@pytest.mark.parametrize("scenario", ["slot_saturation", "repeated_priority_arrivals"])
def test_contention_requests_at_time_zero(scenario, msn):
    reqs = _requests(scenario, msn=msn)
    at_zero = [r for r in reqs if r["arrival_s"] == 0.0]
    assert len([r for r in at_zero if r["group"] == "background"]) >= msn
    later_short = [r for r in reqs if r["arrival_s"] > 0 and r["group"] in ("short", "burst")]
    assert later_short, "short high-priority arrivals must follow the background requests"


def test_counts_scale_with_max_num_seqs_not_hardcoded():
    small = _requests("slot_saturation", msn=4)
    large = _requests("slot_saturation", msn=64)
    assert len(large) - len(small) >= 60


def test_long_prefill_pressure_has_prompts_beyond_token_budget():
    budget = 512
    reqs = _requests("long_prefill_pressure", max_num_batched_tokens=budget)
    assert any(len(r["prompt_token_ids"]) > budget for r in reqs)


def test_kv_pressure_scales_with_budget():
    reqs = _requests("kv_pressure", kv_block_budget=16)
    per_request = [len(r["prompt_token_ids"]) + r["max_tokens"] for r in reqs]
    target = 16 * 256 // 4
    assert all(0.85 * target <= total <= 1.15 * target for total in per_request)


def test_starvation_aging_has_fresh_shorts_after_longs():
    reqs = _requests("starvation_aging")
    first_long = min(r["arrival_s"] for r in reqs if r["group"] == "long")
    late_shorts = [r for r in reqs if r["group"] == "short" and r["arrival_s"] > first_long]
    assert len(late_shorts) > 10


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_standard_profile_size(scenario):
    n = len(_requests(scenario, profile="standard", msn=8))
    assert 60 <= n <= 130


def test_smoke_is_small_and_single_repeat():
    assert PROFILES["smoke"]["repeats"] == 1
    assert len(_requests("no_contention", profile="smoke", msn=4)) <= 16


def test_profiles_repeats():
    assert PROFILES["standard"]["repeats"] == 3
    assert PROFILES["thorough"]["repeats"] == 5


def test_save_load_round_trip(tmp_path):
    trace = build_trace("mixed_prefill_decode", "standard", 8, seed=11,
                        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS)
    path = tmp_path / "nested" / "trace.json"
    save_trace(trace, path)
    loaded = load_trace(path)
    assert loaded == trace
    assert loaded["trace_id"] == trace["trace_id"]


def test_load_detects_tampered_trace(tmp_path):
    import json

    path = tmp_path / "trace.json"
    save_trace(build_trace("no_contention", "smoke", 2, seed=0), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["requests"][0]["max_tokens"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_trace(path)


@pytest.mark.parametrize("kwargs", [
    dict(scenario="nope", profile="standard", max_num_seqs=4, seed=0),
    dict(scenario="no_contention", profile="huge", max_num_seqs=4, seed=0),
    dict(scenario="no_contention", profile="standard", max_num_seqs=0, seed=0),
    dict(scenario="no_contention", profile="standard", max_num_seqs=4, seed=0, load_factor=0),
])
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        build_trace(**kwargs)
