# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for explicit protected bands with horizon-based priority aging."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.request_queue import (
    ExplicitBandPriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
    explicit_band_key,
    explicit_band_of,
    explicit_band_victim_key,
)


@dataclass(eq=False)
class DummyRequest:
    request_id: str
    priority: int
    arrival_time: float
    sampling_params: object = None


def make_request(name, priority, arrival, band=None):
    params = (
        SimpleNamespace(extra_args={"priority_band": band})
        if band is not None else None
    )
    return DummyRequest(name, priority, arrival, params)


def test_band_is_independent_of_original_numeric_priority():
    q = ExplicitBandPriorityRequestQueue(7000, 7)
    q.add_request(make_request("ordinary", -10**12, 0, 2))
    q.add_request(make_request("protected", 10**12, 1000, 1))
    assert [r.request_id for r in q] == ["protected", "ordinary"]


def test_band_1_uses_native_strict_priority_and_arrival():
    q = ExplicitBandPriorityRequestQueue(7000, 7)
    q.add_request(make_request("weak", 10, 0, 1))
    q.add_request(make_request("strong_late", -1, 9, 1))
    q.add_request(make_request("strong_early", -1, 2, 1))
    assert [r.request_id for r in q] == ["strong_early", "strong_late", "weak"]


def test_band_2_crossover_uses_reference_horizon():
    q = ExplicitBandPriorityRequestQueue(7000, 7)
    q.add_request(make_request("old_p15", 15, 0, 2))
    q.add_request(make_request("new_p8", 8, 8, 2))
    assert q.pop_request().request_id == "old_p15"
    q.add_request(make_request("p8_at_6", 8, 6, 2))
    assert q.pop_request().request_id == "p8_at_6"


def test_no_hard_cap_or_float_tie_for_far_apart_priorities():
    a = make_request("a", 2**62, 0, 2)
    b = make_request("b", 2**62 + 1, 0, 2)
    assert explicit_band_key(a, 7000, 7) < explicit_band_key(b, 7000, 7)


def test_default_band_is_configurable():
    q = ExplicitBandPriorityRequestQueue(7000, 7, default_band=1)
    q.add_request(make_request("unmarked", 100, 30))
    q.add_request(make_request("marked_b2", -100, 0, 2))
    assert q.pop_request().request_id == "unmarked"


def test_invalid_band_rejected_and_bool_not_accepted():
    q = ExplicitBandPriorityRequestQueue(7000, 7)
    for bad in (0, 3, True, False, "1", 1.0):
        r = make_request("bad", 0, 0)
        r.sampling_params = SimpleNamespace(extra_args={"priority_band": bad})
        with pytest.raises(ValueError, match="priority_band"):
            q.add_request(r)


def test_victim_prefers_evicting_band_2():
    a = make_request("protected", 999, 0, 1)
    b = make_request("ordinary", -999, 0, 2)
    assert max([a, b], key=explicit_band_victim_key) is b


def test_factory_is_opt_in():
    native = create_request_queue(SchedulingPolicy.PRIORITY)
    assert not isinstance(native, ExplicitBandPriorityRequestQueue)
    enabled = create_request_queue(
        SchedulingPolicy.PRIORITY,
        priority_band_fairness_horizon_ms=7000,
        priority_band_reference_gap=7,
        priority_band_default=2,
    )
    assert isinstance(enabled, ExplicitBandPriorityRequestQueue)


def test_large_queue_remove_requeue_iteration():
    q = ExplicitBandPriorityRequestQueue(7000, 7)
    requests = [make_request(f"r{i}", 8 + i % 8, i * 0.01, 2)
                for i in range(1000)]
    for r in requests:
        q.add_request(r)
    assert len(q) == 1000
    first = q.peek_request()
    assert first is next(iter(q))
    q.remove_request(first)
    q.prepend_request(first)
    assert first is q.peek_request()
    q.remove_requests(set(requests))
    assert not q


def test_invalid_queue_configuration():
    for h, gap, b in ((0, 7, 2), (7000, 0, 2), (7000, 7, 3)):
        with pytest.raises(ValueError):
            ExplicitBandPriorityRequestQueue(h, gap, b)
