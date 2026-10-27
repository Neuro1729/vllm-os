"""Deterministic benchmark traces.

Traces are generated without a GPU and without a tokenizer. Prompt tokens are
uniform random ids in [1000, 10000) drawn from a seeded random.Random. Arrivals
are open-loop Poisson (seeded exponential gaps), except where a scenario fixes
arrival times explicitly (for example, background requests at t=0).

Every request has: request_id, arrival_s, prompt_token_ids, max_tokens, group.
The group is "long" or "short" by prompt class, "burst" for repeated priority
waves, and "background" for the pre-filled long requests that create contention.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

SCENARIOS = (
    "no_contention",
    "slot_saturation",
    "mixed_prefill_decode",
    "short_vs_long",
    "long_prefill_pressure",
    "kv_pressure",
    "starvation_aging",
    "repeated_priority_arrivals",
)
GROUPS = ("long", "short", "burst", "background")

TOKEN_ID_LOW = 1000
TOKEN_ID_HIGH = 10000
KV_BLOCK_TOKENS = 256  # nano-vLLM default kvcache_block_size
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_KV_BLOCK_BUDGET = 64
WAVE_PERIOD_S = 2.0
WAVE_SPACING_S = 0.001

# (low, high) bounds, in tokens, before length_scale is applied.
SHORT_PROMPT = (32, 128)
SHORT_OUTPUT = (16, 64)
LONG_PROMPT = (1024, 2048)
LONG_OUTPUT = (128, 256)
DECODE_HEAVY_PROMPT = (32, 64)
DECODE_HEAVY_OUTPUT = (256, 512)
PREFILL_HEAVY_OUTPUT = (8, 16)

PROFILES = {
    "smoke": {
        "repeats": 1,
        "count_scale": 0.125,
        "length_scale": 0.25,
        "arrival_rate_rps": 4.0,
        "load_factors": (1.0,),
        "description": "Tiny traces for wiring checks. Not for measurement.",
    },
    "standard": {
        "repeats": 3,
        "count_scale": 1.0,
        "length_scale": 1.0,
        "arrival_rate_rps": 4.0,
        "load_factors": (1.0,),
        "description": "About 64-128 requests per trial, 3 repeats.",
    },
    "thorough": {
        "repeats": 5,
        "count_scale": 2.0,
        "length_scale": 1.0,
        "arrival_rate_rps": 4.0,
        "load_factors": (0.5, 1.0, 2.0),
        "description": "Twice the requests of standard, 5 repeats. The first load factor is used by run_matrix.",
    },
}

PARAM_KEYS = (
    "scenario",
    "profile",
    "max_num_seqs",
    "seed",
    "max_num_batched_tokens",
    "kv_block_budget",
    "load_factor",
)


class _Spec(NamedTuple):
    arrival_s: float
    group: str
    prompt_len: int
    max_tokens: int


@dataclass(frozen=True)
class _Ctx:
    rng: random.Random
    count_scale: float
    length_scale: float
    rate: float
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_block_budget: int


# ---------------------------------------------------------------- helpers

def _count(ctx, base, minimum=1):
    return max(minimum, round(base * ctx.count_scale))


def _length(ctx, bounds):
    low, high = bounds
    lo = max(1, int(low * ctx.length_scale))
    hi = max(lo, int(high * ctx.length_scale))
    return ctx.rng.randint(lo, hi)


def _short_shape(ctx):
    return _length(ctx, SHORT_PROMPT), _length(ctx, SHORT_OUTPUT)


def _long_shape(ctx):
    return _length(ctx, LONG_PROMPT), _length(ctx, LONG_OUTPUT)


def _poisson(ctx, count, rate, start=0.0):
    times = []
    t = start
    for _ in range(count):
        t += ctx.rng.expovariate(rate)
        times.append(t)
    return times


def _short_requests(ctx, times, group="short"):
    return [_Spec(t, group, *_short_shape(ctx)) for t in times]


def _long_requests(ctx, times, group="long"):
    return [_Spec(t, group, *_long_shape(ctx)) for t in times]


def _merge(streams):
    tagged = [
        (spec.arrival_s, stream_idx, pos, spec)
        for stream_idx, stream in enumerate(streams)
        for pos, spec in enumerate(stream)
    ]
    tagged.sort(key=lambda item: item[:3])
    return [item[3] for item in tagged]


# ---------------------------------------------------------------- scenarios

def _no_contention(ctx):
    # Light load: short requests never fill the slots.
    return [_short_requests(ctx, _poisson(ctx, _count(ctx, 64), ctx.rate * 0.5))]


def _slot_saturation(ctx):
    # max_num_seqs long requests occupy every slot at t=0; short high-priority
    # requests then arrive open-loop and must wait for a slot.
    background = [_Spec(0.0, "background", *_long_shape(ctx)) for _ in range(ctx.max_num_seqs)]
    shorts = _short_requests(ctx, _poisson(ctx, _count(ctx, 64), ctx.rate))
    return [background, shorts]


def _mixed_prefill_decode(ctx):
    specs = []
    for t in _poisson(ctx, _count(ctx, 96), ctx.rate):
        if ctx.rng.random() < 0.5:
            # Prefill-heavy: long prompt, few output tokens.
            specs.append(_Spec(t, "long", _length(ctx, LONG_PROMPT), _length(ctx, PREFILL_HEAVY_OUTPUT)))
        else:
            # Decode-heavy: short prompt, long output.
            specs.append(_Spec(t, "short", _length(ctx, DECODE_HEAVY_PROMPT), _length(ctx, DECODE_HEAVY_OUTPUT)))
    return [specs]


def _short_vs_long(ctx):
    longs = _long_requests(ctx, _poisson(ctx, _count(ctx, 16), ctx.rate * 0.25))
    shorts = _short_requests(ctx, _poisson(ctx, _count(ctx, 80), ctx.rate * 2))
    return [longs, shorts]


def _long_prefill_pressure(ctx):
    # Some prompts are longer than the per-step token budget, so they need
    # several chunked prefill steps and compete with short prompts.
    budget = ctx.max_num_batched_tokens
    specs = []
    for t in _poisson(ctx, _count(ctx, 64), ctx.rate):
        if ctx.rng.random() < 0.3:
            specs.append(_Spec(t, "long", ctx.rng.randint(budget + 1, 2 * budget), _length(ctx, SHORT_OUTPUT)))
        else:
            specs.append(_Spec(t, "short", _length(ctx, SHORT_PROMPT), _length(ctx, SHORT_OUTPUT)))
    return [specs]


def _kv_pressure(ctx):
    # Each request needs about a quarter of the KV budget, so only a handful
    # run at once even when many slots are free. length_scale is not applied:
    # sizes are defined relative to the KV budget.
    total = max(2, ctx.kv_block_budget * KV_BLOCK_TOKENS // 4)
    specs = []
    for t in _poisson(ctx, _count(ctx, 64), ctx.rate):
        size = max(2, int(total * ctx.rng.uniform(0.9, 1.1)))
        prompt = max(1, int(size * 0.6))
        specs.append(_Spec(t, "long", prompt, max(1, size - prompt)))
    return [specs]


def _starvation_aging(ctx):
    # Long requests arrive first; short requests keep arriving at a higher rate,
    # so older long requests stay queued while fresh short ones are admitted.
    longs = _long_requests(ctx, _poisson(ctx, _count(ctx, 8, minimum=4), ctx.rate * 0.5))
    shorts = _short_requests(ctx, _poisson(ctx, _count(ctx, 96), ctx.rate * 3))
    return [longs, shorts]


def _repeated_priority_arrivals(ctx):
    # max_num_seqs background requests at t=0, then repeated bursts of short
    # high-priority requests, one burst every WAVE_PERIOD_S seconds.
    background = [_Spec(0.0, "background", *_long_shape(ctx)) for _ in range(ctx.max_num_seqs)]
    wave_size = _count(ctx, 8, minimum=2)
    waves = []
    for wave in range(_count(ctx, 8, minimum=2)):
        base = (wave + 1) * WAVE_PERIOD_S
        for j in range(wave_size):
            waves.append(_Spec(round(base + j * WAVE_SPACING_S, 6), "burst", *_short_shape(ctx)))
    return [background, waves]


_BUILDERS = {
    "no_contention": _no_contention,
    "slot_saturation": _slot_saturation,
    "mixed_prefill_decode": _mixed_prefill_decode,
    "short_vs_long": _short_vs_long,
    "long_prefill_pressure": _long_prefill_pressure,
    "kv_pressure": _kv_pressure,
    "starvation_aging": _starvation_aging,
    "repeated_priority_arrivals": _repeated_priority_arrivals,
}


# ---------------------------------------------------------------- public API

def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_trace_id(params: dict, requests: list) -> str:
    return hashlib.sha256(canonical_json({"params": params, "requests": requests})).hexdigest()


def build_trace(scenario, profile, max_num_seqs, seed, max_num_batched_tokens=None,
                kv_block_budget=None, load_factor=None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario!r}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile!r}")
    if not isinstance(max_num_seqs, int) or isinstance(max_num_seqs, bool) or max_num_seqs < 1:
        raise ValueError(f"max_num_seqs must be a positive int, got {max_num_seqs!r}")
    prof = PROFILES[profile]
    load = prof["load_factors"][0] if load_factor is None else float(load_factor)
    if load <= 0:
        raise ValueError("load_factor must be positive")
    budget = DEFAULT_MAX_NUM_BATCHED_TOKENS if max_num_batched_tokens is None else int(max_num_batched_tokens)
    kv_blocks = DEFAULT_KV_BLOCK_BUDGET if kv_block_budget is None else int(kv_block_budget)
    ctx = _Ctx(
        rng=random.Random(seed),
        count_scale=prof["count_scale"],
        length_scale=prof["length_scale"],
        rate=prof["arrival_rate_rps"] * load,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        kv_block_budget=kv_blocks,
    )
    specs = _merge(_BUILDERS[scenario](ctx))
    requests = []
    for i, spec in enumerate(specs):
        requests.append({
            "request_id": f"{scenario}-{i:05d}",
            "arrival_s": round(spec.arrival_s, 6),
            "prompt_token_ids": [ctx.rng.randint(TOKEN_ID_LOW, TOKEN_ID_HIGH - 1) for _ in range(spec.prompt_len)],
            "max_tokens": spec.max_tokens,
            "group": spec.group,
        })
    params = {
        "scenario": scenario,
        "profile": profile,
        "max_num_seqs": max_num_seqs,
        "seed": seed,
        "max_num_batched_tokens": budget,
        "kv_block_budget": kv_blocks,
        "load_factor": load,
    }
    trace = {"trace_id": compute_trace_id(params, requests)}
    trace.update(params)
    trace["requests"] = requests
    return trace


def save_trace(trace: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=1)
        f.write("\n")


def load_trace(path) -> dict:
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    params = {key: trace[key] for key in PARAM_KEYS}
    if compute_trace_id(params, trace["requests"]) != trace["trace_id"]:
        raise ValueError(f"trace_id does not match contents: {path}")
    return trace
