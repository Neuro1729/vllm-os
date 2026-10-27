"""Shared CPU-only fixtures for the benchmark tests. Nothing here touches CUDA.

FakeEngine mimics the scheduler-facing surface the replay loop uses. FakeClock
replaces wall time so arrivals, token timestamps and timeouts are exact.
write_fixture_run builds a synthetic results tree with the real writers from
benchmarks.run_trial. Its numbers are invented and the tests check that the
report says so.
"""
import csv
import json
import random
from dataclasses import dataclass
from enum import Enum
from itertools import count
from types import SimpleNamespace

from benchmarks import run_trial
from benchmarks.workloads import PARAM_KEYS, build_trace, compute_trace_id, save_trace


class FakeStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3
    CANCELLED = 4
    REJECTED = 5


@dataclass
class FakeSeq:
    seq_id: int
    max_tokens: int
    status: FakeStatus = FakeStatus.WAITING
    num_completion_tokens: int = 0
    feedback: object = None


class FakeClock:
    """Simulated time. Used as both the clock and the sleep function."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.now += seconds


class FakeEngine:
    """Round-robin fake. Each step admits waiting work into free slots and gives every running seq one token.

    v0_style=True makes add_request return None (as V0 does) and drops feedback.
    never_finish=True keeps sequences running forever (for timeout tests).
    """

    def __init__(self, clock, *, slots=2, step_time=0.1, never_finish=False, v0_style=False,
                 reject_prompts_longer_than=None, event_sink=None, kv_blocks=8):
        self.clock = clock
        self.slots = slots
        self.step_time = step_time
        self.never_finish = never_finish
        self.v0_style = v0_style
        self.reject_prompts_longer_than = reject_prompts_longer_than
        self.event_sink = event_sink
        self.steps = 0
        self._ids = count()
        stats = None if v0_style else SimpleNamespace(
            to_dict=lambda: {"slot_limit_hits": 3, "kv_limit_hits": 1, "token_budget_hits": 0, "steps": self.steps})
        self.scheduler = SimpleNamespace(
            waiting=[], running=[], stats=stats,
            block_manager=SimpleNamespace(blocks=[None] * kv_blocks))

    def _emit(self, event, seq):
        if self.event_sink is None:
            return
        self.event_sink({"step": self.steps, "event": event, "seq_id": seq.seq_id, "level": 0,
                         "running": len(self.scheduler.running), "waiting": len(self.scheduler.waiting),
                         "free_blocks": len(self.scheduler.block_manager.blocks)})

    def add_request(self, prompt_token_ids, params):
        seq = FakeSeq(next(self._ids), params.max_tokens,
                      feedback=None if self.v0_style else SimpleNamespace(preemptions=0))
        if self.reject_prompts_longer_than is not None and len(prompt_token_ids) > self.reject_prompts_longer_than:
            seq.status = FakeStatus.REJECTED
            return seq
        self.scheduler.waiting.append(seq)
        self._emit("arrive", seq)
        return None if self.v0_style else seq

    def step(self):
        self.steps += 1
        while self.scheduler.waiting and len(self.scheduler.running) < self.slots:
            seq = self.scheduler.waiting.pop(0)
            seq.status = FakeStatus.RUNNING
            self.scheduler.running.append(seq)
            self._emit("admit", seq)
        for seq in list(self.scheduler.running):
            seq.num_completion_tokens += 1
            if not self.never_finish and seq.num_completion_tokens >= seq.max_tokens:
                seq.status = FakeStatus.FINISHED
                self.scheduler.running.remove(seq)
                self._emit("finish", seq)
        self.clock.advance(self.step_time)

    def is_finished(self):
        return not self.scheduler.waiting and not self.scheduler.running


def fake_params(max_tokens):
    return SimpleNamespace(max_tokens=max_tokens, temperature=0.6, ignore_eos=True)


def fake_device():
    return {"name": None, "total_memory_bytes": None, "peak_allocated_bytes": None}


def make_trace_file(tmp_path, requests=None, scenario="no_contention", max_num_seqs=2, seed=0):
    """Save a real trace. If `requests` is given, replace the generated requests (for hand-made timing tests)."""
    trace = build_trace(scenario, "smoke", max_num_seqs, seed)
    if requests is not None:
        trace["requests"] = requests
        trace["trace_id"] = compute_trace_id({k: trace[k] for k in PARAM_KEYS}, requests)
    path = tmp_path / "trace.json"
    save_trace(trace, path)
    return path


def trial_args(trace_path, out_dir, variant="fcfs", extra=()):
    argv = ["--variant", variant, "--model", "fake-model", "--trace", str(trace_path),
            "--out-dir", str(out_dir), "--max-num-seqs", "2", "--max-num-batched-tokens", "256",
            "--max-model-len", "512", "--gpu-memory-utilization", "0.5", *extra]
    return run_trial.parse_args(argv)


# ---------------------------------------------------------------- synthetic results tree

VARIANT_SPEED = {"v0": 1.10, "fcfs": 1.00, "mlfq": 0.90, "mlfq_preempt": 0.95}


def _fixture_records(rng, scenario, variant, n_requests, zero_preemptions, drop_first_token):
    records = []
    t = 0.0
    speed = VARIANT_SPEED[variant]
    for i in range(n_requests):
        t += rng.expovariate(4.0)
        group = "short" if i % 2 == 0 else "long"
        wait = rng.uniform(0.01, 0.2) * speed
        first = t + wait + rng.uniform(0.05, 0.3) * speed
        itl = rng.uniform(0.02, 0.05) * speed
        n_tokens = rng.randint(8, 16) if group == "short" else rng.randint(32, 64)
        token_times = [round(first + k * itl, 6) for k in range(n_tokens)]
        if drop_first_token:
            token_times = []
        if variant == "v0":
            preemptions = None
        else:
            preemptions = 0 if (variant == "mlfq_preempt" and zero_preemptions) else rng.randint(1, 3)
        records.append({
            "request_id": f"{scenario}-{i:05d}",
            "group": group,
            "arrival_s": round(t, 6),
            "arrival_time": round(t + 0.001, 6),
            "admit_time": round(t + wait, 6),
            "first_token_time": None if drop_first_token else round(first, 6),
            "finish_time": round(token_times[-1], 6) if token_times else round(first + n_tokens * itl, 6),
            "n_tokens": n_tokens,
            "status": "finished",
            "preemptions": preemptions,
            "token_times": token_times,
        })
    return records


def _write_events(path, variant, kv_blocks):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(run_trial.EVENT_COLUMNS)
        if variant == "v0":
            return
        for step, free in enumerate([kv_blocks, kv_blocks - 3, kv_blocks - 6, kv_blocks - 2]):
            writer.writerow([step, "admit", step, 0, 2, 3, free, "", "", ""])


def write_fixture_run(root, scenarios=("no_contention", "slot_saturation"), msns=(4, 8), repeats=2,
                      variants=("v0", "fcfs", "mlfq", "mlfq_preempt"), drop_first_token_for=None,
                      synthetic=True, seed=1234):
    """Create a results tree under `root`: configuration.json plus raw/<scenario>/msn<N>/<variant>/rep<k>/.

    drop_first_token_for: (variant, repeat) whose requests have no token timestamps, to exercise missing values.
    The first scenario's mlfq_preempt trials have zero preemptions, to exercise the "not exercised" sentence.
    """
    from pathlib import Path

    root = Path(root)
    rng = random.Random(seed)
    kv_blocks = 8
    config = {
        "synthetic_fixture": synthetic,
        "profile": "smoke",
        "scenarios": list(scenarios),
        "max_num_seqs_list": list(msns),
        "variants": list(variants),
        "repeats": repeats,
        "model": "fixture-model",
        "git_head": None,
    }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    for scenario in scenarios:
        for msn in msns:
            for variant in variants:
                for rep in range(repeats):
                    out = root / "raw" / scenario / f"msn{msn}" / variant / f"rep{rep}"
                    out.mkdir(parents=True, exist_ok=True)
                    drop = drop_first_token_for == (variant, rep)
                    records = _fixture_records(rng, scenario, variant, 6,
                                               zero_preemptions=(scenario == scenarios[0]), drop_first_token=drop)
                    run_trial.write_requests_csv(out / "requests.csv", records)
                    _write_events(out / run_trial.EVENTS_CSV, variant, kv_blocks)
                    trial = {
                        "variant": variant,
                        "model": "fixture-model",
                        "trace_id": f"fixture-{scenario}-{msn}-{variant}-{rep}",
                        "scenario": scenario,
                        "profile": "smoke",
                        "seed": rep,
                        "repeat_index": rep,
                        "max_num_seqs": msn,
                        "max_num_batched_tokens": 256,
                        "max_model_len": 512,
                        "gpu_memory_utilization": 0.5,
                        "enforce_eager": False,
                        "device": {"name": "fixture", "total_memory_bytes": 16 * 2 ** 30,
                                   "peak_allocated_bytes": int((4 + msn / 4 + rep * 0.1) * 2 ** 30)},
                        "kv_blocks_total": kv_blocks,
                        "status": "ok",
                        "error": None,
                        "wall_time_s": 10.0 + rep,
                        "steps": 40,
                        "max_running": 2,
                        "max_concurrent_active": 4,
                        "saturation": None if variant == "v0" else {
                            "slot_limit_hits": 3, "kv_limit_hits": 1, "token_budget_hits": 0},
                        "scheduler_stats": None,
                        "request_summary": None,
                        "events_written": 0 if variant == "v0" else 4,
                        "events_truncated": False,
                    }
                    run_trial.write_trial_json(out / "trial.json", trial)
    return root
