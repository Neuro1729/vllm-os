"""Run ONE benchmark trial in ONE process.

    python -m benchmarks.run_trial --variant fcfs --model PATH --trace T.json \
        --out-dir DIR --max-num-seqs 32 --max-num-batched-tokens 2048 \
        --max-model-len 4096 --gpu-memory-utilization 0.9

Writes into --out-dir: trial.json, requests.csv, scheduler_events.csv.
Refuses (exit 2) when trial.json already exists. Never deletes files.
Exit 0 for status ok or timeout, 1 for failed, 2 for bad arguments.

nanovllm is imported only inside functions, after --nanovllm-root has been put
at sys.path[0]. That keeps --help and the CPU tests free of torch kernels.
"""
import argparse
import atexit
import csv
import functools
import gc
import json
import sys
import time
import traceback
from pathlib import Path

from benchmarks import metrics, replay
from benchmarks.workloads import load_trace

VARIANTS = ("v0", "fcfs", "mlfq", "mlfq_preempt")
EVENT_ROW_LIMIT = 2_000_000
EVENT_COLUMNS = ("step", "event", "seq_id", "level", "running", "waiting",
                 "free_blocks", "reason", "tokens", "cached_blocks")
REQUEST_COLUMNS = ("request_id", "group", "arrival_s", "arrival_time", "admit_time",
                   "first_token_time", "finish_time", "n_tokens", "status",
                   "preemptions", "token_times")
TRIAL_JSON = "trial.json"
REQUESTS_CSV = "requests.csv"
EVENTS_CSV = "scheduler_events.csv"


def variant_kwargs(variant):
    """Scheduling kwargs for LLM(). V0 must receive none of them."""
    if variant == "v0":
        return {}
    if variant == "fcfs":
        return {"scheduling_policy": "fcfs"}
    if variant == "mlfq":
        return {"scheduling_policy": "mlfq", "enable_mlfq_preemption": False}
    if variant == "mlfq_preempt":
        return {"scheduling_policy": "mlfq", "enable_mlfq_preemption": True}
    raise ValueError(f"unknown variant: {variant!r}")


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run_trial",
                                     description="Run one benchmark trial in one process.")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--model", required=True, help="Model directory on disk.")
    parser.add_argument("--trace", required=True, help="Trace JSON from benchmarks.workloads.")
    parser.add_argument("--out-dir", required=True, help="Directory for trial outputs (created if missing).")
    parser.add_argument("--max-num-seqs", required=True, type=_positive_int)
    parser.add_argument("--max-num-batched-tokens", required=True, type=_positive_int)
    parser.add_argument("--max-model-len", required=True, type=_positive_int)
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--nanovllm-root", default=None,
                        help="Checkout to import nanovllm from (used for the V0 baseline).")
    parser.add_argument("--timeout-s", type=_positive_float, default=1800.0,
                        help="Replay budget in seconds, measured from the first step.")
    parser.add_argument("--repeat-index", type=int, default=0)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------- output files

class EventWriter:
    """Streams scheduler events to CSV. Stops after EVENT_ROW_LIMIT rows."""

    def __init__(self, path):
        self.rows = 0
        self.truncated = False
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(EVENT_COLUMNS)

    def __call__(self, event):
        if self.rows >= EVENT_ROW_LIMIT:
            self.truncated = True
            return
        self._writer.writerow([event.get(column) for column in EVENT_COLUMNS])
        self.rows += 1

    def close(self):
        self._fh.close()


def write_requests_csv(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in REQUEST_COLUMNS if key != "token_times"}
            row["token_times"] = json.dumps(record.get("token_times") or [])
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def write_trial_json(path, trial):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trial, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------- engine glue

def build_engine(args, event_sink):
    """Real engine factory. LLM is imported here, never at module level."""
    from nanovllm import LLM

    kwargs = {
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    kwargs.update(variant_kwargs(args.variant))
    if args.variant != "v0":
        kwargs["event_sink"] = event_sink
    return LLM(args.model, **kwargs)


def real_sampling_params(max_tokens):
    from nanovllm import SamplingParams

    return SamplingParams(temperature=0.6, max_tokens=max_tokens, ignore_eos=True)


def probe_device():
    """Device facts from torch.cuda. Fields stay None when CUDA is unavailable."""
    info = {"name": None, "total_memory_bytes": None, "peak_allocated_bytes": None}
    try:
        import torch

        if not torch.cuda.is_available():
            return info
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["total_memory_bytes"] = props.total_memory
        info["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    except Exception:
        pass
    return info


def release_engine(engine):
    """Shut the engine down and return cached GPU memory."""
    if engine is not None:
        exit_fn = getattr(engine, "exit", None)
        if callable(exit_fn):
            try:
                atexit.unregister(exit_fn)  # LLM registers exit() at import time
            except Exception:
                pass
            exit_fn()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def engine_snapshot(engine):
    """Scheduler counters after the run. Missing pieces are None, never 0."""
    snap = {"kv_blocks_total": None, "scheduler_stats": None, "saturation": None}
    if engine is None:
        return snap
    scheduler = getattr(engine, "scheduler", None)
    if scheduler is None:
        return snap
    try:
        snap["kv_blocks_total"] = len(scheduler.block_manager.blocks)
    except AttributeError:
        pass
    stats = getattr(scheduler, "stats", None)
    if stats is not None:
        stats_dict = stats.to_dict()
        snap["scheduler_stats"] = stats_dict
        snap["saturation"] = {
            "slot_limit_hits": stats_dict.get("slot_limit_hits"),
            "kv_limit_hits": stats_dict.get("kv_limit_hits"),
            "token_budget_hits": stats_dict.get("token_budget_hits"),
        }
    return snap


# ---------------------------------------------------------------- trial driver

def _new_trial(args):
    return {
        "variant": args.variant,
        "model": args.model,
        "trace_id": None,
        "scenario": None,
        "profile": None,
        "seed": None,
        "repeat_index": args.repeat_index,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "device": None,
        "kv_blocks_total": None,
        "status": "failed",
        "error": None,
        "wall_time_s": None,
        "steps": None,
        "max_running": None,
        "max_concurrent_active": None,
        "saturation": None,
        "scheduler_stats": None,
        "request_summary": None,
        "events_written": 0,
        "events_truncated": False,
    }


def execute(args, out_dir, engine_factory, make_params, clock=time.perf_counter,
            sleep=time.sleep, device_probe=probe_device):
    """Run one trial. Always writes requests.csv and trial.json.

    engine_factory(event_sink) returns an engine. Injected so CPU tests can use
    a fake engine. make_params(max_tokens) returns sampling parameters.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial = _new_trial(args)
    events = EventWriter(out_dir / EVENTS_CSV)
    event_sink = events if args.variant != "v0" else None
    started = clock()
    records = []
    engine = None
    replayer = None
    try:
        trace = load_trace(args.trace)
        trial.update(trace_id=trace["trace_id"], scenario=trace["scenario"],
                     profile=trace["profile"], seed=trace["seed"])
        records = replay.new_records(trace["requests"])
        engine = engine_factory(event_sink)
        replayer = replay.Replay(engine, trace["requests"], make_params, args.timeout_s,
                                 clock=clock, sleep=sleep)
        replayer.run()
        trial["status"] = "timeout" if replayer.timed_out else "ok"
    except Exception:
        trial["status"] = "failed"
        trial["error"] = traceback.format_exc()
    finally:
        trial["wall_time_s"] = clock() - started
        if replayer is not None:
            records = replayer.records
            trial.update(steps=replayer.steps, max_running=replayer.max_running,
                         max_concurrent_active=replayer.max_concurrent_active)
            if replayer.wall_time_s:
                trial["wall_time_s"] = replayer.wall_time_s
        trial.update(engine_snapshot(engine))
        trial["device"] = device_probe()
        try:
            release_engine(engine)
        except Exception:
            trial["error"] = (trial["error"] or "") + "\nengine cleanup failed:\n" + traceback.format_exc()
        engine = None
        if replayer is not None:
            replayer.engine = None
        events.close()
        trial["events_written"] = events.rows
        trial["events_truncated"] = events.truncated
        trial["request_summary"] = metrics.summarize_requests(records, trial["wall_time_s"])
        write_requests_csv(out_dir / REQUESTS_CSV, records)
        write_trial_json(out_dir / TRIAL_JSON, trial)
    return 1 if trial["status"] == "failed" else 0


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if (out_dir / TRIAL_JSON).exists():
        print(f"refusing to overwrite existing {out_dir / TRIAL_JSON}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.nanovllm_root:
        sys.path.insert(0, str(Path(args.nanovllm_root).resolve()))
    engine_factory = functools.partial(build_engine, args)
    return execute(args, out_dir, engine_factory, real_sampling_params)


if __name__ == "__main__":
    sys.exit(main())
