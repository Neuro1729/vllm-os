"""Orchestrate many trials. Each trial is a separate `python -m benchmarks.run_trial` process.

    python -m benchmarks.run_matrix --output-dir RUN1 --model PATH \
        --nanovllm-root-v0 PATH_TO_E1DA1F5 --max-num-seqs-list 8,32 --profile standard

Layout under --output-dir (must not exist):
    traces/<scenario>/msn<N>/rep<k>.json            one trace per (scenario, N, repeat)
    raw/<scenario>/msn<N>/<variant>/rep<k>/         measured trial outputs
    raw/warmup/<scenario>/msn<N>/<variant>/rep<k>/  untimed warmup outputs (skipped by aggregate)
    logs/raw/...                                    stdout.log / stderr.log for each run
    plan.json, configuration.json, summaries/...

A failed or timed-out run keeps its logs and gets a failed.json; the loop continues
unless --fail-fast is given. --plan-only writes traces and plan.json, then exits
without starting any subprocess.
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks import aggregate
from benchmarks.run_trial import VARIANTS
from benchmarks.workloads import (DEFAULT_MAX_NUM_BATCHED_TOKENS, PROFILES, SCENARIOS,
                                  build_trace, save_trace)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBPROCESS_GRACE_S = 900.0  # model load and teardown come on top of the replay budget
PLAN_FILE = "plan.json"
CONFIG_FILE = "configuration.json"


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def _positive_float(text):
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def _csv_items(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _scenario_list(text):
    names = _csv_items(text)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown scenario(s): {unknown or text!r}; choose from {', '.join(SCENARIOS)}")
    return names


def _variant_list(text):
    names = _csv_items(text)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"unknown variant(s): {unknown or text!r}; choose from {', '.join(VARIANTS)}")
    return names


def _msn_list(text):
    values = [_positive_int(part) for part in _csv_items(text)]
    if not values:
        raise argparse.ArgumentTypeError("--max-num-seqs-list is empty")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"--max-num-seqs-list has duplicates: {text}")
    return values


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m benchmarks.run_matrix",
                                     description="Run every scenario x max_num_seqs x variant x repeat.")
    parser.add_argument("--output-dir", required=True, help="New directory for this run. Must not exist.")
    parser.add_argument("--model", default=None, help="Model directory. Required unless --plan-only.")
    parser.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS), type=_scenario_list)
    parser.add_argument("--max-num-seqs-list", default="32", type=_msn_list)
    parser.add_argument("--variants", default=",".join(VARIANTS), type=_variant_list)
    parser.add_argument("--repeats", default=None, type=_positive_int,
                        help="Measured repeats. Defaults to the profile's value.")
    parser.add_argument("--warmup", default=0, type=_nonnegative_int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-num-batched-tokens", default=DEFAULT_MAX_NUM_BATCHED_TOKENS, type=_positive_int)
    parser.add_argument("--kv-block-budget", default=None, type=_positive_int,
                        help="KV blocks assumed by kv_pressure traces. Set it to the engine's kv_blocks_total.")
    parser.add_argument("--max-model-len", default=4096, type=_positive_int)
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--nanovllm-root-v0", default=None,
                        help="Checkout of the original nano-vLLM (commit e1da1f5). Required unless --skip-v0.")
    parser.add_argument("--timeout-s", default=1800.0, type=_positive_float)
    parser.add_argument("--skip-v0", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.skip_v0:
        args.variants = [v for v in args.variants if v != "v0"]
        if not args.variants:
            parser.error("--skip-v0 removes every requested variant")
    elif args.nanovllm_root_v0 is None:
        parser.error("--nanovllm-root-v0 is required unless --skip-v0 is given")
    elif not Path(args.nanovllm_root_v0).is_dir():
        parser.error(f"--nanovllm-root-v0 is not a directory: {args.nanovllm_root_v0}")
    if not args.plan_only and args.model is None:
        parser.error("--model is required unless --plan-only is given")
    if Path(args.output_dir).exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")
    if args.repeats is None:
        args.repeats = PROFILES[args.profile]["repeats"]
    return args


# ---------------------------------------------------------------- plan

def _rel(path, root):
    return path.relative_to(root).as_posix()


def _trace_jobs(args, out):
    """(scenario, N, rep) -> trace file. One trace per rep, shared by every variant."""
    traces = {}
    reps = max(args.repeats, args.warmup)
    for scenario in args.scenarios:
        for msn in args.max_num_seqs_list:
            for rep in range(reps):
                path = out / "traces" / scenario / f"msn{msn}" / f"rep{rep}.json"
                trace = build_trace(scenario, args.profile, msn, args.seed + rep,
                                    max_num_batched_tokens=args.max_num_batched_tokens,
                                    kv_block_budget=args.kv_block_budget)
                save_trace(trace, path)
                traces[(scenario, msn, rep)] = (path, trace["trace_id"])
    return traces


def _entry(args, out, traces, kind, scenario, msn, variant, rep):
    base = Path("raw") if kind == "measured" else Path("raw") / "warmup"
    rel_dir = base / scenario / f"msn{msn}" / variant / f"rep{rep}"
    trace_path, trace_id = traces[(scenario, msn, rep)]
    out_dir = out / rel_dir
    command = [
        sys.executable, "-m", "benchmarks.run_trial",
        "--variant", variant,
        "--model", str(args.model),
        "--trace", str(trace_path),
        "--out-dir", str(out_dir),
        "--max-num-seqs", str(msn),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--timeout-s", str(args.timeout_s),
        "--repeat-index", str(rep),
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if variant == "v0":
        command += ["--nanovllm-root", str(Path(args.nanovllm_root_v0).resolve())]
    return {
        "kind": kind,
        "scenario": scenario,
        "max_num_seqs": msn,
        "variant": variant,
        "repeat": rep,
        "trace": _rel(trace_path, out),
        "trace_id": trace_id,
        "out_dir": rel_dir.as_posix(),
        "log_dir": (Path("logs") / rel_dir).as_posix(),
        "command": command,
    }


def build_plan(args, out, traces):
    trials = []
    for scenario in args.scenarios:
        for msn in args.max_num_seqs_list:
            for rep in range(args.warmup):
                for variant in args.variants:
                    trials.append(_entry(args, out, traces, "warmup", scenario, msn, variant, rep))
            for rep in range(args.repeats):
                for variant in args.variants:
                    trials.append(_entry(args, out, traces, "measured", scenario, msn, variant, rep))
    counts = {
        "measured": sum(t["kind"] == "measured" for t in trials),
        "warmup": sum(t["kind"] == "warmup" for t in trials),
        "total": len(trials),
    }
    return {"profile": args.profile, "counts": counts, "trials": trials}


# ---------------------------------------------------------------- execution

def _write_failed(out_dir, entry, status, returncode, error):
    payload = {
        "status": status,
        "returncode": returncode,
        "error": error,
        "scenario": entry["scenario"],
        "max_num_seqs": entry["max_num_seqs"],
        "variant": entry["variant"],
        "repeat": entry["repeat"],
        "kind": entry["kind"],
        "trace_id": entry["trace_id"],
    }
    with open(out_dir / "failed.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def run_entry(entry, out, timeout_s):
    """Run one trial subprocess. Returns True only if trial.json was written and the exit code was 0."""
    out_dir = out / entry["out_dir"]
    log_dir = out / entry["log_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    returncode = None
    status = "failed"
    error = None
    with open(log_dir / "stdout.log", "wb") as stdout, open(log_dir / "stderr.log", "wb") as stderr:
        try:
            proc = subprocess.run(entry["command"], cwd=str(REPO_ROOT), stdout=stdout, stderr=stderr,
                                  timeout=timeout_s + SUBPROCESS_GRACE_S, check=False)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
            error = f"subprocess exceeded {timeout_s + SUBPROCESS_GRACE_S:.0f}s and was killed"
    ok = returncode == 0 and (out_dir / "trial.json").exists()
    if not ok:
        if error is None:
            error = f"subprocess exited with code {returncode}; trial.json present: {(out_dir / 'trial.json').exists()}"
        _write_failed(out_dir, entry, status, returncode, error)
    return ok


def git_head():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True,
                              text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and head else None


def build_configuration(args, argv, started, finished, counts):
    return {
        "synthetic_fixture": False,
        "command_line": list(argv) if argv is not None else sys.argv[1:],
        "output_dir": str(Path(args.output_dir).resolve()),
        "model": args.model,
        "profile": args.profile,
        "scenarios": args.scenarios,
        "max_num_seqs_list": args.max_num_seqs_list,
        "variants": args.variants,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seed": args.seed,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_block_budget": args.kv_block_budget,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "timeout_s": args.timeout_s,
        "nanovllm_root_v0": args.nanovllm_root_v0,
        "skip_v0": args.skip_v0,
        "fail_fast": args.fail_fast,
        "git_head": git_head(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "started_at": started,
        "finished_at": finished,
        "trial_counts": counts,
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True)
    traces = _trace_jobs(args, out)
    plan = build_plan(args, out, traces)
    _write_json(out / PLAN_FILE, plan)
    if args.plan_only:
        print(f"wrote {plan['counts']['total']} planned trials to {out / PLAN_FILE}")
        return 0

    started = _now()
    failures = 0
    aborted = False
    for entry in plan["trials"]:
        ok = run_entry(entry, out, args.timeout_s)
        if not ok:
            failures += 1
            print(f"FAILED: {entry['kind']} {entry['out_dir']}", file=sys.stderr)
            if args.fail_fast:
                aborted = True
                break
    finished = _now()
    counts = dict(plan["counts"], failed=failures, aborted=aborted)
    _write_json(out / CONFIG_FILE, build_configuration(args, argv, started, finished, counts))
    if aborted:
        return 1
    aggregate.main([str(out)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
