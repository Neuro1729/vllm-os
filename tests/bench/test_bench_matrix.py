import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import run_matrix
from benchmarks.workloads import load_trace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_argv(tmp_path, *extra):
    return ["--output-dir", str(tmp_path / "run"), "--model", "fake-model",
            "--nanovllm-root-v0", str(tmp_path), *extra]


def _no_subprocess(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError(f"subprocess must not run: {args}")

    monkeypatch.setattr(subprocess, "run", boom)


# ---------------------------------------------------------------- CLI parsing

def test_valid_arguments_parse(tmp_path):
    args = run_matrix.parse_args(_base_argv(tmp_path, "--scenarios", "no_contention,kv_pressure",
                                            "--max-num-seqs-list", "8,32", "--variants", "fcfs,mlfq",
                                            "--repeats", "2", "--warmup", "1"))
    assert args.scenarios == ["no_contention", "kv_pressure"]
    assert args.max_num_seqs_list == [8, 32]
    assert args.variants == ["fcfs", "mlfq"]
    assert args.repeats == 2 and args.warmup == 1


def test_repeats_default_comes_from_profile(tmp_path):
    args = run_matrix.parse_args(_base_argv(tmp_path, "--profile", "thorough"))
    assert args.repeats == 5


@pytest.mark.parametrize("extra", [
    ["--scenarios", "not_a_scenario"],
    ["--variants", "fcfs,bogus"],
    ["--repeats", "0"],
    ["--warmup", "-1"],
    ["--max-num-seqs-list", "8,8"],
    ["--max-num-seqs-list", "0"],
    ["--max-num-seqs-list", "4,x"],
    ["--profile", "gigantic"],
])
def test_invalid_arguments_exit_2(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        run_matrix.parse_args(_base_argv(tmp_path, *extra))
    assert exc.value.code == 2


def test_missing_nanovllm_root_without_skip_v0_exits_2(tmp_path):
    argv = ["--output-dir", str(tmp_path / "run"), "--model", "m"]
    with pytest.raises(SystemExit) as exc:
        run_matrix.parse_args(argv)
    assert exc.value.code == 2


def test_skip_v0_drops_v0_and_needs_no_root(tmp_path):
    argv = ["--output-dir", str(tmp_path / "run"), "--model", "m", "--skip-v0"]
    args = run_matrix.parse_args(argv)
    assert "v0" not in args.variants
    assert args.nanovllm_root_v0 is None


def test_existing_output_dir_exits_2(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(SystemExit) as exc:
        run_matrix.parse_args(_base_argv(tmp_path))
    assert exc.value.code == 2


def test_model_required_unless_plan_only(tmp_path):
    argv = ["--output-dir", str(tmp_path / "run"), "--skip-v0"]
    with pytest.raises(SystemExit) as exc:
        run_matrix.parse_args(argv)
    assert exc.value.code == 2
    args = run_matrix.parse_args(argv + ["--plan-only"])
    assert args.plan_only is True


def test_cli_help_exits_zero():
    result = subprocess.run([sys.executable, "-m", "benchmarks.run_matrix", "--help"], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "--plan-only" in result.stdout


def test_cli_bad_scenario_exits_2_from_shell():
    result = subprocess.run([sys.executable, "-m", "benchmarks.run_matrix", "--output-dir", "unused-x",
                             "--scenarios", "nope"], cwd=str(REPO_ROOT), capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 2


# ---------------------------------------------------------------- plan-only

def test_plan_only_writes_plan_and_launches_nothing(tmp_path, monkeypatch):
    _no_subprocess(monkeypatch)
    scenarios = "no_contention,slot_saturation"
    code = run_matrix.main(_base_argv(tmp_path, "--scenarios", scenarios, "--max-num-seqs-list", "4,8",
                                      "--repeats", "2", "--plan-only"))
    assert code == 0
    out = tmp_path / "run"
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    expected = 2 * 2 * 4 * 2  # scenarios x max_num_seqs x variants x repeats
    assert plan["counts"]["measured"] == expected
    assert plan["counts"]["warmup"] == 0
    assert len(plan["trials"]) == expected
    assert not (out / "configuration.json").exists()
    assert not (out / "raw").exists()


def test_plan_only_counts_warmup_separately(tmp_path, monkeypatch):
    _no_subprocess(monkeypatch)
    run_matrix.main(_base_argv(tmp_path, "--scenarios", "no_contention", "--max-num-seqs-list", "4",
                               "--repeats", "3", "--warmup", "1", "--plan-only"))
    plan = json.loads((tmp_path / "run" / "plan.json").read_text(encoding="utf-8"))
    assert plan["counts"] == {"measured": 12, "warmup": 4, "total": 16}
    warm = [t for t in plan["trials"] if t["kind"] == "warmup"]
    assert all("/raw/warmup/" in ("/" + t["out_dir"]) for t in warm)


def test_plan_only_traces_are_shared_across_variants(tmp_path, monkeypatch):
    _no_subprocess(monkeypatch)
    run_matrix.main(_base_argv(tmp_path, "--scenarios", "kv_pressure", "--max-num-seqs-list", "8",
                               "--repeats", "2", "--plan-only"))
    plan = json.loads((tmp_path / "run" / "plan.json").read_text(encoding="utf-8"))
    by_rep = {}
    for trial in plan["trials"]:
        by_rep.setdefault(trial["repeat"], set()).add(trial["trace"])
    assert all(len(paths) == 1 for paths in by_rep.values())
    for rel in {t["trace"] for t in plan["trials"]}:
        trace = load_trace(tmp_path / "run" / rel)
        assert trace["scenario"] == "kv_pressure" and trace["max_num_seqs"] == 8


def test_v0_command_gets_nanovllm_root_and_others_do_not(tmp_path, monkeypatch):
    _no_subprocess(monkeypatch)
    run_matrix.main(_base_argv(tmp_path, "--scenarios", "no_contention", "--max-num-seqs-list", "4",
                               "--repeats", "1", "--plan-only"))
    plan = json.loads((tmp_path / "run" / "plan.json").read_text(encoding="utf-8"))
    for trial in plan["trials"]:
        has_root = "--nanovllm-root" in trial["command"]
        assert has_root == (trial["variant"] == "v0")


# ---------------------------------------------------------------- execution with a stubbed subprocess

def _stub_subprocess(monkeypatch, behaviour):
    """behaviour(command) returns a CompletedProcess-like result or raises. git calls return code 128."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 128, stdout="", stderr="")
        return behaviour(command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_failed_and_timed_out_runs_keep_logs_and_loop_continues(tmp_path, monkeypatch):
    def behaviour(command):
        variant = command[command.index("--variant") + 1]
        if variant == "fcfs":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=b"boom")

    calls = _stub_subprocess(monkeypatch, behaviour)
    code = run_matrix.main(_base_argv(tmp_path, "--scenarios", "no_contention", "--max-num-seqs-list", "4",
                                      "--repeats", "1", "--variants", "fcfs,mlfq"))
    assert code == 0
    run_calls = [c for c in calls if c[0] != "git"]
    assert len(run_calls) == 2
    out = tmp_path / "run"
    fcfs_failed = json.loads((out / "raw/no_contention/msn4/fcfs/rep0/failed.json").read_text(encoding="utf-8"))
    assert fcfs_failed["status"] == "timeout"
    mlfq_failed = json.loads((out / "raw/no_contention/msn4/mlfq/rep0/failed.json").read_text(encoding="utf-8"))
    assert mlfq_failed["status"] == "failed" and mlfq_failed["returncode"] == 1
    assert (out / "logs/raw/no_contention/msn4/mlfq/rep0/stderr.log").exists()
    config = json.loads((out / "configuration.json").read_text(encoding="utf-8"))
    assert config["trial_counts"]["failed"] == 2
    assert config["git_head"] is None
    assert (out / "summaries" / "metrics_by_scenario.csv").exists()


def test_fail_fast_stops_after_first_failure(tmp_path, monkeypatch):
    calls = _stub_subprocess(monkeypatch, lambda command: subprocess.CompletedProcess(command, 3))
    code = run_matrix.main(_base_argv(tmp_path, "--scenarios", "no_contention", "--max-num-seqs-list", "4",
                                      "--repeats", "2", "--variants", "fcfs,mlfq", "--fail-fast"))
    assert code == 1
    assert len([c for c in calls if c[0] != "git"]) == 1
    config = json.loads((tmp_path / "run" / "configuration.json").read_text(encoding="utf-8"))
    assert config["trial_counts"]["aborted"] is True


def test_successful_run_writes_trial_json_then_aggregates(tmp_path, monkeypatch):
    def behaviour(command):
        out_dir = Path(command[command.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        trial = {"status": "ok", "scenario": "no_contention", "max_num_seqs": 4, "variant": "fcfs",
                 "repeat_index": 0, "trace_id": "stub", "device": None, "wall_time_s": 1.0}
        (out_dir / "trial.json").write_text(json.dumps(trial), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    _stub_subprocess(monkeypatch, behaviour)
    code = run_matrix.main(_base_argv(tmp_path, "--scenarios", "no_contention", "--max-num-seqs-list", "4",
                                      "--repeats", "1", "--variants", "fcfs"))
    assert code == 0
    assert not (tmp_path / "run" / "raw/no_contention/msn4/fcfs/rep0/failed.json").exists()
    assert (tmp_path / "run" / "summaries" / "improvements_vs_fcfs.csv").exists()
