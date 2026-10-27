import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import bench_fixtures as fx
from benchmarks import replay, run_trial
from benchmarks.workloads import save_trace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(i, arrival, max_tokens=3, prompt_len=4):
    return {"request_id": f"r{i}", "arrival_s": arrival, "prompt_token_ids": [1000 + j for j in range(prompt_len)],
            "max_tokens": max_tokens, "group": "short"}


def _run(requests, engine_kwargs=None, timeout_s=50.0):
    clock = fx.FakeClock()
    engine = fx.FakeEngine(clock, **(engine_kwargs or {}))
    rp = replay.Replay(engine, requests, fx.fake_params, timeout_s, clock=clock, sleep=clock.sleep)
    rp.run()
    return rp, engine


def test_arrivals_are_honored():
    requests = [_request(0, 0.0), _request(1, 1.0), _request(2, 2.5)]
    rp, _ = _run(requests)
    for req, rec in zip(requests, rp.records):
        assert rec["arrival_time"] is not None
        assert rec["arrival_time"] >= req["arrival_s"] - 1e-9
        # Idle sleeps are at most 5 ms, so a request is added within one idle sleep or one step of its arrival.
        assert rec["arrival_time"] - req["arrival_s"] <= 0.11


def test_token_timestamps_recorded_at_end_of_step():
    rp, _ = _run([_request(0, 0.0, max_tokens=3)])
    rec = rp.records[0]
    assert rec["status"] == "finished"
    assert rec["n_tokens"] == 3
    assert len(rec["token_times"]) == 3
    assert rec["token_times"] == sorted(rec["token_times"])
    assert rec["first_token_time"] == rec["token_times"][0]
    assert rec["finish_time"] == rec["token_times"][-1]
    assert rec["admit_time"] is not None and rec["admit_time"] <= rec["first_token_time"]


def test_admit_time_waits_for_a_slot():
    requests = [_request(0, 0.0, max_tokens=5), _request(1, 0.0, max_tokens=5), _request(2, 0.0, max_tokens=5)]
    rp, _ = _run(requests, engine_kwargs={"slots": 2})
    assert rp.records[2]["admit_time"] > rp.records[0]["admit_time"]
    assert rp.max_running == 2
    assert rp.max_concurrent_active == 3


def test_unfinished_at_timeout_is_recorded():
    rp, _ = _run([_request(0, 0.0, max_tokens=5), _request(1, 5.0)], engine_kwargs={"never_finish": True},
                 timeout_s=1.0)
    assert rp.timed_out is True
    assert rp.records[0]["status"] == "unfinished"
    assert rp.records[0]["n_tokens"] >= 1
    assert rp.records[0]["token_times"]
    assert rp.records[1]["status"] == "unfinished"
    assert rp.records[1]["arrival_time"] is None


def test_v0_style_engine_uses_waiting_tail():
    rp, engine = _run([_request(0, 0.0, max_tokens=2)], engine_kwargs={"v0_style": True})
    rec = rp.records[0]
    assert rec["status"] == "finished"
    assert rec["preemptions"] is None  # V0 has no feedback state


def test_rejected_request_is_marked():
    rp, _ = _run([_request(0, 0.0, prompt_len=10), _request(1, 0.0, prompt_len=2)],
                 engine_kwargs={"reject_prompts_longer_than": 5})
    assert rp.records[0]["status"] == "rejected"
    assert rp.records[0]["n_tokens"] == 0
    assert rp.records[1]["status"] == "finished"


def test_preemptions_read_from_feedback():
    rp, engine = _run([_request(0, 0.0, max_tokens=2)])
    assert rp.records[0]["preemptions"] == 0


def _trace(tmp_path, requests):
    return fx.make_trace_file(tmp_path, requests=requests)


def test_execute_writes_all_outputs_on_success(tmp_path):
    requests = [_request(i, 0.1 * i, max_tokens=3) for i in range(4)]
    trace_path = _trace(tmp_path, requests)
    out = tmp_path / "out"
    args = fx.trial_args(trace_path, out, variant="mlfq")
    clock = fx.FakeClock()
    engine_holder = {}

    def factory(sink):
        engine_holder["e"] = fx.FakeEngine(clock, event_sink=sink)
        return engine_holder["e"]

    code = run_trial.execute(args, out, factory, fx.fake_params, clock=clock, sleep=clock.sleep,
                             device_probe=fx.fake_device)
    assert code == 0
    trial = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert trial["status"] == "ok" and trial["error"] is None
    assert trial["variant"] == "mlfq" and trial["max_num_seqs"] == 2
    assert trial["saturation"]["slot_limit_hits"] == 3
    assert trial["request_summary"]["counts"]["finished"] == 4
    with open(out / "requests.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and rows[0]["status"] == "finished"
    assert json.loads(rows[0]["token_times"]) and rows[0]["preemptions"] == "0"
    with open(out / "scheduler_events.csv", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    assert {e["event"] for e in events} >= {"arrive", "admit", "finish"}
    assert trial["events_written"] == len(events)
    assert engine_holder["e"] is not None


def test_execute_failure_writes_trial_json_with_traceback(tmp_path):
    trace_path = _trace(tmp_path, [_request(0, 0.0)])
    out = tmp_path / "out"
    args = fx.trial_args(trace_path, out, variant="fcfs")

    def factory(sink):
        raise RuntimeError("simulated OOM while loading weights")

    code = run_trial.execute(args, out, factory, fx.fake_params, device_probe=fx.fake_device)
    assert code == 1
    trial = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert trial["status"] == "failed"
    assert "simulated OOM" in trial["error"] and "RuntimeError" in trial["error"]
    with open(out / "requests.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "unfinished" and rows[0]["arrival_time"] == ""


def test_execute_timeout_exit_zero_and_unfinished(tmp_path):
    trace_path = _trace(tmp_path, [_request(0, 0.0, max_tokens=50)])
    out = tmp_path / "out"
    args = fx.trial_args(trace_path, out, variant="fcfs", extra=("--timeout-s", "0.5"))
    clock = fx.FakeClock()

    def factory(sink):
        return fx.FakeEngine(clock, never_finish=True, event_sink=sink)

    code = run_trial.execute(args, out, factory, fx.fake_params, clock=clock, sleep=clock.sleep,
                             device_probe=fx.fake_device)
    assert code == 0
    trial = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert trial["status"] == "timeout"
    assert trial["request_summary"]["counts"]["unfinished"] == 1


def test_v0_events_file_has_header_only(tmp_path):
    trace_path = _trace(tmp_path, [_request(0, 0.0, max_tokens=2)])
    out = tmp_path / "out"
    args = fx.trial_args(trace_path, out, variant="v0")
    clock = fx.FakeClock()
    code = run_trial.execute(args, out, lambda sink: fx.FakeEngine(clock, v0_style=True), fx.fake_params,
                             clock=clock, sleep=clock.sleep, device_probe=fx.fake_device)
    assert code == 0
    lines = (out / "scheduler_events.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines == [",".join(run_trial.EVENT_COLUMNS)]
    trial = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert trial["saturation"] is None and trial["scheduler_stats"] is None
    assert trial["kv_blocks_total"] == 8


def test_event_rows_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(run_trial, "EVENT_ROW_LIMIT", 3)
    writer = run_trial.EventWriter(tmp_path / "events.csv")
    for i in range(5):
        writer({"step": i, "event": "admit", "seq_id": i})
    writer.close()
    assert writer.rows == 3 and writer.truncated is True
    assert len((tmp_path / "events.csv").read_text(encoding="utf-8").strip().splitlines()) == 4


def test_variant_kwargs_mapping():
    assert run_trial.variant_kwargs("v0") == {}
    assert run_trial.variant_kwargs("fcfs") == {"scheduling_policy": "fcfs"}
    assert run_trial.variant_kwargs("mlfq") == {"scheduling_policy": "mlfq", "enable_mlfq_preemption": False}
    assert run_trial.variant_kwargs("mlfq_preempt") == {"scheduling_policy": "mlfq", "enable_mlfq_preemption": True}


def test_main_refuses_existing_trial_json(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "trial.json").write_text("{}", encoding="utf-8")
    trace_path = _trace(tmp_path, [_request(0, 0.0)])
    code = run_trial.main(["--variant", "fcfs", "--model", "m", "--trace", str(trace_path),
                           "--out-dir", str(out), "--max-num-seqs", "2", "--max-num-batched-tokens", "64",
                           "--max-model-len", "128", "--gpu-memory-utilization", "0.5"])
    assert code == 2
    assert (out / "trial.json").read_text(encoding="utf-8") == "{}"


def test_trace_file_round_trip_used_by_replay(tmp_path):
    from benchmarks.workloads import build_trace

    trace = build_trace("no_contention", "smoke", 2, seed=5)
    path = tmp_path / "t.json"
    save_trace(trace, path)
    assert json.loads(path.read_text(encoding="utf-8"))["trace_id"] == trace["trace_id"]


@pytest.mark.parametrize("argv", [
    ["--variant", "bogus"],
    ["--variant", "fcfs", "--unknown-flag", "1"],
])
def test_run_trial_bad_args_exit_2(argv):
    result = subprocess.run([sys.executable, "-m", "benchmarks.run_trial", *argv], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
