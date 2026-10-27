import csv
import json
from pathlib import Path

import pytest

import bench_fixtures as fx
from benchmarks import aggregate, generate_report

BANNER = "TEST FIXTURE DATA"
NOT_EXERCISED = "MLFQ preemption was not exercised; this run cannot establish its effect."
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


@pytest.fixture
def fixture_run(tmp_path):
    return fx.write_fixture_run(tmp_path / "run", drop_first_token_for=("mlfq", 0))


def test_aggregate_writes_expected_tables(fixture_run):
    written = aggregate.aggregate(fixture_run)
    assert written["trial_results"].exists() and written["requests"].exists()
    trials = json.loads(written["trial_results"].read_text(encoding="utf-8"))
    assert len(trials) == 2 * 2 * 4 * 2
    rows = _read_csv(written["metrics_by_scenario"])
    assert len(rows) == 2 * 2 * 4
    assert {r["variant"] for r in rows} == {"v0", "fcfs", "mlfq", "mlfq_preempt"}
    improvements = _read_csv(written["improvements"])
    assert {r["variant"] for r in improvements} == {"v0", "mlfq", "mlfq_preempt"}
    assert all(r["baseline_variant"] == "fcfs" for r in improvements)


def test_aggregate_is_deterministic(fixture_run):
    aggregate.aggregate(fixture_run)
    first = {p: p.read_bytes() for p in (fixture_run / "summaries").glob("*.csv")}
    first_raw = (fixture_run / "raw" / "requests.csv").read_bytes()
    aggregate.aggregate(fixture_run)
    second = {p: p.read_bytes() for p in (fixture_run / "summaries").glob("*.csv")}
    assert first == second
    assert first_raw == (fixture_run / "raw" / "requests.csv").read_bytes()


def test_aggregate_skips_warmup_and_counts_failed_only_dirs(tmp_path):
    root = fx.write_fixture_run(tmp_path / "run", scenarios=("no_contention",), msns=(4,), repeats=1)
    warm = root / "raw" / "warmup" / "no_contention" / "msn4" / "fcfs" / "rep0"
    warm.mkdir(parents=True)
    (warm / "trial.json").write_text("{not even json", encoding="utf-8")  # would crash if read
    failed = root / "raw" / "no_contention" / "msn4" / "mlfq" / "rep9"
    failed.mkdir(parents=True)
    (failed / "failed.json").write_text(json.dumps({
        "status": "timeout", "scenario": "no_contention", "max_num_seqs": 4, "variant": "mlfq", "repeat": 9,
        "returncode": None, "error": "killed"}), encoding="utf-8")
    written = aggregate.aggregate(root)
    trials = json.loads(written["trial_results"].read_text(encoding="utf-8"))
    assert all("warmup" not in t["path"] for t in trials)
    timed_out = [t for t in trials if t["status"] == "timeout"]
    assert len(timed_out) == 1 and timed_out[0]["failed_trials"] == 1
    assert timed_out[0]["ttft_p95_s"] is None


def test_report_has_fixture_banner_and_no_urls(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    out = tmp_path / "report"
    generate_report.generate(fixture_run, out)
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    html_text = (out / "REPORT.html").read_text(encoding="utf-8")
    assert BANNER in md
    assert BANNER in html_text
    assert "http://" not in html_text and "https://" not in html_text
    assert 'name="viewport"' in html_text
    summary = json.loads((out / "SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["synthetic_fixture"] is True


def test_real_run_has_no_fixture_banner(tmp_path):
    root = fx.write_fixture_run(tmp_path / "run", synthetic=False, scenarios=("no_contention",), msns=(4,),
                                repeats=2)
    aggregate.aggregate(root)
    out = generate_report.generate(root, tmp_path / "report")
    assert BANNER not in (out / "REPORT.md").read_text(encoding="utf-8")


def test_report_sections_in_order_and_na_cells(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    out = generate_report.generate(fixture_run, tmp_path / "report")
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    headings = [line for line in md.splitlines() if line.startswith("## ")]
    assert headings == [
        "## 1. Environment and configuration",
        "## 2. Executive summary",
        "## 3. Comparison",
        "## 4. Improvement vs FCFS",
        "## 5. Charts",
        "## 6. Interpretation",
        "## 7. Overall conclusion",
        "## 8. Limitations",
    ]
    assert "n/a (not measured)" in md
    assert "| P95 TTFT (s) |" in md


def test_not_exercised_sentence_printed_exactly(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    out = generate_report.generate(fixture_run, tmp_path / "report")
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert f"- {NOT_EXERCISED}" in md
    html_text = (out / "REPORT.html").read_text(encoding="utf-8")
    assert NOT_EXERCISED in html_text


def test_pngs_generated_with_some_metrics_missing(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    out = generate_report.generate(fixture_run, tmp_path / "report")
    pngs = sorted((out / "plots").glob("*.png"))
    assert len(pngs) >= 5
    for png in pngs:
        assert png.read_bytes()[:8] == PNG_MAGIC
    summary = json.loads((out / "SUMMARY.json").read_text(encoding="utf-8"))
    names = {c["file"] for c in summary["charts"]}
    assert "plots/p95_ttft_by_policy.png" in names


def test_charts_skip_when_no_data(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    aggregate.aggregate(root)
    out = generate_report.generate(root, tmp_path / "report")
    summary = json.loads((out / "SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["charts"] == []
    assert len(summary["charts_skipped"]) >= 5
    assert "n/a (not measured)" in (out / "REPORT.md").read_text(encoding="utf-8")


def test_input_directory_is_not_modified(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    before = _snapshot(fixture_run)
    generate_report.generate(fixture_run, tmp_path / "report")
    assert _snapshot(fixture_run) == before


def test_output_inside_input_is_refused(fixture_run):
    aggregate.aggregate(fixture_run)
    code = generate_report.main(["--input", str(fixture_run), "--output", str(fixture_run / "report")])
    assert code == 2
    assert not (fixture_run / "report").exists()


def test_missing_aggregates_is_refused(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    code = generate_report.main(["--input", str(empty), "--output", str(tmp_path / "rep")])
    assert code == 2


def test_cli_main_writes_report(fixture_run, tmp_path):
    aggregate.aggregate(fixture_run)
    code = generate_report.main(["--input", str(fixture_run), "--output", str(tmp_path / "cli")])
    assert code == 0
    assert (tmp_path / "cli" / "REPORT.html").exists()


def test_aggregate_main_rejects_missing_dir(tmp_path):
    assert aggregate.main([str(tmp_path / "missing")]) == 2
