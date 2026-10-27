"""Shell entry points: syntax, --help, and argument validation. Runs on CPU; no GPU or install."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def bash(script, *args):
    return subprocess.run([BASH, str(SCRIPTS / script), *args], capture_output=True, text=True, cwd=REPO, timeout=60)


@pytest.mark.parametrize("script", ["common.sh", "setup_gpu_env.sh", "run_gpu_tests.sh", "run_full_experiment.sh"])
def test_syntax_is_valid(script):
    result = subprocess.run([BASH, "-n", str(SCRIPTS / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", ["setup_gpu_env.sh", "run_gpu_tests.sh", "run_full_experiment.sh"])
def test_help_exits_zero(script):
    result = bash(script, "--help")
    assert result.returncode == 0
    assert "Usage" in result.stdout


def test_full_pipeline_requires_a_sequence_limit():
    result = bash("run_full_experiment.sh")
    assert result.returncode != 0
    assert "--max-num-seqs" in result.stderr


@pytest.mark.parametrize("args, message", [
    (["--max-num-seqs", "8", "--max-num-seqs-list", "4,8"], "not both"),
    (["--max-num-seqs-list", "4,4"], "duplicate"),
    (["--max-num-seqs", "0"], "positive integers"),
    (["--max-num-seqs", "8", "--profile", "huge"], "smoke, standard or thorough"),
    (["--max-num-seqs", "8", "--gpu-memory-utilization", "1.5"], "(0, 1]"),
    (["--max-num-seqs", "8", "--repeats", "0"], "positive integer"),
    (["--max-num-seqs", "8", "--bogus"], "unknown argument"),
    (["--setup-only", "--skip-setup"], "cannot be combined"),
])
def test_invalid_pipeline_arguments_fail_clearly(args, message):
    result = bash("run_full_experiment.sh", *args)
    assert result.returncode != 0
    assert message in result.stderr


def test_gpu_test_runner_rejects_unknown_policy():
    result = bash("run_gpu_tests.sh", "--policy", "round-robin", "--skip-setup")
    assert result.returncode != 0
    assert "fcfs or mlfq" in result.stderr


def test_setup_refuses_unknown_argument():
    result = bash("setup_gpu_env.sh", "--sudo")
    assert result.returncode != 0
    assert "unknown argument" in result.stderr


def test_no_script_uses_sudo_or_driver_install():
    for path in SCRIPTS.glob("*.sh"):
        text = path.read_text()
        assert "sudo " not in text, path.name
        assert "apt-get" not in text and "apt install" not in text, path.name
        assert "nvidia-driver" not in text.lower() or "do not" in text.lower(), path.name
