#!/usr/bin/env python3
"""One-command local launcher for the vLLM Scheduler Playground (Windows / macOS / Linux).

    python run_local.py                  # create .venv, start vLLM (if installed) + the UI, open the browser
    python run_local.py --no-vllm        # UI only; point it at a vLLM server you already run
    python run_local.py --vllm-url http://127.0.0.1:8000
    python run_local.py --dry-run        # print the commands, start nothing

Everything binds to 127.0.0.1 (this computer only). There is no tunnel or public URL.
vLLM itself needs Linux + an NVIDIA GPU (on Windows, run this inside WSL2).
Ctrl+C stops both processes gracefully (SIGTERM first, so the GPU is released cleanly).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"


def venv_python():
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def ensure_ui_env(dry):
    """UI deps (fastapi, uvicorn, httpx) live in ./.venv; vLLM stays in its own environment."""
    if dry:
        return venv_python() if venv_python().exists() else Path(sys.executable)
    if not venv_python().exists():
        print("Creating .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([str(venv_python()), "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    return venv_python()


def vllm_command(a):
    exe = a.vllm_bin or shutil.which("vllm")
    if not exe:
        return None
    cmd = [exe, "serve", a.model, "--host", "127.0.0.1", "--port", str(a.vllm_port),
           "--dtype", "float16", "--max-model-len", str(a.max_model_len),
           "--scheduling-policy", "priority", "--enable-chunked-prefill",
           "--max-num-batched-tokens", "2048", "--max-num-seqs", str(a.max_num_seqs),
           "--no-enable-prefix-caching"]
    if a.mitigate:   # per-prefill cap + adaptive floor: the "after" comparison
        cmd += ["--long-prefill-token-threshold", "512", "--long-prefill-token-threshold-adaptive"]
    return cmd


def wait_for(url, timeout, what):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    print(f"Timed out waiting for {what} at {url}")
    return False


def stop(proc, name):
    if proc.poll() is not None:
        return
    print(f"Stopping {name} ...")
    proc.terminate()                      # SIGTERM: lets vLLM release the GPU cleanly
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print(f"{name} did not exit in 90 s; killing it")
        proc.kill()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="chat/instruct model served by vLLM")
    p.add_argument("--max-model-len", type=int, default=16384, help="use 32768 with a 7B model for 16K-24K prompts")
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--mitigate", action="store_true", help="enable the 512-token prefill cap + adaptive floor")
    p.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for vLLM (e.g. 0)")
    p.add_argument("--vllm-bin", default=None, help="path to the vllm executable if it is not on PATH")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--vllm-url", default=None, help="use an already-running server; do not start vLLM")
    p.add_argument("--no-vllm", action="store_true", help="start only the UI (uses --vllm-url or http://127.0.0.1:8000)")
    p.add_argument("--ui-port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    external = a.no_vllm or a.vllm_url
    vllm_url = (a.vllm_url or f"http://127.0.0.1:{a.vllm_port}").rstrip("/")
    py = ensure_ui_env(a.dry_run)
    ui_cmd = [str(py), str(ROOT / "app.py"), "--host", "127.0.0.1", "--port", str(a.ui_port)]
    v_cmd = None if external else vllm_command(a)
    if not external and v_cmd is None:
        print("vLLM was not found on PATH. Install it (Linux/WSL2 + NVIDIA GPU: `pip install vllm`), pass\n"
              "--vllm-bin, or run the UI against a server you already have with --vllm-url / --no-vllm.\n"
              "Starting the UI only.")
    if a.dry_run:
        print("vLLM:", " ".join(v_cmd) if v_cmd else "(not started)")
        print("UI:  ", " ".join(ui_cmd), f"(VLLM_URL={vllm_url})")
        return 0

    procs = []
    try:
        if v_cmd:
            env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER=os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "0"))
            if a.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
            print("Starting vLLM (first run downloads the model) ...")
            procs.append((subprocess.Popen(v_cmd, env=env), "vLLM"))
            if not wait_for(f"{vllm_url}/v1/models", 1800, "vLLM"):
                return 1
        procs.append((subprocess.Popen(ui_cmd, env=dict(os.environ, VLLM_URL=vllm_url)), "UI"))
        ui = f"http://127.0.0.1:{a.ui_port}"
        wait_for(f"{ui}/api/health", 60, "the UI")
        print(f"\nPlayground ready: {ui}   (this computer only; Ctrl+C to stop)\n")
        if not a.no_browser:
            webbrowser.open(ui)
        while all(proc.poll() is None for proc, _ in procs):
            time.sleep(1)
        print("A process exited; shutting down.")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for proc, name in reversed(procs):
            stop(proc, name)


if __name__ == "__main__":
    sys.exit(main())
