@echo off
rem Diagnose the GPU decode path: runs the GPU test-1 workload in eager mode, then with CUDA graphs.
rem Mounts the current repo over /workspace, so a git pull is enough (no image rebuild).
rem Usage (from the repo root):  scripts\diagnose_gpu.bat
setlocal
cd /d "%~dp0.."

set "IMAGE=nanovllm-mlfq"
set "MODEL_DIR=%USERPROFILE%\huggingface\Qwen3-0.6B"

echo === eager mode (no CUDA graphs) ===
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% python scripts/diagnose_gpu_decode.py --model /models/Qwen3-0.6B --eager

echo.
echo === cuda-graph mode ===
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% python scripts/diagnose_gpu_decode.py --model /models/Qwen3-0.6B

echo.
echo === chunked prefill vs unchunked (each run in its own process) ===
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% python scripts/diagnose_chunked_prefill.py --model /models/Qwen3-0.6B --budget 16384 --label unchunked-A

echo.
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% python scripts/diagnose_chunked_prefill.py --model /models/Qwen3-0.6B --budget 16384 --label unchunked-B

echo.
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% python scripts/diagnose_chunked_prefill.py --model /models/Qwen3-0.6B --budget 128 --label chunked-128

echo.
echo Diagnostic finished. Paste all outputs.
exit /b 0
