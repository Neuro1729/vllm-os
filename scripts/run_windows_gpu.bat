@echo off
rem One-command GPU run for Windows with Docker Desktop.
rem Needs only Docker Desktop (GPU enabled) and the nanovllm-mlfq image built from the Dockerfile.
rem Usage (from the repo root):  scripts\run_windows_gpu.bat
setlocal
cd /d "%~dp0.."

set "IMAGE=nanovllm-mlfq"
set "MODEL_DIR=%USERPROFILE%\huggingface\Qwen3-0.6B"

rem 1. Check that the image sees the GPU and the key dependencies.
docker run --rm --gpus all %IMAGE% python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
if errorlevel 1 goto fail

rem 2. Download the model to the host folder (skipped if already there). Uses the image's Python, so no host Python is needed.
if exist "%MODEL_DIR%\model.safetensors" goto model_ready
mkdir "%MODEL_DIR%" 2>nul
docker run --rm -v "%MODEL_DIR%:/models/Qwen3-0.6B" %IMAGE% python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='/models/Qwen3-0.6B')"
if errorlevel 1 goto fail
:model_ready

rem 3. GPU correctness tests (mlfq policy).
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" %IMAGE% bash -c "bash scripts/run_gpu_tests.sh --gpu 0 --model /models/Qwen3-0.6B --policy mlfq --skip-setup"
if errorlevel 1 goto fail

rem 4. Smoke benchmark. Results are written to .\results on the host.
if not exist "%cd%\results" mkdir "%cd%\results"
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" -v "%cd%\results:/workspace/results" %IMAGE% bash -c "bash scripts/run_full_experiment.sh --model /models/Qwen3-0.6B --max-num-seqs 4 --profile smoke --skip-setup"
if errorlevel 1 goto fail

echo.
echo All steps finished. Results are in %cd%\results
exit /b 0

:fail
echo.
echo FAILED. Read the output above for the first error.
exit /b 1
