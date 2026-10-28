@echo off
rem Benchmark run for Windows with Docker Desktop: FCFS vs MLFQ vs MLFQ+preemption (plus V0 baseline).
rem Needs the model from run_windows_gpu.bat (%USERPROFILE%\huggingface\Qwen3-0.6B).
rem Correctness-test failures do not stop the run; results are labeled unverified in that case.
rem Usage (from the repo root):
rem   scripts\run_windows_benchmark.bat            quick smoke run (default)
rem   scripts\run_windows_benchmark.bat standard   full comparison, 3 repeats (much longer)
setlocal
cd /d "%~dp0.."

set "IMAGE=nanovllm-mlfq"
set "MODEL_DIR=%USERPROFILE%\huggingface\Qwen3-0.6B"
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=smoke"
if "%PROFILE%"=="standard" (set "REPEATS=3") else (set "REPEATS=1")

if not exist "%MODEL_DIR%\model.safetensors" (
    echo Model not found at %MODEL_DIR%. Run scripts\run_windows_gpu.bat first to download it.
    exit /b 1
)

if not exist "%cd%\results" mkdir "%cd%\results"
docker run --rm --gpus all --ipc=host --shm-size=16g -v "%cd%:/workspace" -v "%MODEL_DIR%:/models/Qwen3-0.6B:ro" -v "%cd%\results:/workspace/results" %IMAGE% bash -c "bash scripts/run_full_experiment.sh --model /models/Qwen3-0.6B --max-num-seqs 4 --profile %PROFILE% --repeats %REPEATS% --skip-setup --continue-on-test-failure"
if errorlevel 1 (
    echo.
    echo Benchmark failed. Read the output above for the first error.
    exit /b 1
)

echo.
echo Benchmark finished. Newest folder under %cd%\results\ has reports\REPORT.md
exit /b 0
