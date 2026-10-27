# GPU testing

**Status: none of the GPU code in this repository has been executed.** The development machine has no NVIDIA GPU. Everything below is written to run on a Linux machine with an NVIDIA GPU (about 24 GB VRAM target) and has not been verified there.

## Prerequisites

- An NVIDIA driver that already works (`nvidia-smi` prints the GPU). The scripts never install drivers, CUDA toolkits, or system packages, and never use `sudo`.
- Python 3.10–3.12 (`pyproject.toml` requires `<3.13`).
- Network access, or a local copy of the model (`--model /path/to/Qwen3-0.6B`).

## Setup

```bash
bash scripts/setup_gpu_env.sh            # creates .venv-gpu, installs requirements-gpu.txt
bash scripts/setup_gpu_env.sh --check-only
```

`setup_gpu_env.sh` is idempotent: a hash of `requirements-gpu.txt` is stored in the venv, and a repeat run skips pip if nothing changed. It validates the CUDA runtime, triton, flash-attn, and transformers, and it checks free disk space.

The pins in `requirements-gpu.txt` (torch 2.6.0 with the cu124 wheel index, triton 3.2.0, flash-attn 2.7.4.post1, transformers 4.56.0) are a best guess for a torch 2.6 + CUDA 12.x stack. They are **unverified**. If flash-attn does not install from a prebuilt wheel, install a wheel that matches your torch and CUDA version, then rerun with `--skip-setup`.

## Run the GPU correctness tests only

```bash
bash scripts/run_gpu_tests.sh --gpu 0 --model ~/huggingface/Qwen3-0.6B --policy mlfq
bash scripts/run_gpu_tests.sh --gpu 0 --policy fcfs
```

`--policy` selects the scheduler the shared engine is created with. Run it twice to cover both.

## What the GPU tests check

File: `tests/gpu/test_gpu_correctness.py`. Each test is skipped when CUDA is absent.

| Test | Checks |
|---|---|
| `test_mixed_prompt_lengths_all_complete` | prompts from 3 to 1200 tokens (several longer than the 512-token Q0 prefill allowance) all finish with the requested length |
| `test_memory_integrity_after_run` | after a run, every KV block is free and no block is marked used |
| `test_chunked_prefill_matches_unchunked_prefill` | the same prompts under a 128-token budget and a 16384-token budget give (near-)identical tokens; passes if at least 90% of tokens match, to allow bf16 numerical noise at near-ties |
| `test_mixed_prefill_and_decode_in_one_step` | a long prompt admitted while short requests decode shares a step with them |
| `test_shared_prefix_populates_prefix_cache` | prompts sharing a full block populate the prefix cache |

The chunking test changes `llm.scheduler.max_num_batched_tokens` in place. nano-vLLM creates a process group for each engine, so a second engine in the same process would fail.

## Known gaps

- The unified mixed-batch attention path (`ModelRunner.prepare_unified`) and CUDA-graph replay for pure decode batches are not exercised by a test that checks logits directly. The tests above check outputs and memory integrity, which is weaker.
- No test forces KV pressure on the real engine. The benchmark's `kv_pressure` scenario is the first real exercise of that path.
- Tensor-parallel execution (`tensor_parallel_size > 1`) is not tested.
