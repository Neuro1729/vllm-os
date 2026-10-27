import sys
import types

# nanovllm/__init__.py imports LLM -> ModelRunner -> triton / flash-attn / CUDA.
# The scheduler, block manager and sequence never touch those, so stub the
# LLM module out to let the scheduler tests import on a CPU-only machine.
_llm_stub = types.ModuleType("nanovllm.llm")
_llm_stub.LLM = object
sys.modules.setdefault("nanovllm.llm", _llm_stub)
