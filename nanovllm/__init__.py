from nanovllm.sampling_params import SamplingParams


def __getattr__(name):
    # LLM pulls in torch, triton and flash-attn. Import it lazily so the scheduler,
    # block manager and MLFQ policy stay importable on CPU-only machines.
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    raise AttributeError(f"module 'nanovllm' has no attribute {name!r}")
