"""Step-time model used by the CPU harness only to place request arrivals between steps.

The CPU sweep uses a constant step time (see cpu_sweep.py); results are reported in steps."""
from dataclasses import dataclass


@dataclass
class CostModel:
    """Rough Llama-3.1-8B bf16 on one H100-80GB. Only used to turn steps into milliseconds."""
    base_ms: float = 6.0            # weight read / launch floor per step
    per_tok_ms: float = 0.032       # dense GEMM cost per batched token (~31K tok/s prefill)
    attn_prefill_ms: float = 1.31e-6  # per (chunk_tok * attended_ctx) pair  (4*L*H flops @400TF)
    attn_decode_ms: float = 4.4e-5    # per context token read by a decode (131KB/tok @3TB/s)
    overhead_ms: float = 1.0        # scheduler + sampling CPU overhead

    def step_ms(self, prefill_chunks, decode_ctx_sum, n_tokens):
        t = max(self.base_ms, n_tokens * self.per_tok_ms) + self.overhead_ms
        for c, n in prefill_chunks:  # c = tokens already computed, n = chunk size
            t += self.attn_prefill_ms * n * (c + n / 2)
        t += self.attn_decode_ms * decode_ctx_sum
        return t

    def alone_ttft_ms(self, prompt, budget):
        """TTFT if the request had the GPU to itself (cap disabled when alone, as in vLLM)."""
        t, c = 0.0, 0
        while c < prompt:
            n = min(budget, prompt - c)
            t += self.step_ms([(c, n)], 0, n)
            c += n
        return t
