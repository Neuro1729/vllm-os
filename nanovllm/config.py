import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Scheduling policy: "fcfs" (admission order) or "mlfq" (four-level feedback queue).
    scheduling_policy: str = "fcfs"
    # Proactive, priority-driven preemption. Only meaningful with scheduling_policy="mlfq".
    enable_mlfq_preemption: bool = False
    # Per-level service allowances for Q0..Q2. Q3 is the bottom level and never demotes.
    mlfq_prefill_quanta: tuple[int, ...] = (512, 1024, 2048)
    mlfq_decode_quanta: tuple[int, ...] = (16, 48, 192)
    # Seconds of continuous waiting before a waiting request is promoted one level.
    mlfq_aging_interval: float = 5.0
    # Promote a request to Q0 when its first output token is produced (off by default).
    mlfq_first_token_promotion: bool = False
    # Proactive preemption limits (anti-thrashing).
    max_proactive_preemptions_per_step: int = 1
    preemption_cooldown_steps: int = 4
    max_preemptions_per_request_window: int = 2
    preemption_window_steps: int = 64

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.scheduling_policy in ("fcfs", "mlfq"), self.scheduling_policy
        assert not self.enable_mlfq_preemption or self.scheduling_policy == "mlfq", \
            "enable_mlfq_preemption requires scheduling_policy='mlfq'"
        assert len(self.mlfq_prefill_quanta) == 3 and len(self.mlfq_decode_quanta) == 3
        assert all(q > 0 for q in self.mlfq_prefill_quanta + self.mlfq_decode_quanta)
        assert self.mlfq_aging_interval > 0
        assert self.max_proactive_preemptions_per_step >= 1
        assert self.preemption_cooldown_steps >= 0
        assert self.max_preemptions_per_request_window >= 0
        assert self.preemption_window_steps >= 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
