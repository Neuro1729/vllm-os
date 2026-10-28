import os
from dataclasses import dataclass, field
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
    num_kvcache_blocks: int = -1    # > 0 caps the GPU KV cache (useful to create KV pressure)
    # scheduling, see docs/SCHEDULER.md
    scheduling_policy: str = "fcfs"    # fcfs | priority | mlfq
    # MLFQ level i holds sequences of up to mlfq_level_tokens[i] tokens (prompt + generated);
    # anything longer is in the last level
    mlfq_level_tokens: tuple[int, ...] = (1024, 4096, 8192)
    mlfq_priority_scale: float = 1.0    # level bands *= scale ** -priority: important requests stay up longer
    mlfq_max_levels: dict[int, int] = field(default_factory=dict)    # priority -> deepest level it can sink to
    aging_steps: int = 0    # waiting this many steps promotes one MLFQ level (0 = off)
    starvation_steps: int = 0    # waiting this many steps promotes one priority tier (0 = off)
    preempt_after_steps: int = 0    # steps a better request waits at a full batch before it may evict
    min_resident_steps: int = 0    # decode steps a seq runs before it may be preempted for a newcomer
    preemption_mode: str = "recompute"    # recompute | swap | auto
    swap_space: float = 0    # GiB of pinned host memory for swapped-out KV blocks
    swap_min_tokens: int = 0    # auto mode: swap only sequences with at least this many cached tokens
    num_cpu_blocks: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.scheduling_policy in ("fcfs", "priority", "mlfq")
        assert self.preemption_mode in ("recompute", "swap", "auto")
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
