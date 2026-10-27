"""Feedback-policy state for the four-level MLFQ scheduler.

This module is pure bookkeeping: it never touches tensors or the model runner,
so it can be unit-tested on a CPU-only machine.

Levels: Q0 (highest) .. Q3 (lowest). New requests enter Q0. A request that
exhausts its level-specific prefill or decode allowance is demoted one level.
Waiting requests are promoted one level per `aging_interval` seconds of
continuous waiting. Q3 never demotes.
"""
from dataclasses import dataclass, field

NUM_LEVELS = 4
BOTTOM_LEVEL = NUM_LEVELS - 1


@dataclass
class FeedbackState:
    level: int = 0
    # Service consumed at the current level. Reset whenever the level changes.
    level_prefill_service: int = 0
    level_decode_service: int = 0
    # Lifetime service totals. Never reset; used for reporting.
    prefill_service: int = 0
    decode_service: int = 0
    # Position in the FIFO of the current queue (set on enqueue and level change).
    queue_seq: int = 0
    # Admission order across all sequences; breaks ties deterministically.
    admit_order: int = -1
    waiting_since: float | None = None
    last_scheduled_step: int = -1
    admitted_step: int = -1
    cooldown_until_step: int = 0
    preemptions: int = 0
    proactive_preempt_steps: list[int] = field(default_factory=list)
    aging_history: list[tuple[int, int, int]] = field(default_factory=list)  # (step, from, to)
    demotion_history: list[tuple[int, int, int]] = field(default_factory=list)  # (step, from, to)


@dataclass(frozen=True)
class MLFQConfig:
    prefill_quanta: tuple[int, ...] = (512, 1024, 2048)
    decode_quanta: tuple[int, ...] = (16, 48, 192)
    aging_interval: float = 5.0
    first_token_promotion: bool = False

    @classmethod
    def from_config(cls, config) -> "MLFQConfig":
        return cls(
            prefill_quanta=tuple(config.mlfq_prefill_quanta),
            decode_quanta=tuple(config.mlfq_decode_quanta),
            aging_interval=config.mlfq_aging_interval,
            first_token_promotion=config.mlfq_first_token_promotion,
        )

    def prefill_quantum(self, level: int) -> int | None:
        return None if level >= BOTTOM_LEVEL else self.prefill_quanta[level]

    def decode_quantum(self, level: int) -> int | None:
        return None if level >= BOTTOM_LEVEL else self.decode_quanta[level]


def _set_level(state: FeedbackState, new_level: int, step: int, history: list, queue_seq: int):
    old_level = state.level
    state.level = new_level
    state.level_prefill_service = 0
    state.level_decode_service = 0
    state.queue_seq = queue_seq
    history.append((step, old_level, new_level))


def account_service(state: FeedbackState, cfg: MLFQConfig, *, decode: bool, tokens: int, step: int, queue_seq: int) -> bool:
    """Charge `tokens` of service to the request. Returns True if it was demoted.

    `decode=True` means one generated token (a plain decode step); otherwise
    `tokens` prompt or recomputed tokens were computed. A computed token is
    charged exactly once, so callers must not also charge it as decode.
    """
    if decode:
        state.decode_service += tokens
        state.level_decode_service += tokens
        quantum = cfg.decode_quantum(state.level)
        exhausted = quantum is not None and state.level_decode_service >= quantum
    else:
        state.prefill_service += tokens
        state.level_prefill_service += tokens
        quantum = cfg.prefill_quantum(state.level)
        exhausted = quantum is not None and state.level_prefill_service >= quantum

    if not exhausted or state.level >= BOTTOM_LEVEL:
        return False
    _set_level(state, state.level + 1, step, state.demotion_history, queue_seq)
    return True


def first_token(state: FeedbackState, cfg: MLFQConfig, *, step: int, queue_seq: int) -> bool:
    """Optional promotion to Q0 on the first output token. Returns True if promoted."""
    if not cfg.first_token_promotion or state.level == 0:
        return False
    _set_level(state, 0, step, state.aging_history, queue_seq)
    return True


def age(state: FeedbackState, cfg: MLFQConfig, *, now: float, step: int, queue_seq: int) -> bool:
    """Promote a waiting request one level if it has waited `aging_interval` seconds.

    The wait clock restarts on every promotion, so a request climbs at most one
    level per interval and cannot jump several levels in one scheduling step.
    Returns True if promoted.
    """
    if state.waiting_since is None:
        state.waiting_since = now
        return False
    if state.level == 0 or now - state.waiting_since < cfg.aging_interval:
        return False
    _set_level(state, state.level - 1, step, state.aging_history, queue_seq)
    state.waiting_since = now
    return True
