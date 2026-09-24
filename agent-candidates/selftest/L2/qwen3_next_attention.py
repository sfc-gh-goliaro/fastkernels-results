"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.qwen3_next_attention import Qwen3NextAttention as _BaseQwen3NextAttention


class Qwen3NextAttention(_BaseQwen3NextAttention):
    pass
