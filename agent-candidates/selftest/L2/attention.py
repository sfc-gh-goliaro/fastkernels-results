"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.attention import LlamaAttention as _BaseLlamaAttention


class LlamaAttention(_BaseLlamaAttention):
    pass
