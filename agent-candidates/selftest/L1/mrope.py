"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.mrope import MRotaryEmbedding as _BaseMRotaryEmbedding


class MRotaryEmbedding(_BaseMRotaryEmbedding):
    pass
