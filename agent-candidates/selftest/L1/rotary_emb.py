"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.rotary_emb import RotaryEmbedding as _BaseRotaryEmbedding


class RotaryEmbedding(_BaseRotaryEmbedding):
    pass
