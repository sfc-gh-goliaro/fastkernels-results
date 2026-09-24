"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.oasis_rotary import OasisRotaryEmbedding as _BaseOasisRotaryEmbedding


class OasisRotaryEmbedding(_BaseOasisRotaryEmbedding):
    pass
