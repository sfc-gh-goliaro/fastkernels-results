"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.attention_impl import Attention as _BaseAttention


class Attention(_BaseAttention):
    pass
