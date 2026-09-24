"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.t5_attention import T5SelfAttention as _BaseT5SelfAttention


class T5SelfAttention(_BaseT5SelfAttention):
    pass
