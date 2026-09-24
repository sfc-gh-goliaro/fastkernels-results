"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.mla_attention_impl import MLAAttention as _BaseMLAAttention


class MLAAttention(_BaseMLAAttention):
    pass
