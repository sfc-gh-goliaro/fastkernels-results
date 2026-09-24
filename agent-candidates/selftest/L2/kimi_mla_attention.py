"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.kimi_mla_attention import KimiMLAAttention as _BaseKimiMLAAttention


class KimiMLAAttention(_BaseKimiMLAAttention):
    pass
