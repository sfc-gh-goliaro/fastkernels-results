"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.kimi_delta_attention import KimiDeltaAttention as _BaseKimiDeltaAttention


class KimiDeltaAttention(_BaseKimiDeltaAttention):
    pass
