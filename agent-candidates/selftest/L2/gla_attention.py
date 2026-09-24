"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.gla_attention import GatedLinearAttention as _BaseGatedLinearAttention


class GatedLinearAttention(_BaseGatedLinearAttention):
    pass
