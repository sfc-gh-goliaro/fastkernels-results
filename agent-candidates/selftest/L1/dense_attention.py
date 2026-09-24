"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.dense_attention import DenseAttention as _BaseDenseAttention


class DenseAttention(_BaseDenseAttention):
    pass
