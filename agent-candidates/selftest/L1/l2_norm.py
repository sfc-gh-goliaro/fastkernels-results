"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.l2_norm import L2Norm as _BaseL2Norm


class L2Norm(_BaseL2Norm):
    pass
