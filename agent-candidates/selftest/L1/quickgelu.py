"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.quickgelu import QuickGELU as _BaseQuickGELU


class QuickGELU(_BaseQuickGELU):
    pass
