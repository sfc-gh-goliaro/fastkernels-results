"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.gelu import GELU as _BaseGELU


class GELU(_BaseGELU):
    pass
