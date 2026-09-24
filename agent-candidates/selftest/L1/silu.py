"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.silu import SiLU as _BaseSiLU


class SiLU(_BaseSiLU):
    pass
