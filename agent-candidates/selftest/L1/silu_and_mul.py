"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.silu_and_mul import SiluAndMul as _BaseSiluAndMul


class SiluAndMul(_BaseSiluAndMul):
    pass
