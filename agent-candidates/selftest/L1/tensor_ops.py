"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.tensor_ops import Pad as _BasePad


class Pad(_BasePad):
    pass
