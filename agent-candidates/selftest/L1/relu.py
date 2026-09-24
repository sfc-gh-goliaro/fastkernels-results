"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.relu import ReLU as _BaseReLU


class ReLU(_BaseReLU):
    pass
