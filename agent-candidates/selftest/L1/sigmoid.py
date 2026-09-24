"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.sigmoid import Sigmoid as _BaseSigmoid


class Sigmoid(_BaseSigmoid):
    pass
