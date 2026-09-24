"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.softmax import Softmax as _BaseSoftmax


class Softmax(_BaseSoftmax):
    pass
