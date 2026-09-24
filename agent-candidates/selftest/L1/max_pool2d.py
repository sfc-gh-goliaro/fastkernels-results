"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.max_pool2d import MaxPool2d as _BaseMaxPool2d


class MaxPool2d(_BaseMaxPool2d):
    pass
