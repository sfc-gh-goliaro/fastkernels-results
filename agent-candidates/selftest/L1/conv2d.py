"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.conv2d import Conv2d as _BaseConv2d


class Conv2d(_BaseConv2d):
    pass
