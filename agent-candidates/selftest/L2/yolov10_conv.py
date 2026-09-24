"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_conv import YOLOConv as _BaseYOLOConv


class YOLOConv(_BaseYOLOConv):
    pass
