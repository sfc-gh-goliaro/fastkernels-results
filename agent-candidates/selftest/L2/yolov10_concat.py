"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_concat import YOLOConcat as _BaseYOLOConcat


class YOLOConcat(_BaseYOLOConcat):
    pass
