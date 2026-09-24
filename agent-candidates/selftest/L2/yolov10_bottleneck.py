"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_bottleneck import YOLOBottleneck as _BaseYOLOBottleneck


class YOLOBottleneck(_BaseYOLOBottleneck):
    pass
