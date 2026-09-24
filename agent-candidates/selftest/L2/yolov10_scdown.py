"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_scdown import YOLOSCDown as _BaseYOLOSCDown


class YOLOSCDown(_BaseYOLOSCDown):
    pass
