"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_dfl import YOLODFL as _BaseYOLODFL


class YOLODFL(_BaseYOLODFL):
    pass
