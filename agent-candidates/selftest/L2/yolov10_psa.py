"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_psa import YOLOPSA as _BaseYOLOPSA


class YOLOPSA(_BaseYOLOPSA):
    pass
