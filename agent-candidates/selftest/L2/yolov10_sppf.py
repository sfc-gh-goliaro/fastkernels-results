"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_sppf import YOLOSPPF as _BaseYOLOSPPF


class YOLOSPPF(_BaseYOLOSPPF):
    pass
