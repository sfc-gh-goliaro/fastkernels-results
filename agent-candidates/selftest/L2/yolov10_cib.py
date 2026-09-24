"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_cib import YOLOCIB as _BaseYOLOCIB


class YOLOCIB(_BaseYOLOCIB):
    pass
