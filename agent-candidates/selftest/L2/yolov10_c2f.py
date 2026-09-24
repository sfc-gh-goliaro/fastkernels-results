"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_c2f import YOLOC2f as _BaseYOLOC2f
from fastkernels.tasks.baseline.L2.yolov10_c2f import YOLOC2fCIB as _BaseYOLOC2fCIB


class YOLOC2f(_BaseYOLOC2f):
    pass


class YOLOC2fCIB(_BaseYOLOC2fCIB):
    pass
