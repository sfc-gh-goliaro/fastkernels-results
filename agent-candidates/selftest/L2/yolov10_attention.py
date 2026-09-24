"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.yolov10_attention import YOLOAttention as _BaseYOLOAttention


class YOLOAttention(_BaseYOLOAttention):
    pass
