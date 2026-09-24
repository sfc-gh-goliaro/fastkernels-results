"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.yolov10_neck import YOLOv10Neck as _BaseYOLOv10Neck


class YOLOv10Neck(_BaseYOLOv10Neck):
    pass
