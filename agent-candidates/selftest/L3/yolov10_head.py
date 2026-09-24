"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.yolov10_head import YOLOv10DetectHead as _BaseYOLOv10DetectHead


class YOLOv10DetectHead(_BaseYOLOv10DetectHead):
    pass
