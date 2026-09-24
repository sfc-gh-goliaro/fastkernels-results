"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.yolov10_backbone import YOLOv10Backbone as _BaseYOLOv10Backbone


class YOLOv10Backbone(_BaseYOLOv10Backbone):
    pass
