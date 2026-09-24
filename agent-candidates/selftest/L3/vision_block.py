"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.vision_block import VisionBlock as _BaseVisionBlock


class VisionBlock(_BaseVisionBlock):
    pass
