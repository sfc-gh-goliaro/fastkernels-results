"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.vision_mlp import VisionMLP as _BaseVisionMLP


class VisionMLP(_BaseVisionMLP):
    pass
