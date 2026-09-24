"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.vision_attention import VisionAttention as _BaseVisionAttention


class VisionAttention(_BaseVisionAttention):
    pass
