"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.vision_pos_embed_interpolate import VisionPosEmbedInterpolate as _BaseVisionPosEmbedInterpolate


class VisionPosEmbedInterpolate(_BaseVisionPosEmbedInterpolate):
    pass
