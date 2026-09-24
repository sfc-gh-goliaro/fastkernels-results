"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.vision_patch_embed import VisionPatchEmbed as _BaseVisionPatchEmbed


class VisionPatchEmbed(_BaseVisionPatchEmbed):
    pass
