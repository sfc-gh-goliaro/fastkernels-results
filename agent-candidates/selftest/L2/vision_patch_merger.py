"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.vision_patch_merger import VisionPatchMerger as _BaseVisionPatchMerger


class VisionPatchMerger(_BaseVisionPatchMerger):
    pass
