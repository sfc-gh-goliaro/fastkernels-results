"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.oasis_patch_embed import OasisPatchEmbed as _BaseOasisPatchEmbed


class OasisPatchEmbed(_BaseOasisPatchEmbed):
    pass
