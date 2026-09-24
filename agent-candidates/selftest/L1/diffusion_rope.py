"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.diffusion_rope import DiffusionRoPE as _BaseDiffusionRoPE


class DiffusionRoPE(_BaseDiffusionRoPE):
    pass
