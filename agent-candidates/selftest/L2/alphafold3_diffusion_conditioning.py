"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_diffusion_conditioning import DiffusionConditioning as _BaseDiffusionConditioning


class DiffusionConditioning(_BaseDiffusionConditioning):
    pass
