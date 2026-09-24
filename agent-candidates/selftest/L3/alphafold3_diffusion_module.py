"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.alphafold3_diffusion_module import DiffusionModule as _BaseDiffusionModule
from fastkernels.tasks.baseline.L3.alphafold3_diffusion_module import SampleDiffusion as _BaseSampleDiffusion


class DiffusionModule(_BaseDiffusionModule):
    pass


class SampleDiffusion(_BaseSampleDiffusion):
    pass
