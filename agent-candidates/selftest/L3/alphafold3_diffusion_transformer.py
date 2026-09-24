"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.alphafold3_diffusion_transformer import DiffusionTransformer as _BaseDiffusionTransformer


class DiffusionTransformer(_BaseDiffusionTransformer):
    pass
