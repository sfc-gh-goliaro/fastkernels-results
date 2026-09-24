"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.alphafold3_msa_module import MSAModuleStack as _BaseMSAModuleStack


class MSAModuleStack(_BaseMSAModuleStack):
    pass
