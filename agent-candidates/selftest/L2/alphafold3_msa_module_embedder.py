"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_msa_module_embedder import MSAModuleEmbedder as _BaseMSAModuleEmbedder


class MSAModuleEmbedder(_BaseMSAModuleEmbedder):
    pass
