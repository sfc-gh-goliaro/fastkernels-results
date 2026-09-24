"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_input_embedder import InputEmbedder as _BaseInputEmbedder


class InputEmbedder(_BaseInputEmbedder):
    pass
