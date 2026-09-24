"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_of3_attention import OF3Attention as _BaseOF3Attention


class OF3Attention(_BaseOF3Attention):
    pass
