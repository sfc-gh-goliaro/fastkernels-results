"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_msa_attention import MSARowAttentionWithPairBias as _BaseMSARowAttentionWithPairBias


class MSARowAttentionWithPairBias(_BaseMSARowAttentionWithPairBias):
    pass
