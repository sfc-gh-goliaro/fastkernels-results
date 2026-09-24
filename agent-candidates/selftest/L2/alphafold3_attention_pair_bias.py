"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_attention_pair_bias import AttentionPairBias as _BaseAttentionPairBias
from fastkernels.tasks.baseline.L2.alphafold3_attention_pair_bias import CrossAttentionPairBias as _BaseCrossAttentionPairBias


class AttentionPairBias(_BaseAttentionPairBias):
    pass


class CrossAttentionPairBias(_BaseCrossAttentionPairBias):
    pass
