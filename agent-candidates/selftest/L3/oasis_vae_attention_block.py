"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.oasis_vae_attention_block import OasisVAEAttentionBlock as _BaseOasisVAEAttentionBlock


class OasisVAEAttentionBlock(_BaseOasisVAEAttentionBlock):
    pass
