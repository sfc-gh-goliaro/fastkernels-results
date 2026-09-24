"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.oasis_vae_attention import OasisVAEAttention as _BaseOasisVAEAttention


class OasisVAEAttention(_BaseOasisVAEAttention):
    pass
