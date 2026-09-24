"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.layer_norm import LayerNorm as _BaseLayerNorm


class LayerNorm(_BaseLayerNorm):
    pass
