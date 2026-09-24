"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.ada_layer_norm_continuous import AdaLayerNormContinuous as _BaseAdaLayerNormContinuous


class AdaLayerNormContinuous(_BaseAdaLayerNormContinuous):
    pass
