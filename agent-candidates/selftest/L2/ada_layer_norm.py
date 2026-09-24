"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.ada_layer_norm import AdaLayerNormZero as _BaseAdaLayerNormZero
from fastkernels.tasks.baseline.L2.ada_layer_norm import AdaLayerNormZeroSingle as _BaseAdaLayerNormZeroSingle


class AdaLayerNormZero(_BaseAdaLayerNormZero):
    pass


class AdaLayerNormZeroSingle(_BaseAdaLayerNormZeroSingle):
    pass
