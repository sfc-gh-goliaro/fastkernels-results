"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.t5_layer_norm import T5LayerNorm as _BaseT5LayerNorm


class T5LayerNorm(_BaseT5LayerNorm):
    pass
