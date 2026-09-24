"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.flux_attention import FluxAttention as _BaseFluxAttention


class FluxAttention(_BaseFluxAttention):
    pass
