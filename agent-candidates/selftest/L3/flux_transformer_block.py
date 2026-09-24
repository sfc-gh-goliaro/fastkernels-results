"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.flux_transformer_block import FluxSingleTransformerBlock as _BaseFluxSingleTransformerBlock
from fastkernels.tasks.baseline.L3.flux_transformer_block import FluxTransformerBlock as _BaseFluxTransformerBlock


class FluxSingleTransformerBlock(_BaseFluxSingleTransformerBlock):
    pass


class FluxTransformerBlock(_BaseFluxTransformerBlock):
    pass
