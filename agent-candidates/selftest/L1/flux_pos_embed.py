"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.flux_pos_embed import FluxPosEmbed as _BaseFluxPosEmbed


class FluxPosEmbed(_BaseFluxPosEmbed):
    pass
