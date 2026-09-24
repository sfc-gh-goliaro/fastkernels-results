"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.rms_norm_gated import RMSNormGated as _BaseRMSNormGated


class RMSNormGated(_BaseRMSNormGated):
    pass
