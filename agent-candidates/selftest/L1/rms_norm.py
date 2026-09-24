"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm as _BaseRMSNorm


class RMSNorm(_BaseRMSNorm):
    pass
