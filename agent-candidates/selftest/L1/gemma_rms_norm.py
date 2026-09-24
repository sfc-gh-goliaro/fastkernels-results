"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.gemma_rms_norm import GemmaRMSNorm as _BaseGemmaRMSNorm


class GemmaRMSNorm(_BaseGemmaRMSNorm):
    pass
