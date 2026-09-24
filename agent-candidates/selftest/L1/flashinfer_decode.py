"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.flashinfer_decode import TRTLLMDecode as _BaseTRTLLMDecode


class TRTLLMDecode(_BaseTRTLLMDecode):
    pass
