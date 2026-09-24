"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.flashinfer_mla_decode import FlashInferMLADecode as _BaseFlashInferMLADecode


class FlashInferMLADecode(_BaseFlashInferMLADecode):
    pass
