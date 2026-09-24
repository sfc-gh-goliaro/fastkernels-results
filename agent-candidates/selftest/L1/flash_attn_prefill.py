"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.flash_attn_prefill import FlashAttnPrefill as _BaseFlashAttnPrefill


class FlashAttnPrefill(_BaseFlashAttnPrefill):
    pass
