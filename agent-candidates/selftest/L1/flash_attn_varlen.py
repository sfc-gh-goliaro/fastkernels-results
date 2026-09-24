"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.flash_attn_varlen import FlashAttnVarlen as _BaseFlashAttnVarlen


class FlashAttnVarlen(_BaseFlashAttnVarlen):
    pass
