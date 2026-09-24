"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.fused_recurrent_gla import FusedRecurrentGLA as _BaseFusedRecurrentGLA


class FusedRecurrentGLA(_BaseFusedRecurrentGLA):
    pass
