"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.fused_experts import FusedExperts as _BaseFusedExperts


class FusedExperts(_BaseFusedExperts):
    pass
