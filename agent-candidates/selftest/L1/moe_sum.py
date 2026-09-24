"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.moe_sum import MoeSum as _BaseMoeSum


class MoeSum(_BaseMoeSum):
    pass
