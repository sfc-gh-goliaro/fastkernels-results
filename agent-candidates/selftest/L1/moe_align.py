"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.moe_align import MoeAlign as _BaseMoeAlign


class MoeAlign(_BaseMoeAlign):
    pass
