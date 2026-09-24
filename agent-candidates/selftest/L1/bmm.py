"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.bmm import BatchMatMul as _BaseBatchMatMul


class BatchMatMul(_BaseBatchMatMul):
    pass
