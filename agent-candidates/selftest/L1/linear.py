"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.linear import BMM as _BaseBMM
from fastkernels.tasks.baseline.L1.linear import Linear as _BaseLinear
from fastkernels.tasks.baseline.L1.linear import Matmul as _BaseMatmul


class BMM(_BaseBMM):
    pass


class Linear(_BaseLinear):
    pass


class Matmul(_BaseMatmul):
    pass
