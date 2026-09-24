"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.fp8_linear import Fp8Linear as _BaseFp8Linear
from fastkernels.tasks.baseline.L1.fp8_linear import PerTokenGroupQuantFp8 as _BasePerTokenGroupQuantFp8


class Fp8Linear(_BaseFp8Linear):
    pass


class PerTokenGroupQuantFp8(_BasePerTokenGroupQuantFp8):
    pass
