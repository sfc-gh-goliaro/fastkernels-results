"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.t5_dense import T5DenseGatedActDense as _BaseT5DenseGatedActDense


class T5DenseGatedActDense(_BaseT5DenseGatedActDense):
    pass
