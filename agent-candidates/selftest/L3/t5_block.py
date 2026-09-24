"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.t5_block import T5Block as _BaseT5Block


class T5Block(_BaseT5Block):
    pass
