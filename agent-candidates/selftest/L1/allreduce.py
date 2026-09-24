"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.allreduce import AllReduce as _BaseAllReduce


class AllReduce(_BaseAllReduce):
    pass
