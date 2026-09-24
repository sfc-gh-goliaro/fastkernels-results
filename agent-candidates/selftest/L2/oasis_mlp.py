"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.oasis_mlp import OasisMLP as _BaseOasisMLP


class OasisMLP(_BaseOasisMLP):
    pass
