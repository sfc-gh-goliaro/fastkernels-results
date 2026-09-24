"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.oasis_dit import OasisDiT as _BaseOasisDiT


class OasisDiT(_BaseOasisDiT):
    pass
