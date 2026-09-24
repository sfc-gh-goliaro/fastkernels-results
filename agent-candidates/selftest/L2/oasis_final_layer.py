"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.oasis_final_layer import OasisFinalLayer as _BaseOasisFinalLayer


class OasisFinalLayer(_BaseOasisFinalLayer):
    pass
