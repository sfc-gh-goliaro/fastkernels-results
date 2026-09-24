"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.oasis_block import SpatioTemporalDiTBlock as _BaseSpatioTemporalDiTBlock


class SpatioTemporalDiTBlock(_BaseSpatioTemporalDiTBlock):
    pass
