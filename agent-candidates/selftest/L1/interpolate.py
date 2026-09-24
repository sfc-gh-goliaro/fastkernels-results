"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.interpolate import Interpolate as _BaseInterpolate


class Interpolate(_BaseInterpolate):
    pass
