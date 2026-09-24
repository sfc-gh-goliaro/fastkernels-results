"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.gate_linear import GateLinear as _BaseGateLinear


class GateLinear(_BaseGateLinear):
    pass
