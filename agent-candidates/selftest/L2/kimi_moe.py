"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.kimi_moe import KimiMoE as _BaseKimiMoE


class KimiMoE(_BaseKimiMoE):
    pass
