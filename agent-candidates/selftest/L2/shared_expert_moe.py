"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.shared_expert_moe import SharedExpertMoE as _BaseSharedExpertMoE


class SharedExpertMoE(_BaseSharedExpertMoE):
    pass
