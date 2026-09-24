"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.qwen3_moe import Qwen3MoE as _BaseQwen3MoE


class Qwen3MoE(_BaseQwen3MoE):
    pass
