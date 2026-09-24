"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.gpt_oss_moe import GptOssMoE as _BaseGptOssMoE


class GptOssMoE(_BaseGptOssMoE):
    pass
