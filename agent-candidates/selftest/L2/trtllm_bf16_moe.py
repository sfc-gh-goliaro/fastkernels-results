"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.trtllm_bf16_moe import TrtLlmBf16MoE as _BaseTrtLlmBf16MoE


class TrtLlmBf16MoE(_BaseTrtLlmBf16MoE):
    pass
