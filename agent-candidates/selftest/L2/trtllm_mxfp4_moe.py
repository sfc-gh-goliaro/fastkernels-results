"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.trtllm_mxfp4_moe import TrtLlmMxfp4MoE as _BaseTrtLlmMxfp4MoE


class TrtLlmMxfp4MoE(_BaseTrtLlmMxfp4MoE):
    pass
