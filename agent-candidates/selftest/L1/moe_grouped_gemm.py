"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.moe_grouped_gemm import MoeGroupedGemm as _BaseMoeGroupedGemm


class MoeGroupedGemm(_BaseMoeGroupedGemm):
    pass
