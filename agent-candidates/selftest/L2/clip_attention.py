"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.clip_attention import CLIPAttention as _BaseCLIPAttention


class CLIPAttention(_BaseCLIPAttention):
    pass
