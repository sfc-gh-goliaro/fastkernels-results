"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.qwen3_next_gdn_attention import Qwen3NextGDNAttention as _BaseQwen3NextGDNAttention


class Qwen3NextGDNAttention(_BaseQwen3NextGDNAttention):
    pass
