"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.yarn_rotary_emb import YaRNRotaryEmbedding as _BaseYaRNRotaryEmbedding


class YaRNRotaryEmbedding(_BaseYaRNRotaryEmbedding):
    pass
