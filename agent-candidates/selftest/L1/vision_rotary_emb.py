"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.vision_rotary_emb import VisionRotaryEmbedding as _BaseVisionRotaryEmbedding


class VisionRotaryEmbedding(_BaseVisionRotaryEmbedding):
    pass
