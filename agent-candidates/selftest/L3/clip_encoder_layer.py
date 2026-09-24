"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.clip_encoder_layer import CLIPEncoderLayer as _BaseCLIPEncoderLayer


class CLIPEncoderLayer(_BaseCLIPEncoderLayer):
    pass
