"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.qwen3_moe_decoder import Qwen3MoEDecoderLayer as _BaseQwen3MoEDecoderLayer


class Qwen3MoEDecoderLayer(_BaseQwen3MoEDecoderLayer):
    pass
