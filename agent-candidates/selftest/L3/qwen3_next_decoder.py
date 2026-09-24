"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.qwen3_next_decoder import Qwen3NextDecoderLayer as _BaseQwen3NextDecoderLayer


class Qwen3NextDecoderLayer(_BaseQwen3NextDecoderLayer):
    pass
