"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.llama_decoder import LlamaDecoderLayer as _BaseLlamaDecoderLayer


class LlamaDecoderLayer(_BaseLlamaDecoderLayer):
    pass
