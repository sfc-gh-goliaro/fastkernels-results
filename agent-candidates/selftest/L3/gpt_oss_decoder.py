"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.gpt_oss_decoder import GptOssDecoderLayer as _BaseGptOssDecoderLayer


class GptOssDecoderLayer(_BaseGptOssDecoderLayer):
    pass
