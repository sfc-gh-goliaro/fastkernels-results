"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.kimi_linear_decoder import KimiLinearDecoderLayer as _BaseKimiLinearDecoderLayer


class KimiLinearDecoderLayer(_BaseKimiLinearDecoderLayer):
    pass
