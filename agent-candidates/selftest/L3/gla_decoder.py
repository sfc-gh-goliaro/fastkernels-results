"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.gla_decoder import GLADecoderLayer as _BaseGLADecoderLayer


class GLADecoderLayer(_BaseGLADecoderLayer):
    pass
