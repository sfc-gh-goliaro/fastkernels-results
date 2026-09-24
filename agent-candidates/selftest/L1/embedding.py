"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.embedding import Embedding as _BaseEmbedding


class Embedding(_BaseEmbedding):
    pass
