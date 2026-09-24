"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.chunk_gla import ChunkGLA as _BaseChunkGLA


class ChunkGLA(_BaseChunkGLA):
    pass
