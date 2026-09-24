"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.oasis_timestep_embedder import OasisTimestepEmbedder as _BaseOasisTimestepEmbedder


class OasisTimestepEmbedder(_BaseOasisTimestepEmbedder):
    pass
