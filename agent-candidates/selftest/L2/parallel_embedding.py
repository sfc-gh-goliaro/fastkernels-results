"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.parallel_embedding import ParallelLMHead as _BaseParallelLMHead
from fastkernels.tasks.baseline.L2.parallel_embedding import VocabParallelEmbedding as _BaseVocabParallelEmbedding


class ParallelLMHead(_BaseParallelLMHead):
    pass


class VocabParallelEmbedding(_BaseVocabParallelEmbedding):
    pass
