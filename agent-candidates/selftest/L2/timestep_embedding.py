"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.timestep_embedding import CombinedTimestepGuidanceTextProjEmbeddings as _BaseCombinedTimestepGuidanceTextProjEmbeddings
from fastkernels.tasks.baseline.L2.timestep_embedding import TimestepEmbedding as _BaseTimestepEmbedding
from fastkernels.tasks.baseline.L2.timestep_embedding import Timesteps as _BaseTimesteps


class CombinedTimestepGuidanceTextProjEmbeddings(_BaseCombinedTimestepGuidanceTextProjEmbeddings):
    pass


class TimestepEmbedding(_BaseTimestepEmbedding):
    pass


class Timesteps(_BaseTimesteps):
    pass
