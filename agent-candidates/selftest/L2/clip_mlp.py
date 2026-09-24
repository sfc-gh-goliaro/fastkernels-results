"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.clip_mlp import CLIPMLP as _BaseCLIPMLP
from fastkernels.tasks.baseline.L2.clip_mlp import CLIPTextEmbeddings as _BaseCLIPTextEmbeddings


class CLIPMLP(_BaseCLIPMLP):
    pass


class CLIPTextEmbeddings(_BaseCLIPTextEmbeddings):
    pass
