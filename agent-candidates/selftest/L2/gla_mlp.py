"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.gla_mlp import GLAMLP as _BaseGLAMLP


class GLAMLP(_BaseGLAMLP):
    pass
