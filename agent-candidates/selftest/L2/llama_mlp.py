"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.llama_mlp import LlamaMLP as _BaseLlamaMLP


class LlamaMLP(_BaseLlamaMLP):
    pass
