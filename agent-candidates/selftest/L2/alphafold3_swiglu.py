"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_swiglu import AdaLN as _BaseAdaLN
from fastkernels.tasks.baseline.L2.alphafold3_swiglu import SwiGLU as _BaseSwiGLU


class AdaLN(_BaseAdaLN):
    pass


class SwiGLU(_BaseSwiGLU):
    pass
