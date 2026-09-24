"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_swiglu_transition import ConditionedTransitionBlock as _BaseConditionedTransitionBlock
from fastkernels.tasks.baseline.L2.alphafold3_swiglu_transition import SwiGLUTransition as _BaseSwiGLUTransition


class ConditionedTransitionBlock(_BaseConditionedTransitionBlock):
    pass


class SwiGLUTransition(_BaseSwiGLUTransition):
    pass
