"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.merge_attn_states import MergeAttnStates as _BaseMergeAttnStates


class MergeAttnStates(_BaseMergeAttnStates):
    pass
