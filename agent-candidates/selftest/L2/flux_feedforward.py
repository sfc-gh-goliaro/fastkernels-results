"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.flux_feedforward import FeedForward as _BaseFeedForward


class FeedForward(_BaseFeedForward):
    pass
