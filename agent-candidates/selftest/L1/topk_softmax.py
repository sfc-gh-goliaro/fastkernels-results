"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.topk_softmax import TopKSoftmax as _BaseTopKSoftmax


class TopKSoftmax(_BaseTopKSoftmax):
    pass
