"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.log_sigmoid import LogSigmoid as _BaseLogSigmoid


class LogSigmoid(_BaseLogSigmoid):
    pass
