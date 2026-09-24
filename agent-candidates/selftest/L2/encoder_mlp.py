"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderIntermediate as _BaseEncoderIntermediate
from fastkernels.tasks.baseline.L2.encoder_mlp import EncoderOutput as _BaseEncoderOutput


class EncoderIntermediate(_BaseEncoderIntermediate):
    pass


class EncoderOutput(_BaseEncoderOutput):
    pass
