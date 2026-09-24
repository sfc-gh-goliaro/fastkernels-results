"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.alphafold3_heads import AuxiliaryHeads as _BaseAuxiliaryHeads


class AuxiliaryHeads(_BaseAuxiliaryHeads):
    pass
