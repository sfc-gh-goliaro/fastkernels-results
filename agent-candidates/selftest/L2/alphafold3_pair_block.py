"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_pair_block import PairBlock as _BasePairBlock


class PairBlock(_BasePairBlock):
    pass
