"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L3.alphafold3_pairformer import PairFormerStack as _BasePairFormerStack


class PairFormerStack(_BasePairFormerStack):
    pass
