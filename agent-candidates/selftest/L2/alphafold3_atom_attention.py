"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_atom_attention import AtomAttentionDecoder as _BaseAtomAttentionDecoder
from fastkernels.tasks.baseline.L2.alphafold3_atom_attention import AtomAttentionEncoder as _BaseAtomAttentionEncoder


class AtomAttentionDecoder(_BaseAtomAttentionDecoder):
    pass


class AtomAttentionEncoder(_BaseAtomAttentionEncoder):
    pass
