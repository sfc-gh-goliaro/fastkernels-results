"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_triangle_attention import TriangleAttention as _BaseTriangleAttention


class TriangleAttention(_BaseTriangleAttention):
    pass
