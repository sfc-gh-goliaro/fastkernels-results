"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.alphafold3_outer_product_mean import OuterProductMean as _BaseOuterProductMean


class OuterProductMean(_BaseOuterProductMean):
    pass
