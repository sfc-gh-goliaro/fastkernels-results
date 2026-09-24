"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L2.parallel_linear import ColumnParallelLinear as _BaseColumnParallelLinear
from fastkernels.tasks.baseline.L2.parallel_linear import MergedColumnParallelLinear as _BaseMergedColumnParallelLinear
from fastkernels.tasks.baseline.L2.parallel_linear import QKVParallelLinear as _BaseQKVParallelLinear
from fastkernels.tasks.baseline.L2.parallel_linear import ReplicatedLinear as _BaseReplicatedLinear
from fastkernels.tasks.baseline.L2.parallel_linear import RowParallelLinear as _BaseRowParallelLinear


class ColumnParallelLinear(_BaseColumnParallelLinear):
    pass


class MergedColumnParallelLinear(_BaseMergedColumnParallelLinear):
    pass


class QKVParallelLinear(_BaseQKVParallelLinear):
    pass


class ReplicatedLinear(_BaseReplicatedLinear):
    pass


class RowParallelLinear(_BaseRowParallelLinear):
    pass
