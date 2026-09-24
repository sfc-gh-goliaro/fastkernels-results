"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.store_kvcache_fp8_mla import GatherAndDequantKVCacheMLA as _BaseGatherAndDequantKVCacheMLA
from fastkernels.tasks.baseline.L1.store_kvcache_fp8_mla import StoreKVCacheFP8MLA as _BaseStoreKVCacheFP8MLA


class GatherAndDequantKVCacheMLA(_BaseGatherAndDequantKVCacheMLA):
    pass


class StoreKVCacheFP8MLA(_BaseStoreKVCacheFP8MLA):
    pass
