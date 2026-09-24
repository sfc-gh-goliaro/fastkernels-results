"""Self-test candidate: subclasses the baseline unchanged (swap only)."""

from fastkernels.tasks.baseline.L1.store_kvcache import StoreKVCacheHND as _BaseStoreKVCacheHND


class StoreKVCacheHND(_BaseStoreKVCacheHND):
    pass
