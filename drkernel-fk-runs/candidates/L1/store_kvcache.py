import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _store_kvcache_2d_tiled_kernel(
    key_ptr,               # *dtype, logical shape [N, D]
    key_stride_n,          # int: stride along N in elements
    value_ptr,             # *dtype, logical shape [N, D]
    value_stride_n,        # int: stride along N in elements
    k_cache_ptr,           # *dtype, logical shape [B, D] contiguous
    v_cache_ptr,           # *dtype, logical shape [B, D] contiguous
    slot_mapping_ptr,      # *int64, shape [N]
    D: tl.constexpr,       # int: row length to copy
    BLOCK: tl.constexpr,   # int: tile width
):
    # 2D launch: pid0 over N (rows), pid1 over tiles in D
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)

    # column indices for this tile
    col = pid_t * BLOCK + tl.arange(0, BLOCK)
    mask = col < D

    # load destination slot (int64)
    slot = tl.load(slot_mapping_ptr + pid_n).to(tl.int64)

    # base source offset for this row n: n * stride_n
    row_start = (pid_n * key_stride_n).to(tl.int64)
    col64 = col.to(tl.int64)
    D64 = tl.full((), D, dtype=tl.int64)

    # compute source and dest element offsets (int64)
    src_off = row_start + col64
    dst_base = slot * D64
    dst_off = dst_base + col64

    # load key and value
    k = tl.load(key_ptr + src_off, mask=mask, other=0)
    v = tl.load(value_ptr + (pid_n * value_stride_n + col64), mask=mask, other=0)

    # store into caches
    tl.store(k_cache_ptr + dst_off, k, mask=mask)
    tl.store(v_cache_ptr + dst_off, v, mask=mask)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original Model that stores key/value
    into a paged KV cache.

    Interprets:
    - key, value: [N, H, D] where D = head_dim; copies flattened rows of length D_tot = H*D
    - k_cache, v_cache: arbitrary shape but views them as [B, D_tot] for writing
    - slot_mapping: [N] (int32 or int64)

    Forward signature:
      forward(key, value, k_cache, v_cache, slot_mapping)
    """
    def __init__(self, *args, **kwargs):
        # Accept and ignore any constructor args to be robust with harness
        super().__init__()

    def forward(self, key, value, k_cache, v_cache, slot_mapping):
        """
        Stores key rows into k_cache and value rows into v_cache according to slot_mapping.
        """
        # Input checks
        assert key.shape == value.shape and key.dim() == 3, f"Expected key/value [N, H, D], got {key.shape}"
        N, H, Dh = key.shape

        # Compute total flattened width D_tot = H * Dh
        D = int(H * Dh)

        # Ensure slot_mapping is int64
        if slot_mapping.dtype != torch.int64:
            slot_mapping = slot_mapping.to(torch.int64)

        # --- CPU fallback ---
        if key.device.type != "cuda" or value.device.type != "cuda":
            # Reshape to [N, D]
            kv = key.reshape(N, D).contiguous()
            vv = value.reshape(N, D).contiguous()
            # View caches as [B, D] and copy by index
            B_k = k_cache.numel() // D if k_cache.dim() > 1 else k_cache.shape[0]
            # But to be safe, flatten last_dims to make 2D
            k2d = k_cache.view(-1, D)
            v2d = v_cache.view(-1, D)
            B = k2d.shape[0]
            assert B == int(slot_mapping.max().item()) + 1 or True, "Assume B large enough; using given slots"
            # Just in case, verify slots range
            min_slot = int(slot_mapping.min().item())
            max_slot = int(slot_mapping.max().item())
            assert min_slot >= 0 and max_slot < B, f"slot out of range: [{min_slot}, {max_slot}] vs B={B}"
            k2d.index_copy_(0, slot_mapping, kv)
            v2d.index_copy_(0, slot_mapping, vv)
            return

        # --- CUDA + Triton ---
        # Get strides (in elements) for key and value along N
        key_stride_n = int(key.stride(0))
        value_stride_n = int(value.stride(0))

        # View caches as 2D [B, D] contiguous
        # If cache is 3D [B,H,Pg,Dh], flattened D = H*Pg*Dh -> view as [B, D]
        k2d = k_cache.view(-1, D).contiguous()
        v2d = v_cache.view(-1, D).contiguous()
        B = k2d.shape[0]
        assert v2d.shape[0] == B, f"Cache batch dim mismatch: {k2d.shape} vs {v2d.shape}"

        # Launch config
        BLOCK = 128
        grid = (N, triton.cdiv(D, BLOCK))

        _store_kvcache_2d_tiled_kernel[grid](
            key, key_stride_n,
            value, value_stride_n,
            k2d, v2d,
            slot_mapping,
            D,
            BLOCK=BLOCK,
            num_warps=4,
        )

StoreKVCacheHND = ModelNew
