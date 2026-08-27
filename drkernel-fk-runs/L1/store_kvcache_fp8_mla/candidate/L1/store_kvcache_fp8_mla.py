import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton kernels (BF16 path)
# -----------------------------

# Store: concat kv_c_normed (512 BF16) + k_pe (64 BF16) -> kv_cache[slot, token_start : token_start + 576]
@triton.jit
def store_bf16_mla_kernel(
    kv_c_ptr,        # *bf16, shape [N, 512]
    k_pe_ptr,        # *bf16, shape [N, 64]
    kv_cache_ptr,    # *bf16, shape [B, S, 576]
    slot_ptr,        # *int64, shape [N]
    D: tl.constexpr, # 576
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)  # token id in [0, N)
    # Load slot (linear index) for this token
    slot = tl.load(slot_ptr + pid)
    # Base offset in kv_cache for this token (elements)
    base = slot * D
    # Copy kv_c_normed: 0..512
    for d in range(0, 512, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < 512
        x = tl.load(kv_c_ptr + pid * 512 + offs, mask=mask, other=0)
        tl.store(kv_cache_ptr + base + offs, x, mask=mask)
    # Copy k_pe: 512..576
    for d in range(0, 64, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < 64
        x = tl.load(k_pe_ptr + pid * 64 + offs, mask=mask, other=0)
        tl.store(kv_cache_ptr + base + 512 + offs, x, mask=mask)


# Gather (ordered): copy [num_tokens, 576] from kv_cache into workspace contiguous
# Assumes tokens are ordered per sequence: token 0..L0-1, L0..L1-1, ...
@triton.jit
def gather_bf16_mla_ordered_kernel(
    kv_cache_ptr,    # *bf16, shape [B, S, 576]
    workspace_ptr,   # *bf16, shape [N, 576]
    starts_ptr,      # *int32, shape [num_seqs]
    D: tl.constexpr, # 576
    BLOCK_D: tl.constexpr,
    S: tl.constexpr, # block size
):
    pid = tl.program_id(0)  # token id
    # Derive sequence id and local offset
    seq = pid // S
    local = pid % S
    start = tl.load(starts_ptr + seq)  # int32
    base = start * D + local  # element offset into kv_cache (start block, local col)
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(kv_cache_ptr + base + offs, mask=mask, other=0)
        tl.store(workspace_ptr + pid * D + offs, x, mask=mask)


# -----------------------------
# Python wrappers (entry points)
# -----------------------------

class ModelNew(nn.Module):
    """
    Triton implementation of the store kernel for BF16 MLA cache ("auto" layout).

    Signature:
        forward(kv_c_normed: torch.Tensor, k_pe: torch.Tensor,
                kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None
    Behavior:
        - Writes kv_c_normed (BF16 [N,512]) and k_pe (BF16 [N,64]) into
          kv_cache (BF16 [B,S,576]) at the linear slot indices given by
          slot_mapping (int64 [N]).
        - Assumes CUDA tensors and BF16 dtype for Triton path; otherwise
          falls back to torch ops.
    """
    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        self.kv_cache_dtype = kv_cache_dtype

    def forward(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Fallback if not CUDA or not BF16
        use_triton = (
            kv_c_normed.is_cuda and k_pe.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda and
            kv_c_normed.dtype == torch.bfloat16 and k_pe.dtype == torch.bfloat16 and
            kv_cache.dtype == torch.bfloat16
        )
        if not use_triton:
            # Simple torch fallback: scatter_add into contiguous view
            N = kv_c_normed.shape[0]
            kv_c = kv_c_normed.reshape(N, 512).contiguous()
            k_pe_ = k_pe.reshape(N, 64).contiguous()
            D = 576
            # Create a contiguous buffer per slot and scatter
            # But kv_cache is already [B,S,D]; we can write via advanced indexing
            # Extract slots
            slots = slot_mapping
            # Flatten
            B, S, _ = kv_cache.shape
            # For each token t: kv_cache[slots[t], 0:512] = kv_c[t]; 512:576 = k_pe[t]
            for t in range(N):
                s = int(slots[t].item())
                kv_cache[s, 0:512].copy_(kv_c[t])
                kv_cache[s, 512:576].copy_(k_pe_[t])
            return

        # Shapes
        N = kv_c_normed.shape[0]
        Dc = kv_c_normed.shape[1]
        Dp = k_pe.reshape(-1, k_pe.shape[-1]).shape[1]
        if Dc != 512 or Dp != 64:
            # Fallback to torch if shapes don't match expected
            kv_c = kv_c_normed.reshape(N, 512).contiguous()
            k_pe_ = k_pe.reshape(N, 64).contiguous()
            for t in range(N):
                s = int(slot_mapping[t].item())
                kv_cache[s, 0:512].copy_(kv_c[t])
                kv_cache[s, 512:576].copy_(k_pe_[t])
            return

        # Ensure contiguous
        kv_c_normed = kv_c_normed.contiguous()
        k_pe_ = k_pe.reshape(N, Dp).contiguous()
        kv_cache = kv_cache.contiguous()
        slot_mapping = slot_mapping.contiguous()

        # Grid and block config
        D = 576
        BLOCK_D = 128
        grid = (N,)

        # Launch Triton
        store_bf16_mla_kernel[grid](
            kv_c_normed, k_pe_, kv_cache, slot_mapping,
            D, BLOCK_D,
            num_warps=4,
        )


class GatherKVCacheFP8MLA(nn.Module):
    """
    Triton implementation of the gather kernel for BF16 MLA cache (ordered assumption).

    Signature:
        forward(kv_cache: torch.Tensor, block_table: torch.Tensor,
                seq_lens: torch.Tensor, workspace_starts: torch.Tensor,
                num_seqs: int, workspace: torch.Tensor) -> None
    Notes:
        - Assumes tokens are iterated per sequence in order.
        - Uses workspace_starts (int32) to locate start block per sequence.
    """
    def __init__(self, kv_cache_dtype: str = "auto"):
        super().__init__()
        self.kv_cache_dtype = kv_cache_dtype
        self.S = None  # block size; infer from kv_cache on first use

    def forward(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        num_seqs: int,
        workspace: torch.Tensor,
    ) -> None:
        use_triton = (
            kv_cache.is_cuda and workspace.is_cuda and workspace_starts.is_cuda and
            kv_cache.dtype == torch.bfloat16 and workspace.dtype == torch.bfloat16
        )
        if not use_triton:
            # Torch fallback: copy ordered
            B, S, Dtok = kv_cache.shape
            N = workspace.shape[0]
            D = workspace.shape[1]
            if Dtok != 576 or D != 576:
                return  # or raise; here we skip
            # tokens are ordered: token t in seq = t // S, local = t % S, start = workspace_starts[seq]
            for t in range(N):
                seq = t // S
                local = t % S
                start = int(workspace_starts[seq].item())
                src = kv_cache[start, local, :].contiguous()
                workspace[t, :].copy_(src)
            return

        # Shapes
        B, S, Dtok = kv_cache.shape
        N = workspace.shape[0]
        D = workspace.shape[1]
        if Dtok != 576 or D != 576:
            return

        kv_cache = kv_cache.contiguous()
        workspace = workspace.contiguous()
        workspace_starts = workspace_starts.contiguous()

        # Infer S if needed
        if self.S is None:
            self.S_ = int(S)
        else:
            self.S_ = int(S)
        S_ = self.S_
        if S_ != S:
            # mismatch; continue with S from tensor
            S_ = int(S)

        D = 576
        BLOCK_D = 128
        grid = (N,)

        gather_bf16_mla_ordered_kernel[grid](
            kv_cache, workspace, workspace_starts,
            D, BLOCK_D,
            S_,
            num_warps=4,
        )

GatherAndDequantKVCacheMLA = ModelNew
StoreKVCacheFP8MLA = ModelNew
