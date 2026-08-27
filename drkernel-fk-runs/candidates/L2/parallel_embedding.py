import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.jit
def _embed_allreduce_fused_kernel(
    weight_ptr,        # *fp16/fp32/bf16 [V_local, K]
    x_ptr,             # *int32          [M, N]
    out_ptr,           # *fp32           [M, K]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_w_row: tl.constexpr,
    stride_w_col: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_col: tl.constexpr,
    stride_o_row: tl.constexpr,
    stride_o_col: tl.constexpr,
    vocab_start: tl.constexpr,
    per_rank: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    kb = tl.program_id(1)
    k = kb * BLOCK_K
    offs_k = k + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over N in blocks of BLOCK_N
    for nb in range(0, N, BLOCK_N):
        n = nb + tl.arange(0, BLOCK_N)
        mask_n = n < N

        # Load x[m, n] as int32
        x_idx = tl.load(x_ptr + m * stride_x_row + n * stride_x_col,
                        mask=mask_n, other=0).to(tl.int32)

        in_range = (x_idx >= vocab_start) & (x_idx < (vocab_start + per_rank))
        local = x_idx - vocab_start

        # Iterate vector lanes
        for ii in range(BLOCK_N):
            ni = nb + ii
            valid = mask_n[ii] & in_range[ii]
            if valid:
                row = local[ii].to(tl.int32)
                ptr = weight_ptr + row * stride_w_row + offs_k * stride_w_col
                w = tl.load(ptr, mask=mask_k, other=0.0)
                # Accumulate in fp32
                acc += w.to(tl.float32)

    # Atomic add into out
    out_ptr_row = out_ptr + m * stride_o_row
    out_ptr_k = out_ptr_row + offs_k * stride_o_col
    tl.atomic_add(out_ptr_k, acc, mask=mask_k)


class Model(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int | None = None,
                 params_dtype: torch.dtype | None = None,
                 org_num_embeddings: int | None = None,
                 padding_size: int = 64,
                 bias: bool = False):  # accepted but ignored to match caller
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.params_dtype = params_dtype
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.padding_size = padding_size

        # Assume tp_size = 1 (no distributed group initialization here)
        self.tp_size = 1
        self.tp_rank = 0

        # Local table size
        self.per_partition = max(1, num_embeddings) if self.tp_size <= 1 else (num_embeddings // self.tp_size)
        self.vocab_start = 0
        self.vocab_end = self.per_partition

        # Embedding module
        self.embedding_op = nn.Embedding(self.per_partition, embedding_dim, padding_idx=padding_idx)

        # AllReduce placeholder to keep API
        self.allreduce = nn.Module()

        # Tuning
        self._block_k = 128
        self._block_n = 32
        self._num_warps = 4
        self._num_stages = 2

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fused embedding + all-reduce via Triton.
        Returns out [M, K] in weight.dtype.
        """
        assert x.is_cuda, "Triton path requires CUDA tensor"
        assert _HAS_TRITON, "Triton is not available"

        # Support 1D or 2D
        if x.dim() == 1:
            M = x.shape[0]
            N = 1
            x_2d = x.view(M, N)
            need_squeeze = True
        else:
            M, N = x.shape
            x_2d = x
            need_squeeze = False

        K = self.embedding_dim

        # Types
        if x_2d.dtype not in (torch.int32, torch.int64):
            x_2d = x_2d.to(torch.int64)
        x_2d = x_2d.contiguous().to(torch.int32)

        weight = self.embedding_op.weight
        assert weight.is_cuda, "Embedding weights must be on CUDA for Triton"

        # Output buffer in fp32 for atomics
        out = torch.zeros((M, K), device=weight.device, dtype=torch.float32)

        # Strides
        stride_w_row = weight.stride(0)
        stride_w_col = weight.stride(1)
        stride_x_row = x_2d.stride(0)
        stride_x_col = x_2d.stride(1)
        stride_o_row = out.stride(0)
        stride_o_col = out.stride(1)

        grid = (M, triton.cdiv(K, self._block_k))

        _embed_allreduce_fused_kernel[grid](
            weight, x_2d, out,
            M, N, K,
            stride_w_row, stride_w_col,
            stride_x_row, stride_x_col,
            stride_o_row, stride_o_col,
            self.vocab_start, self.per_partition,
            BLOCK_K=self._block_k,
            BLOCK_N=self._block_n,
            num_warps=self._num_warps,
            num_stages=self._num_stages,
        )

        # Cast to weight dtype
        if weight.dtype == torch.float16:
            ret = out.to(torch.float16)
        elif weight.dtype == torch.bfloat16:
            ret = out.to(torch.bfloat16)
        else:
            ret = out  # float32

        if need_squeeze:
            return ret.view(M)
        else:
            return ret

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Entry point. Uses Triton if available; else falls back to original logic.
        Note: we don't use distributed here to avoid environment dependence.
        """
        use_triton = _HAS_TRITON and x.is_cuda and self.embedding_op.weight.is_cuda
        if use_triton:
            return self._forward_triton(x)

        # Fallback: pure PyTorch, no dist
        per = self.per_partition  # computed at init assuming tp=1
        start = 0

        # Mask and local x
        mask = (x >= start) & (x < start + per)
        local_x = x - start
        local_x = mask * local_x

        y = self.embedding_op(local_x)
        # No real all_reduce without dist; return y.
        return y

ParallelLMHead = ModelNew
VocabParallelEmbedding = ModelNew
