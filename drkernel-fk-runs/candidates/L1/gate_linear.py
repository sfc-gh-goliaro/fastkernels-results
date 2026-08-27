import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M: minimize overhead
        triton.Config({"BLOCK_M": 1,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=2, num_stages=3),

        # Small/medium M
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),

        # Larger N utilization
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),

        # Very large M
        triton.Config({"BLOCK_M": 128,"BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bf16_fp32_kernel_v3(
    a_ptr,  # *bf16, [M, K]
    w_ptr,  # *bf16, [N, K] (we index as W[n, k])
    c_ptr,  # *fp32, [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-block
    # A tile: [BM, BK], k is fast axis
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # B tile loaded as [BK, BN] for better coalescing: k fast, n slow
    b_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load with masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)       # [BM, BK]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)       # [BK, BN]

        # Accumulate: dot(a[BM,BK], b[BK,BN]) -> [BM,BN]
        acc += tl.dot(a, b)

        # Advance
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_wk

    # Store
    c = acc
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    store_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=store_mask)


class ModelNew(nn.Module):
    """
    Triton-optimized router gate matmul: y = x @ weight^T
    - BF16 I/O, FP32 accumulation
    - Shape-aware, autotuned tiling
    - Correct grid that depends on chosen BLOCK sizes
    - CPU fallback
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Compute router logits y = x @ weight^T.

        Args:
            x: (M, K) tensor, BF16
            weight: (N, K) tensor, BF16 (same K as x)
            out_dtype: desired output dtype; if None return float32

        Returns:
            (M, N) tensor
        """
        # CPU fallback
        if not x.is_cuda or not weight.is_cuda:
            out = torch.nn.functional.linear(x, weight)
            return out.to(torch.float32 if out_dtype is None else out_dtype)

        # Validate shapes/dtypes
        assert x.dim() == 2 and weight.dim() == 2, "x and weight must be 2D"
        M, K = x.shape
        N, Kw = weight.shape
        assert Kw == K, f"Weight second dim must match x.shape[1]; got {Kw} vs {K}"
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, \
            f"Expected BF16 tensors, got {x.dtype} and {weight.dtype}"

        # Output (FP32)
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides
        stride_am = x.stride(0)
        stride_ak = x.stride(1)
        # weight is (N, K); use strides wn=stride(0), wk=stride(1)
        stride_wn = weight.stride(0)
        stride_wk = weight.stride(1)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)

        # Grid must depend on the chosen BLOCK sizes (META). This is critical.
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

        _matmul_bf16_fp32_kernel_v3[grid](
            x, weight, out,
            M, N, K,
            stride_am, stride_ak,
            stride_wn, stride_wk,
            stride_cm, stride_cn,
        )

        # Cast if requested
        if out_dtype is not None and out_dtype != torch.float32:
            out = out.to(out_dtype)
        return out

GateLinear = ModelNew
