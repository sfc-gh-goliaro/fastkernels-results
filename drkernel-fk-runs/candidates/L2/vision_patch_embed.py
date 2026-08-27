import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def _to_triton_dtype(torch_dtype):
    if torch_dtype == torch.float16:
        return tl.float16
    if torch_dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


@triton.jit
def _matmul_bias_kernel(
    A, B, Bias, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    OUT_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)  # tile index along M
    pid_n = tl.program_id(1)  # tile index along N
    pid_kg = tl.program_id(2) # group index along K (each group covers BLOCK_K)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K is covered in groups; local k within this group
    offs_k = pid_kg * BLOCK_K + tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in steps of BLOCK_K
    for kk in range(0, (K + BLOCK_K - 1) // BLOCK_K):
        k_ids = offs_k + kk * BLOCK_K  # absolute k indices for this iteration

        # Compute pointers for A[M,K] and B[K,N]
        A_ptrs = A + (offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak)
        B_ptrs = B + (k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks to guard OOB
        a_mask = (offs_m[:, None] < M) & (k_ids[None, :] < K)
        b_mask = (k_ids[:, None] < K) & (offs_n[None, :] < N)

        # Loads; cast to fp32
        a = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias: Bias is [N]
    if Bias is not None:
        bias = tl.load(Bias + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # Store result to C[M,N]
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(OUT_dtype), mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int, bias: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_size = in_channels * temporal_patch_size * patch_size * patch_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        # Keep Conv3d to mirror API; its weights will be used in forward as linear proj.
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel,
                              stride=temporal_patch_size if kernel[0] is None else kernel,
                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten as in original
        x = x.view(x.shape[0], self.input_size)

        # Get weight and bias
        W = self.proj.weight.view(self.embed_dim, self.input_size)  # [N, K]
        b = self.proj.bias  # [N] or None

        # If not CUDA, fall back to torch F.linear
        if (not x.is_cuda) or (not W.is_cuda):
            return F.linear(x, W, b)

        # Shapes
        M, K = x.shape
        N = W.shape[0]
        assert W.shape == (N, K), f"Weight shape must be [N, K]; got {W.shape}"
        assert x.shape == (M, K), f"Input shape must be [M, K]; got {x.shape}"

        # Allocate output
        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Strides in elements
        stride_am = x.stride(0)
        stride_ak = x.stride(1)
        # W is [N,K]; we index as B[k,n] = W[n,k]
        stride_bk = W.stride(1)  # step over k
        stride_bn = W.stride(0)  # step over n
        stride_cm = y.stride(0)
        stride_cn = y.stride(1)

        # Tile sizes (good starting point for these shapes)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        # Grid: ( ceil_div(M,BM), ceil_div(N,BN), ceil_div(K,BK) )
        grid = (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
            triton.cdiv(K, BLOCK_K),
        )

        OUT_dtype = _to_triton_dtype(y.dtype)

        # Launch kernel
        _matmul_bias_kernel[grid](
            x, W, b if b is not None else tl.zeros((), dtype=tl.float32), y,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            OUT_dtype,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        return y

VisionPatchEmbed = ModelNew
