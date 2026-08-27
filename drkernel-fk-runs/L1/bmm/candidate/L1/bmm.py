import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


@triton.autotune(
    configs=[
        # Balanced中小型
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 64,  'BLOCK_P': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 128, 'BLOCK_P': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64,  'BLOCK_P': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128, 'BLOCK_P': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64,  'BLOCK_P': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128, 'BLOCK_P': 128}, num_warps=8, num_stages=3),
        # Larger tiles for big N/P
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64,  'BLOCK_P': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_P': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128, 'BLOCK_P': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K', 'P'],
)
@triton.jit
def bmm_kernel(
    A, B, C,
    BATCH: tl.constexpr, N: tl.constexpr, K: tl.constexpr, P: tl.constexpr,
    sA_b: tl.constexpr, sA_m: tl.constexpr, sA_k: tl.constexpr,
    sB_b: tl.constexpr, sB_k: tl.constexpr, sB_p: tl.constexpr,
    sC_b: tl.constexpr, sC_m: tl.constexpr, sC_p: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # 0=fp32, 1=bf16, 2=fp16
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_P: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    # Base pointers for this batch
    A_b = A + pid_b * sA_b
    B_b = B + pid_b * sB_b
    C_b = C + pid_b * sC_b

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_P), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for this K-slice
        A_ptr = A_b + (offs_m[:, None] * sA_m) + (offs_k[None, :] * sA_k)  # [BM, BK]
        B_ptr = B_b + (offs_k[:, None] * sB_k) + (offs_p[None, :] * sB_p)  # [BK, BP]

        # In-bounds masks
        a_mask = (offs_m[:, None] < N) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_p[None, :] < P)

        # Load in native dtype (bf16/fp16) to enable faster multiply; no upcast before dot.
        a = tl.load(A_ptr, mask=a_mask, other=0)
        b = tl.load(B_ptr, mask=b_mask, other=0)

        # Dot product: a:[BM,BK], b:[BK,BP] -> [BM,BP], accumulated in fp32
        acc += tl.dot(a, b)

    # Store result to C in requested output dtype
    C_ptr = C_b + (offs_m[:, None] * sC_m) + (offs_p[None, :] * sC_p)
    c_mask = (offs_m[:, None] < N) & (offs_p[None, :] < P)

    if OUT_DTYPE == 1:
        out = acc.to(tl.bfloat16)
    elif OUT_DTYPE == 2:
        out = acc.to(tl.float16)
    else:
        out = acc  # fp32

    tl.store(C_ptr, out, mask=c_mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Fallbacks
        if (not _HAS_TRITON) or (not a.is_cuda) or (not b.is_cuda):
            return torch.bmm(a, b)

        assert a.dim() == 3 and b.dim() == 3, f"Expected 3D tensors, got {a.shape} and {b.shape}"
        B_a, N, K_a = a.shape
        B_b, K_b, P = b.shape
        assert B_a == B_b, f"Batch mismatch: {B_a} vs {B_b}"
        assert K_a == K_b, f"Inner dim mismatch: {K_a} vs {K_b}"
        B, N, K, P = B_a, N, K_a, P

        if a.dtype != b.dtype:
            raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
        if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return torch.bmm(a, b)

        # Allocate output in input dtype
        out = torch.empty((B, N, P), device=a.device, dtype=a.dtype)

        # Strides in elements
        sA_b, sA_m, sA_k = a.stride(0), a.stride(1), a.stride(2)
        sB_b, sB_k, sB_p = b.stride(0), b.stride(1), b.stride(2)
        sC_b, sC_m, sC_p = out.stride(0), out.stride(1), out.stride(2)

        # Output dtype code for kernel
        if a.dtype == torch.bfloat16:
            out_dtype_code = 1
        elif a.dtype == torch.float16:
            out_dtype_code = 2
        else:
            out_dtype_code = 0  # float32

        # Meta-dependent grid so it matches the selected BLOCK sizes
        def grid(meta):
            return (
                triton.cdiv(N, meta['BLOCK_M']),
                triton.cdiv(P, meta['BLOCK_P']),
                B,
            )

        bmm_kernel[grid](
            a, b, out,
            B, N, K, P,
            sA_b, sA_m, sA_k,
            sB_b, sB_k, sB_p,
            sC_b, sC_m, sC_p,
            OUT_DTYPE=out_dtype_code,
        )

        return out


# Optional quick test
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    for (bn, n, k, p) in [
        (16, 1, 128, 512),
        (16, 64, 128, 512),
        (16, 64, 512, 128),
        (16, 188, 512, 128),
        (16, 997, 512, 128),
    ]:
        for dtype in (torch.bfloat16, torch.float16):
            a = torch.randn(bn, n, k, device=dev, dtype=dtype)
            b = torch.randn(bn, k, p, device=dev, dtype=dtype)
            ref = torch.bmm(a, b)
            mdl = ModelNew().to(dev)
            out = mdl(a, b)
            print(f"shape {out.shape} dtype {out.dtype} max abs err: ",
                  (out.float() - ref.float()).abs().max().item())

BatchMatMul = ModelNew
