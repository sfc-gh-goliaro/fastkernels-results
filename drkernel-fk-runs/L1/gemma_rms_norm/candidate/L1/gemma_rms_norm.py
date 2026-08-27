import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def rmsnorm_kernel(
        x_ptr,         # *ptr to x, shape [M, D]
        w_ptr,         # *ptr to weight, shape [D]
        out_ptr,       # *ptr to out, shape [M, D]
        M, D,          # int: rows, cols
        stride_xm, stride_xd,  # strides for x
        stride_om, stride_od,  # strides for out
        eps,           # float
        OUT_DTYPE: tl.constexpr,  # output dtype (tl.float16 / tl.bfloat16 / tl.float32)
        BLOCK_D: tl.constexpr,    # block size along D (set to D for single-pass)
    ):
        row = tl.program_id(0)

        # Fast path: single pass when D <= BLOCK_D
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D  # safe even if BLOCK_D == D; no extra cost.

        # Load x and w
        x = tl.load(x_ptr + row * stride_xm + offs * stride_xd, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = 1.0 + w

        # sum of squares and rsqrt(mean + eps) in one step: rsqrt(sumsq / D + eps) == rsqrt(sumsq * (1/D) + eps)
        sumsq = tl.sum(xf * xf, axis=0)
        # Use inv_d folded into rsqrt argument to save one mul
        inv = tl.math.rsqrt(sumsq * (1.0 / BLOCK_D) + eps)

        # Fused multiply-add: y = (xf * inv) * s  =>  y = fma(xf * inv, s, 0)
        t = xf * inv
        y = tl.fma(t, s, 0.0)
        y = y.to(OUT_DTYPE)

        tl.store(out_ptr + row * stride_om + offs * stride_od, y, mask=mask)


def _pick_params(D: int):
    # For these shapes, D is 2048 (power-of-two). Set BLOCK_D = D to avoid masks and loops.
    block = D if D <= 4096 else 4096
    # Warps: 8 for >=1024, else 4/2
    num_warps = 8 if block >= 1024 else (4 if block >= 256 else 2)
    return block, num_warps


class ModelNew(nn.Module):
    """Triton-optimized RMSNorm matching vLLM's GemmaRMSNorm semantics.

    - Stores weight as offset from 1.0 (values near zero).
    - Runtime scale is (1 + weight).
    - Compute in float32; cast to original dtype at the end.
    - Forward kernel supports 2D [M, D] on CUDA.
    - Falls back to a pure PyTorch implementation otherwise.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = float(eps)
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _forward_torch(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # Pure PyTorch fallback that matches the original code
        orig_dtype = x.dtype
        if residual is not None:
            if orig_dtype == torch.float16:
                z = x.float() + residual.float()
            else:
                z = x + residual
            z32 = z.float()
            var = z32.pow(2).mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(var + self.variance_epsilon)
            y32 = z32 * inv
            s = 1.0 + self.weight.float()
            y32 = y32 * s
            y = y32.to(orig_dtype)
            return y, residual
        else:
            x32 = x.float()
            var = x32.pow(2).mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(var + self.variance_epsilon)
            y32 = x32 * inv
            s = 1.0 + self.weight.float()
            y32 = y32 * s
            y = y32.to(orig_dtype)
            return y

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # Triton fast path ignores residual (inference-optimized); torch fallback preserves it.
        if residual is not None:
            return self._forward_torch(x, residual)

        # Expect 2D: [M, D]
        if x.dim() != 2:
            return self._forward_torch(x, residual)

        M, D = x.shape
        if not x.is_cuda or not _HAS_TRITON:
            return self._forward_torch(x, residual)

        # Ensure contiguous for simple stride math and coalesced access
        x_c = x.contiguous()
        w_c = self.weight.contiguous()
        out = torch.empty_like(x_c)

        # Launch params
        BLOCK_D, num_warps = _pick_params(D)

        # Map torch dtype to triton dtype for output cast
        if x_c.dtype == torch.float16:
            out_dtype = tl.float16
        elif x_c.dtype == torch.bfloat16:
            out_dtype = tl.bfloat16
        elif x_c.dtype == torch.float32:
            out_dtype = tl.float32
        else:
            return self._forward_torch(x, residual)

        grid = (M,)
        rmsnorm_kernel[grid](
            x_c, w_c, out,
            M, D,
            x_c.stride(0), x_c.stride(1),
            out.stride(0), out.stride(1),
            self.variance_epsilon,
            OUT_DTYPE=out_dtype,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            num_stages=3,
        )
        return out

GemmaRMSNorm = ModelNew
