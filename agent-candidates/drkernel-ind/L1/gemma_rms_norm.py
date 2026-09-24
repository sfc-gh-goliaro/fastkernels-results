import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _choose_block_and_warps(D: int, max_block: int = 4096):
    # Choose a power-of-two block, capped at max_block
    block = min(_next_power_of_two(D), max_block)
    # Heuristic for warps
    if block >= 1024:
        warps = 8
    elif block >= 256:
        warps = 4
    else:
        warps = 2
    return block, warps


@triton.jit
def _rmsnorm_kernel_single_pass(
    x_ptr,         # *T, [B, D]
    w_ptr,         # *f32, [D]
    y_ptr,         # *T, [B, D]
    D: tl.constexpr,
    eps,           # f32
    stride_x_row,  # int
    stride_x_col,  # int
    stride_y_row,  # int
    stride_y_col,  # int
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load x in native dtype, promote to fp32
    x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
    x32 = x.to(tl.float32)
    # Load weights
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)  # fp32

    # Reduce sum of squares
    sumsq = tl.sum(x32 * x32, axis=0)
    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Normalize and scale, store (cast to output dtype automatically)
    y32 = x32 * inv_rms * (1.0 + w)
    tl.store(y_ptr + row * stride_y_row + offs * stride_y_col, y32, mask=mask)


@triton.jit
def _rmsnorm_with_res_kernel_single_pass(
    x_ptr,          # *T, [B, D]
    resid_ptr,      # *T, [B, D]
    w_ptr,          # *f32, [D]
    y_ptr,          # *T, [B, D]
    resid_out_ptr,  # *T, [B, D]
    D: tl.constexpr,
    eps,            # f32
    stride_x_row,   # int
    stride_x_col,   # int
    stride_res_row, # int
    stride_res_col, # int
    stride_y_row,   # int
    stride_y_col,   # int
    stride_ro_row,  # int
    stride_ro_col,  # int
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load x and resid in native dtype, promote to fp32
    x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
    resid = tl.load(resid_ptr + row * stride_res_row + offs * stride_res_col, mask=mask, other=0)
    x32 = x.to(tl.float32)
    resid32 = resid.to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)  # fp32

    z = x32 + resid32
    sumsq = tl.sum(z * z, axis=0)
    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Normalize and scale
    y32 = z * inv_rms * (1.0 + w)
    # resid_out in original dtype, using already-loaded x/resid to avoid reload
    resid_out = x + resid

    tl.store(y_ptr + row * stride_y_row + offs * stride_y_col, y32, mask=mask)
    tl.store(resid_out_ptr + row * stride_ro_row + offs * stride_ro_col, resid_out, mask=mask)


@triton.jit
def _rmsnorm_kernel_two_pass(
    x_ptr,         # *T, [B, D]
    w_ptr,         # *f32, [D]
    y_ptr,         # *T, [B, D]
    D: tl.constexpr,
    eps,           # f32
    stride_x_row,  # int
    stride_x_col,  # int
    stride_y_row,  # int
    stride_y_col,  # int
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: sum of squares
    sumsq = tl.zeros((), dtype=tl.float32)
    col = 0
    while col < D:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
        x32 = x.to(tl.float32)
        sumsq += tl.sum(x32 * x32, axis=0)
        col += BLOCK_SIZE

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Pass 2: normalize+scale and store
    col = 0
    while col < D:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
        x32 = x.to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        y32 = x32 * inv_rms * (1.0 + w)
        tl.store(y_ptr + row * stride_y_row + offs * stride_y_col, y32, mask=mask)
        col += BLOCK_SIZE


@triton.jit
def _rmsnorm_with_res_kernel_two_pass(
    x_ptr,          # *T, [B, D]
    resid_ptr,      # *T, [B, D]
    w_ptr,          # *f32, [D]
    y_ptr,          # *T, [B, D]
    resid_out_ptr,  # *T, [B, D]
    D: tl.constexpr,
    eps,            # f32
    stride_x_row,   # int
    stride_x_col,   # int
    stride_res_row, # int
    stride_res_col, # int
    stride_y_row,   # int
    stride_y_col,   # int
    stride_ro_row,  # int
    stride_ro_col,  # int
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: sum of squares of (x + resid)
    sumsq = tl.zeros((), dtype=tl.float32)
    col = 0
    while col < D:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
        resid = tl.load(resid_ptr + row * stride_res_row + offs * stride_res_col, mask=mask, other=0)
        z32 = x.to(tl.float32) + resid.to(tl.float32)
        sumsq += tl.sum(z32 * z32, axis=0)
        col += BLOCK_SIZE

    mean = sumsq / D
    inv_rms = tl.rsqrt(mean + eps)

    # Pass 2: normalize+scale and store, also store x+resid
    col = 0
    while col < D:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(x_ptr + row * stride_x_row + offs * stride_x_col, mask=mask, other=0)
        resid = tl.load(resid_ptr + row * stride_res_row + offs * stride_res_col, mask=mask, other=0)
        z32 = x.to(tl.float32) + resid.to(tl.float32)

        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        y32 = z32 * inv_rms * (1.0 + w)

        tl.store(y_ptr + row * stride_y_row + offs * stride_y_col, y32, mask=mask)
        resid_out = x + resid
        tl.store(resid_out_ptr + row * stride_ro_row + offs * stride_ro_col, resid_out, mask=mask)
        col += BLOCK_SIZE


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = float(eps)
        # Weight is the offset from 1.0 (Gemma convention): scale = 1 + weight
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def _forward_no_res_triton(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Triton path requires CUDA tensor"
        B, D = x.shape
        x_c = x.contiguous()
        y = torch.empty_like(x_c)

        block, warps = _choose_block_and_warps(D, max_block=4096)

        grid = (B,)
        if block >= D:
            _rmsnorm_kernel_single_pass[grid](
                x_c, self.weight, y,
                D, self.variance_epsilon,
                x_c.stride(0), x_c.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE=block,
                num_warps=warps,
                num_stages=3,  # improved pipelining
            )
        else:
            _rmsnorm_kernel_two_pass[grid](
                x_c, self.weight, y,
                D, self.variance_epsilon,
                x_c.stride(0), x_c.stride(1),
                y.stride(0), y.stride(1),
                BLOCK_SIZE=block,
                num_warps=warps,
                num_stages=3,
            )
        return y

    def _forward_with_res_triton(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.is_cuda and residual.is_cuda, "Triton path requires CUDA tensors"
        B, D = x.shape
        x_c = x.contiguous()
        resid_c = residual.contiguous()
        y = torch.empty_like(x_c)
        resid_out = torch.empty_like(x_c)

        block, warps = _choose_block_and_warps(D, max_block=4096)

        grid = (B,)
        if block >= D:
            _rmsnorm_with_res_kernel_single_pass[grid](
                x_c, resid_c, self.weight, y, resid_out,
                D, self.variance_epsilon,
                x_c.stride(0), x_c.stride(1),
                resid_c.stride(0), resid_c.stride(1),
                y.stride(0), y.stride(1),
                resid_out.stride(0), resid_out.stride(1),
                BLOCK_SIZE=block,
                num_warps=warps,
                num_stages=3,
            )
        else:
            _rmsnorm_with_res_kernel_two_pass[grid](
                x_c, resid_c, self.weight, y, resid_out,
                D, self.variance_epsilon,
                x_c.stride(0), x_c.stride(1),
                resid_c.stride(0), resid_c.stride(1),
                y.stride(0), y.stride(1),
                resid_out.stride(0), resid_out.stride(1),
                BLOCK_SIZE=block,
                num_warps=warps,
                num_stages=3,
            )
        return y, resid_out

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # Fallback to native if Triton not available or CPU
        if (not _TRITON_AVAILABLE) or (not x.is_cuda):
            w = self.weight
            orig_dtype = x.dtype
            if residual is None:
                x32 = x.float()
                var = x32.pow(2).mean(dim=-1, keepdim=True)
                y32 = x32 * torch.rsqrt(var + self.variance_epsilon)
                y32 = y32 * (1.0 + w.float())
                return y32.to(orig_dtype)
            else:
                x_plus = x.float() + residual.float()
                resid_out = x_plus.to(x.dtype)
                x32 = x_plus.float()
                var = x32.pow(2).mean(dim=-1, keepdim=True)
                y32 = x32 * torch.rsqrt(var + self.variance_epsilon)
                y32 = y32 * (1.0 + self.weight.float())
                return y32.to(x.dtype), resid_out

        # Triton path
        if residual is None:
            return self._forward_no_res_triton(x)
        else:
            return self._forward_with_res_triton(x, residual)

GemmaRMSNorm = ModelNew
