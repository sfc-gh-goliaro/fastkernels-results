import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


def _validate_pad_general(ndim: int, pad):
    if not isinstance(pad, (tuple, list)):
        raise TypeError(f"pad must be tuple/list, got {type(pad)}")
    pad = tuple(int(p) for p in pad)
    if len(pad) != 2 * ndim:
        # Let PyTorch handle unusual pad formats
        return None
    for p in pad:
        if p < 0:
            raise ValueError(f"negative padding not supported: {pad}")
    return pad


@triton.jit
def _pad_const_1d_kernel_2d(
    x_ptr, out_ptr, value,
    L_in: tl.constexpr,  # int
    L_out: tl.constexpr, # int
    pad0: tl.constexpr,  # int
    pad1: tl.constexpr,  # int
    BLOCK_W: tl.constexpr,
):
    # 2D grid: pid0 over output elements (ceil-div by BLOCK_W), pid1 over W-chunks
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    w = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < L_out

    base = pid0 * BLOCK_W
    offs = base + w
    in_idx = offs - pad0
    in_range = (in_idx >= 0) & (in_idx < L_in)
    x_val = tl.load(x_ptr + in_idx, mask=in_range & mask, other=value)
    tl.store(out_ptr + offs, x_val, mask=mask)


@triton.jit
def _pad_const_3d_kernel_2d(
    x_ptr, out_ptr, value,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,     # input shape
    ON: tl.constexpr, OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,  # output shape
    pN0: tl.constexpr, pC0: tl.constexpr, pH0: tl.constexpr, pW0: tl.constexpr,  # pad before
    pN1: tl.constexpr, pC1: tl.constexpr, pH1: tl.constexpr, pW1: tl.constexpr,  # pad after
    BLOCK_W: tl.constexpr,
):
    # Map linear pid0 -> (n, c, h); vectorize over w
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decode (n, c, h) from pid0 without div/mod per element
    # total_rows = N * C * H
    # n = pid0 // (C*H)
    # t = pid0 % (C*H)
    # c = t // H
    # h = t % H
    CH = C * H
    n = pid0 // CH
    t = pid0 % CH
    c = t // H
    h = t % H

    w = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < OW

    # Input coords
    in_n = n - pN0
    in_c = c - pC0
    in_h = h - pH0
    in_w = w - pW0  # pW0==0, pW1 unused (after) for 3D here

    # Bounds
    valid = (in_n >= 0) & (in_n < N) & \
            (in_c >= 0) & (in_c < C) & \
            (in_h >= 0) & (in_h < H) & \
            (in_w >= 0) & (in_w < W)

    # Input linear index base for this (n,c,h): base_in = n*(C*H*W) + c*(H*W) + h*W
    base_in = n * (C * H * W) + c * (H * W) + h * W
    in_idx = base_in + in_w

    x_val = tl.load(x_ptr + in_idx, mask=valid & mask, other=value)
    # Output linear index base: (n*OC + c*OH + h)*OW + w
    out_base = ((n * OC + c * OH) + h) * OW  # simplifies to n*OC*OH*OW + c*OH*OW + h*OW + w but we add w separately
    # Correct out base: n*(OC*OH*OW) + c*(OH*OW) + h*OW
    out_base = n * (OC * OH * OW) + c * (OH * OW) + h * OW
    out_idx = out_base + w
    tl.store(out_ptr + out_idx, x_val, mask=mask)


@triton.jit
def _pad_const_4d_kernel_2d(
    x_ptr, out_ptr, value,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,     # input shape
    ON: tl.constexpr, OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,  # output shape
    pN0: tl.constexpr, pC0: tl.constexpr, pH0: tl.constexpr, pW0: tl.constexpr,  # pad before
    pN1: tl.constexpr, pC1: tl.constexpr, pH1: tl.constexpr, pW1: tl.constexpr,  # pad after
    BLOCK_W: tl.constexpr,
):
    # 2D grid: pid0 over N*C*H rows, pid1 over W tiles
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    CH = C * H
    n = pid0 // CH
    t = pid0 % CH
    c = t // H
    h = t % H

    w = pid1 * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = w < OW

    # Input coords
    in_n = n - pN0
    in_c = c - pC0
    in_h = h - pH0
    in_w = w - pW0

    # Bounds
    valid = (in_n >= 0) & (in_n < N) & \
            (in_c >= 0) & (in_c < C) & \
            (in_h >= 0) & (in_h < H) & \
            (in_w >= 0) & (in_w < W)

    # Input base index for this (n,c,h): n*(C*H*W) + c*(H*W) + h*W
    base_in = n * (C * H * W) + c * (H * W) + h * W
    in_idx = base_in + in_w

    x_val = tl.load(x_ptr + in_idx, mask=valid & mask, other=value)

    # Output base index: n*(OC*OH*OW) + c*(OH*OW) + h*OW
    out_base = n * (OC * OH * OW) + c * (OH * OW) + h * OW
    out_idx = out_base + w
    tl.store(out_ptr + out_idx, x_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, pad: tuple[int, ...], value: float = 0.0) -> torch.Tensor:
        # Fallback if Triton not available or tensor not on CUDA
        if (not _HAS_TRITON) or (not x.is_cuda):
            return F.pad(x, pad, value=value)

        ndim = x.ndim
        pad_tup = _validate_pad_general(ndim, pad)
        if pad_tup is None:
            # Let PyTorch handle unusual pad
            return F.pad(x, pad, value=value)
        pad = pad_tup

        # Dtype support
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32,
                           torch.int64, torch.int32):
            return F.pad(x, pad, value=value)

        # Make sure tensor is contiguous
        if not x.is_contiguous():
            x = x.contiguous()

        shape = list(x.shape)

        # Compute out shape
        out_shape = shape.copy()
        pad_before = pad[0::2]
        pad_after = pad[1::2]
        for k in range(ndim):
            out_shape[k] += pad[2 * k] + pad[2 * k + 1]

        # Prepare value
        if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            val = float(value)
        else:
            val = int(value)

        if ndim == 1:
            L_in = shape[0]
            L_out = out_shape[0]
            p0, p1 = pad_before[0], pad_after[0]
            out = torch.empty((L_out,), device=x.device, dtype=x.dtype)

            BLOCK_W = 128
            grid = (triton.cdiv(L_out, BLOCK_W), 1)
            _pad_const_1d_kernel_2d[grid](
                x, out, val,
                L_in, L_out,
                p0, p1,
                BLOCK_W=BLOCK_W,
            )
            return out

        # 3D and 4D: use 2D tiling over W
        if ndim == 3:
            # Expect (N, C, H); W=1
            N, C, H = shape
            ON, OC, OH = out_shape
            pN0, pC0, pH0 = pad_before
            pN1, pC1, pH1 = pad_after
            W = 1
            OW = 1

            out = torch.empty((ON, OC, OH), device=x.device, dtype=x.dtype)

            BLOCK_W = 128
            grid = (N * C * H, triton.cdiv(OW, BLOCK_W))
            _pad_const_3d_kernel_2d[grid](
                x, out, val,
                N, C, H, W,
                ON, OC, OH, OW,
                pN0, pC0, pH0, 0,
                pN1, pC1, pH1, 0,
                BLOCK_W=BLOCK_W,
            )
            return out

        # ndim == 4
        N, C, H, W = shape
        ON, OC, OH, OW = out_shape
        pN0, pC0, pH0, pW0 = pad_before
        pN1, pC1, pH1, pW1 = pad_after

        out = torch.empty((ON, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 128
        grid = (N * C * H, triton.cdiv(OW, BLOCK_W))
        _pad_const_4d_kernel_2d[grid](
            x, out, val,
            N, C, H, W,
            ON, OC, OH, OW,
            pN0, pC0, pH0, pW0,
            pN1, pC1, pH1, pW1,
            BLOCK_W=BLOCK_W,
        )
        return out

    # Fallback for other ranks/dtypes is handled above.

Pad = ModelNew
