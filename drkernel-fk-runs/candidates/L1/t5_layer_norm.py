import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def _t5_rmsnorm_kernel_v2(
    X,             # *ptr to input [M, H]
    W,             # *ptr to weight [H]
    OUT,           # *ptr to output [M, H]
    YTMP,          # *ptr to tmp [M, H] (used if OUT_DTYPE < f32); can be same as OUT
    M, H,          # int: rows, cols
    stride_xm, stride_xh,
    stride_w,
    stride_om, stride_oh,
    stride_ym, stride_yh,
    eps,           # float32
    OUT_DTYPE: tl.constexpr,  # 0=f32, 1=f16, 2=bf16
    BLOCK: tl.constexpr
):
    m = tl.program_id(0)
    if m >= M:
        return

    # Accumulate sum of squares over H in float32
    sum_squares = 0.0
    off = 0
    while off < H:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H
        x = tl.load(X + m * stride_xm + idx * stride_xh, mask=mask, other=0.0)
        xf = x.to(tl.float32)
        sum_squares += tl.sum(xf * xf)
        off += BLOCK

    # Compute inv_std for the whole row
    hs = tl.full((), H, dtype=tl.float32)
    mean_sq = sum_squares / hs
    inv_std = 1.0 / tl.sqrt(mean_sq + eps)  # float32

    # Case A: OUT is float32 -> we can write final OUT = y*w during this pass (no second pass needed)
    if OUT_DTYPE == 0:
        off = 0
        while off < H:
            idx = off + tl.arange(0, BLOCK)
            mask = idx < H

            x = tl.load(X + m * stride_xm + idx * stride_xh, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + idx * stride_w, mask=mask, other=1.0).to(tl.float32)

            y = x * inv_std         # float32
            z = y * w               # float32

            tl.store(OUT + m * stride_om + idx * stride_oh, z, mask=mask)
            off += BLOCK
        return

    # Case B: OUT is f16/bf16
    # Pass 1B: while looping, compute y = x * inv_std and store to YTMP in reduced precision
    off = 0
    while off < H:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H

        x = tl.load(X + m * stride_xm + idx * stride_xh, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_std

        if OUT_DTYPE == 1:
            yt = y.to(tl.float16)
        else:  # 2 -> bf16
            yt = y.to(tl.bfloat16)

        tl.store(YTMP + m * stride_ym + idx * stride_yh, yt, mask=mask)
        off += BLOCK

    # Pass 2B: read YTMP, weight, compute out = cast(YTMP) * weight, store
    off = 0
    while off < H:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H

        if OUT_DTYPE == 1:
            yt = tl.load(YTMP + m * stride_ym + idx * stride_yh, mask=mask, other=tl.zeros((), dtype=tl.float16))
            yt_f = yt.to(tl.float32)
        else:
            yt = tl.load(YTMP + m * stride_ym + idx * stride_yh, mask=mask, other=tl.zeros((), dtype=tl.bfloat16))
            yt_f = yt.to(tl.float32)

        w = tl.load(W + idx * stride_w, mask=mask, other=1.0).to(tl.float32)

        z = yt_f * w
        # z is float32; cast to OUT dtype
        if OUT_DTYPE == 1:
            zo = z.to(tl.float16)
        else:
            zo = z.to(tl.bfloat16)

        tl.store(OUT + m * stride_om + idx * stride_oh, zo, mask=mask)
        off += BLOCK


def _torch_dtype_to_out_code(dtype: torch.dtype) -> int:
    # 0=f32, 1=f16, 2=bf16
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x - 1).bit_length())


def _choose_block_and_warps(H: int):
    block = min(1024, _next_power_of_two(H))
    block = max(64, block)
    if block <= 128:
        warps = 4
    elif block <= 256:
        warps = 4
    elif block <= 512:
        warps = 8
    else:
        warps = 8
    return block, warps


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, S, H] (or [..., H])
        Returns: same shape, dtype = weight.dtype
        """
        # CPU / non-CUDA fallback: vectorized PyTorch, preserving numerics
        if not hidden_states.is_cuda:
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=False)
            y = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                y = y.to(self.weight.dtype)
            return self.weight * y

        assert hidden_states.dim() >= 2, "Expected at least 2D tensor"
        *prefix, H = hidden_states.shape
        M = 1
        for d in prefix:
            M *= d
        X = hidden_states.view(M, H)

        out = torch.empty_like(X, dtype=self.weight.dtype, device=X.device)

        # Strides in elements
        stride_xm, stride_xh = X.stride(0), X.stride(1)
        stride_w = self.weight.stride(0)
        stride_om, stride_oh = out.stride(0), out.stride(1)

        BLOCK, num_warps = _choose_block_and_warps(H)
        grid = (M,)
        eps = float(self.variance_epsilon)

        out_code = _torch_dtype_to_out_code(out.dtype)

        # If OUT is f32, we can use OUT as YTMP (since pass 1 writes OUT and pass 2 is skipped).
        if out_code == 0:
            ytmp = out
        else:
            # Allocate a tmp tensor for y in reduced precision
            ytmp = torch.empty_like(X, dtype=_torch_code_to_dtype(out_code), device=X.device)

        stride_ym, stride_yh = ytmp.stride(0), ytmp.stride(1)

        _t5_rmsnorm_kernel_v2[grid](
            X, self.weight, out, ytmp,
            M, H,
            stride_xm, stride_xh,
            stride_w,
            stride_om, stride_oh,
            stride_ym, stride_yh,
            eps,
            out_code,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=2,
        )

        return out.view(*prefix, H)


def _torch_code_to_dtype(code: int) -> torch.dtype:
    if code == 1:
        return torch.float16
    if code == 2:
        return torch.bfloat16
    return torch.float32

T5LayerNorm = ModelNew
