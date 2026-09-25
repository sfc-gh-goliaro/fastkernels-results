from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < n_rows) & (cols[None, :] < N)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(
            residual_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        # vLLM stores the fused add in the input dtype before normalization.
        x = (x + residual).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_ptr + offsets, x, mask=mask)

    variance = tl.sum(x * x, axis=1) / N
    inv_rms = tl.rsqrt(variance + eps)
    weight = tl.load(
        weight_ptr + cols[None, :], mask=cols[None, :] < N, other=0.0
    ).to(tl.float32)
    out = x * inv_rms[:, None] * weight
    tl.store(out_ptr + offsets, out, mask=mask)


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_buffer(
                "_unit_weight", torch.ones(hidden_size), persistent=False
            )

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = (x * torch.rsqrt(variance + eps)).to(orig_dtype)
        if weight is not None:
            x = x * weight
        return x if residual is None else (x, residual)

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self.forward_native(
                x,
                self.weight if self.elementwise_affine else None,
                self.eps,
                self.hidden_size,
                residual,
            )

        weight = self.weight if self.elementwise_affine else self._unit_weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)

        n = x.shape[-1]
        rows = x.numel() // n
        if n <= 128:
            block_m = min(16, triton.next_power_of_2(rows))
            if block_m == 1:
                num_warps = 1
            else:
                num_warps = 8 if block_m >= 8 else 4
        elif n <= 512:
            block_m = 8
            num_warps = 8
        elif n <= 2560:
            block_m = 1
            num_warps = 4
        else:
            block_m = 2 if rows > 1 else 1
            num_warps = 8

        out = x if residual is not None else torch.empty_like(x)
        _rms_norm_kernel[(triton.cdiv(rows, block_m),)](
            x,
            residual,
            weight,
            out,
            rows,
            self.eps,
            N=n,
            BLOCK_N=triton.next_power_of_2(n),
            BLOCK_M=block_m,
            HAS_RESIDUAL=residual is not None,
            num_warps=num_warps,
        )
        return out if residual is None else (out, residual)
