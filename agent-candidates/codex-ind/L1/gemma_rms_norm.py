from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemma_rms_norm_2048(x, weight, out, eps: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 2048)
    offsets = row * 2048 + cols

    values = tl.load(x + offsets).to(tl.float32)
    variance = tl.sum(values * values) * (1.0 / 2048.0)
    rstd = tl.rsqrt(variance + eps)
    scales = 1.0 + tl.load(weight + cols).to(tl.float32)
    tl.store(out + offsets, values * rstd * scales)


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            orig_dtype = x.dtype
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x.to(orig_dtype) if x.dtype != orig_dtype else x
            values = x.float()
            variance = values.pow(2).mean(dim=-1, keepdim=True)
            values *= torch.rsqrt(variance + self.variance_epsilon)
            values *= 1.0 + self.weight.float()
            return values.to(orig_dtype), residual

        if x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == 2048:
            rows = x.numel() // 2048
            out = torch.empty_like(x)
            _gemma_rms_norm_2048[(rows,)](
                x,
                self.weight,
                out,
                eps=self.variance_epsilon,
                num_warps=2,
            )
            return out

        values = x.float()
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        values *= torch.rsqrt(variance + self.variance_epsilon)
        values *= 1.0 + self.weight.float()
        return values.to(x.dtype)
