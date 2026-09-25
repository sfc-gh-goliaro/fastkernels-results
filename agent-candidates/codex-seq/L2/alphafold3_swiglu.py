"""SwiGLU activation and AdaLN for AlphaFold3 (L2 composites).

SwiGLU: SiLU(linear_a(x)) * linear_b(x)
AdaLN: Adaptive Layer Normalization

Reference: openfold3/core/model/primitives/activations.py SwiGLU
           openfold3/core/model/primitives/normalization.py AdaLN
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _swiglu_kernel(
    x_ptr,
    wa_ptr,
    wb_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_a = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        wa = tl.load(
            wa_ptr + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        wb = tl.load(
            wb_ptr + offs_n[None, :] * K + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc_a += tl.dot(x, wa)
        acc_b += tl.dot(x, wb)
    a = acc_a.to(tl.bfloat16)
    b = acc_b.to(tl.bfloat16)

    # Match the inexpensive SiLU approximation used by the frozen L1 winner.
    af = a.to(tl.float32)
    a2 = af * af
    even = a2 * (0.2395166094 + a2 * (-0.0138038741 + a2 * 0.0004331403))
    silu = tl.maximum(-0.28, tl.minimum(0.5 * af + even, tl.maximum(af, 0.0)))
    silu = silu.to(tl.bfloat16)
    out = silu * b
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _adaln_kernel(
    a_ptr,
    s_ptr,
    ln_s_weight_ptr,
    wg_ptr,
    bg_ptr,
    ws_ptr,
    out_ptr,
    M: tl.constexpr,
    CA: tl.constexpr,
    CS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_CA: tl.constexpr,
    BLOCK_CS: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ka = tl.arange(0, BLOCK_CA)
    ks = tl.arange(0, BLOCK_CS)

    # Normalizing here avoids materializing either normalized input. For CA >
    # BLOCK_N, each output tile repeats the small row reduction from L2 cache.
    a_full = tl.load(
        a_ptr + rows[:, None] * CA + ka[None, :],
        mask=(rows[:, None] < M) & (ka[None, :] < CA),
        other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(a_full, axis=1) / CA
    a_centered = tl.where(ka[None, :] < CA, a_full - a_mean[:, None], 0.0)
    a_var = tl.sum(a_centered * a_centered, axis=1) / CA
    a_rstd = tl.rsqrt(a_var + EPS)

    s_full = tl.load(
        s_ptr + rows[:, None] * CS + ks[None, :],
        mask=(rows[:, None] < M) & (ks[None, :] < CS),
        other=0.0,
    ).to(tl.float32)
    s_mean = tl.sum(s_full, axis=1) / CS
    s_centered = tl.where(ks[None, :] < CS, s_full - s_mean[:, None], 0.0)
    s_var = tl.sum(s_centered * s_centered, axis=1) / CS
    s_rstd = tl.rsqrt(s_var + EPS)
    ln_weight = tl.load(
        ln_s_weight_ptr + ks,
        mask=ks < CS,
        other=0.0,
    ).to(tl.float32)
    s_norm = (s_centered * s_rstd[:, None] * ln_weight[None, :]).to(tl.bfloat16)

    wg = tl.load(
        wg_ptr + cols[None, :] * CS + ks[:, None],
        mask=(cols[None, :] < CA) & (ks[:, None] < CS),
        other=0.0,
    )
    ws = tl.load(
        ws_ptr + cols[None, :] * CS + ks[:, None],
        mask=(cols[None, :] < CA) & (ks[:, None] < CS),
        other=0.0,
    )
    gate_linear = tl.dot(s_norm, wg)
    gate_linear += tl.load(bg_ptr + cols, mask=cols < CA, other=0.0)[None, :]
    gate_linear = gate_linear.to(tl.bfloat16)
    shift = tl.dot(s_norm, ws).to(tl.bfloat16)
    gate = (1.0 / (1.0 + tl.exp(-gate_linear.to(tl.float32)))).to(tl.bfloat16)

    a_tile = tl.load(
        a_ptr + rows[:, None] * CA + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < CA),
        other=0.0,
    ).to(tl.float32)
    a_norm = ((a_tile - a_mean[:, None]) * a_rstd[:, None]).to(tl.bfloat16)
    combined = (a_norm + shift).to(tl.bfloat16)
    result = gate * combined
    tl.store(
        out_ptr + rows[:, None] * CA + cols[None, :],
        result,
        mask=(rows[:, None] < M) & (cols[None, :] < CA),
    )


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(Wa x) * Wb x.

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.silu = SiLU()
        self.linear_a = Linear(c_in, c_out, bias=False)
        self.linear_b = Linear(c_in, c_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and x.is_contiguous()
            and x.dtype == torch.bfloat16
            and x.shape[-1] == self.linear_a.weight.shape[1]
        ):
            m = x.numel() // x.shape[-1]
            k = x.shape[-1]
            n = self.linear_a.weight.shape[0]
            if (k, n) in ((64, 256), (128, 256), (384, 1536), (768, 1536)):
                out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
                if m <= 16:
                    block_m, block_n, warps = 16, 32, 4
                else:
                    block_m, block_n, warps = 64, 64, 4
                _swiglu_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
                    x,
                    self.linear_a.weight,
                    self.linear_b.weight,
                    out,
                    M=m,
                    N=n,
                    K=k,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=min(128, triton.next_power_of_2(k)),
                    num_warps=warps,
                    num_stages=3,
                )
                return out
        return self.silu(self.linear_a(x)) * self.linear_b(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization matching the reference AdaLN.

    Submodule structure matches checkpoint keys:
    - layer_norm_s: LayerNorm(c_s), weight-only
    - linear_g: Linear(c_s, c_a, bias=True) — gating
    - linear_s: Linear(c_s, c_a, bias=False) — additive conditioning

    Reference: openfold3/core/model/primitives/normalization.py AdaLN

    Args:
        c_a: Activation channel dimension
        c_s: Conditioning channel dimension
    """

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s

        self.layer_norm_a = LayerNorm(c_a, create_scale=False, create_offset=False)
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_s = Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if (
            a.is_cuda
            and a.is_contiguous()
            and s.is_contiguous()
            and a.dtype == torch.bfloat16
            and s.dtype == torch.bfloat16
            and (self.c_a, self.c_s) in ((128, 128), (768, 384))
            and a.numel() // self.c_a == s.numel() // self.c_s
        ):
            m = a.numel() // self.c_a
            out = torch.empty_like(a)
            if self.c_a == 128:
                block_m, block_n, warps = 16, 128, 4
            else:
                block_m, block_n, warps = 16, 32, 8
            _adaln_kernel[
                (triton.cdiv(m, block_m), triton.cdiv(self.c_a, block_n))
            ](
                a,
                s,
                self.layer_norm_s.weight,
                self.linear_g.weight,
                self.linear_g.bias,
                self.linear_s.weight,
                out,
                M=m,
                CA=self.c_a,
                CS=self.c_s,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_CA=triton.next_power_of_2(self.c_a),
                BLOCK_CS=triton.next_power_of_2(self.c_s),
                EPS=self.layer_norm_a.eps,
                num_warps=warps,
                num_stages=2,
            )
            return out
        s_norm = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s_norm))
        a_norm = self.layer_norm_a(a)
        return g * (a_norm + self.linear_s(s_norm))
