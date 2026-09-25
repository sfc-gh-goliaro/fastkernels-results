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

    acc_a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idx[None, :],
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        wa = tl.load(
            wa_ptr + offs_n[None, :] * K + k_idx[:, None],
            mask=(offs_n[None, :] < N) & (k_idx[:, None] < K),
            other=0.0,
        )
        wb = tl.load(
            wb_ptr + offs_n[None, :] * K + k_idx[:, None],
            mask=(offs_n[None, :] < N) & (k_idx[:, None] < K),
            other=0.0,
        )
        acc_a += tl.dot(x, wa)
        acc_b += tl.dot(x, wb)

    # Match the BF16 materialization points in Linear -> SiLU -> multiply.
    a = acc_a.to(tl.bfloat16)
    b = acc_b.to(tl.bfloat16)
    silu = (a.to(tl.float32) * tl.sigmoid(a.to(tl.float32))).to(tl.bfloat16)
    out = (silu * b).to(tl.bfloat16)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _adaln_kernel(
    a_ptr,
    s_ptr,
    ln_weight_ptr,
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ca_cols = tl.arange(0, BLOCK_CA)
    cs_cols = tl.arange(0, BLOCK_CS)

    s = tl.load(
        s_ptr + rows[:, None] * CS + cs_cols[None, :],
        mask=(rows[:, None] < M) & (cs_cols[None, :] < CS),
        other=0.0,
    ).to(tl.float32)
    s_mean = tl.sum(s, axis=1) / CS
    s_centered = tl.where(cs_cols[None, :] < CS, s - s_mean[:, None], 0.0)
    s_var = tl.sum(s_centered * s_centered, axis=1) / CS
    ln_weight = tl.load(
        ln_weight_ptr + cs_cols,
        mask=cs_cols < CS,
        other=0.0,
    ).to(tl.float32)
    s_norm = (
        s_centered * tl.rsqrt(s_var[:, None] + 1.0e-5) * ln_weight[None, :]
    ).to(tl.bfloat16)

    wg = tl.load(
        wg_ptr + cols[None, :] * CS + cs_cols[:, None],
        mask=(cols[None, :] < CA) & (cs_cols[:, None] < CS),
        other=0.0,
    )
    ws = tl.load(
        ws_ptr + cols[None, :] * CS + cs_cols[:, None],
        mask=(cols[None, :] < CA) & (cs_cols[:, None] < CS),
        other=0.0,
    )
    gate_linear = tl.dot(s_norm, wg)
    shift = tl.dot(s_norm, ws)
    bias = tl.load(bg_ptr + cols, mask=cols < CA, other=0.0)
    gate_linear = (gate_linear + bias[None, :]).to(tl.bfloat16)
    gate = tl.sigmoid(gate_linear.to(tl.float32)).to(tl.bfloat16)
    shift = shift.to(tl.bfloat16)

    a_all = tl.load(
        a_ptr + rows[:, None] * CA + ca_cols[None, :],
        mask=(rows[:, None] < M) & (ca_cols[None, :] < CA),
        other=0.0,
    ).to(tl.float32)
    a_mean = tl.sum(a_all, axis=1) / CA
    a_centered = tl.where(ca_cols[None, :] < CA, a_all - a_mean[:, None], 0.0)
    a_var = tl.sum(a_centered * a_centered, axis=1) / CA
    a = tl.load(
        a_ptr + rows[:, None] * CA + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < CA),
        other=0.0,
    ).to(tl.float32)
    a_norm = ((a - a_mean[:, None]) * tl.rsqrt(a_var[:, None] + 1.0e-5)).to(
        tl.bfloat16
    )

    summed = (a_norm + shift).to(tl.bfloat16)
    out = (gate * summed).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * CA + cols[None, :],
        out,
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
        x = x.contiguous()
        m = x.numel() // self.linear_a.weight.shape[1]
        n, k = self.linear_a.weight.shape
        out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
        if m <= 16:
            block_k = 256 if k > 384 else 128
            stages = 2 if block_k == 256 else 3
            block_m, block_n, warps = 16, 64, 8
        elif m <= 128:
            block_m, block_n, block_k, warps, stages = 32, 64, 128, 4, 3
        else:
            block_m, block_n, block_k, warps, stages = 32, 64, 128, 4, 3
        _swiglu_kernel[
            (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        ](
            x,
            self.linear_a.weight,
            self.linear_b.weight,
            out,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        return out


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
        a = a.contiguous()
        s = s.contiguous()
        m = a.numel() // self.c_a
        out = torch.empty_like(a)
        if self.c_a == 768:
            block_m, block_n, warps, stages = 16, 32, 8, 3
        else:
            block_m, block_n, warps, stages = 32, 64, 4, 3
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
            num_warps=warps,
            num_stages=stages,
        )
        return out
