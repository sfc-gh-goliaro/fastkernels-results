"""Fused SwiGLU transition kernels for the captured AlphaFold3 shapes."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN, SwiGLU


@triton.jit
def _ln_swiglu_kernel(
    x_ptr, ln_w_ptr, wa_ptr, wb_ptr, h_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BKR: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    kr = tl.arange(0, BKR)

    x = tl.load(
        x_ptr + rows[:, None] * K + kr[None, :],
        mask=(rows[:, None] < M) & (kr[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    centered = tl.where(kr[None, :] < K, x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / K
    inv_std = tl.rsqrt(var + 1.0e-5)
    acc_a = tl.zeros((BM, BN), tl.float32)
    acc_b = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        kk = start + tl.arange(0, BK)
        xv = tl.load(
            x_ptr + rows[:, None] * K + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        ln_w = tl.load(ln_w_ptr + kk, mask=kk < K, other=0.0).to(tl.float32)
        norm = ((xv - mean[:, None]) * inv_std[:, None] * ln_w[None, :]).to(tl.bfloat16)
        wa = tl.load(
            wa_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        wb = tl.load(
            wb_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc_a += tl.dot(norm, wa)
        acc_b += tl.dot(norm, wb)
    av = acc_a.to(tl.bfloat16)
    bv = acc_b.to(tl.bfloat16)
    af = av.to(tl.float32)
    silu = (af * tl.sigmoid(af)).to(tl.bfloat16)
    hidden = (silu * bv).to(tl.bfloat16)
    tl.store(
        h_ptr + rows[:, None] * N + cols[None, :],
        hidden,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _swiglu_kernel(
    x_ptr, wa_ptr, wb_ptr, h_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc_a = tl.zeros((BM, BN), tl.float32)
    acc_b = tl.zeros((BM, BN), tl.float32)
    for start in range(0, K, BK):
        kk = start + tl.arange(0, BK)
        x = tl.load(
            x_ptr + rows[:, None] * K + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        wa = tl.load(
            wa_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        wb = tl.load(
            wb_ptr + cols[None, :] * K + kk[:, None],
            mask=(cols[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc_a += tl.dot(x, wa)
        acc_b += tl.dot(x, wb)
    av = acc_a.to(tl.bfloat16)
    bv = acc_b.to(tl.bfloat16)
    af = av.to(tl.float32)
    silu = (af * tl.sigmoid(af)).to(tl.bfloat16)
    hidden = (silu * bv).to(tl.bfloat16)
    tl.store(
        h_ptr + rows[:, None] * N + cols[None, :],
        hidden,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _ln_swiglu_output_kernel(
    x_ptr, ln_w_ptr, wa_ptr, wb_ptr, wo_ptr, mask_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
    BKR: tl.constexpr, BK: tl.constexpr, BH: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, HAS_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    kr = tl.arange(0, BKR)
    x = tl.load(
        x_ptr + rows[:, None] * K + kr[None, :],
        mask=(rows[:, None] < M) & (kr[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    centered = tl.where(kr[None, :] < K, x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / K
    inv_std = tl.rsqrt(var + 1.0e-5)
    out_acc = tl.zeros((BM, BN), tl.float32)

    for hs in range(0, H, BH):
        hh = hs + tl.arange(0, BH)
        acc_a = tl.zeros((BM, BH), tl.float32)
        acc_b = tl.zeros((BM, BH), tl.float32)
        for start in range(0, K, BK):
            kk = start + tl.arange(0, BK)
            xv = tl.load(
                x_ptr + rows[:, None] * K + kk[None, :],
                mask=(rows[:, None] < M) & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            lw = tl.load(ln_w_ptr + kk, mask=kk < K, other=0.0).to(tl.float32)
            norm = ((xv - mean[:, None]) * inv_std[:, None] * lw[None, :]).to(tl.bfloat16)
            wa = tl.load(
                wa_ptr + hh[None, :] * K + kk[:, None],
                mask=(hh[None, :] < H) & (kk[:, None] < K),
                other=0.0,
            )
            wb = tl.load(
                wb_ptr + hh[None, :] * K + kk[:, None],
                mask=(hh[None, :] < H) & (kk[:, None] < K),
                other=0.0,
            )
            acc_a += tl.dot(norm, wa)
            acc_b += tl.dot(norm, wb)
        av = acc_a.to(tl.bfloat16)
        bv = acc_b.to(tl.bfloat16)
        af = av.to(tl.float32)
        hidden = ((af * tl.sigmoid(af)).to(tl.bfloat16) * bv).to(tl.bfloat16)
        wo = tl.load(
            wo_ptr + cols[None, :] * H + hh[:, None],
            mask=(cols[None, :] < K) & (hh[:, None] < H),
            other=0.0,
        )
        out_acc += tl.dot(hidden, wo)

    y = out_acc.to(tl.bfloat16)
    if HAS_MASK:
        mask = tl.load(mask_ptr + rows, mask=rows < M, other=0.0)
        y = (y * mask[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * K + cols[None, :],
        y,
        mask=(rows[:, None] < M) & (cols[None, :] < K),
    )


@triton.jit
def _swiglu_output_kernel(
    x_ptr, wa_ptr, wb_ptr, wo_ptr, gate_ptr, mask_ptr, out_ptr,
    M: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
    BK: tl.constexpr, BH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    out_acc = tl.zeros((BM, BN), tl.float32)
    for hs in range(0, H, BH):
        hh = hs + tl.arange(0, BH)
        acc_a = tl.zeros((BM, BH), tl.float32)
        acc_b = tl.zeros((BM, BH), tl.float32)
        for start in range(0, K, BK):
            kk = start + tl.arange(0, BK)
            x = tl.load(
                x_ptr + rows[:, None] * K + kk[None, :],
                mask=(rows[:, None] < M) & (kk[None, :] < K),
                other=0.0,
            )
            wa = tl.load(
                wa_ptr + hh[None, :] * K + kk[:, None],
                mask=(hh[None, :] < H) & (kk[:, None] < K),
                other=0.0,
            )
            wb = tl.load(
                wb_ptr + hh[None, :] * K + kk[:, None],
                mask=(hh[None, :] < H) & (kk[:, None] < K),
                other=0.0,
            )
            acc_a += tl.dot(x, wa)
            acc_b += tl.dot(x, wb)
        av = acc_a.to(tl.bfloat16)
        bv = acc_b.to(tl.bfloat16)
        af = av.to(tl.float32)
        hidden = ((af * tl.sigmoid(af)).to(tl.bfloat16) * bv).to(tl.bfloat16)
        wo = tl.load(
            wo_ptr + cols[None, :] * H + hh[:, None],
            mask=(cols[None, :] < K) & (hh[:, None] < H),
            other=0.0,
        )
        out_acc += tl.dot(hidden, wo)
    y = out_acc.to(tl.bfloat16)
    gate = tl.load(
        gate_ptr + rows[:, None] * K + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < K),
        other=0.0,
    )
    y = (y * gate).to(tl.bfloat16)
    if HAS_MASK:
        mask = tl.load(mask_ptr + rows, mask=rows < M, other=0.0)
        y = (y * mask[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * K + cols[None, :],
        y,
        mask=(rows[:, None] < M) & (cols[None, :] < K),
    )


@triton.jit
def _output_kernel(
    h_ptr, w_ptr, mask_ptr, gate_ptr, out_ptr,
    M: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    HAS_MASK: tl.constexpr, HAS_GATE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(0, H, BK):
        kk = start + tl.arange(0, BK)
        h = tl.load(
            h_ptr + rows[:, None] * H + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < H),
            other=0.0,
        )
        w = tl.load(
            w_ptr + cols[None, :] * H + kk[:, None],
            mask=(cols[None, :] < K) & (kk[:, None] < H),
            other=0.0,
        )
        acc += tl.dot(h, w)
    y = acc.to(tl.bfloat16)
    if HAS_GATE:
        gate = tl.load(
            gate_ptr + rows[:, None] * K + cols[None, :],
            mask=(rows[:, None] < M) & (cols[None, :] < K),
            other=0.0,
        )
        y = (y * gate).to(tl.bfloat16)
    if HAS_MASK:
        mask = tl.load(mask_ptr + rows, mask=rows < M, other=0.0)
        y = (y * mask[:, None]).to(tl.bfloat16)
    tl.store(
        out_ptr + rows[:, None] * K + cols[None, :],
        y,
        mask=(rows[:, None] < M) & (cols[None, :] < K),
    )


@triton.jit
def _condition_kernel(
    a_ptr, s_ptr, lns_w_ptr,
    adag_w_ptr, adag_b_ptr, shift_w_ptr,
    outg_w_ptr, outg_b_ptr,
    conditioned_ptr, out_gate_ptr,
    M: tl.constexpr, KA: tl.constexpr, KS: tl.constexpr,
    BKA: tl.constexpr, BKS: tl.constexpr,
    BK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    ka = tl.arange(0, BKA)
    ks = tl.arange(0, BKS)

    a = tl.load(
        a_ptr + rows[:, None] * KA + ka[None, :],
        mask=(rows[:, None] < M) & (ka[None, :] < KA),
        other=0.0,
    ).to(tl.float32)
    amean = tl.sum(a, axis=1) / KA
    ac = tl.where(ka[None, :] < KA, a - amean[:, None], 0.0)
    avar = tl.sum(ac * ac, axis=1) / KA

    s = tl.load(
        s_ptr + rows[:, None] * KS + ks[None, :],
        mask=(rows[:, None] < M) & (ks[None, :] < KS),
        other=0.0,
    ).to(tl.float32)
    smean = tl.sum(s, axis=1) / KS
    sc = tl.where(ks[None, :] < KS, s - smean[:, None], 0.0)
    svar = tl.sum(sc * sc, axis=1) / KS

    adag_b = tl.load(adag_b_ptr + cols, mask=cols < KA, other=0.0)
    outg_b = tl.load(outg_b_ptr + cols, mask=cols < KA, other=0.0)
    inv_s = tl.rsqrt(svar + 1.0e-5)
    acc_adag = tl.zeros((BM, BN), tl.float32)
    acc_shift = tl.zeros((BM, BN), tl.float32)
    acc_outg = tl.zeros((BM, BN), tl.float32)
    for start in range(0, KS, BK):
        kk = start + tl.arange(0, BK)
        sv = tl.load(
            s_ptr + rows[:, None] * KS + kk[None, :],
            mask=(rows[:, None] < M) & (kk[None, :] < KS),
            other=0.0,
        ).to(tl.float32)
        wnorm = tl.load(lns_w_ptr + kk, mask=kk < KS, other=0.0).to(tl.float32)
        sn = ((sv - smean[:, None]) * inv_s[:, None] * wnorm[None, :]).to(tl.bfloat16)
        adag_w = tl.load(
            adag_w_ptr + cols[None, :] * KS + kk[:, None],
            mask=(cols[None, :] < KA) & (kk[:, None] < KS),
            other=0.0,
        )
        shift_w = tl.load(
            shift_w_ptr + cols[None, :] * KS + kk[:, None],
            mask=(cols[None, :] < KA) & (kk[:, None] < KS),
            other=0.0,
        )
        outg_w = tl.load(
            outg_w_ptr + cols[None, :] * KS + kk[:, None],
            mask=(cols[None, :] < KA) & (kk[:, None] < KS),
            other=0.0,
        )
        acc_adag += tl.dot(sn, adag_w)
        acc_shift += tl.dot(sn, shift_w)
        acc_outg += tl.dot(sv.to(tl.bfloat16), outg_w)

    adag = (acc_adag + adag_b[None, :]).to(tl.bfloat16)
    shift = acc_shift.to(tl.bfloat16)
    outg = (acc_outg + outg_b[None, :]).to(tl.bfloat16)
    adag = tl.sigmoid(adag.to(tl.float32)).to(tl.bfloat16)
    outg = tl.sigmoid(outg.to(tl.float32)).to(tl.bfloat16)

    a_tile = tl.load(
        a_ptr + rows[:, None] * KA + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < KA),
        other=0.0,
    ).to(tl.float32)
    anorm = (
        (a_tile - amean[:, None]) * tl.rsqrt(avar[:, None] + 1.0e-5)
    ).to(tl.bfloat16)
    conditioned = ((anorm + shift).to(tl.bfloat16) * adag).to(tl.bfloat16)
    valid = (rows[:, None] < M) & (cols[None, :] < KA)
    tl.store(conditioned_ptr + rows[:, None] * KA + cols[None, :], conditioned, mask=valid)
    tl.store(out_gate_ptr + rows[:, None] * KA + cols[None, :], outg, mask=valid)


def _matmul_grid(m: int, n: int, bm: int, bn: int):
    return (triton.cdiv(m, bm), triton.cdiv(n, bn))


class SwiGLUTransition(nn.Module):
    def __init__(self, c_in: int, n: int):
        super().__init__()
        self.c_in = c_in
        self.n = n
        self.layer_norm = LayerNorm(c_in)
        self.swiglu = SwiGLU(c_in, n * c_in)
        self.linear_out = Linear(n * c_in, c_in, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        ckpt_chunk_size: int | None = None,
    ) -> torch.Tensor:
        k = self.c_in
        n = self.n * k
        m = x.numel() // k
        out = torch.empty((m, k), device=x.device, dtype=x.dtype)
        bm = 16 if m <= 16 else 32
        bn = 64 if k == 384 else 128
        bk = 64 if k == 64 else 128
        if k == 64:
            _ln_swiglu_output_kernel[(triton.cdiv(m, bm),)](
                x, self.layer_norm.weight,
                self.swiglu.linear_a.weight, self.swiglu.linear_b.weight,
                self.linear_out.weight, mask, out,
                M=m, K=k, H=n, BKR=triton.next_power_of_2(k),
                BK=bk, BH=64, BM=bm, BN=bn, HAS_MASK=mask is not None,
                num_warps=4, num_stages=2,
            )
            return out.view(x.shape)
        hidden = torch.empty((m, n), device=x.device, dtype=x.dtype)
        _ln_swiglu_kernel[_matmul_grid(m, n, bm, bn)](
            x, self.layer_norm.weight,
            self.swiglu.linear_a.weight, self.swiglu.linear_b.weight, hidden,
            M=m, K=k, N=n, BKR=triton.next_power_of_2(k), BK=bk, BM=bm, BN=bn,
            num_warps=4, num_stages=2,
        )
        _output_kernel[_matmul_grid(m, k, bm, bn)](
            hidden, self.linear_out.weight, mask, out, out,
            M=m, H=n, K=k, HAS_MASK=mask is not None, HAS_GATE=False,
            BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=2,
        )
        return out.view(x.shape)


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_a: int, c_s: int, n: int):
        super().__init__()
        self.layer_norm = AdaLN(c_a=c_a, c_s=c_s)
        self.swiglu = SwiGLU(c_a, n * c_a)
        self.sigmoid = Sigmoid()
        self.linear_g = Linear(c_s, c_a, bias=True)
        self.linear_out = Linear(n * c_a, c_a, bias=False)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        ka = a.shape[-1]
        ks = s.shape[-1]
        m = a.numel() // ka
        n = self.swiglu.linear_a.weight.shape[0]
        conditioned = torch.empty((m, ka), device=a.device, dtype=a.dtype)
        out_gate = torch.empty_like(conditioned)
        out = torch.empty_like(conditioned)
        bm = 16
        bn = 64
        bk = 128
        _condition_kernel[_matmul_grid(m, ka, bm, bn)](
            a, s, self.layer_norm.layer_norm_s.weight,
            self.layer_norm.linear_g.weight, self.layer_norm.linear_g.bias,
            self.layer_norm.linear_s.weight,
            self.linear_g.weight, self.linear_g.bias,
            conditioned, out_gate,
            M=m, KA=ka, KS=ks,
            BKA=triton.next_power_of_2(ka), BKS=triton.next_power_of_2(ks),
            BK=bk, BM=bm, BN=bn, num_warps=4, num_stages=2,
        )
        if ka == 128:
            _swiglu_output_kernel[(triton.cdiv(m, bm),)](
                conditioned, self.swiglu.linear_a.weight,
                self.swiglu.linear_b.weight, self.linear_out.weight,
                out_gate, mask, out,
                M=m, K=ka, H=n, BK=bk, BH=64, BM=bm, BN=128,
                HAS_MASK=mask is not None, num_warps=4, num_stages=2,
            )
            return out.view(a.shape)
        hidden = torch.empty((m, n), device=a.device, dtype=a.dtype)
        _swiglu_kernel[_matmul_grid(m, n, bm, bn)](
            conditioned, self.swiglu.linear_a.weight,
            self.swiglu.linear_b.weight, hidden,
            M=m, K=ka, N=n, BK=bk, BM=bm, BN=bn,
            num_warps=4, num_stages=2,
        )
        _output_kernel[_matmul_grid(m, ka, bm, bn)](
            hidden, self.linear_out.weight, mask, out_gate, out,
            M=m, H=n, K=ka, HAS_MASK=mask is not None, HAS_GATE=True,
            BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=2,
        )
        return out.view(a.shape)
