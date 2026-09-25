"""Specialized recurrent GLA decode kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fla.ops.gla import fused_recurrent_gla


@triton.jit
def _gla_decode(
    q,
    k,
    v,
    gk,
    h0,
    o,
    ht,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = i_nh * K * V + o_k[:, None] * V + o_v[None, :]

    b_q = tl.load(q + i_nh * K + o_k).to(tl.float32) * 0.0625
    b_k = tl.load(k + i_nh * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_nh * V + o_v).to(tl.float32)
    b_g = tl.exp(tl.load(gk + i_nh * K + o_k).to(tl.float32))
    b_h = tl.load(h0 + p_h).to(tl.float32)
    b_h = b_h * b_g[:, None] + b_k[:, None] * b_v[None, :]

    b_o = tl.sum(b_h * b_q[:, None], axis=0)
    tl.store(ht + p_h, b_h)
    tl.store(o + i_nh * V + o_v, b_o)


@triton.jit
def _gla_decode_partial(
    q,
    k,
    v,
    gk,
    h0,
    o_partial,
    ht,
    NH: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_nh = tl.program_id(2)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = i_nh * K * V + o_k[:, None] * V + o_v[None, :]

    b_q = tl.load(q + i_nh * K + o_k).to(tl.float32) * 0.0625
    b_k = tl.load(k + i_nh * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_nh * V + o_v).to(tl.float32)
    b_g = tl.exp(tl.load(gk + i_nh * K + o_k).to(tl.float32))
    b_h = tl.load(h0 + p_h).to(tl.float32)
    b_h = b_h * b_g[:, None] + b_k[:, None] * b_v[None, :]

    b_o = tl.sum(b_h * b_q[:, None], axis=0)
    tl.store(ht + p_h, b_h)
    tl.store(o_partial + (i_k * NH + i_nh) * V + o_v, b_o)


@triton.jit
def _reduce_decode(o_partial, o, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(o_partial + offs, mask=mask, other=0.0)
    x += tl.load(o_partial + N + offs, mask=mask, other=0.0)
    tl.store(o + offs, x, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _gla_prefill(
    q,
    k,
    v,
    gk,
    o_partial,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_h = tl.program_id(2)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    b_h = tl.zeros((BK, BV), tl.float32)

    for i_t in range(T):
        qkv_k = i_t * H * K + i_h * K + o_k
        qv_v = i_t * H * V + i_h * V + o_v
        b_q = tl.load(q + qkv_k).to(tl.float32) * 0.0625
        b_k = tl.load(k + qkv_k).to(tl.float32)
        b_v = tl.load(v + qv_v).to(tl.float32)
        b_g = tl.exp(tl.load(gk + qkv_k).to(tl.float32))
        b_h = b_h * b_g[:, None] + b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        p_o = ((i_k * T + i_t) * H + i_h) * V + o_v
        tl.store(o_partial + p_o, b_o)

    p_h = i_h * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(ht + p_h, b_h)


@triton.jit
def _reduce_prefill(o_partial, o, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(o_partial + offs, mask=mask, other=0.0)
    x += tl.load(o_partial + N + offs, mask=mask, other=0.0)
    x += tl.load(o_partial + 2 * N + offs, mask=mask, other=0.0)
    x += tl.load(o_partial + 3 * N + offs, mask=mask, other=0.0)
    tl.store(o + offs, x, mask=mask)


class FusedRecurrentGLA(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor | None = None,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            q.shape[1] == 1
            and q.shape[2:] == (5, 256)
            and v.shape[-1] == 512
            and gk is not None
            and initial_state is not None
            and cu_seqlens is None
            and output_final_state
            and (scale is None or scale == 0.0625)
        ):
            ht = torch.empty_like(initial_state)
            o = torch.empty_like(v)
            nh = q.shape[0] * 5
            if q.shape[0] >= 128:
                o_partial = torch.empty((2, *v.shape), device=q.device, dtype=torch.float32)
                _gla_decode_partial[(16, 2, nh)](
                    q, k, v, gk, initial_state, o_partial, ht,
                    NH=nh, K=256, V=512, BK=128, BV=32,
                    num_warps=4,
                )
                n = v.numel()
                _reduce_decode[(triton.cdiv(n, 512),)](
                    o_partial, o, N=n, BLOCK=512, num_warps=4,
                )
            else:
                _gla_decode[(32, nh)](
                    q, k, v, gk, initial_state, o, ht,
                    H=5, K=256, V=512, BV=16,
                    num_warps=4,
                )
            return o, ht

        if (
            q.shape == (1, 42, 5, 256)
            and v.shape[-1] == 512
            and gk is not None
            and initial_state is None
            and cu_seqlens is None
            and output_final_state
            and (scale is None or scale == 0.0625)
        ):
            o = torch.empty_like(v)
            ht = torch.empty((1, 5, 256, 512), device=q.device, dtype=torch.float32)
            o_partial = torch.empty((4, *v.shape), device=q.device, dtype=torch.float32)
            _gla_prefill[(8, 4, 5)](
                q, k, v, gk, o_partial, ht,
                T=42, H=5, K=256, V=512, BK=64, BV=64,
                num_warps=4,
            )
            n = v.numel()
            _reduce_prefill[(triton.cdiv(n, 512),)](
                o_partial, o, N=n, BLOCK=512, num_warps=4,
            )
            return o, ht

        return fused_recurrent_gla(
            q=q,
            k=k,
            v=v,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
