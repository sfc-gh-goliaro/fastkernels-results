"""Specialized Chunk GLA forward kernels."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


_RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _packed_cumsum(
    g,
    cu,
    gc,
    H: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    DENSE: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_h = tl.program_id(1)
    i_n = tl.program_id(2)

    if DENSE:
        bos = i_n * T
        seq_len = T
    else:
        bos = tl.load(cu + i_n).to(tl.int64)
        seq_len = tl.load(cu + i_n + 1).to(tl.int64) - bos
    o_t = tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    mask = (o_t[:, None] < seq_len) & (o_k[None, :] < K)
    x = tl.load(
        g + (bos + o_t[:, None]) * H * K + i_h * K + o_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.cumsum(x, axis=0) * _RCP_LN2
    tl.store(
        gc + ((i_n * BT + o_t[:, None]) * H + i_h) * K + o_k[None, :],
        x,
        mask=mask,
    )


@triton.jit
def _packed_output(
    q,
    k,
    v,
    gc,
    h0,
    cu,
    out,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DENSE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    i_n = tl.program_id(2)

    if DENSE:
        bos = i_n * T
        seq_len = T
    else:
        bos = tl.load(cu + i_n).to(tl.int64)
        seq_len = tl.load(cu + i_n + 1).to(tl.int64) - bos
    o_t = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < seq_len
    m_v = o_v < V
    scale = 1.0 / tl.sqrt(float(K))
    acc = tl.zeros((BT, BV), dtype=tl.float32)
    attn = tl.zeros((BT, BT), dtype=tl.float32)

    for i_k in range(0, K, BK):
        o_k = i_k + tl.arange(0, BK)
        m_tk = m_t[:, None] & (o_k[None, :] < K)
        qk = tl.load(
            q + (bos + o_t[:, None]) * H * K + i_h * K + o_k[None, :],
            mask=m_tk,
            other=0.0,
        )
        kk = tl.load(
            k + (bos + o_t[:, None]) * H * K + i_h * K + o_k[None, :],
            mask=m_tk,
            other=0.0,
        )
        gg = tl.load(
            gc + ((i_n * BT + o_t[:, None]) * H + i_h) * K + o_k[None, :],
            mask=m_tk,
            other=0.0,
        )
        gl = tl.load(
            gc + ((i_n * BT + (seq_len - 1)) * H + i_h) * K + o_k,
            mask=o_k < K,
            other=0.0,
        )

        qg = (qk * tl.exp2(gg)).to(tl.bfloat16)
        hb = tl.load(
            h0 + (i_n * H + i_h) * K * V + o_k[:, None] * V + o_v[None, :],
            mask=(o_k[:, None] < K) & m_v[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(qg, hb)

        qa = qk * tl.exp2(gg - gl[None, :])
        ka = kk * tl.exp2(gl[None, :] - gg)
        attn += tl.dot(qa, tl.trans(ka))

    causal = o_t[:, None] >= o_t[None, :]
    attn = tl.where(causal & m_t[:, None] & m_t[None, :], attn * scale, 0.0)
    vv = tl.load(
        v + (bos + o_t[:, None]) * H * V + i_h * V + o_v[None, :],
        mask=m_t[:, None] & m_v[None, :],
        other=0.0,
    )
    acc = acc * scale + tl.dot(attn.to(tl.bfloat16), vv)
    tl.store(
        out + (bos + o_t[:, None]) * H * V + i_h * V + o_v[None, :],
        acc,
        mask=m_t[:, None] & m_v[None, :],
    )


@triton.jit
def _packed_state(
    k,
    v,
    gate,
    gc,
    h0,
    cu,
    ht,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DENSE: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H

    if DENSE:
        bos = i_n * T
        seq_len = T
    else:
        bos = tl.load(cu + i_n).to(tl.int64)
        seq_len = tl.load(cu + i_n + 1).to(tl.int64) - bos
    o_t = tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask = (o_k[:, None] < K) & (o_v[None, :] < V)
    m_t = o_t < seq_len
    offset = i_nh * K * V + o_k[:, None] * V + o_v[None, :]
    old = tl.load(h0 + offset, mask=mask, other=0.0)
    gl = tl.load(
        gc + ((i_n * BT + (seq_len - 1)) * H + i_h) * K + o_k,
        mask=(seq_len > 0) & (o_k < K),
        other=0.0,
    )
    kk = tl.load(
        k + (bos + o_t[None, :]) * H * K + i_h * K + o_k[:, None],
        mask=(o_k[:, None] < K) & m_t[None, :],
        other=0.0,
    )
    gg = tl.load(
        gc + ((i_n * BT + o_t[None, :]) * H + i_h) * K + o_k[:, None],
        mask=(o_k[:, None] < K) & m_t[None, :],
        other=0.0,
    )
    ks = (kk * tl.exp2(gl[:, None] - gg)).to(tl.bfloat16)
    vv = tl.load(
        v + (bos + o_t[:, None]) * H * V + i_h * V + o_v[None, :],
        mask=m_t[:, None] & (o_v[None, :] < V),
        other=0.0,
    )
    if DENSE and T == 65:
        g63 = tl.load(
            gc + ((i_n * BT + 63) * H + i_h) * K + o_k,
            mask=o_k < K,
            other=0.0,
        )
        first = o_t < 64
        ks_first = tl.where(first[None, :], kk * tl.exp2(g63[:, None] - gg), 0.0)
        state64 = old * tl.exp2(g63)[:, None] + tl.dot(ks_first.to(tl.bfloat16), vv)
        g64 = tl.load(
            gate + (bos + 64) * H * K + i_h * K + o_k,
            mask=o_k < K,
            other=0.0,
        ).to(tl.float32) * _RCP_LN2
        ks_last = tl.where((o_t == 64)[None, :], kk, 0.0)
        vv_last = tl.where((o_t == 64)[:, None], vv, 0.0)
        updated = state64 * tl.exp2(g64)[:, None] + tl.dot(
            ks_last.to(tl.bfloat16), vv_last
        )
    else:
        updated = old * tl.exp2(gl)[:, None] + tl.dot(ks, vv)
    tl.store(ht + offset, updated, mask=mask)


@triton.jit
def _dense_global_cumsum(
    g,
    gc,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    CT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    o_i = tl.arange(0, CT)
    carry = tl.zeros((BK,), dtype=tl.float32)
    base = (i_b * T * H + i_h) * K
    for i_c in range(0, triton.cdiv(T, CT)):
        o_t = i_c * CT + o_i
        mask = (o_t[:, None] < T) & (o_k[None, :] < K)
        x = tl.load(
            g + base + o_t[:, None] * H * K + o_k[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(x, axis=0)
        tl.store(
            gc + base + o_t[:, None] * H * K + o_k[None, :],
            (prefix + carry[None, :]) * _RCP_LN2,
            mask=mask,
        )
        carry += tl.sum(x, axis=0)


@triton.jit
def _dense_local_output(
    q,
    k,
    v,
    gc,
    out,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    W: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    S: tl.constexpr = W + BT
    o_q = i_c * BT + tl.arange(0, BT)
    o_s = i_c * BT - W + tl.arange(0, S)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    m_s = (o_s >= 0) & (o_s < T)
    m_v = o_v < V
    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * H + i_h) * V
    attn = tl.zeros((BT, S), dtype=tl.float32)

    for i_k in range(0, K, BK):
        o_k = i_k + tl.arange(0, BK)
        m_k = o_k < K
        b_q = tl.load(
            q + base_k + o_q[:, None] * H * K + o_k[None, :],
            mask=m_q[:, None] & m_k[None, :],
            other=0.0,
        )
        b_k = tl.load(
            k + base_k + o_s[:, None] * H * K + o_k[None, :],
            mask=m_s[:, None] & m_k[None, :],
            other=0.0,
        )
        g_q = tl.load(
            gc + base_k + o_q[:, None] * H * K + o_k[None, :],
            mask=m_q[:, None] & m_k[None, :],
            other=0.0,
        )
        g_k = tl.load(
            gc + base_k + o_s[:, None] * H * K + o_k[None, :],
            mask=m_s[:, None] & m_k[None, :],
            other=0.0,
        )
        anchor_pos = min((i_c + 1) * BT, T) - 1
        anchor = tl.load(
            gc + base_k + anchor_pos * H * K + o_k,
            mask=m_k,
            other=0.0,
        )
        q_scaled = b_q * tl.exp2(g_q - anchor[None, :])
        k_scaled = b_k * tl.exp2(anchor[None, :] - g_k)
        attn += tl.dot(q_scaled, tl.trans(k_scaled))

    causal = o_q[:, None] >= o_s[None, :]
    score = tl.where(
        causal & m_q[:, None] & m_s[None, :],
        attn * (1.0 / tl.sqrt(float(K))),
        0.0,
    ).to(tl.bfloat16)
    b_v = tl.load(
        v + base_v + o_s[:, None] * H * V + o_v[None, :],
        mask=m_s[:, None] & m_v[None, :],
        other=0.0,
    )
    result = tl.dot(score, b_v)
    tl.store(
        out + base_v + o_q[:, None] * H * V + o_v[None, :],
        result,
        mask=m_q[:, None] & m_v[None, :],
    )


@triton.jit
def _dense_tail_state(
    k,
    v,
    g,
    ht,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    ST: tl.constexpr,
    LT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, ST)
    o_t = T - LT + o_i
    m_t = o_i < LT
    m_k = o_k < K
    m_v = o_v < V
    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * H + i_h) * V

    gx = tl.load(
        g + base_k + o_t[:, None] * H * K + o_k[None, :],
        mask=m_t[:, None] & m_k[None, :],
        other=0.0,
    ).to(tl.float32)
    prefix = tl.cumsum(gx, axis=0) * _RCP_LN2
    gg = tl.trans(prefix)
    gl = tl.sum(gx, axis=0) * _RCP_LN2
    kk = tl.load(
        k + base_k + o_t[None, :] * H * K + o_k[:, None],
        mask=m_k[:, None] & m_t[None, :],
        other=0.0,
    )
    ks = (kk * tl.exp2(gl[:, None] - gg)).to(tl.bfloat16)
    vv = tl.load(
        v + base_v + o_t[:, None] * H * V + o_v[None, :],
        mask=m_t[:, None] & m_v[None, :],
        other=0.0,
    )
    state = tl.dot(ks, vv)
    tl.store(
        ht + i_bh * K * V + o_k[:, None] * V + o_v[None, :],
        state,
        mask=m_k[:, None] & m_v[None, :],
    )


class ChunkGLA(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Captured packed calls contain short sequences (one 64-token chunk
        # each). Avoid FLA's dynamic indices and full-length intermediates.
        if (
            cu_seqlens is not None
            and q.shape[0] == 1
            and initial_state is not None
            and scale is None
            and q.shape[-1] == 256
            and v.shape[-1] == 512
        ):
            H, K, V = q.shape[2], q.shape[3], v.shape[3]
            N, BT = initial_state.shape[0], 64
            gc = torch.empty((N, BT, H, K), device=g.device, dtype=torch.float32)
            _packed_cumsum[(triton.cdiv(K, 32), H, N)](
                g, cu_seqlens, gc,
                H=H, K=K, T=q.shape[1], BT=BT, BK=32, DENSE=False,
                num_warps=2,
            )

            out = torch.zeros_like(v)
            _packed_output[(triton.cdiv(V, 128), H, N)](
                q, k, v, gc, initial_state, cu_seqlens, out,
                H=H, K=K, V=V, T=q.shape[1], BT=BT, BK=64, BV=128,
                DENSE=False,
                num_warps=8, num_stages=3,
            )

            ht = torch.empty_like(initial_state) if output_final_state else None
            if ht is not None:
                _packed_state[
                    (triton.cdiv(K, 32), triton.cdiv(V, 128), N * H)
                ](
                    k, v, g, gc, initial_state, cu_seqlens, ht,
                    H=H, K=K, V=V, T=q.shape[1], BT=BT, BK=32, BV=128,
                    DENSE=False,
                    num_warps=4, num_stages=3,
                )
            return out, ht

        if (
            cu_seqlens is None
            and initial_state is not None
            and scale is None
            and q.shape[-1] == 256
            and v.shape[-1] == 512
            and q.shape[1] <= 65
        ):
            B, T, H, K = q.shape
            V = v.shape[-1]
            BT = 64 if T <= 64 else 128
            gc = torch.empty((B, BT, H, K), device=g.device, dtype=torch.float32)
            _packed_cumsum[(triton.cdiv(K, 32), H, B)](
                g, g, gc,
                H=H, K=K, T=T, BT=BT, BK=32, DENSE=True,
                num_warps=4,
            )

            out = torch.empty_like(v)
            bv = 128 if BT == 64 else 32
            _packed_output[(triton.cdiv(V, bv), H, B)](
                q, k, v, gc, initial_state, g, out,
                H=H, K=K, V=V, T=T, BT=BT, BK=64, BV=bv, DENSE=True,
                num_warps=4 if BT == 64 else 8, num_stages=3,
            )

            ht = torch.empty_like(initial_state) if output_final_state else None
            if ht is not None:
                _packed_state[(triton.cdiv(K, 32), triton.cdiv(V, 128), B * H)](
                    k, v, g, gc, initial_state, g, ht,
                    H=H, K=K, V=V, T=T, BT=BT, BK=32, BV=128, DENSE=True,
                    num_warps=4, num_stages=3,
                )
            return out, ht

        if (
            cu_seqlens is None
            and initial_state is None
            and output_final_state
            and scale is None
            and q.shape[-1] == 256
            and v.shape[-1] == 512
            and q.shape[0] >= 32
            and q.shape[1] >= 64
        ):
            B, T, H, K = q.shape
            V = v.shape[-1]
            gc = torch.empty_like(g, dtype=torch.float32)
            _dense_global_cumsum[(triton.cdiv(K, 64), H, B)](
                g, gc, T=T, H=H, K=K, CT=64, BK=64,
                num_warps=4,
            )
            out = torch.empty_like(v)
            _dense_local_output[
                (triton.cdiv(V, 128), triton.cdiv(T, 16), B * H)
            ](
                q, k, v, gc, out,
                T=T, H=H, K=K, V=V,
                BT=16, W=16, BK=64, BV=128,
                num_warps=4, num_stages=2,
            )
            ht = torch.empty((B, H, K, V), device=q.device, dtype=torch.float32)
            _dense_tail_state[
                (triton.cdiv(K, 32), triton.cdiv(V, 64), B * H)
            ](
                k, v, g, ht,
                T=T, H=H, K=K, V=V,
                ST=64, LT=T % 64, BK=32, BV=64,
                num_warps=4, num_stages=3,
            )
            return out, ht

        raise NotImplementedError("unsupported ChunkGLA shape")
