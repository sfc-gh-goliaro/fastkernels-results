"""Fused AlphaFold3 attention for the captured small-sequence workloads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear


@triton.jit
def _qkvg_kernel(
    q_x,
    kv_x,
    w_q,
    w_k,
    w_v,
    w_g,
    q_bias,
    q_out,
    k_out,
    v_out,
    g_out,
    M_Q: tl.constexpr,
    M_KV: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    HAS_Q_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    is_q = pid_m < Q_BLOCKS
    local_m = tl.where(is_q, pid_m, pid_m - Q_BLOCKS)

    offs_m = local_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if is_q:
        for kb in tl.static_range(0, C // BLOCK_K):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            x = tl.load(
                q_x + offs_m[:, None] * C + offs_k[None, :],
                mask=(offs_m[:, None] < M_Q) & (offs_k[None, :] < C),
                other=0.0,
            )
            q_w = tl.load(
                w_q + offs_n[None, :] * C + offs_k[:, None],
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
                other=0.0,
            )
            g_w = tl.load(
                w_g + offs_n[None, :] * C + offs_k[:, None],
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
                other=0.0,
            )
            acc0 += tl.dot(x, q_w)
            acc1 += tl.dot(x, g_w)
        if HAS_Q_BIAS:
            acc0 += tl.load(q_bias + offs_n[None, :], mask=offs_n[None, :] < N)
        mask = (offs_m[:, None] < M_Q) & (offs_n[None, :] < N)
        tl.store(q_out + offs_m[:, None] * N + offs_n[None, :], acc0, mask=mask)
        tl.store(g_out + offs_m[:, None] * N + offs_n[None, :], acc1, mask=mask)
    else:
        for kb in tl.static_range(0, C // BLOCK_K):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            x = tl.load(
                kv_x + offs_m[:, None] * C + offs_k[None, :],
                mask=(offs_m[:, None] < M_KV) & (offs_k[None, :] < C),
                other=0.0,
            )
            k_w = tl.load(
                w_k + offs_n[None, :] * C + offs_k[:, None],
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
                other=0.0,
            )
            v_w = tl.load(
                w_v + offs_n[None, :] * C + offs_k[:, None],
                mask=(offs_n[None, :] < N) & (offs_k[:, None] < C),
                other=0.0,
            )
            acc0 += tl.dot(x, k_w)
            acc1 += tl.dot(x, v_w)
        mask = (offs_m[:, None] < M_KV) & (offs_n[None, :] < N)
        tl.store(k_out + offs_m[:, None] * N + offs_n[None, :], acc0, mask=mask)
        tl.store(v_out + offs_m[:, None] * N + offs_n[None, :], acc1, mask=mask)


@triton.jit
def _attention_kernel(
    q,
    k,
    v,
    out,
    Q: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_block = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    head = bh % H
    offs_q = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    qv = tl.load(
        q + ((batch * Q + offs_q[:, None]) * H + head) * D + offs_d[None, :],
        mask=(offs_q[:, None] < Q) & (offs_d[None, :] < D),
        other=0.0,
    )
    qv = (qv * SCALE).to(tl.bfloat16)
    kv = tl.load(
        k + ((batch * K + offs_k[:, None]) * H + head) * D + offs_d[None, :],
        mask=(offs_k[:, None] < K) & (offs_d[None, :] < D),
        other=0.0,
    )
    scores = tl.dot(qv, tl.trans(kv)).to(tl.bfloat16).to(tl.float32)
    scores = tl.where(offs_k[None, :] < K, scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probs = tl.exp(scores)
    probs /= tl.sum(probs, axis=1)[:, None]
    probs = probs.to(tl.bfloat16)

    vv = tl.load(
        v + ((batch * K + offs_k[:, None]) * H + head) * D + offs_d[None, :],
        mask=(offs_k[:, None] < K) & (offs_d[None, :] < D),
        other=0.0,
    )
    ov = tl.dot(probs, vv)
    tl.store(
        out + ((batch * Q + offs_q[:, None]) * H + head) * D + offs_d[None, :],
        ov,
        mask=(offs_q[:, None] < Q) & (offs_d[None, :] < D),
    )


@triton.jit
def _gated_output_kernel(
    attn,
    gate,
    weight,
    out,
    M: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for kb in tl.static_range(0, N // BLOCK_K):
        offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            attn + offs_m[:, None] * N + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < N),
            other=0.0,
        )
        g = tl.load(
            gate + offs_m[:, None] * N + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < N),
            other=0.0,
        )
        g = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)
        a = (a * g).to(tl.bfloat16)
        w = tl.load(
            weight + offs_n[None, :] * N + offs_k[:, None],
            mask=(offs_n[None, :] < C) & (offs_k[:, None] < N),
            other=0.0,
        )
        acc += tl.dot(a, w)

    tl.store(
        out + offs_m[:, None] * C + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < C),
    )


@triton.jit
def _attention_output_128_kernel(
    q,
    k,
    v,
    gate,
    weight,
    out,
    Q: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_q = tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)

    for head in tl.static_range(0, H):
        qv = tl.load(
            q + ((batch * Q + offs_q[:, None]) * H + head) * D + offs_d[None, :],
            mask=offs_q[:, None] < Q,
            other=0.0,
        )
        qv = (qv * SCALE).to(tl.bfloat16)
        kv = tl.load(
            k + ((batch * K + offs_k[:, None]) * H + head) * D + offs_d[None, :],
            mask=offs_k[:, None] < K,
            other=0.0,
        )
        scores = tl.dot(qv, tl.trans(kv)).to(tl.bfloat16).to(tl.float32)
        scores = tl.where(offs_k[None, :] < K, scores, -float("inf"))
        scores -= tl.max(scores, axis=1)[:, None]
        probs = tl.exp(scores)
        probs /= tl.sum(probs, axis=1)[:, None]
        probs = probs.to(tl.bfloat16)
        vv = tl.load(
            v + ((batch * K + offs_k[:, None]) * H + head) * D + offs_d[None, :],
            mask=offs_k[:, None] < K,
            other=0.0,
        )
        head_out = tl.dot(probs, vv).to(tl.bfloat16)
        gv = tl.load(
            gate + (batch * Q + offs_q[:, None]) * (H * D)
            + head * D + offs_d[None, :],
            mask=offs_q[:, None] < Q,
            other=0.0,
        )
        gv = tl.sigmoid(gv.to(tl.float32)).to(tl.bfloat16)
        head_out = (head_out * gv).to(tl.bfloat16)
        w = tl.load(
            weight + offs_n[None, :] * (H * D) + head * D + offs_d[:, None],
            mask=offs_n[None, :] < H * D,
            other=0.0,
        )
        acc += tl.dot(head_out, w)

    tl.store(
        out + (batch * Q + offs_q[:, None]) * (H * D) + offs_n[None, :],
        acc,
        mask=(offs_q[:, None] < Q) & (offs_n[None, :] < H * D),
    )


class OF3Attention(nn.Module):
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)
        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)
        self._workspace = None
        self._launch = None

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        if self._workspace is None:
            c = self.c_q
            n = self.no_heads * self.c_hidden
            q_len = q_x.shape[-2]
            kv_len = kv_x.shape[-2]
            batch = q_x.numel() // (q_len * c)
            m_q = batch * q_len
            m_kv = batch * kv_len
            q = torch.empty((m_q, n), device=q_x.device, dtype=q_x.dtype)
            k = torch.empty((m_kv, n), device=q_x.device, dtype=q_x.dtype)
            v = torch.empty_like(k)
            g = torch.empty_like(q)
            attn = torch.empty_like(q)
            out = torch.empty((m_q, c), device=q_x.device, dtype=q_x.dtype)

            if c == 128 and m_q == m_kv:
                block_m, block_n, block_k, projection_warps = 32, 128, 32, 8
            elif c == 128:
                block_m, block_n, block_k, projection_warps = 32, 32, 32, 4
            elif c == 384:
                block_m, block_n, block_k, projection_warps = 16, 32, 64, 8
            else:
                block_m, block_n, block_k, projection_warps = 16, 32, 64, 4
            q_blocks = triton.cdiv(m_q, block_m)
            kv_blocks = triton.cdiv(m_kv, block_m)
            projection_grid = (triton.cdiv(n, block_n), q_blocks + kv_blocks)

            block_q = 32 if q_len == 32 else 16
            attention_grid = (triton.cdiv(q_len, block_q), batch * self.no_heads)
            block_k_attn = triton.next_power_of_2(kv_len)
            block_d = triton.next_power_of_2(self.c_hidden)

            if c == 128 and m_q == m_kv:
                out_block_m, out_block_n, out_block_k, output_warps = 64, 64, 64, 8
            elif c == 128:
                out_block_m, out_block_n, out_block_k, output_warps = 32, 32, 64, 4
            elif c == 384:
                out_block_m, out_block_n, out_block_k, output_warps = 8, 32, 64, 4
            else:
                out_block_m, out_block_n, out_block_k, output_warps = 8, 128, 64, 4
            output_grid = (
                triton.cdiv(m_q, out_block_m),
                triton.cdiv(c, out_block_n),
            )
            fused_block_n = 32
            fused_grid = (batch, triton.cdiv(c, fused_block_n))
            self._workspace = (q, k, v, g, attn, out, out.view(q_x.shape))
            self._launch = (
                m_q, m_kv, c, n, q_len, kv_len, batch,
                projection_grid, q_blocks, block_m, block_n, block_k,
                projection_warps, attention_grid, block_q, block_k_attn,
                block_d, output_grid, out_block_m, out_block_n, out_block_k,
                output_warps, fused_grid, fused_block_n,
            )

        q, k, v, g, attn, out, out_view = self._workspace
        (
            m_q, m_kv, c, n, q_len, kv_len, batch,
            projection_grid, q_blocks, block_m, block_n, block_k,
            projection_warps, attention_grid, block_q, block_k_attn,
            block_d, output_grid, out_block_m, out_block_n, out_block_k,
            output_warps, fused_grid, fused_block_n,
        ) = self._launch
        q_bias = self.linear_q.bias
        if q_bias is None:
            q_bias = self.linear_q.weight
        _qkvg_kernel[projection_grid](
            q_x,
            kv_x,
            self.linear_q.weight,
            self.linear_k.weight,
            self.linear_v.weight,
            self.linear_g.weight,
            q_bias,
            q,
            k,
            v,
            g,
            M_Q=m_q,
            M_KV=m_kv,
            C=c,
            N=n,
            Q_BLOCKS=q_blocks,
            HAS_Q_BIAS=self.linear_q.bias is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=projection_warps,
        )

        if c == 128:
            _attention_output_128_kernel[fused_grid](
                q,
                k,
                v,
                g,
                self.linear_o.weight,
                out,
                Q=q_len,
                K=kv_len,
                H=self.no_heads,
                D=self.c_hidden,
                SCALE=1.0 / math.sqrt(self.c_hidden),
                BLOCK_Q=block_q,
                BLOCK_K=block_k_attn,
                BLOCK_N=fused_block_n,
                num_warps=4,
            )
        else:
            _attention_kernel[attention_grid](
                q,
                k,
                v,
                attn,
                Q=q_len,
                K=kv_len,
                D=self.c_hidden,
                H=self.no_heads,
                SCALE=1.0 / math.sqrt(self.c_hidden),
                BLOCK_Q=block_q,
                BLOCK_K=block_k_attn,
                BLOCK_D=block_d,
                num_warps=4,
            )
            _gated_output_kernel[output_grid](
                attn,
                g,
                self.linear_o.weight,
                out,
                M=m_q,
                C=c,
                N=n,
                BLOCK_M=out_block_m,
                BLOCK_N=out_block_n,
                BLOCK_K=out_block_k,
                num_warps=output_warps,
            )
        return out_view
