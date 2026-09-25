"""GLA / RetNet decoder layer.

Pre-norm residual:
  attn_norm -> GatedLinearAttention -> residual
  mlp_norm  -> GLAMLP -> residual

Forward signature mirrors FLA's ``GLABlock.forward`` (returns a tuple of
``(hidden_states, attentions, past_key_values)``) so that the same L3
block backs both GLA and RetNet — RetNet just uses ``decay_mode="fixed_per_head"``
and ``use_rotary=True`` in the attention layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.rms_norm import RMSNorm
from ..L2.gla_attention import GatedLinearAttention
from ..L2.gla_mlp import GLAMLP

_RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _exact_logsigmoid_div_kernel(
    x,
    y,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    z = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.minimum(z, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(z)))
    tl.store(y + offsets, value * 0.0625, mask=mask)


@triton.jit
def _exact_norm_silu_mul_kernel(
    x,
    gate,
    weight,
    out,
    rows: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, N)
    offsets = row[:, None] * N + col[None, :]
    mask = row[:, None] < rows
    value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(value * value, axis=1) / N
    w = tl.load(weight + col).to(tl.float32)
    norm = (value * tl.rsqrt(variance[:, None] + eps) * w[None, :]).to(
        tl.bfloat16
    )
    z = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    silu = (z * tl.sigmoid(z)).to(tl.bfloat16)
    tl.store(out + offsets, norm * silu, mask=mask)


@triton.jit
def _exact_decode_output_kernel(
    qkvg,
    weight,
    out,
    eps: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // 5
    head = row % 5
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    base = token * 7680
    qk = tl.load(qkvg + base + head * K + o_k).to(tl.float32)
    kk = tl.load(qkvg + base + 1280 + head * K + o_k).to(tl.float32)
    vv = tl.load(qkvg + base + 2560 + head * V + o_v).to(tl.float32)
    value = (tl.sum(qk * kk, axis=0) * 0.0625 * vv).to(
        tl.bfloat16
    ).to(tl.float32)
    variance = tl.sum(value * value, axis=0) / V
    w = tl.load(weight + o_v).to(tl.float32)
    norm = (value * tl.rsqrt(variance + eps) * w).to(tl.bfloat16)
    z = tl.load(qkvg + base + 5120 + head * V + o_v).to(tl.float32)
    silu = (z * tl.sigmoid(z)).to(tl.bfloat16)
    tl.store(out + row * V + o_v, norm * silu)


@triton.jit
def _add_rms_norm_kernel(
    residual,
    update,
    weight,
    norm,
    rows: tl.constexpr,
    eps: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    mask = col < N
    offsets = row * N + col
    x = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(update + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x.to(tl.bfloat16)
    tl.store(update + offsets, x, mask=mask)
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    w = tl.load(weight + col, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        norm + offsets,
        x * tl.rsqrt(variance + eps) * w,
        mask=mask,
    )


@triton.jit
def _logsigmoid_chunk_cumsum_kernel(
    logits,
    g_cumsum,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_c * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    offsets = (i_b * T * H + o_t[:, None] * H + i_h) * K + o_k[None, :]
    mask = (o_t[:, None] < T) & (o_k[None, :] < K)
    z = tl.load(logits + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.minimum(z, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(z)))
    values = (values * 0.0625).to(tl.bfloat16).to(tl.float32)
    values = tl.cumsum(values, axis=0) * _RCP_LN2
    tl.store(g_cumsum + offsets, values, mask=mask)


@triton.jit
def _chunk_state_kernel(
    k,
    v,
    g_cumsum,
    states,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * H + i_h) * V
    state = tl.zeros((BK, BV), tl.float32)

    for i_c in range(0, tl.cdiv(T, BT)):
        state_offset = ((i_c * B * H + i_bh) * K + o_k[:, None]) * V
        tl.store(states + state_offset + o_v[None, :], state)

        o_t = i_c * BT + tl.arange(0, BT)
        m_t = o_t < T
        kk = tl.load(
            k + base_k + o_t[None, :] * H * K + o_k[:, None],
            mask=m_t[None, :],
            other=0.0,
        )
        prefix = tl.load(
            g_cumsum + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        vv = tl.load(
            v + base_v + o_t[:, None] * H * V + o_v[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        last_t = min((i_c + 1) * BT, T) - 1
        last = tl.load(
            g_cumsum + base_k + last_t * H * K + o_k
        ).to(tl.float32)
        decayed_k = (kk * tl.exp2(last[:, None] - tl.trans(prefix))).to(
            tl.bfloat16
        )
        state = state * tl.exp2(last)[:, None] + tl.dot(decayed_k, vv)


@triton.jit
def _chunk_diagonal_kernel(
    q,
    k,
    g_cumsum,
    diagonal,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_s = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = i_c * BT + i_s * BC + tl.arange(0, BC)
    o_k = tl.arange(0, K)
    mask = o_i[:, None] < T
    base = (i_b * T * H + i_h) * K
    qv = tl.load(
        q + base + o_i[:, None] * H * K + o_k[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gq = tl.load(
        g_cumsum + base + o_i[:, None] * H * K + o_k[None, :],
        mask=mask,
        other=0.0,
    )
    k_ptr = k + base + (i_c * BT + i_s * BC) * H * K + o_k
    g_ptr = (
        g_cumsum
        + base
        + (i_c * BT + i_s * BC) * H * K
        + o_k
    )
    out_base = ((i_b * T + o_i) * H + i_h) * BC
    for j in range(0, BC):
        valid_j = i_c * BT + i_s * BC + j < T
        kv = tl.load(k_ptr, mask=valid_j, other=0.0).to(tl.float32)
        gk = tl.load(g_ptr, mask=valid_j, other=0.0)
        score = tl.sum(
            qv * kv[None, :] * tl.exp2(gq - gk[None, :]), axis=1
        ) * 0.0625
        tl.store(
            diagonal + out_base + j,
            score,
            mask=(o_i < T) & valid_j,
        )
        k_ptr += H * K
        g_ptr += H * K


@triton.jit
def _chunk_output_kernel(
    q,
    k,
    v,
    g_cumsum,
    states,
    diagonal,
    out,
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    o_t = i_c * BT + o_i
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    base_k = (i_b * T * H + i_h) * K
    base_v = (i_b * T * H + i_h) * V
    attn = tl.zeros((BT, BT), tl.float32)
    inter = tl.zeros((BT, BV), tl.float32)

    for start_k in range(0, K, BK):
        o_k = start_k + tl.arange(0, BK)
        qk = tl.load(
            q + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        kk = tl.load(
            k + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        prefix = tl.load(
            g_cumsum + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        q_scaled = (qk * tl.exp2(prefix)).to(tl.bfloat16)
        q_local = qk.to(tl.float32) * tl.exp2(prefix) * 0.0625
        k_local = kk.to(tl.float32) * tl.exp2(-prefix)
        attn += tl.dot(q_local, tl.trans(k_local), input_precision="tf32")

        state_offset = ((i_c * B * H + i_bh) * K + o_k[:, None]) * V
        state = tl.load(states + state_offset + o_v[None, :])
        inter += tl.dot(q_scaled, state)

    o_j = tl.arange(0, BT)
    diagonal_values = tl.load(
        diagonal
        + ((i_b * T + o_t[:, None]) * H + i_h) * 16
        + (o_j[None, :] % 16),
        mask=m_t[:, None],
        other=0.0,
    )
    same_subchunk = (o_i[:, None] // 16) == (o_j[None, :] // 16)
    attn = tl.where(same_subchunk, diagonal_values, attn)
    causal = o_i[:, None] >= o_i[None, :]
    score = tl.where(
        causal & m_t[:, None] & m_t[None, :], attn, 0.0
    ).to(tl.bfloat16)
    vv = tl.load(
        v + base_v + o_t[:, None] * H * V + o_v[None, :],
        mask=m_t[:, None],
        other=0.0,
    )
    local = tl.dot(score, vv)
    tl.store(
        out + base_v + o_t[:, None] * H * V + o_v[None, :],
        local + inter * 0.0625,
        mask=m_t[:, None],
    )


class GLADecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GatedLinearAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            decay_mode=getattr(config, "decay_mode", "learned_low_rank"),
            gate_low_rank_dim=getattr(config, "gate_low_rank_dim", 16),
            gate_logit_normalizer=getattr(config, "gate_logit_normalizer", 16),
            use_rotary=getattr(config, "use_rotary", False),
            rotary_base=getattr(config, "rotary_base", 10000.0),
            rotary_max_position=getattr(config, "max_position_embeddings", 8192),
            norm_eps=config.norm_eps,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GLAMLP(config.hidden_size, config.intermediate_size)
        self._qkvg_weight = None

    @staticmethod
    def _long_prefill(q, k, v, gk_logits):
        B, T, H, K = q.shape
        V = v.shape[-1]
        chunks = triton.cdiv(T, 64)
        g_cumsum = torch.empty_like(gk_logits, dtype=torch.float32)
        _logsigmoid_chunk_cumsum_kernel[(4, chunks, B * H)](
            gk_logits,
            g_cumsum,
            T=T,
            H=H,
            K=K,
            BT=64,
            BK=64,
            num_warps=4,
        )
        states = torch.empty(
            (chunks, B, H, K, V), device=v.device, dtype=torch.bfloat16
        )
        _chunk_state_kernel[(triton.cdiv(V, 128), 4, B * H)](
            k,
            v,
            g_cumsum,
            states,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=64,
            BK=64,
            BV=128,
            num_warps=8,
            num_stages=2,
        )
        diagonal = torch.empty(
            (B, T, H, 16), device=v.device, dtype=torch.float32
        )
        _chunk_diagonal_kernel[(chunks, 4, B * H)](
            q,
            k,
            g_cumsum,
            diagonal,
            T=T,
            H=H,
            K=K,
            BT=64,
            BC=16,
            num_warps=4,
        )
        out = torch.empty_like(v)
        _chunk_output_kernel[
            (triton.cdiv(V, 128), chunks, B * H)
        ](
            q,
            k,
            v,
            g_cumsum,
            states,
            diagonal,
            out,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=64,
            BK=64,
            BV=128,
            num_warps=8,
            num_stages=3,
        )
        return out

    def _attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values,
        use_cache: bool,
        kwargs: dict,
    ):
        attn = self.attn
        if not (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and attention_mask is None
            and past_key_values is None
            and attn.decay_mode == "learned_low_rank"
            and not attn.use_rotary
            and (attn.num_heads, attn.head_k_dim, attn.head_v_dim)
            == (5, 256, 512)
        ):
            return attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")

        if T == 1 and (cu_seqlens is None or B == 1):
            if self._qkvg_weight is None:
                self._qkvg_weight = torch.cat(
                    (
                        attn.q_proj.weight,
                        attn.k_proj.weight,
                        attn.v_proj.weight,
                        attn.g_proj.weight,
                    ),
                    dim=0,
                )
            qkvg = torch.mm(
                hidden_states.view(-1, hidden_states.shape[-1]),
                self._qkvg_weight.t(),
            )
            o = torch.empty(
                (B, T, 5, 512),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            _exact_decode_output_kernel[(B * 5,)](
                qkvg,
                attn.g_norm_swish_gate.weight,
                o,
                eps=attn.g_norm_swish_gate.eps,
                K=256,
                V=512,
                num_warps=4,
            )
        else:
            q = attn.q_proj(hidden_states).view(B, T, 5, 256)
            k = attn.k_proj(hidden_states).view(B, T, 5, 256)
            v = attn.v_proj(hidden_states).view(B, T, 5, 512)
            gate = attn.g_proj(hidden_states)
            gk_logits = attn.gk_proj(hidden_states)
            dense_long = cu_seqlens is None and B >= 32
            single_packed = (
                B == 1
                and cu_seqlens is not None
                and cu_seqlens.numel() == 2
            )
            if T >= 64 and (dense_long or single_packed):
                o = self._long_prefill(
                    q,
                    k,
                    v,
                    gk_logits.view(B, T, 5, 256),
                )
            else:
                normalized = torch.empty_like(gk_logits)
                block = (
                    8192
                    if gk_logits.numel() >= 16 * 1024 * 1024
                    else 1024
                )
                _exact_logsigmoid_div_kernel[
                    (triton.cdiv(gk_logits.numel(), block),)
                ](
                    gk_logits,
                    normalized,
                    gk_logits.numel(),
                    BLOCK=block,
                    num_warps=8 if block == 8192 else 4,
                )
                gk = normalized.view(B, T, 5, 256)
                if T >= 64:
                    o, _ = attn.chunk(
                        q=q,
                        k=k,
                        v=v,
                        g=gk,
                        initial_state=None,
                        output_final_state=False,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, _ = attn.fused_recurrence(
                        q=q,
                        k=k,
                        v=v,
                        gk=gk,
                        initial_state=None,
                        output_final_state=False,
                        cu_seqlens=cu_seqlens,
                    )

            rows = o.numel() // 512
            fused = torch.empty_like(o)
            block_m = 8 if rows >= 8 else 4
            _exact_norm_silu_mul_kernel[
                (triton.cdiv(rows, block_m),)
            ](
                o,
                gate,
                attn.g_norm_swish_gate.weight,
                fused,
                rows=rows,
                eps=attn.g_norm_swish_gate.eps,
                N=512,
                BLOCK_M=block_m,
                num_warps=8,
            )
            o = fused

        return attn.o_proj(o.view(B, T, attn.value_dim)), None, past_key_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        residual = hidden_states
        h = self.attn_norm(
            hidden_states.reshape(-1, hidden_states.size(-1))
        ).reshape_as(hidden_states)
        h, attentions, past_key_values = self._attention_forward(
            h,
            attention_mask,
            past_key_values,
            use_cache,
            kwargs,
        )
        if (
            h.is_cuda
            and h.dtype == torch.bfloat16
            and h.is_contiguous()
            and residual.is_contiguous()
            and h.size(-1) == 2560
        ):
            normed = torch.empty_like(h)
            rows = h.numel() // 2560
            _add_rms_norm_kernel[(rows,)](
                residual,
                h,
                self.mlp_norm.weight,
                normed,
                rows=rows,
                eps=self.mlp_norm.eps,
                N=2560,
                BLOCK_N=4096,
                num_warps=4,
            )
            hidden_states = h
            h = normed
        else:
            hidden_states = residual + h
            h = self.mlp_norm(
                hidden_states.reshape(-1, hidden_states.size(-1))
            ).reshape_as(hidden_states)
        hidden_states = hidden_states + self.mlp(h)
        return hidden_states, attentions, past_key_values
