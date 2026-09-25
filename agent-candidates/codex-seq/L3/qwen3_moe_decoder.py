"""Qwen3 MoE decoder layer: QK-norm attention + MoE with RMSNorm residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from fastkernels.infra.cuda_ext import lazy_op

from ..L1.rms_norm import RMSNorm
from ..L1.mrope import MRotaryEmbedding
from ..L2.attention import LlamaAttention
from ..L2.qwen3_moe import Qwen3MoE


_CUDA = lazy_op("qwen3_moe_decoder", "qwen3_moe_decoder.cu")


@triton.jit
def _mrope_direct_kernel(
    q_ptr,
    k_ptr,
    positions_ptr,
    cache_ptr,
    num_tokens,
    positions_axis_stride,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    section_h: tl.constexpr,
    section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
):
    token = tl.program_id(0)
    half_hd: tl.constexpr = hd // 2
    dims = tl.arange(0, pad_hd // 2)

    if is_interleaved:
        h_mask = ((dims % 3) == 1) & (dims <= 3 * section_h)
        w_mask = ((dims % 3) == 2) & (dims <= 3 * section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        section_t: tl.constexpr = half_hd - section_h - section_w
        t_mask = dims < section_t
        h_mask = (dims >= section_t) & (dims < section_t + section_h)
        w_mask = (dims >= section_t + section_h) & (dims < half_hd)

    pos_t = tl.load(positions_ptr + token)
    pos_h = tl.load(positions_ptr + positions_axis_stride + token)
    pos_w = tl.load(positions_ptr + 2 * positions_axis_stride + token)
    t_cache = cache_ptr + pos_t * hd
    h_cache = cache_ptr + pos_h * hd
    w_cache = cache_ptr + pos_w * hd
    valid_dim = dims < half_hd

    t_cos = tl.load(t_cache + dims, mask=t_mask & valid_dim, other=0)
    h_cos = tl.load(h_cache + dims, mask=h_mask & valid_dim, other=0)
    w_cos = tl.load(w_cache + dims, mask=w_mask & valid_dim, other=0)
    t_sin = tl.load(t_cache + half_hd + dims, mask=t_mask & valid_dim, other=0)
    h_sin = tl.load(h_cache + half_hd + dims, mask=h_mask & valid_dim, other=0)
    w_sin = tl.load(w_cache + half_hd + dims, mask=w_mask & valid_dim, other=0)
    cos = t_cos + h_cos + w_cos
    sin = t_sin + h_sin + w_sin

    q_heads = tl.arange(0, pad_n_qh)[:, None]
    k_heads = tl.arange(0, pad_n_kh)[:, None]
    q_offsets = q_heads * hd + dims[None, :]
    k_offsets = k_heads * hd + dims[None, :]
    q_mask = (q_heads < n_qh) & valid_dim[None, :]
    k_mask = (k_heads < n_kh) & valid_dim[None, :]
    q_base = q_ptr + token * n_qh * hd
    k_base = k_ptr + token * n_kh * hd

    q1 = tl.load(q_base + q_offsets, mask=q_mask, other=0).to(sin.dtype)
    q2 = tl.load(
        q_base + q_offsets + half_hd, mask=q_mask, other=0
    ).to(sin.dtype)
    k1 = tl.load(k_base + k_offsets, mask=k_mask, other=0).to(sin.dtype)
    k2 = tl.load(
        k_base + k_offsets + half_hd, mask=k_mask, other=0
    ).to(sin.dtype)

    tl.store(q_base + q_offsets, q1 * cos - q2 * sin, mask=q_mask)
    tl.store(
        q_base + q_offsets + half_hd, q2 * cos + q1 * sin, mask=q_mask
    )
    tl.store(k_base + k_offsets, k1 * cos - k2 * sin, mask=k_mask)
    tl.store(
        k_base + k_offsets + half_hd, k2 * cos + k1 * sin, mask=k_mask
    )


class _FusedMRotaryEmbedding(nn.Module):
    def __init__(self, rotary_emb: nn.Module):
        super().__init__()
        self.head_dim = rotary_emb.head_dim
        self.rotary_dim = rotary_emb.rotary_dim
        self.mrope_section = rotary_emb.mrope_section
        self.mrope_interleaved = rotary_emb.mrope_interleaved
        self.register_buffer(
            "cos_sin_cache", rotary_emb.cos_sin_cache, persistent=False
        )
        self._q_buffer = None
        self._k_buffer = None

    def forward(self, positions, query, key):
        if positions.ndim == 1:
            return MRotaryEmbedding._apply_sgl_rope(self, positions, query, key)

        num_tokens = positions.shape[-1]
        hd = self.head_dim
        n_qh = query.shape[1] // hd if query.ndim == 2 else query.shape[1]
        n_kh = key.shape[1] // hd if key.ndim == 2 else key.shape[1]
        q_view = query.reshape(num_tokens, -1)
        k_view = key.reshape(num_tokens, -1)
        if (
            self._q_buffer is None
            or self._q_buffer.shape[0] < num_tokens
            or self._q_buffer.shape[1] != q_view.shape[1]
        ):
            self._q_buffer = torch.empty_like(q_view)
            self._k_buffer = torch.empty_like(k_view)
        q_flat = self._q_buffer[:num_tokens]
        k_flat = self._k_buffer[:num_tokens]
        q_flat.copy_(q_view)
        k_flat.copy_(k_view)

        _mrope_direct_kernel[(num_tokens,)](
            q_flat,
            k_flat,
            positions,
            self.cos_sin_cache,
            num_tokens,
            positions.stride(0),
            n_qh,
            n_kh,
            hd,
            triton.next_power_of_2(n_qh),
            triton.next_power_of_2(n_kh),
            triton.next_power_of_2(hd),
            self.mrope_section[1],
            self.mrope_section[2],
            self.mrope_interleaved,
        )
        return q_flat.view_as(query), k_flat.view_as(key)


class _DecoderRMSNorm(RMSNorm):
    def forward(self, x, residual=None):
        weight = self.weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        if x.dtype == torch.bfloat16 and x.shape[-1] == 4096:
            if residual is None:
                return _CUDA.rms_norm_4096(x, weight, self.eps)
            _CUDA.fused_add_rms_norm_4096(x, residual, weight, self.eps)
            return x, residual
        return super().forward(x, residual)


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = LlamaAttention(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        if (
            rotary_emb is not None
            and hasattr(rotary_emb, "mrope_section")
            and hasattr(rotary_emb, "cos_sin_cache")
        ):
            self.self_attn.rotary_emb = _FusedMRotaryEmbedding(rotary_emb)
        self.mlp = Qwen3MoE(config, quant_config=quant_config)
        self.input_layernorm = _DecoderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = _DecoderRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @staticmethod
    def _ensure_linear_buffers(projection, tokens):
        linear = getattr(projection, "linear_op", None)
        weight = getattr(projection, "weight", None)
        if (
            tokens >= 32
            and linear is not None
            and weight is not None
            and hasattr(linear, "_ensure_buffers")
            and (
                linear._a_buf is None
                or linear._a_buf.shape[0] < tokens
            )
        ):
            linear._ensure_buffers(
                tokens, weight.shape[1], weight.shape[0], weight.device
            )

    def forward(self, positions, hidden_states, residual):
        tokens = hidden_states.shape[0]
        self._ensure_linear_buffers(self.self_attn.qkv_proj, tokens)
        self._ensure_linear_buffers(self.self_attn.o_proj, tokens)

        rope = self.self_attn.rotary_emb
        if (
            rope is not None
            and hasattr(rope, "cos_sin_cache")
            and rope.cos_sin_cache.dtype != hidden_states.dtype
        ):
            rope.cos_sin_cache = rope.cos_sin_cache.to(hidden_states.dtype)

        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
