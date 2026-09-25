"""Multi-dimensional Rotary Position Embedding (M-RoPE) for Qwen VL models.

Handles 3D position tensors (3, seq_len) representing temporal/height/width
dimensions. Each dimension's positions index into a shared cos/sin cache,
and the resulting embeddings are assembled by section into the rotary dim.

Uses a Triton kernel for multimodal prefill (3D positions differ across dims)
and a custom CUDA kernel for decode / text-only (all 3 dims identical -> standard RoPE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from . import rotary_emb as _rotary_emb_reg  # noqa: F401 — registers fastkernels_rope ops


@triton.jit
def _mrope_fused_kernel(
    positions_ptr, q_ptr, k_ptr, cache_ptr, axis_ptr,
    num_tokens, positions_axis_stride,
    n_qh: tl.constexpr, n_kh: tl.constexpr,
    hd: tl.constexpr,
    block_qh: tl.constexpr, pad_n_kh: tl.constexpr,
):
    token = tl.program_id(0)
    q_head_block = tl.program_id(1)
    half_hd: tl.constexpr = hd // 2
    dims = tl.arange(0, half_hd)
    axis = tl.load(axis_ptr + dims)

    pos_t = tl.load(positions_ptr + token)
    pos_h = tl.load(positions_ptr + positions_axis_stride + token)
    pos_w = tl.load(positions_ptr + 2 * positions_axis_stride + token)
    positions = tl.where(axis == 0, pos_t, tl.where(axis == 1, pos_h, pos_w))
    cache_offsets = positions * hd + dims
    cos_row = tl.load(cache_ptr + cache_offsets)
    sin_row = tl.load(cache_ptr + cache_offsets + half_hd)

    q_heads = q_head_block * block_qh + tl.arange(0, block_qh)[:, None]
    q_offsets = q_heads * hd + dims[None, :]
    q_mask = q_heads < n_qh
    q_base = q_ptr + token * (n_qh * hd)
    q1 = tl.load(q_base + q_offsets, mask=q_mask, other=0.0)
    q2 = tl.load(q_base + q_offsets + half_hd, mask=q_mask, other=0.0)
    tl.store(q_base + q_offsets, q1 * cos_row - q2 * sin_row, mask=q_mask)
    tl.store(q_base + q_offsets + half_hd, q2 * cos_row + q1 * sin_row, mask=q_mask)

    k_heads = tl.arange(0, pad_n_kh)[:, None]
    k_offsets = k_heads * hd + dims[None, :]
    k_mask = (k_heads < n_kh) & (q_head_block == 0)
    k_base = k_ptr + token * (n_kh * hd)
    k1 = tl.load(k_base + k_offsets, mask=k_mask, other=0.0)
    k2 = tl.load(k_base + k_offsets + half_hd, mask=k_mask, other=0.0)
    tl.store(k_base + k_offsets, k1 * cos_row - k2 * sin_row, mask=k_mask)
    tl.store(k_base + k_offsets + half_hd, k2 * cos_row + k1 * sin_row, mask=k_mask)


class MRotaryEmbedding(nn.Module):
    """M-RoPE for Qwen2-VL / Qwen3-VL.

    positions can be either:
      - 1D (seq_len,) for text-only (all 3 dims identical -> standard RoPE)
      - 2D (3, seq_len) for multimodal (T/H/W positions differ)

    mrope_section: list of 3 ints [t, h, w] summing to rotary_dim // 2
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        mrope_section: list[int],
        mrope_interleaved: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        assert sum(mrope_section) == head_dim // 2

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        t = torch.arange(max_position_embeddings * 4, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        axis = torch.zeros(head_dim // 2, dtype=torch.int8)
        if mrope_interleaved:
            axis[1:mrope_section[1] * 3:3] = 1
            axis[2:mrope_section[2] * 3:3] = 2
        else:
            axis[mrope_section[0]:mrope_section[0] + mrope_section[1]] = 1
            axis[mrope_section[0] + mrope_section[1]:] = 2
        self.register_buffer("mrope_axis", axis, persistent=False)

    def _apply_sgl_rope(self, positions_1d, query, key):
        """Apply standard RoPE for 1D positions (decode or text-only)."""
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)
        if torch.compiler.is_compiling():
            from .rotary_emb import RotaryEmbedding
            return RotaryEmbedding.forward_native(
                positions_1d,
                query.view(query.shape[0], -1),
                key.view(key.shape[0], -1),
                self.head_dim, cache,
            )
        torch.ops.fastkernels_rope.rotary_embedding(
            positions_1d,
            query.view(query.shape[0], -1),
            key.view(key.shape[0], -1),
            self.head_dim,
            cache,
            True,
        )
        return query, key

    def forward_native_2d(self, positions, query, key):
        """Pure PyTorch MRoPE for (3, seq_len) positions -- Inductor-friendly.

        Mirrors the Triton _mrope_kernel: splits q/k into first/second half,
        gathers cos/sin per T/H/W section, and applies the standard neox-style
        rotation to all head_dim elements.
        """
        cache = self.cos_sin_cache
        if cache.dtype != query.dtype:
            cache = cache.to(query.dtype)

        num_tokens = query.shape[0]
        cos_sin = cache[positions]          # (3, seq_len, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1) # each (3, seq_len, head_dim/2)

        if self.mrope_interleaved:
            cos = self._apply_interleaved(cos)
            sin = self._apply_interleaved(sin)
        else:
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
        # cos, sin: (seq_len, head_dim/2)

        hd = self.head_dim
        half = hd // 2
        q_shape = query.shape
        k_shape = key.shape
        q = query.view(num_tokens, -1, hd)
        k = key.view(num_tokens, -1, hd)

        cos = cos.unsqueeze(1)  # (seq_len, 1, head_dim/2)
        sin = sin.unsqueeze(1)

        q1 = q[..., :half]
        q2 = q[..., half:]
        k1 = k[..., :half]
        k2 = k[..., half:]

        new_q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        new_k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return new_q.view(q_shape), new_k.view(k_shape)

    def forward(self, positions, query, key):
        """Apply M-RoPE in-place.

        Args:
            positions: (seq_len,) or (3, seq_len) int64 tensor
            query: (seq_len, num_heads, head_dim)
            key: (seq_len, num_kv_heads, head_dim)
        """
        if positions.ndim == 1:
            return self._apply_sgl_rope(positions, query, key)

        if torch.compiler.is_compiling():
            return self.forward_native_2d(positions, query, key)

        # 2D M-RoPE: positions (3, seq_len) with potentially different T/H/W dims (multimodal prefill)
        num_tokens = positions.shape[-1]
        hd = self.head_dim
        q_was_2d = query.ndim == 2
        if q_was_2d:
            n_qh = query.shape[1] // hd
            n_kh = key.shape[1] // hd
        else:
            n_qh = query.shape[1]
            n_kh = key.shape[1]

        q_flat = query.reshape(num_tokens, -1).contiguous()
        k_flat = key.reshape(num_tokens, -1).contiguous()
        pad_n_kh = triton.next_power_of_2(n_kh)
        block_qh = 16

        _mrope_fused_kernel[(num_tokens, triton.cdiv(n_qh, block_qh))](
            positions, q_flat, k_flat, self.cos_sin_cache, self.mrope_axis,
            num_tokens, positions.stride(0), n_qh, n_kh, hd,
            block_qh, pad_n_kh,
            num_warps=2,
        )

        return q_flat.view_as(query), k_flat.view_as(key)

    def _apply_interleaved(self, x):
        """Reorganize from [TTT...HHH...WWW] to interleaved [THWTHW...]."""
        s = self.mrope_section
        result = x[0].clone()
        result[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        result[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return result
