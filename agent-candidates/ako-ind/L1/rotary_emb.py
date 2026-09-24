"""Rotary position embeddings (RoPE), with optional Llama 3.1-style frequency scaling.

Uses a custom CUDA kernel for high-performance in-place RoPE.  The pybind11
op is wrapped into a ``torch.library`` custom op with in-place mutation
annotations so ``torch.compile`` handles functionalization automatically.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import lazy_op

# NOTE (candidate isolation): the harness imports the baseline module and this
# candidate into the SAME process.  Both the extension name and the
# ``torch.library`` namespace must therefore differ from the baseline's
# ("rotary_emb" / "fastkernels_rope"):
#   * ``lazy_op`` keys its ninja build directory and ``.so`` on the extension
#     *name* alone, so a shared name makes the candidate silently rebuild over
#     -- and then load -- the baseline's artifact (a no-op candidate).
#   * a ``torch.library`` namespace accepts exactly one "DEF" per process, so a
#     shared namespace raises on import.
_C = lazy_op("rotary_emb_ako", "rotary_emb_ako.cu")

_DEBUG = bool(os.environ.get("FK_ROPE_DEBUG", ""))
if _DEBUG:
    print("[fk-rope-ako] candidate module imported from", __file__, flush=True)

# ---------------------------------------------------------------------------
# Register in-place rotary embedding op for torch.compile compatibility.
# Uses Tensor(a!) annotations so Inductor auto-functionalizes correctly.
# ---------------------------------------------------------------------------

_lib = torch.library.Library("fastkernels_rope_ako", "DEF")

_lib.define(
    "rotary_embedding(Tensor positions, Tensor(a!) query, "
    "Tensor(b!)? key, int head_size, Tensor cos_sin_cache, "
    "bool is_neox) -> ()"
)

def _rotary_embedding_impl(
    positions, query, key, head_size, cos_sin_cache, is_neox,
):
    _C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

_lib.impl("rotary_embedding", _rotary_embedding_impl, "CUDA")

@torch.library.impl(_lib, "rotary_embedding", "Meta")
def _rotary_embedding_meta(positions, query, key, head_size, cos_sin_cache, is_neox):
    pass


def _compute_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    low_wl = original_max_position_embeddings / low_freq_factor
    high_wl = original_max_position_embeddings / high_freq_factor
    wl = 2 * math.pi / inv_freq
    if low_freq_factor != high_freq_factor:
        smooth = (original_max_position_embeddings / wl - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = torch.zeros_like(inv_freq)
    return torch.where(
        wl < high_wl,
        inv_freq,
        torch.where(
            wl > low_wl,
            inv_freq / scaling_factor,
            (1 - smooth) * inv_freq / scaling_factor + smooth * inv_freq,
        ),
    )


class RotaryEmbedding(nn.Module):
    """RoPE with optional Llama 3.1-style frequency scaling.

    When rope_scaling_factor == 1.0 (default), behaves as standard RoPE.
    When rope_scaling_factor != 1.0, applies the Llama 3.1 piecewise
    frequency scaling controlled by low/high freq factors.
    """

    # Memo for :meth:`_typed_cache`.  Class-level defaults so subclasses that
    # bypass ``RotaryEmbedding.__init__`` (see
    # ``Gemma4ProportionalRotaryEmbedding``) still read a sane value.
    _cs_src = None
    _cs_cast = None

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling_factor: float = 1.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 1.0,
        rope_original_max_position_embeddings: int | None = None,
        is_neox_style: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))

        if rope_scaling_factor != 1.0 and rope_original_max_position_embeddings is not None:
            inv_freq = _compute_scaled_inv_freq(
                inv_freq,
                rope_scaling_factor,
                rope_low_freq_factor,
                rope_high_freq_factor,
                rope_original_max_position_embeddings,
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def forward_native(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch NeOX-style RoPE matching the CUDA kernel.

        The cache stores [cos, sin] each with embed_dim = head_dim/2 entries.
        Rotation pairs elements (i, i + embed_dim) across the full head,
        exactly matching the CUDA kernel's IS_NEOX=true path:
          out[i]            = x[i]*cos[i] - x[i+embed_dim]*sin[i]
          out[i+embed_dim]  = x[i+embed_dim]*cos[i] + x[i]*sin[i]
        """
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim]
        sin = cos_sin[..., embed_dim:]

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q1, q2 = q[..., :embed_dim], q[..., embed_dim:]
        k1, k2 = k[..., :embed_dim], k[..., embed_dim:]

        query = torch.cat([q1 * cos - q2 * sin,
                           q2 * cos + q1 * sin], dim=-1).view(q_shape)
        key = torch.cat([k1 * cos - k2 * sin,
                         k2 * cos + k1 * sin], dim=-1).view(k_shape)
        return query, key

    @staticmethod
    def forward_native_interleaved(positions, query, key, head_dim, cos_sin_cache):
        """Pure PyTorch GPT-J/interleaved RoPE matching CUDA IS_NEOX=false."""
        cos_sin = cos_sin_cache[positions]
        embed_dim = cos_sin.shape[-1] // 2
        cos = cos_sin[..., :embed_dim].unsqueeze(1)
        sin = cos_sin[..., embed_dim:].unsqueeze(1)

        q_shape = query.shape
        k_shape = key.shape
        q = query.view(q_shape[0], -1, head_dim)
        k = key.view(k_shape[0], -1, head_dim)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_rot = torch.stack(
            (q_even * cos - q_odd * sin,
             q_odd * cos + q_even * sin),
            dim=-1,
        ).flatten(-2)
        k_rot = torch.stack(
            (k_even * cos - k_odd * sin,
             k_odd * cos + k_even * sin),
            dim=-1,
        ).flatten(-2)
        return q_rot.view(q_shape), k_rot.view(k_shape)

    def _typed_cache(self, dtype):
        """``cos_sin_cache`` in *dtype*, converted at most once.

        The table is a constant, but ``cache.to(query.dtype)`` on every forward
        re-converts all of it: for Llama-3.1-8B that is a 131072x128 fp32
        buffer, i.e. 67 MB read + 33.5 MB write *per call* -- an order of
        magnitude more traffic than the RoPE itself at the decode shapes that
        dominate the capture.  Memoize the converted copy instead.

        Keyed on the buffer *object* rather than on dtype/device alone: every
        ``Module._apply`` (``.to(device)``, ``.float()``, ...) rebinds
        ``_buffers['cos_sin_cache']`` to a fresh tensor, so an identity check
        against the source invalidates a stale copy for free.
        """
        src = self.cos_sin_cache
        if src.dtype == dtype:
            return src
        cast = self._cs_cast
        if cast is not None and self._cs_src is src and cast.dtype == dtype:
            return cast
        cast = src.to(dtype)
        self._cs_src = src
        self._cs_cast = cast
        return cast

    def forward_cuda(self, positions, query, key):
        """CUDA kernel path for eager mode."""
        torch.ops.fastkernels_rope_ako.rotary_embedding(
            positions, query, key, self.head_dim,
            self._typed_cache(query.dtype), self.is_neox_style,
        )
        return query, key

    def forward(self, positions, query, key):
        if torch.compiler.is_compiling():
            # Left functional on purpose: under tracing the memo in
            # ``_typed_cache`` would be a side effect on ``self``, and Inductor
            # hoists this constant conversion out of the graph anyway.
            cache = self.cos_sin_cache
            if cache.dtype != query.dtype:
                cache = cache.to(query.dtype)
            if self.is_neox_style:
                return self.forward_native(
                    positions, query, key, self.head_dim, cache,
                )
            return self.forward_native_interleaved(
                positions, query, key, self.head_dim, cache,
            )
        return self.forward_cuda(positions, query, key)


class Gemma4ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma4 proportional RoPE.

    Gemma4 full-attention layers use a partial rotary factor, but the
    frequency exponents are divided by the full head dimension and the
    non-rotated angle pairs are represented as identity rotation.  This
    matches HF/vLLM's proportional RoPE instead of rotating a compact
    leading slice with ``rotary_dim`` as the denominator.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ):
        nn.Module.__init__(self)
        self.head_dim = head_dim
        self.is_neox_style = True
        rope_angles = rotary_dim // 2
        nope_angles = (head_dim // 2) - rope_angles

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim
            )
        )
        if nope_angles > 0:
            inv_freq = torch.cat(
                [inv_freq, torch.zeros(nope_angles, dtype=torch.float)],
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).float()
        self.register_buffer("cos_sin_cache", cache, persistent=False)
