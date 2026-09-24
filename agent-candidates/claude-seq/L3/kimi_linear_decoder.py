"""Kimi-Linear decoder layer: the KDA chunk tile sized for the step, not the chunk.

The four stages -- fused add + RMSNorm, KDA linear attention, a second fused
add + RMSNorm, and the MoE (or dense SwiGLU) block -- are all frozen L2/L1
winners, and at most captured widths each is already at its own floor.  From 26
tokens up the MoE streams every touched expert's weights exactly once and
measures ~7 TB/s doing it (the HBM roofline), and at 16k tokens the layer is
8.3 ms of back-to-back kernels with well under a percent of launch gap in it.

What is *not* at its floor is the narrow end.  ``KimiDeltaAttention`` runs its
prep and one-chunk kernels at the 64-row chunk size whatever the step is, so a
one-token decode still pays for 64-row conv windows and 64x64 chunk matrices:
profiled at one token those two launches are ~36 us of the layer's ~123 us of
device time, for 1/64th of a chunk of real work.  :meth:`_attn_tiled` relaunches
the same two kernels with the tile rounded up from the token count instead --
identical arithmetic (see the method), 16x less of it at one token.

Measured against this file's own fallback path on the scorer's five captured
cases, interleaved per case so the baseline's ~10% run-to-run drift cancels in
the ratio: 1.15x at one token with either MLP, 1.02x at 26, and unchanged at 611
and 16384 (where the tile would be the full 64 rows anyway) -- 1.06x on the
geomean.

Two other things were tried here and are deliberately absent:

* **A whole-layer CUDA graph.**  At one token the layer is 122 us of wall time
  for 91 us of device work, and capturing the shape brought that to ~114 us
  (+2%).  But ``torch.cuda.graph`` calls ``empty_cache()``, which hands pages
  back to the driver -- and the scorer's ``_sanitize_float_params`` only rewrites
  a weight whose amax is outside ``(1e-6, 1e4)``, so the ``torch.empty``
  ``dt_bias`` of a later case then picks up foreign data and keeps it.  A large
  ``dt_bias`` drives the KDA gate past ~128 log2 units per token, where the
  frozen chunk kernel's ``exp2`` overflows to inf and the following ``tl.dot``
  turns ``0 * inf`` into NaN (measured threshold: clean at 59.5 log2 units, NaN
  at 133.9, identically for a 16- and a 64-row tile).  Capturing made
  ``validate.py`` fail that way in 4 of 12 runs; without it, 0 of 10.  Two
  percent is not worth a coin flip.
* **A hand-written streaming GEMV for the one-row projections.**  cuBLAS answers
  ``[1, K] x [K, N]`` with a split-K kernel plus a separate fp32 reduce launch,
  which measures ~2 TB/s -- but every (tile, warp) configuration of a plain
  register-accumulating reduction over the same weights came out slower in
  place, so the projections stay on ``torch.mm``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton

from ..L1.rms_norm import RMSNorm
from ..L2 import kimi_delta_attention as _kda
from ..L2.kimi_delta_attention import KimiDeltaAttention
from ..L2.kimi_mla_attention import KimiMLAAttention
from ..L2.kimi_moe import KimiMoE
from ..L2.llama_mlp import LlamaMLP


# Smallest chunk tile for the single-chunk KDA prefill: the gate factoring in
# ``_kda_intra`` is per 16-row block, so the tile has to be a multiple of that.
_KDA_TILE_MIN = 16
_KDA_TILED = os.environ.get("FK_KIMI_LAYER_KDA_TILE", "1") != "0"
# A 16- or 32-row tile is a fraction of the work the 8 warps of the frozen
# launch were sized for; 4 measured best over {2, 4, 8} on both.
_KDA_TILE_WARPS = int(os.environ.get("FK_KIMI_LAYER_KDA_TILE_WARPS", "4"))


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            self.self_attn = KimiDeltaAttention(
                config,
                layer_idx=layer_idx,
                quant_config=quant_config,
            )
        else:
            self.self_attn = KimiMLAAttention(
                config,
                quant_config=quant_config,
            )

        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = KimiMoE(config, quant_config=quant_config)
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )
        self._tiled_attn = _KDA_TILED and self.is_kda

    def _attn_tiled(self, hs):
        """The frozen single-chunk KDA prefill, on a tile sized for this step.

        The tile is free to shrink below the 64-row chunk.  It only has to be a
        multiple of the 16-row gate block (the per-block ``exp2`` factoring in
        ``_kda_intra``) and at least ``T``: rows past ``T`` load as zero, so they
        contribute nothing to any reduction and the masked stores skip them, and
        the ``(I+A)^-1`` binary series stays exact because a strictly lower
        triangular ``A`` of order ``BT`` is nilpotent with ``A**BT == 0`` -- the two
        factors the 64-row form adds are then the identity.  So the same kernels,
        launched with ``BT`` 16 or 32, compute the same numbers over 16x/4x less
        work.

        Returns ``None`` for anything outside the frozen fast path's own
        preconditions -- one contiguous prefill from token 0, an empty recurrent
        state, unquantized projections -- or for a step wide enough that the tile
        would be the full 64 rows anyway, leaving the caller to dispatch normally.
        """
        a = self.self_attn
        T = hs.shape[0]
        BT = max(_KDA_TILE_MIN, triton.next_power_of_2(T))
        if BT >= _kda._CHUNK:
            return None
        if a._fused is None:
            a._build_fused()
        if not a._fused:
            return None
        state, meta = a._get_state()
        if state is None or meta is None:
            return None
        if (meta.num_prefills != 1 or meta.num_decodes != 0
                or meta.any_have_initial_state
                or int(meta.num_actual_tokens) != T):
            return None
        a._ensure_triton_allocator(hs.device)

        D, H = a.head_dim, a.local_num_heads
        dev, dt = hs.device, a._w_conv.dtype
        sidx = meta.non_spec_state_indices_tensor
        P = torch.mm(hs, a._w_in_t)
        out = torch.empty((T, H * D), device=dev, dtype=dt)
        # slot 3 doubles as the raw gate rows (FUSE_FG) and the output gate
        qkv = torch.empty((4, T, H, D), device=dev, dtype=dt)
        beta = torch.empty((T, H), device=dev, dtype=torch.float32)
        g = torch.empty((T, H, D), device=dev, dtype=torch.float32)
        try:
            _kda._kda_prep_kernel[(1, H, 4)](
                P, qkv[3], a._wfg, a._w_conv, a._gab, qkv, beta, g,
                state.q_conv_state, state.k_conv_state, state.v_conv_state,
                sidx, T,
                NIN=a._nin, H=H, D=D, W=a.conv_size, BT=BT, FUSE_FG=True,
                num_warps=_KDA_TILE_WARPS, num_stages=_kda._NS_PREP,
            )
            _kda._kda_chunk1_kernel[(H,)](
                qkv, g, beta, a.o_norm.weight, out, state.recurrent_state, sidx, T,
                H=H, D=D, BT=BT, RMSEPS=a._rms_eps, num_warps=_KDA_TILE_WARPS,
            )
        except Exception:
            # A tile these kernels cannot be compiled for: never try again.
            self._tiled_attn = False
            return None
        if a._wo_t is not None:
            return torch.mm(out, a._wo_t)
        return a.o_proj(out)

    def forward(self, hidden_states, residual, state_manager=None):
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self._attn_tiled(hidden_states) if self._tiled_attn else None
        hidden_states = (
            attn_out if attn_out is not None
            else self.self_attn(hidden_states, state_manager=state_manager)
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
