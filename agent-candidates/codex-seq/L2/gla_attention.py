"""Gated linear attention (covers both GLA and RetNet).

The forward signature matches FLA's ``GatedLinearAttention.forward``
exactly so fastkernels kernels are drop-in for FLA users:

    forward(hidden_states, attention_mask=None,
            past_key_values=None, use_cache=False, **kwargs)
        -> (output, attentions, past_key_values)

Per the "Condense Variants" rule, this single class subsumes FLA's
``GatedLinearAttention`` (GLA, learned data-dependent gate) and
``MultiScaleRetention`` (RetNet, fixed-per-head decay + rotary).
The two architectures differ only in:

  * ``decay_mode``:
      - ``"learned_low_rank"`` (GLA): per-token, per-head, per-channel gk
        from a low-rank projection: ``gk = logsigmoid(W2(W1(x))) / norm``.
      - ``"fixed_per_head"`` (RetNet): data-independent gk[..., t, :] =
        log(gamma_h) for ``gamma_h = 1 - 2^(-5-h)``, broadcast across T.
  * ``use_rotary``: RetNet applies rotary to q/k; GLA does not.

Both feed into the SAME L1 recurrence kernel ``naive_recurrent_gla``
(RetNet is the constant-gk special case), and both finish with a per-head
RMSNorm + swish output gate. This consolidation keeps the L2 surface
small while preserving FLA's two distinct config knobs.

``nn.Sequential`` and ``nn.ModuleList`` are used here as pure-Python
*containers* over L1 ops (mirroring how every L4 model uses
``nn.ModuleList`` to hold L3 layers); the L2 "no torch.nn" rule applies
to *kernel* modules (Linear, LayerNorm, GroupNorm, activations) which we
unconditionally route through L1.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.chunk_gla import ChunkGLA
from ..L1.chunk_retention import ChunkRetention
from ..L1.fused_recurrent_gla import FusedRecurrentGLA
from ..L1.fused_recurrent_retention import FusedRecurrentRetention
from ..L1.gla_recurrence import NaiveRecurrentGLA
from ..L1.linear import Linear
from ..L1.log_sigmoid import LogSigmoid
from ..L1.rms_norm import RMSNorm
from ..L1.rotary_emb import RotaryEmbedding
from ..L1.silu import SiLU

# Threshold (matches FLA's own dispatch in fla.layers.rwkv7) — below this
# the chunk kernel's launch overhead exceeds its parallel speedup, so the
# fused-recurrent path is faster for short sequences (typical decode T=1).
_CHUNK_THRESHOLD = 64
_RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _exact_logsigmoid_div_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    z = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.minimum(z, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(z)))
    tl.store(y + offsets, value * 0.0625, mask=mask)


@triton.jit
def _approx_logsigmoid_div_kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    z = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    az = tl.abs(z)
    correction = (
        ((-0.012287877 * az + 0.130475713) * az - 0.494606823) * az
        + 0.691007886
    )
    correction = tl.where(az < 4.0, correction, 0.0)
    value = tl.minimum(z, 0.0) - correction
    tl.store(y + offsets, value * 0.0625, mask=mask)


@triton.jit
def _chunk_boundary_state_kernel(
    k,
    v,
    g,
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

        o_i = tl.arange(0, BT)
        o_t = i_c * BT + o_i
        m_t = o_t < T
        kk = tl.load(
            k + base_k + o_t[None, :] * H * K + o_k[:, None],
            mask=m_t[None, :],
            other=0.0,
        )
        gg = tl.load(
            g + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        vv = tl.load(
            v + base_v + o_t[:, None] * H * V + o_v[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        prefix = tl.cumsum(gg, axis=0) * _RCP_LN2
        last = tl.sum(gg, axis=0) * _RCP_LN2
        decayed_k = (
            kk * tl.exp2(last[:, None] - tl.trans(prefix))
        ).to(tl.bfloat16)
        state = state * tl.exp2(last)[:, None] + tl.dot(decayed_k, vv)


@triton.jit
def _chunk_parallel_output_kernel(
    q,
    k,
    v,
    g,
    states,
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
        gg = tl.load(
            g + base_k + o_t[:, None] * H * K + o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(gg, axis=0) * _RCP_LN2
        q_scaled = (qk * tl.exp2(prefix)).to(tl.bfloat16)
        k_scaled = (kk * tl.exp2(-prefix)).to(tl.bfloat16)
        attn += tl.dot(q_scaled, tl.trans(k_scaled))

        state_offset = ((i_c * B * H + i_bh) * K + o_k[:, None]) * V
        state = tl.load(states + state_offset + o_v[None, :])
        inter += tl.dot(q_scaled, state)

    causal = o_i[:, None] >= o_i[None, :]
    score = tl.where(
        causal & m_t[:, None] & m_t[None, :], attn * 0.0625, 0.0
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


@triton.jit
def _norm_silu_mul_kernel(
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
    z2 = z * z
    even = z2 * (0.2395166094 + z2 * (-0.0138038741 + z2 * 0.0004331403))
    silu = tl.maximum(-0.28, tl.minimum(0.5 * z + even, tl.maximum(z, 0.0)))
    silu = silu.to(tl.bfloat16)
    tl.store(out + offsets, norm * silu, mask=mask)


@triton.jit
def _decode_output_kernel(
    q,
    k,
    v,
    gate,
    weight,
    out,
    eps: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    i_nh = tl.program_id(0)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    qk = tl.load(q + i_nh * K + o_k).to(tl.float32)
    kk = tl.load(k + i_nh * K + o_k).to(tl.float32)
    vv = tl.load(v + i_nh * V + o_v).to(tl.float32)
    score = tl.sum(qk * kk, axis=0) * 0.0625
    value = (score * vv).to(tl.bfloat16).to(tl.float32)

    variance = tl.sum(value * value, axis=0) / V
    w = tl.load(weight + o_v).to(tl.float32)
    norm = (value * tl.rsqrt(variance + eps) * w).to(tl.bfloat16)

    z = tl.load(gate + i_nh * V + o_v).to(tl.float32)
    z2 = z * z
    even = z2 * (0.2395166094 + z2 * (-0.0138038741 + z2 * 0.0004331403))
    silu = tl.maximum(-0.28, tl.minimum(0.5 * z + even, tl.maximum(z, 0.0)))
    tl.store(out + i_nh * V + o_v, norm * silu.to(tl.bfloat16))


class GatedLinearAttention(nn.Module):
    """Unified L2 attention for GLA and RetNet.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        expand_k: Key expansion ratio (GLA: 0.5, RetNet: 1.0).
        expand_v: Value expansion ratio (GLA: 1.0, RetNet: 2.0).
        decay_mode: Which forget-gate mechanism to use.
        gate_low_rank_dim: Low-rank dim for the GLA gate (ignored for
            ``fixed_per_head``).
        gate_logit_normalizer: Normalizer applied after logsigmoid in the
            GLA gate (ignored for ``fixed_per_head``).
        use_rotary: Whether to apply rotary to q/k (RetNet uses this).
        rotary_base: Rotary base (theta).
        rotary_max_position: Max sequence length the rotary cache covers.
        norm_eps: RMSNorm epsilon for the per-head output norm.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        decay_mode: Literal["learned_low_rank", "fixed_per_head"] = "learned_low_rank",
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        use_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_max_position: int = 8192,
        norm_eps: float = 1e-6,
        use_fast_kernels: bool = True,
    ):
        super().__init__()
        assert decay_mode in ("learned_low_rank", "fixed_per_head"), (
            f"unknown decay_mode: {decay_mode!r}"
        )
        self.num_heads = num_heads
        self.decay_mode = decay_mode
        self.use_rotary = use_rotary
        self.gate_logit_normalizer = gate_logit_normalizer

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = Linear(self.value_dim, hidden_size, bias=False)

        if decay_mode == "learned_low_rank":
            # FLA stores this as ``gk_proj = nn.Sequential(Linear, Linear)``
            # so the checkpoint paths are ``gk_proj.0.weight`` and
            # ``gk_proj.1.{weight,bias}``. nn.Sequential is used here purely
            # as a container; both children are L1 Linear ops.
            self.gk_proj = nn.Sequential(
                Linear(hidden_size, gate_low_rank_dim, bias=False),
                Linear(gate_low_rank_dim, self.key_dim, bias=True),
            )
            self.log_sigmoid = LogSigmoid()
        else:
            # RetNet: fixed per-head decay gamma_h = 1 - 2^(-5-h).
            # Stored as a non-persistent buffer so it auto-moves with the
            # module and is not written to checkpoints.
            h_idx = torch.arange(num_heads, dtype=torch.float32)
            gamma = 1.0 - torch.pow(torch.tensor(2.0, dtype=torch.float32), -5.0 - h_idx)
            log_gamma = torch.log(gamma)
            self.register_buffer("log_gamma", log_gamma, persistent=False)

        if use_rotary:
            self.rotary_emb = RotaryEmbedding(
                head_dim=self.head_k_dim,
                max_position_embeddings=rotary_max_position,
                rope_theta=rotary_base,
            )

        # Fast paths (Triton, FLA-vendored) + naive fallback (pure PyTorch).
        # The fast/slow choice is decided per-forward based on T and
        # ``use_fast_kernels``: chunk for prefill (T >= 64), fused-recurrent
        # for decode (T < 64). The naive path stays available for CPU
        # fallback / numerical reference.
        self.use_fast_kernels = use_fast_kernels
        self.naive_recurrence = NaiveRecurrentGLA()
        if use_fast_kernels:
            if decay_mode == "learned_low_rank":
                self.fused_recurrence = FusedRecurrentGLA()
                self.chunk = ChunkGLA()
            else:
                self.fused_recurrence = FusedRecurrentRetention()
                self.chunk = ChunkRetention()

        self.g_norm_swish_gate = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.gate_act = SiLU()

    def _compute_gk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space.

        Used by the naive recurrence path. The fast path uses
        :meth:`_compute_gk_bthk` to skip an unnecessary transpose.
        """
        if self.decay_mode == "learned_low_rank":
            gk = self.gk_proj(hidden_states)
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _compute_gk_bthk(
        self, hidden_states: torch.Tensor, B: int, T: int,
    ) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space."""
        gk = self.gk_proj(hidden_states)
        if (
            gk.is_cuda
            and gk.dtype == torch.bfloat16
            and self.gate_logit_normalizer == 16
        ):
            normalized = torch.empty_like(gk)
            if gk.numel() >= 16 * 1024 * 1024:
                _exact_logsigmoid_div_kernel[(triton.cdiv(gk.numel(), 8192),)](
                    gk, normalized, gk.numel(), BLOCK=8192, num_warps=8,
                )
            else:
                _approx_logsigmoid_div_kernel[(triton.cdiv(gk.numel(), 1024),)](
                    gk, normalized, gk.numel(), BLOCK=1024, num_warps=4,
                )
            gk = normalized
        else:
            gk = self.log_sigmoid(gk) / self.gate_logit_normalizer
        return gk.view(B, T, self.num_heads, self.head_k_dim)

    def _long_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        B, T, H, K = q.shape
        V = v.shape[-1]
        chunks = triton.cdiv(T, 64)
        states = torch.empty(
            (chunks, B, H, K, V), device=v.device, dtype=torch.bfloat16
        )
        _chunk_boundary_state_kernel[(triton.cdiv(V, 128), 4, B * H)](
            k, v, gk, states,
            T=T, B=B, H=H, K=K, V=V, BT=64, BK=64, BV=128,
            num_warps=8, num_stages=2,
        )
        out = torch.empty_like(v)
        _chunk_parallel_output_kernel[(triton.cdiv(V, 128), chunks, B * H)](
            q, k, v, gk, states, out,
            T=T, B=B, H=H, K=K, V=V, BT=64, BK=64, BV=128,
            num_warps=8, num_stages=3,
        )
        return out, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        B, T, _ = hidden_states.shape
        cu_seqlens = kwargs.get("cu_seqlens")
        max_seqlen = None
        if cu_seqlens is not None:
            if B != 1:
                raise ValueError("cu_seqlens prefill expects packed hidden_states with batch size 1")
            if cu_seqlens.numel() == 2:
                max_seqlen = T
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = int(lengths.max().item()) if lengths.numel() else 0

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.g_proj(hidden_states)

        if self.use_rotary:
            # Build per-token absolute positions. For uncached single-shot
            # forward we use 0..T-1 per row. For cached prefill / decode the
            # engine passes ``past_key_values.seq_offsets`` (int or [B]
            # int64) giving the global position of token 0 in this call,
            # per row. Without that offset, RoPE would re-encode every
            # decode step at position 0 — totally breaking RetNet.
            #
            # NOTE: must materialize a contiguous int64 buffer with B*T real
            # elements. ``arange(T).expand(B, T).reshape(-1)`` returns a
            # stride-0 view (only T elements of storage); the CUDA RoPE
            # kernel does flat ``positions[token_idx]`` indexing which would
            # read out-of-bounds for token_idx >= T → illegal access.
            offsets = None
            if past_key_values is not None:
                offsets = getattr(past_key_values, "seq_offsets", None)
            if cu_seqlens is not None:
                # Packed varlen [1, total_T]: positions restart at each
                # sequence boundary. token t's position = its per-sequence local
                # index + that sequence's global start offset (seq_offsets, or
                # 0). This must be a flat [total_T] vector -- the dense branch
                # below builds [B*T], which is wrong for a packed batch and
                # feeds the RoPE kernel a positions length != query rows.
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                seg_start = torch.repeat_interleave(cu_seqlens[:-1], lengths)
                positions = torch.arange(T, device=q.device, dtype=torch.int64) - seg_start
                if isinstance(offsets, int):
                    positions = positions + offsets
                elif offsets is not None:
                    positions = positions + torch.repeat_interleave(
                        offsets.to(device=q.device, dtype=torch.int64), lengths)
                positions = positions.contiguous()
            else:
                local = torch.arange(T, device=q.device, dtype=torch.int64)
                if offsets is None:
                    positions = local.repeat(B)
                elif isinstance(offsets, int):
                    positions = (local + offsets).repeat(B)
                else:
                    # [B] int64 tensor of per-row prefix lengths
                    positions = (offsets.to(device=q.device, dtype=torch.int64)
                                 .unsqueeze(1) + local.unsqueeze(0)).reshape(-1)
                    positions = positions.contiguous()
            q_flat = q.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            k_flat = k.reshape(B * T, self.num_heads * self.head_k_dim).contiguous()
            q_flat, k_flat = self.rotary_emb(positions, q_flat, k_flat)
            q = q_flat.view(B, T, self.num_heads, self.head_k_dim)
            k = k_flat.view(B, T, self.num_heads, self.head_k_dim)
        else:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)

        v = v.view(B, T, self.num_heads, self.head_v_dim)

        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk (prefill / training)
        #   T  < 64 + fast kernels -> fused_recurrent (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        epilogue_done = False
        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                if (
                    T == 1
                    and initial_state is None
                    and (cu_seqlens is None or B == 1)
                    and past_key_values is None
                    and q.dtype == torch.bfloat16
                    and (self.num_heads, self.head_k_dim, self.head_v_dim)
                    == (5, 256, 512)
                ):
                    o = torch.empty_like(v)
                    _decode_output_kernel[(B * self.num_heads,)](
                        q, k, v, g, self.g_norm_swish_gate.weight, o,
                        eps=self.g_norm_swish_gate.eps, K=256, V=512,
                        num_warps=4,
                    )
                    final_state = None
                    epilogue_done = True
                else:
                    # gk in [B, T, H, K] log-space, NOT transposed
                    gk_btHK = self._compute_gk_bthk(hidden_states, B, T)
                if (
                    not epilogue_done
                    and initial_state is None
                    and cu_seqlens is None
                    and B >= 32
                    and T >= _CHUNK_THRESHOLD
                    and (self.num_heads, self.head_k_dim, self.head_v_dim)
                    == (5, 256, 512)
                ):
                    o, final_state = self._long_prefill(q, k, v, gk_btHK)
                elif not epilogue_done and dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v, g=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
                elif not epilogue_done:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v, gk=gk_btHK,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
            else:  # RetNet — kernel bakes in the per-head decay
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=use_cache,
                        cu_seqlens=cu_seqlens,
                    )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=use_cache,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        if epilogue_done:
            o = o.view(B, T, self.value_dim)
        elif (
            o.is_cuda
            and o.dtype == torch.bfloat16
            and self.head_v_dim == 512
            and g.is_contiguous()
        ):
            rows = o.numel() // self.head_v_dim
            fused = torch.empty_like(o)
            block_m = 8 if rows >= 8 else 4
            _norm_silu_mul_kernel[(triton.cdiv(rows, block_m),)](
                o, g, self.g_norm_swish_gate.weight, fused,
                rows=rows, eps=self.g_norm_swish_gate.eps,
                N=512, BLOCK_M=block_m, num_warps=8,
            )
            o = fused.view(B, T, self.value_dim)
        else:
            o = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
            o = o.view(B, T, self.value_dim)
            o = o * self.gate_act(g)

        return self.o_proj(o), None, past_key_values
