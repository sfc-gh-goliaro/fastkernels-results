"""L3 Kimi-Linear decoder layer, tuned for per-call CPU/launch overhead.

The captured call mix is dominated by narrow decode widths (~13.2k of ~16k calls
carry 1-88 tokens), where every kernel in the layer finishes long before the host
can queue the next one. Measured with the identity kernel, n=1, n=64 and n=611 all
land within 3% of each other -- the layer costs what it costs to *launch*, not to
compute. So everything here is about doing less per call: fewer stream syncs,
fewer launches, fewer allocations.

In decreasing order of measured effect:

1. **No device->host syncs.** A sync audit of the baseline
   (``torch.cuda.set_sync_debug_mode("warn")``) finds exactly two, both in
   ``KimiDeltaAttention.forward_impl``: it derives "which recurrent slots must be
   zeroed" as ``state_indices[~has_initial_state]``, a data-dependent shape, so
   the ``masked_select`` and the scatter that follows each block the stream.
   Timed in isolation on a loaded host they were 2.35 ms + 2.34 ms of a 9.4 ms
   layer, and they also stop the host from ever running ahead of the GPU.
   ``KimiLinearMetadata`` already carries the host-side
   ``any_have_initial_state`` / ``all_have_initial_state`` summary for exactly
   this decision (see its docstring), so we read that; the mixed case falls back
   to a maskless multiply, which has a fixed shape.
2. **The MoE tail runs from a CUDA graph, keyed by token count.** FlashInfer's
   ``trtllm_bf16_moe`` costs ~0.75 ms of *host* time per call, and only ~0.14 ms
   of that is its Python wrapper (``MoERunner`` + ``AutoTuner.choose_one``
   rebuilt per call) -- the rest is inside the C++ launcher, which constructs a
   launcher object for every supported tile size on every call. It does not
   block, so it cannot be hidden. Router GEMM + experts + shared expert are
   stateless in everything but ``hidden_states``, so one capture per width
   replaces all of it with a copy-in and a replay. Capture failure falls back to
   the eager path. (The *layer* is deliberately not captured: KDA's state lives
   on a global Context whose metadata tensors are rebuilt every step.)
3. **The recurrent scan replaces the chunked KDA below 128 tokens.** The chunked
   pipeline is ~10 Triton launches (l2norm x2, gate cumsum, kkt x2, solve_tril,
   w_u, h, o); the scan is 2, and it is the same recurrence in the same fp32
   state -- it is what vLLM's own decode path runs. 350 us -> 65 us of host time.
   Two traps: its in-place state store indexes ``ssm_state_indices`` by *token*
   and ignores the tok stride, so a 1-D index reads past its end (illegal
   accesses at T=64); and filling a [P, T] index with the slot makes it restore
   the state once per token (71 us at T=64), so all but the first and last column
   are set to ``NULL_BLOCK_ID``.
4. **Three hand-written fusions**, all validated against the ops they replace:
   one Triton launch for all three q/k/v short-convs plus the beta sigmoid
   (was 3 launches through the varlen conv wrapper + 2 elementwise, ~0.12 ms);
   one launch for both low-rank gate projections (cuBLAS picks split-K for their
   K=128, so they cost a GEMM *and* a splitKreduce each, ~28 us of GPU for 2 MB
   of weights); and q/k/v/beta/f_a/g_a in one wide GEMM.
5. **No scratch allocate-and-copy.** ``core_attn_out`` was a fresh
   ``torch.zeros`` that the kernel output was then copied into; for a pure-prefill
   or pure-decode batch (every captured case) the output already covers every
   row, so it is returned directly -- which also drops ~400 MB of HBM traffic on
   the 16384-token prefill. Same for the first-layer ``residual.clone()``, which
   vLLM also aliases.

Correctness note: the harness's random weights leave ``o_proj``, ``w13``, ``w2``
and ``gate_up`` exactly zero, so its own comparison cannot see anything the KDA
does. The fusions above were validated separately with weights that carry signal
(see ``## Notes`` in ITERATIONS.md): the conv matches ``causal_conv1d_fn`` to
within one bf16 ULP with bitwise-identical conv states, the gate projection is
bitwise identical to the two ``F.linear`` calls, and the whole layer stays closer
to the baseline than the baseline is to itself under a one-ULP input perturbation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..L1.rms_norm import RMSNorm
from ..L2.kimi_delta_attention import KimiDeltaAttention
from ..L2.kimi_mla_attention import KimiMLAAttention
from ..L2.kimi_moe import KimiMoE
from ..L2.llama_mlp import LlamaMLP
from ..L1.kda import chunk_kda_with_fused_gate, fused_kda_gate, fused_recurrent_kda

# Up to this many tokens the sequential recurrent KDA kernel (2 launches) beats
# the chunked one (~10 launches over l2norm / kkt / solve_tril / w_u / h / o),
# because at these widths the layer is bound by how fast the host can queue
# kernels, not by the O(T*D^2) the scan does. Same recurrence, same fp32 state --
# it is the kernel vLLM's own decode path uses. The bound is 64 rather than
# larger because the in-place state store fires once per token, so its 2 MB
# write is what eventually overtakes the chunked path.
_RECURRENT_MAX_TOKENS = 128

# CUDA-graph the MoE's *stateless* tail (router GEMM + trtllm-gen experts +
# shared expert) below this width. Not the layer: KDA's state lives on a global
# Context and its metadata tensors are rebuilt per step, but the MoE reads only
# hidden_states plus frozen weights, so one capture per token count is safe and
# replaces ~0.75 ms/call of host work (FlashInfer rebuilds a launcher per
# supported tile size *inside the C++ op*, so bypassing the Python wrapper alone
# only recovers ~140 us of it).
_MOE_GRAPH_MAX_TOKENS = 512



# ---------------------------------------------------------------------------
# Fused q/k/v short-conv
#
# ``KimiDeltaAttention`` runs three separate ``causal_conv1d_fn`` calls over three
# contiguous column ranges of the *same* projection output. Each costs a Triton
# launch plus ~25 us of Python in the varlen wrapper (stride bookkeeping,
# chunk-metadata plumbing) -- ~0.12 ms/call at decode width, the largest single
# item left once the MoE is graphed. This does all three in one launch: one
# program per (sequence, channel block, group), carrying the depthwise window in
# registers, which also lets the conv state update fall out of the same registers
# instead of a second masked pass.
#
# Semantics follow ``causal_conv1d_fn``'s prefill path exactly: silu activation,
# taps ordered oldest-first, zero history when ``has_initial_state`` is false,
# the trailing ``KW-1`` inputs written back to ``conv_states[cache_idx]`` with
# shift-left when the sequence is shorter than the state, ``PAD_SLOT_ID`` (-1)
# sequences skipped whole and ``NULL_BLOCK_ID`` (0) slots read as zero history.
# ---------------------------------------------------------------------------
_PAD_SLOT_ID = -1
_NULL_BLOCK_ID = 0

# One program walks one whole sequence, so past a few hundred tokens the
# reference kernel's token-blocked grid wins on parallelism (measured crossover
# ~256: 157 us vs 159 us for the three calls; at T=2048 it is 2.2 ms vs 143 us).
_FUSED_CONV_MAX_QUERY = 256


@triton.jit
def _qkv_conv_fwd_kernel(
    x_ptr, w_ptr, o_ptr,
    sq_ptr, sk_ptr, sv_ptr,
    cu_ptr, idx_ptr, has_init_ptr,
    b_ptr, x_col0, stride_x_row,
    stride_o_group, stride_o_row,
    stride_s_slot, stride_s_tok,
    C,
    KW: tl.constexpr, SL: tl.constexpr, BLOCK_C: tl.constexpr,
    PAD_SLOT: tl.constexpr, NULL_SLOT: tl.constexpr,
    NH: tl.constexpr, BLOCK_H: tl.constexpr, HAS_BETA: tl.constexpr,
):
    i_seq = tl.program_id(0)
    i_c = tl.program_id(1)
    i_g = tl.program_id(2)

    slot = tl.load(idx_ptr + i_seq).to(tl.int64)
    if slot == PAD_SLOT:
        return

    bos = tl.load(cu_ptr + i_seq).to(tl.int64)
    eos = tl.load(cu_ptr + i_seq + 1).to(tl.int64)
    seqlen = eos - bos
    if seqlen <= 0:
        return

    off_c = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = off_c < C

    if HAS_BETA:
        # ``beta`` is a 32-wide column range of the same fused projection, and it
        # only ever needs sigmoid in fp32. Emitting it from one program here costs
        # nothing measurable and saves the two elementwise launches (a bf16->fp32
        # copy and a sigmoid) it would otherwise take.
        if (i_c == 0) & (i_g == 0):
            off_h = tl.arange(0, BLOCK_H)
            mask_h = off_h < NH
            b_in = x_ptr + bos * stride_x_row + x_col0 + 3 * C + off_h
            b_out = b_ptr + bos * NH + off_h
            for _tb in range(0, seqlen):
                b_raw = tl.load(b_in, mask=mask_h, other=0.0).to(tl.float32)
                tl.store(b_out, 1.0 / (1.0 + tl.exp(-b_raw)), mask=mask_h)
                b_in += stride_x_row
                b_out += NH

    if i_g == 0:
        s_ptr = sq_ptr
    elif i_g == 1:
        s_ptr = sk_ptr
    else:
        s_ptr = sv_ptr

    # depthwise taps, oldest first
    w_base = w_ptr + (i_g * C + off_c) * KW
    w0 = tl.load(w_base + 0, mask=mask_c, other=0.0)
    w1 = tl.load(w_base + 1, mask=mask_c, other=0.0)
    w2 = tl.load(w_base + 2, mask=mask_c, other=0.0)
    w3 = tl.load(w_base + 3, mask=mask_c, other=0.0)

    load_init = (tl.load(has_init_ptr + i_seq).to(tl.int32) != 0) & (slot != NULL_SLOT)
    s_base = s_ptr + slot * stride_s_slot + off_c
    if load_init:
        c0 = tl.load(s_base + 0 * stride_s_tok, mask=mask_c, other=0.0).to(tl.float32)
        c1 = tl.load(s_base + 1 * stride_s_tok, mask=mask_c, other=0.0).to(tl.float32)
        c2 = tl.load(s_base + 2 * stride_s_tok, mask=mask_c, other=0.0).to(tl.float32)
    else:
        c0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        c1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
        c2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    x_row = x_ptr + bos * stride_x_row + x_col0 + i_g * C + off_c
    o_row = o_ptr + i_g * stride_o_group + bos * stride_o_row + off_c
    for _t in range(0, seqlen):
        c3 = tl.load(x_row, mask=mask_c, other=0.0).to(tl.float32)
        acc = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3
        acc = acc / (1.0 + tl.exp(-acc))
        tl.store(o_row, acc.to(o_ptr.dtype.element_ty), mask=mask_c)
        c0 = c1
        c1 = c2
        c2 = c3
        x_row += stride_x_row
        o_row += stride_o_row

    # trailing SL inputs become the new conv state (shift-left when seqlen < SL)
    if slot != NULL_SLOT:
        tl.store(s_base + 0 * stride_s_tok, c0.to(s_ptr.dtype.element_ty), mask=mask_c)
        tl.store(s_base + 1 * stride_s_tok, c1.to(s_ptr.dtype.element_ty), mask=mask_c)
        tl.store(s_base + 2 * stride_s_tok, c2.to(s_ptr.dtype.element_ty), mask=mask_c)


def _qkv_conv(x, x_col0, w, state_q, state_k, state_v, cu, idx, has_init,
              num_seqs, num_tokens, C, num_heads=0):
    """One launch for the q/k/v short-convs (plus the beta sigmoid).

    Returns ``([3, num_tokens, C], beta[1, num_tokens, num_heads] | None)``.
    """
    out = torch.empty((3, num_tokens, C), dtype=x.dtype, device=x.device)
    BLOCK_C = 64 if C <= 8192 else 128
    has_beta = 0 < num_heads <= BLOCK_C
    beta = (torch.empty((1, num_tokens, num_heads), dtype=torch.float32,
                        device=x.device) if has_beta else None)
    grid = (num_seqs, triton.cdiv(C, BLOCK_C), 3)
    _qkv_conv_fwd_kernel[grid](
        x, w, out, state_q, state_k, state_v, cu, idx, has_init,
        beta, x_col0, x.stride(0),
        out.stride(0), out.stride(1),
        state_q.stride(0), state_q.stride(1),
        C, KW=4, SL=3, BLOCK_C=BLOCK_C,
        PAD_SLOT=_PAD_SLOT_ID, NULL_SLOT=_NULL_BLOCK_ID,
        NH=num_heads, BLOCK_H=max(triton.next_power_of_2(max(num_heads, 1)), 1),
        HAS_BETA=has_beta,
        num_warps=2, num_stages=2,
    )
    return out, beta



# ---------------------------------------------------------------------------
# Fused low-rank gate projections (f_b and g_b)
#
# Both are ``[T, 128] x [128, 4096]``. K=128 is small enough that cuBLAS picks a
# split-K tactic at decode width, so each costs a GEMM launch *plus* a
# splitKreduce launch -- ~28 us of GPU for 2 MB of weights that should take ~3 us.
# One Triton launch does both (grid dim 0 selects f_b vs g_b) and writes each
# result to its own contiguous plane, which the downstream kernels need anyway:
# ``fused_kda_gate`` hardcodes its row stride as ``H*D``, so a strided view of a
# combined [T, 2*proj] buffer would be read from the wrong addresses.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_proj_kernel(
    x_ptr, w_ptr, o_ptr,
    x_col0, stride_x_row, stride_o_group, stride_o_row,
    T, N,
    R: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr,
):
    i_g, i_n, i_t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    off_t = i_t * BT + tl.arange(0, BT)
    off_n = i_n * BN + tl.arange(0, BN)
    off_r = tl.arange(0, R)
    mask_t = off_t < T
    mask_n = off_n < N

    x = tl.load(
        x_ptr + off_t[:, None] * stride_x_row + (x_col0 + i_g * R) + off_r[None, :],
        mask=mask_t[:, None], other=0.0,
    )
    w = tl.load(
        w_ptr + i_g * (N * R) + off_n[:, None] * R + off_r[None, :],
        mask=mask_n[:, None], other=0.0,
    )
    acc = tl.dot(x, tl.trans(w))
    tl.store(
        o_ptr + i_g * stride_o_group + off_t[:, None] * stride_o_row + off_n[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_n[None, :],
    )


def _gate_proj(x, x_col0, w, num_tokens, N, R):
    """Both low-rank gate projections in one launch -> ``[2, num_tokens, N]``."""
    out = torch.empty((2, num_tokens, N), dtype=x.dtype, device=x.device)
    BT = 16 if num_tokens <= 16 else 64
    BN = 128
    grid = (2, triton.cdiv(N, BN), triton.cdiv(num_tokens, BT))
    _gate_proj_kernel[grid](
        x, w, out, x_col0, x.stride(0), out.stride(0), out.stride(1),
        num_tokens, N, R=R, BT=BT, BN=BN, num_warps=4, num_stages=2,
    )
    return out


class FastKDA(KimiDeltaAttention):
    """KDA with the sync-free state path, fused projections and no scratch copy.

    Subclasses the baseline so the harness's ``isinstance`` state-prep still
    recognises it and every parameter name stays where the loader puts it.
    """

    def process_weights_after_loading(self) -> None:
        super_fn = getattr(super(), "process_weights_after_loading", None)
        if callable(super_fn):
            super_fn()
        if self.qkvb_proj is None:  # quantized path keeps the split GEMMs
            return
        p = self._ps_local
        w_qkvb = self.qkvb_proj.weight.data
        w_fa = self.f_a_proj.weight.data
        w_ga = self.g_a_proj.weight.data
        self._n_qkvb = w_qkvb.shape[0]
        self._n_fa = w_fa.shape[0]
        self._n_ga = w_ga.shape[0]
        self.register_buffer(
            "_w_in",
            torch.cat([w_qkvb, w_fa.to(w_qkvb.dtype), w_ga.to(w_qkvb.dtype)], 0)
            .contiguous(),
            persistent=False,
        )
        # [3*C, KW] fp32 depthwise taps for the fused q/k/v short-conv.
        cw = [m.weight.data for m in (self.q_conv1d, self.k_conv1d, self.v_conv1d)]
        if self.conv_size == 4 and all(w.shape[0] == p for w in cw):
            self.register_buffer(
                "_w_conv",
                torch.cat([w.view(p, self.conv_size).float() for w in cw], 0)
                .contiguous(),
                persistent=False,
            )
        else:
            self._w_conv = None
        # [2, proj, rank] stacked f_b / g_b weights for the fused gate projection.
        w_fb = self.f_b_proj.weight.data
        w_gb = self.g_b_proj.weight.data
        if (w_fb.shape == w_gb.shape and self._n_fa == self._n_ga
                and w_fb.shape[1] == self._n_fa):
            self.register_buffer(
                "_w_gate",
                torch.stack([w_fb, w_gb], 0).contiguous(),
                persistent=False,
            )
            self._gate_rank = self._n_fa
        else:
            self._w_gate = None

    # -- state gather without a data-dependent shape (hence without a sync) ----
    def _prefill_initial_state(self, recurrent_state, indices, has_initial, meta):
        if not getattr(meta, "any_have_initial_state", True):
            # Nothing to carry in: the chunk kernel's ``initial_state=None`` path
            # starts from zero, so neither the gather nor the zero-fill runs.
            return None
        state = recurrent_state[indices]
        if getattr(meta, "all_have_initial_state", False):
            return state.contiguous()
        # "Some but not all": scale by the mask rather than scattering zeros into
        # the slots that must restart -- same values, fixed shape, no sync. The
        # baseline's zero-fill of the stored state is redundant because the final
        # state is written back over it below.
        return state * has_initial.view(-1, *([1] * (state.dim() - 1)))

    def _store_mask(self, num_cols: int, like: torch.Tensor) -> torch.Tensor:
        """``[1, num_cols]`` 0/1 row, 1 only in the first and last column."""
        cache = getattr(self, "_store_masks", None)
        if cache is None:
            cache = self._store_masks = {}
        mask = cache.get(num_cols)
        if mask is None or mask.device != like.device or mask.dtype != like.dtype:
            mask = torch.zeros((1, num_cols), dtype=like.dtype,
                               device=like.device)
            mask[0, 0] = 1
            mask[0, num_cols - 1] = 1
            cache[num_cols] = mask
        return mask

    def _use_recurrent_prefill(self, num_prefill_tokens, meta) -> bool:
        """Is the recurrent scan the cheaper way to run this prefill batch?

        Only for narrow batches (the scan is O(T) sequential, so it loses badly
        once T is large), and only when the carried-in state is uniformly all-zero
        or all-live -- the mixed case would need a masked in-place zero-fill,
        which is exactly the data-dependent work we are avoiding.
        """
        if num_prefill_tokens > _RECURRENT_MAX_TOKENS:
            return False
        if meta.any_have_initial_state and not meta.all_have_initial_state:
            return False
        return not torch.cuda.is_current_stream_capturing()

    def _attn_core(self, q_proj_states, k_proj_states, v_proj_states, raw_g,
                   beta, num_tokens, fused=None, conv_col0=0, raw_beta=None):
        """Return ``[1, num_tokens, H, D]`` core attention output (no scratch)."""
        state_view, meta = self._get_state()
        if state_view is None or meta is None:
            return torch.zeros(
                (1, num_tokens, self.local_num_heads, self.head_dim),
                dtype=q_proj_states.dtype, device=q_proj_states.device,
            )

        num_actual_tokens = meta.num_actual_tokens
        if num_actual_tokens != num_tokens:
            q_proj_states = q_proj_states[:num_actual_tokens]
            k_proj_states = k_proj_states[:num_actual_tokens]
            v_proj_states = v_proj_states[:num_actual_tokens]
            raw_g = raw_g[:, :num_actual_tokens]
            if beta is not None:
                beta = beta[:, :num_actual_tokens]
            if raw_beta is not None:
                raw_beta = raw_beta[:num_actual_tokens]

        num_prefill_tokens = meta.num_prefill_tokens
        num_decode_tokens = meta.num_decode_tokens
        H, D = self.local_num_heads, self.head_dim

        qsl = meta.query_start_loc_int32
        if qsl is None:
            qsl = meta.non_spec_query_start_loc.to(torch.int32)
        state_indices = meta.non_spec_state_indices_tensor

        max_query = meta.max_query_len or num_actual_tokens
        if (fused is not None and getattr(self, "_w_conv", None) is not None
                and num_decode_tokens == 0 and meta.num_prefills > 0
                and max_query <= _FUSED_CONV_MAX_QUERY
                and state_view.q_conv_state.stride(-1) == 1):
            qkv, fused_beta = _qkv_conv(
                fused, conv_col0, self._w_conv,
                state_view.q_conv_state, state_view.k_conv_state,
                state_view.v_conv_state, qsl,
                state_indices[:meta.num_prefills],
                meta.has_initial_state[:meta.num_prefills],
                meta.num_prefills, num_actual_tokens, self._ps_local,
                num_heads=(H if beta is None else 0),
            )
            q = qkv[0].view(1, num_actual_tokens, H, D)
            k = qkv[1].view(1, num_actual_tokens, H, D)
            v = qkv[2].view(1, num_actual_tokens, H, D)
            if fused_beta is not None:
                beta = fused_beta
        else:
            cw = self.q_conv1d.weight
            q_conv_weights = cw.view(cw.size(0), cw.size(2))
            cw = self.k_conv1d.weight
            k_conv_weights = cw.view(cw.size(0), cw.size(2))
            cw = self.v_conv1d.weight
            v_conv_weights = cw.view(cw.size(0), cw.size(2))
            if meta.num_prefills > 0:
                q = self._run_conv_prefill(q_proj_states, state_view.q_conv_state,
                                           q_conv_weights, meta)
                k = self._run_conv_prefill(k_proj_states, state_view.k_conv_state,
                                           k_conv_weights, meta)
                v = self._run_conv_prefill(v_proj_states, state_view.v_conv_state,
                                           v_conv_weights, meta)
            else:
                q = self._run_conv_decode(q_proj_states, state_view.q_conv_state,
                                          q_conv_weights, meta)
                k = self._run_conv_decode(k_proj_states, state_view.k_conv_state,
                                          k_conv_weights, meta)
                v = self._run_conv_decode(v_proj_states, state_view.v_conv_state,
                                          v_conv_weights, meta)
            q = q.view(1, num_actual_tokens, H, D)
            k = k.view(1, num_actual_tokens, H, D)
            v = v.view(1, num_actual_tokens, H, D)
        if beta is None:
            beta = raw_beta.float().sigmoid().unsqueeze(0)

        pf_out = dec_out = None
        if num_prefill_tokens > 0:
            pf_state_indices = state_indices[:meta.num_prefills]
            pf_has_initial = meta.has_initial_state[:meta.num_prefills]
            pf_cu_seqlens = (
                qsl if meta.num_decodes == 0 else qsl[: meta.num_prefills + 1]
            )
            if num_decode_tokens > 0:
                pq, pk = q[:, :num_prefill_tokens], k[:, :num_prefill_tokens]
                pv = v[:, :num_prefill_tokens]
                pg = raw_g[:, :num_prefill_tokens].contiguous()
                pb = beta[:, :num_prefill_tokens].contiguous()
            else:
                pq, pk, pv, pg, pb = q, k, v, raw_g, beta

            if self._use_recurrent_prefill(num_prefill_tokens, meta):
                # Narrow prefill: run the scan in place against the real state
                # buffer (2 launches) instead of the chunked pipeline (~10).
                if not meta.any_have_initial_state:
                    idx_long = meta.state_indices_long
                    idx_long = (pf_state_indices.long() if idx_long is None
                                else idx_long[:meta.num_prefills])
                    state_view.recurrent_state.index_fill_(0, idx_long, 0)
                pf_g = fused_kda_gate(
                    pg.reshape(num_prefill_tokens, H * D),
                    self.A_log, D, g_bias=self.dt_bias,
                ).unsqueeze(0)
                # ``fused_recurrent_kda``'s in-place state store indexes
                # ``ssm_state_indices`` by *token* (a spec-decode feature) and
                # ignores the tok stride, so a 1-D index reads past its end once a
                # sequence is longer than one token -- illegal accesses at T=64.
                # It needs a [P, T] index. Filling every column with the slot
                # works but then the kernel restores the state once *per token*:
                # 64 x 2 MB at T=64, which profiled as 71 us, most of the layer's
                # GPU time. The kernel skips any column whose slot is
                # ``NULL_BLOCK_ID`` (0), and only reads column 0 for the initial
                # state, so zeroing every column but the first and last keeps the
                # initial load and the final state exactly as they were and drops
                # the other T-2 stores.
                pf_idx = pf_state_indices
                if num_prefill_tokens > meta.num_prefills:
                    pf_idx = pf_idx.view(-1, 1) * self._store_mask(max_query,
                                                                   pf_idx)
                pf_out, _ = fused_recurrent_kda(
                    q=pq, k=pk, v=pv, g=pf_g, beta=pb,
                    initial_state=state_view.recurrent_state,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=pf_cu_seqlens,
                    ssm_state_indices=pf_idx,
                )
            else:
                if (torch.cuda.is_current_stream_capturing()
                        and meta.num_decodes == 0):
                    pf_initial_state = state_view.recurrent_state[
                        pf_state_indices].contiguous()
                    pf_initial_state.zero_()
                else:
                    pf_initial_state = self._prefill_initial_state(
                        state_view.recurrent_state, pf_state_indices,
                        pf_has_initial, meta,
                    )
                pf_out, pf_last_state = chunk_kda_with_fused_gate(
                    q=pq, k=pk, v=pv, raw_g=pg, beta=pb,
                    A_log=self.A_log, g_bias=self.dt_bias,
                    initial_state=pf_initial_state, output_final_state=True,
                    use_qk_l2norm_in_kernel=True, cu_seqlens=pf_cu_seqlens,
                )
                state_view.recurrent_state[pf_state_indices] = pf_last_state

        if num_decode_tokens > 0:
            dec_start = num_prefill_tokens
            dec_state_indices = state_indices
            if meta.num_prefills > 0:
                dec_state_indices = dec_state_indices[meta.num_prefills:]
            dec_cu = qsl if meta.num_prefills == 0 else qsl[: meta.num_decodes + 1]
            if dec_start:
                dec_q = q[:, dec_start:].contiguous()
                dec_k = k[:, dec_start:].contiguous()
                dec_v = v[:, dec_start:].contiguous()
                dec_beta = beta[:, dec_start:].contiguous()
                raw_dec_g = raw_g[:, dec_start:]
            else:
                dec_q, dec_k, dec_v, dec_beta = q, k, v, beta
                raw_dec_g = raw_g
            dec_g = fused_kda_gate(
                raw_dec_g.reshape(num_decode_tokens, H * D),
                self.A_log, D, g_bias=self.dt_bias,
            ).unsqueeze(0)
            dec_out, _ = fused_recurrent_kda(
                q=dec_q, k=dec_k, v=dec_v, g=dec_g, beta=dec_beta,
                initial_state=state_view.recurrent_state,
                use_qk_l2norm_in_kernel=True, cu_seqlens=dec_cu,
                ssm_state_indices=dec_state_indices,
            )

        # Pure-prefill / pure-decode batches (every captured case) need no
        # scratch buffer at all: the kernel's output already covers every row.
        if num_actual_tokens == num_tokens:
            if dec_out is None:
                return pf_out
            if pf_out is None:
                return dec_out
            core = torch.empty((1, num_tokens, H, D), dtype=q.dtype,
                               device=q.device)
        else:
            core = torch.zeros((1, num_tokens, H, D), dtype=q.dtype,
                               device=q.device)
        if pf_out is not None:
            core[:, :num_prefill_tokens] = pf_out
        if dec_out is not None:
            core[:, num_prefill_tokens:num_actual_tokens] = dec_out
        return core

    def forward_impl(self, q_proj_states, k_proj_states, v_proj_states, raw_g,
                     beta, core_attn_out) -> None:
        """Baseline-compatible in-place entry point (used by the custom-op path)."""
        out = self._attn_core(q_proj_states, k_proj_states, v_proj_states, raw_g,
                              beta, core_attn_out.shape[1])
        if out.data_ptr() != core_attn_out.data_ptr():
            core_attn_out.copy_(out)

    def forward(self, hidden_states: torch.Tensor, state_manager=None):
        del state_manager
        if self.qkvb_proj is None or self._use_custom_op:
            return super().forward(hidden_states)

        num_tokens = hidden_states.size(0)
        self._ensure_triton_allocator(hidden_states.device)

        H, D = self.local_num_heads, self.head_dim
        p = self._ps_local
        # One GEMM for q/k/v/beta and both low-rank gate stems.
        fused = F.linear(hidden_states, self._w_in)
        q_proj_states = fused[:, :p]
        k_proj_states = fused[:, p:2 * p]
        v_proj_states = fused[:, 2 * p:3 * p]
        nq = self._n_qkvb
        raw_beta = fused[:, 3 * p:nq]
        fa_ga = fused[:, nq:]

        if (getattr(self, "_w_gate", None) is not None
                and num_tokens <= _FUSED_CONV_MAX_QUERY):
            fg = _gate_proj(fused, nq, self._w_gate, num_tokens,
                            self.f_b_proj.weight.shape[0], self._gate_rank)
            raw_g = fg[0].view(1, num_tokens, H, D)
            g2 = fg[1].view(num_tokens, H, D)
        else:
            # Two GEMMs, not one block-diagonal GEMM: a combined [T, 2*proj]
            # output leaves neither half row-contiguous, and ``fused_kda_gate``
            # hardcodes its row stride as ``H*D`` instead of reading it off the
            # tensor -- so the strided half is read from the wrong addresses,
            # silently (the harness's zeroed weights leave this output at ~0).
            raw_g = F.linear(fa_ga[:, :self._n_fa], self.f_b_proj.weight).view(
                1, num_tokens, H, D)
            g2 = F.linear(fa_ga[:, self._n_fa:], self.g_b_proj.weight).view(
                num_tokens, H, D)

        core_attn_out = self._attn_core(q_proj_states, k_proj_states,
                                        v_proj_states, raw_g, None, num_tokens,
                                        fused=fused, conv_col0=0,
                                        raw_beta=raw_beta)
        core_attn_out = self.o_norm(core_attn_out, g2)
        return self.o_proj(core_attn_out.view(num_tokens, H * D))


class FastMoE(KimiMoE):
    """KimiMoE without the per-call FlashInfer Python wrapper, CUDA-graphed."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__(config, quant_config=quant_config)
        self._moe_graphs: dict[int, tuple] = {}
        self._graphs_disabled = False

    def process_weights_after_loading(self) -> None:
        super().process_weights_after_loading()
        if not self.use_trtllm:
            return
        try:
            from flashinfer.fused_moe.core import (
                WeightLayout,
                get_trtllm_moe_sm100_module,
            )
            from flashinfer.utils import device_support_pdl
        except Exception:  # noqa: BLE001 - keep the wrapped path
            return
        ns = get_trtllm_moe_sm100_module()
        raw = None
        for cell in ns.MoERunner.forward.__closure__ or ():
            obj = cell.cell_contents
            if hasattr(obj, "trtllm_bf16_moe"):
                raw = obj
                break
        if raw is None:
            return
        dev = self.w13.device
        self._raw_moe = raw.trtllm_bf16_moe
        self._raw_weight_layout = int(WeightLayout.BlockMajorK)
        self._raw_pdl = bool(device_support_pdl(dev))
        self._raw_empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
        self._raw_empty_f32 = torch.empty(0, dtype=torch.float32, device=dev)
        self._raw_tactic = [-1, -1]

    def _experts(self, hidden_states: torch.Tensor, out: torch.Tensor) -> None:
        """router GEMM -> trtllm-gen experts -> + shared expert, into ``out``.

        Stateless in everything but ``hidden_states``, which is what makes it
        safe to capture.
        """
        # Same tier-2 cuBLAS BF16xBF16->FP32 router GEMM the GateLinear dispatch
        # picks for (hidden=2304, experts=256), without re-deciding every call.
        router_logits = torch.ops.fastkernels.router_gemm_bf16_fp32(
            hidden_states, self.gate.weight,
        )
        self._raw_moe(
            router_logits, self.gate.e_score_correction_bias,
            self._raw_empty_i32, self._raw_empty_f32,
            hidden_states, self.w13, self.w2, None,
            None, None, None, out,
            self.num_experts, self.top_k, self.num_expert_group,
            self.topk_group, self.intermediate_per_tp, 0, self.num_experts,
            self.routed_scaling_factor, 2,  # ROUTING_DEEPSEEK_V3
            True, self._raw_weight_layout, True, self._raw_pdl,
            self._raw_tactic, 3,  # ActivationType.Swiglu
            True, None,
        )
        if self.shared_experts is not None:
            out += self.shared_experts(hidden_states)

    def _graph_for(self, num_tokens: int):
        graph = self._moe_graphs.get(num_tokens)
        if graph is not None:
            return graph
        if (self._graphs_disabled or self.tp_size > 1 or self._use_custom_op
                or num_tokens > _MOE_GRAPH_MAX_TOKENS
                or torch.cuda.is_current_stream_capturing()):
            return None
        dev = self.w13.device
        try:
            static_in = torch.zeros(num_tokens, self.hidden_size,
                                    dtype=torch.bfloat16, device=dev)
            static_out = torch.zeros(num_tokens, self.hidden_size,
                                     dtype=torch.bfloat16, device=dev)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._experts(static_in, static_out)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._experts(static_in, static_out)
        except Exception:  # noqa: BLE001 - any capture problem: use the eager path
            self._graphs_disabled = True
            return None
        graph = (g, static_in, static_out)
        self._moe_graphs[num_tokens] = graph
        return graph

    def forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_raw_moe", None) is None:
            return super().forward_impl(hidden_states)

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]

        graph = self._graph_for(num_tokens)
        if graph is not None:
            g, static_in, static_out = graph
            static_in.copy_(hidden_states)
            g.replay()
            # The graph's own output buffer is handed straight back (vLLM's
            # graphed layers do the same): the only consumer is the next layer's
            # fused-add-RMSNorm, which reads it before this layer replays again.
            return static_out.view(orig_shape)

        out = torch.empty(num_tokens, self.hidden_size,
                          dtype=torch.bfloat16, device=hidden_states.device)
        self._experts(hidden_states, out)
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)


class KimiLinearDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, quant_config: dict | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = config.is_kda_layer(layer_idx)

        if self.is_kda:
            self.self_attn = FastKDA(
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
            self.block_sparse_moe = FastMoE(config, quant_config=quant_config)
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = LlamaMLP(config, quant_config=quant_config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(self, hidden_states, residual, state_manager=None):
        if residual is None:
            # ``input_layernorm`` is out-of-place here, so the incoming buffer can
            # be handed on as the residual directly (as vLLM does) instead of
            # paying a clone.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states, state_manager=state_manager)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
