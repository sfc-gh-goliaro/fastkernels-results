"""Qwen3-Next Gated Delta Net (GDN) linear attention, prefill-fused.

The baseline is already a tuned eager implementation: one joint input-projection
GEMM, a single Triton launch for the projection deinterleave, vLLM's varlen
causal conv, vLLM's fused post-conv prep, and FlashInfer's chunked recurrence.
This candidate keeps that shape and removes work from two directions at once.

**Host issue is the bottleneck at small N.** Measured on this workspace's B200:
at ``N in {1, 26, 60, 445}`` the full forward costs ~400-430 us of host issue
against 60-90 us of device work, and the CUDA-event window the benchmark reports
is 1.04-1.10x that host time -- ``bench.py::_time_module`` records its
per-iteration events *on the stream around each call*, so a starved GPU runs
``end[i]`` as soon as the host enqueues it. Cutting Python-level stages and
launch count is therefore what moves the score at four of the five scored
shapes; cutting kernel time is worth almost nothing there. At ``N=16384`` the
same ratio is 7.6-17.8, so that shape is device-bound and only kernel time
counts. Both figures come from ``tools/tune.py host``.

**The prologue is redundant bandwidth at large N.** At ``N=16384`` the
deinterleave, conv and post-conv prep together move ~1.9 GB to produce 268 MB of
q/k/v, because ``mixed_qkv`` and ``z`` are materialised in between.

So the prefill path here is five stages and six kernels:

===  =========================================================  ==============
 #   stage                                                      kernel
===  =========================================================  ==============
 1   joint in_proj GEMM                                         cuBLAS
 2   deinterleave + causal conv1d + SiLU + conv-state update    Triton (A)
     + L2-norm(q,k) + g/beta gating, in one pass over ``proj``
 3   FlashInfer ``chunk_gated_delta_rule``                      CuTe-DSL
 4   scatter the final state into its slot                      ``index_copy_``
 5   gated RMSNorm reading ``z`` straight out of ``proj``        Triton (B)
 6   out_proj GEMM                                              cuBLAS
===  =========================================================  ==============

Kernel A never materialises ``mixed_qkv`` or ``z``: the deinterleave is a pure
column permutation of the projection output, so the conv reads its taps directly
from ``proj`` at a computed column, and the L2-norm and gating are local to the
same program. Kernel B reads ``z`` at its strided offset in ``proj`` -- flattened
to ``[N*HV, V]`` that view's row stride is *not* uniform (the per-K-head group
stride 768 differs from ``v_per_k*head_v_dim = 512``), which is exactly why the
frozen contiguous-input ``RMSNormGated`` cannot consume it.

Both new kernels sit behind an explicit, cheap host-side predicate. Anything the
predicate rejects -- decode, absent chunk-plan metadata, prefix-caching metadata,
an unexpected layout, dtype, device or head configuration, a missing joint
projection -- falls through to ``super().forward_impl``, i.e. the baseline chain,
unchanged.

This is a subclass of the baseline rather than a copy of it because
``bench.py::_locate_recurrent_attn`` finds the recurrent attention with
``isinstance(sub, baseline.Qwen3NextGDNAttention)``. A structurally identical but
unrelated class makes that lookup fail, ``_prep_kimi_recurrent`` raise
``_UnsupportedInput``, and every case report SKIPPED rather than PASSED.

Numerics follow the baseline's *intermediate precisions* rather than relying on
the benchmark's 1e-2 headroom, because the recurrence compounds error: fp32 conv
accumulator with input-dtype taps in the reference's association, fp32 SiLU, the
result rounded to bf16 and widened back to fp32 before the L2-norm (reproducing
the bf16 store/reload seam between ``causal_conv1d_fn`` and
``_fused_post_conv_kernel``), fp32 L2-norm with ``eps=1e-6``, ``v`` taken as the
bf16 conv output with no fp32 round trip, fp32 gating with softplus threshold
20.0, the final state rounded to bf16 once inside FlashInfer's epilogue instead
of by a following cast, and fp32 gated-norm intermediates with one rounding on
store.

Every increment is independently selectable at import time (see the flag block
below), so each node of the recorded candidate DAG is reproducible from this one
file by naming its configuration, and a regression bisects to a single change.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from ...baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _BaselineGDN,
    _unpack_qkvz_ba,
)
from ....infra.context import get_context
from ..L1.causal_conv1d import causal_conv1d_fn as _vllm_causal_conv1d_fn
from ..L1.gated_delta_rule import (
    fused_post_conv_prep as _vllm_fused_post_conv_prep,
)
from ..L1.rms_norm_gated import RMSNormGated

__targets__ = ["Qwen3NextGDNAttention"]


# Verbatim from vLLM's ``vllm.v1.attention.backends.utils``, and from
# ``L1.gated_delta_rule``'s ``_fused_post_conv_kernel``. The two sentinels are
# compared against *different* quantities (see ``_gdn_prologue_kernel``); the two
# constants pin the gating and the normalisation to the reference's values.
_PAD_SLOT_ID = -1
_NULL_BLOCK_ID = 0
_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0

# ``compute_causal_conv1d_metadata`` builds a chunk plan for exactly one token
# tile (``for block_m in [8]``), so reusing that plan -- which is what makes the
# varlen mapping free -- fixes the tile. It is not a tuning knob.
_BLOCK_T = 8

_SUPPORTED_CONV_WIDTHS = (2, 3, 4)

# The configuration family the fused kernels have actually been validated on:
# bf16 activations, 128-wide K and V heads, and two V heads per K head. Every
# captured case is in this family, and `tools/correctness.py` covers two members
# of it (the captured 16/32 head split and an 8/16 one standing in for TP>1).
# Anything outside it routes to the baseline chain -- widening the family means
# validating the wider geometry first, not relaxing this check.
_SUPPORTED_DTYPES = (torch.bfloat16,)
_SUPPORTED_HEAD_DIM = 128
_SUPPORTED_V_PER_K = 2


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


# --- Increment selection --------------------------------------------------------
# One switch per change the plan lands, so every node of the recorded DAG is
# reproducible from this file by naming its configuration, and a regression
# bisects to one change. Defaults are the configuration chosen on measurement;
# ``docs/notes/increments.md`` carries the per-node, per-shape latencies.
#
# ``GDN_ROOT_ONLY=1`` is the DAG root: the baseline forward, unmodified, reached
# through the subclass the benchmark's prep hook requires.
_ROOT_ONLY = _flag("GDN_ROOT_ONLY", "0")
_G_EXP_IN_KERNEL = _flag("GDN_G_EXP", "1")
_ZERO_STATE_SHORTCUT = _flag("GDN_ZERO_STATE", "1")
_PREALLOC_OUTPUTS = _flag("GDN_PREALLOC", "1")
_BF16_FINAL_STATE = _flag("GDN_BF16_STATE", "1")
_CACHE_BUFFERS = _flag("GDN_CACHE_BUFFERS", "0")
_FUSE_PROLOGUE = _flag("GDN_FUSE_PROLOGUE", "1")
_FUSE_GATE = _flag("GDN_FUSE_GATE", "1")

# --- Kernel A shape selection ---------------------------------------------------
# ``GDN_QK_PAIRED=1`` puts each K head's Q and K halves in one program reading
# ``2*K`` contiguous projection columns, with two segmented L2 reductions -- the
# grid the plan sketches, ``(tot, H + HV)`` instead of ``(tot, 2*H + HV)``.
# ``GDN_HEADS_PER_PROGRAM`` folds that many consecutive program columns into one
# program, unrolled, for instruction-level parallelism. Both are swept by
# ``tools/tune.py sweep``; see ``docs/notes/tuning.md`` for why the defaults win.
_QK_PAIRED = _flag("GDN_QK_PAIRED", "0")
_HEADS_PER_PROGRAM = int(os.environ.get("GDN_HEADS_PER_PROGRAM", "1"))

# --- Fault injection, for the negative tests -----------------------------------
# Each of these breaks exactly one thing the implementation depends on. They exist
# so ``tools/correctness.py::section_negatives`` can show the corresponding
# positive check has teeth: a test that mutates nothing proves nothing. All
# default off and all are ``tl.constexpr`` inside the kernel, so the shipped
# configuration compiles with the dead branch eliminated and is byte-identical to
# a build without them (verified by the sweep numbers being unchanged).
_MUT_LOCAL_TAIL = _flag("GDN_MUT_LOCAL_TAIL", "0")  # chunk tail, not sequence tail
_MUT_NO_SEAM = _flag("GDN_MUT_NO_SEAM", "0")  # skip the bf16 store/reload seam
_MUT_CONTIG_STATE = _flag("GDN_MUT_CONTIG_STATE", "0")  # assume a contiguous state
_MUT_SWAP_SENTINELS = _flag("GDN_MUT_SWAP_SENTINELS", "0")  # confuse the two ids
_MUT_CONV_REASSOC = _flag("GDN_MUT_CONV_REASSOC", "0")  # reassociate the tap sum
_MUT_SHARED_QK_NORM = _flag("GDN_MUT_SHARED_QK_NORM", "0")  # one norm for q and k


# Swept on the B200 with ``tools/tune.py sweep``; see ``docs/notes/tuning.md``.
# One warp per prologue program is worth 84 us of 2760 at N=16384 against two,
# and 360 us against four: the program's unit of work is 128 bf16 channels, so
# 32 lanes give each lane a 64-bit load while 64 lanes give it a 32-bit one and
# 128 lanes leave most of them idle. ``num_stages`` is within noise at every
# combination, the token loop being latency- rather than issue-bound.
_PROLOGUE_WARPS = int(os.environ.get("GDN_PROLOGUE_WARPS", "1"))
_PROLOGUE_STAGES = int(os.environ.get("GDN_PROLOGUE_STAGES", "1"))
_GATE_BLOCK_T = int(os.environ.get("GDN_GATE_BLOCK_T", "16"))
_GATE_WARPS = int(os.environ.get("GDN_GATE_WARPS", "2"))


@triton.jit
def _prologue_head(
    proj_ptr,
    w_ptr,
    cstate_ptr,
    A_log_ptr,
    dt_bias_ptr,
    qk_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    z_ptr,
    # ---- per-program scalars, resolved by the caller ----
    slot,
    chunk_offset,
    seq_start,
    seqlen,
    segment_len,
    token_offset,
    load_init_state,
    num_cache_lines,
    # ---- which logical head this call handles ----
    i_h,
    i_hv,
    i_p,
    mat,
    # ---- strides ----
    stride_proj_tok,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_w_dim,
    stride_w_width,
    stride_qk_mat,
    stride_qk_tok,
    stride_v_tok,
    stride_z_tok,
    # ---- dims / meta ----
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    VP: tl.constexpr,
    GROUP: tl.constexpr,
    QKVZ: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    L2_EPS: tl.constexpr,
    SP_THRESH: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    WRITE_Z: tl.constexpr,
    IS_QK: tl.constexpr,
    PAIRED: tl.constexpr,
    MUT_LOCAL_TAIL: tl.constexpr,
    MUT_NO_SEAM: tl.constexpr,
    MUT_CONTIG_STATE: tl.constexpr,
    MUT_CONV_REASSOC: tl.constexpr,
    MUT_SHARED_QK_NORM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BD: tl.constexpr,
):
    """One logical head of one chunk: ``proj`` -> q/k, or -> v plus its gating.

    Instantiated twice per kernel -- once with the Q/K column width, once with the
    V one -- so each gets a tile sized to the work it actually does rather than
    to the larger of the two.

    ``IS_QK`` selects the Q/K epilogue (L2-norm, store into the ``[2, N, H, K]``
    buffer) from the V one (store ``v``, plus ``g``/``beta`` and optionally
    ``z``). ``PAIRED`` widens a Q/K call to both 128-column halves of one K head,
    normalising each half with its own reduction.
    """
    d = tl.arange(0, BD)

    # The deinterleave, as address arithmetic. ``in_proj_qkvz`` emits one group
    # per K head as ``[q(K) k(K) v(VP*V) z(VP*V)]``; ``conv1d.weight`` and the
    # conv state are indexed by conv channel, so neither needs permuting.
    HK: tl.constexpr = H * K
    if IS_QK:
        if PAIRED:
            # ``2*K`` contiguous projection columns: the low half is Q, the high
            # half K. Both are stored to the same ``[2, N, H, K]`` buffer, so the
            # matrix selector becomes part of the offset rather than a branch.
            src_col = i_h * GROUP + d
            dst_ch = tl.where(d < K, i_h * K + d, HK + i_h * K + (d - K))
            dst_col = tl.where(d < K, i_h * K + d, i_h * K + (d - K))
            qk_mat = tl.where(d < K, 0, 1)
            dim_w = 2 * K
        else:
            src_col = i_h * GROUP + mat * K + d
            dst_ch = mat * HK + i_h * K + d
            dst_col = i_h * K + d
            qk_mat = mat + 0 * d
            dim_w = K
    else:
        src_col = i_h * GROUP + 2 * K + i_p * V + d
        dst_ch = 2 * HK + i_hv * V + d
        dst_col = i_hv * V + d
        qk_mat = 0 * d
        dim_w = V
    mask_d = d < dim_w

    state_len: tl.constexpr = KERNEL_WIDTH - 1
    if MUT_CONTIG_STATE:
        # Address the conv state as if it were a contiguous
        # ``[S, conv_dim, state_len]``, which is what the shape suggests and what
        # the real (transposed) tensor is not.
        cs_dim = state_len
        cs_tok = 1
    else:
        cs_dim = stride_cs_dim
        cs_tok = stride_cs_tok
    cstate_base = cstate_ptr + slot * stride_cs_seq + dst_ch * cs_dim
    x_seq_base = proj_ptr + seq_start * stride_proj_tok + src_col

    # STEP 1: the prior ``state_len`` taps. Held in registers, so the conv-state
    # overwrite in step 2 cannot disturb them. ``col_j`` is the tap for
    # sequence-relative row ``token_offset - state_len + j``, which is
    # ``cstate[..., j]`` when it comes from the cache.
    if chunk_offset == 0:
        if load_init_state:
            col0 = tl.load(cstate_base, mask_d, 0.0)
            if KERNEL_WIDTH >= 3:
                col1 = tl.load(cstate_base + cs_tok, mask_d, 0.0)
            if KERNEL_WIDTH >= 4:
                col2 = tl.load(cstate_base + 2 * cs_tok, mask_d, 0.0)
        else:
            col0 = tl.zeros((BD,), dtype=proj_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BD,), dtype=proj_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BD,), dtype=proj_ptr.dtype.element_ty)

        # STEP 2: the chunk-0 program writes the new conv state, from
        # *pre-activation* projection rows, in conv-channel order. The rows are
        # the last ``state_len`` of the *whole sequence*, not of this chunk.
        idx_tok = tl.arange(0, NP2_STATELEN)
        mask_store = (idx_tok < state_len)[:, None] & mask_d[None, :]
        target = cstate_base[None, :] + (idx_tok * cs_tok)[:, None]
        if state_len <= seqlen:
            if MUT_LOCAL_TAIL:
                # This chunk's last rows instead of the whole sequence's -- wrong
                # for any sequence longer than one chunk, and invisible to the
                # benchmark, which compares only the returned tensor.
                rows = (tl.minimum(BLOCK_T, seqlen) - state_len) + idx_tok
            else:
                rows = (seqlen - state_len) + idx_tok
            mask_x = (
                (rows >= 0)[:, None] & (rows < seqlen)[:, None] & mask_d[None, :]
            )
            loaded = tl.load(
                proj_ptr
                + ((seq_start + rows) * stride_proj_tok)[:, None]
                + src_col[None, :],
                mask_x,
                0.0,
            )
            # The reference needs this barrier and so does this kernel: without
            # it the store can be reordered ahead of loads of the same slot.
            tl.debug_barrier()
            tl.store(target, loaded, mask_store)
        else:
            # Shorter than the state: shift the old state left and join the new
            # rows onto its tail, or lead with zeros when there is no old state.
            # ``N=1`` is a scored shape and takes this branch.
            rows = idx_tok - (state_len - seqlen)
            mask_x = (
                (rows >= 0)[:, None] & (rows < seqlen)[:, None] & mask_d[None, :]
            )
            loaded = tl.load(
                proj_ptr
                + ((seq_start + rows) * stride_proj_tok)[:, None]
                + src_col[None, :],
                mask_x,
                0.0,
            )
            if load_init_state:
                keep = (
                    (slot < num_cache_lines)
                    & ((idx_tok + seqlen) < state_len)[:, None]
                    & mask_d[None, :]
                )
                old = tl.load(
                    cstate_base[None, :]
                    + ((idx_tok + seqlen) * cs_tok)[:, None],
                    keep,
                    other=0.0,
                )
                # ``tl.where`` does not itself order against the loads feeding
                # it; the reference carries the same barrier for the same reason.
                tl.debug_barrier()
                new_state = tl.where(keep, old, loaded)
            else:
                new_state = loaded
            tl.store(target, new_state, mask_store)
    else:
        # Later chunks read their taps from the previous rows of the projection.
        # ``token_offset >= BLOCK_T >= state_len`` here, so these rows are always
        # inside this sequence and never reach into the one packed before it.
        prior = x_seq_base + (token_offset - state_len) * stride_proj_tok
        col0 = tl.load(prior, mask_d, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH >= 3:
            col1 = tl.load(
                prior + stride_proj_tok, mask_d, 0.0, cache_modifier=".ca"
            )
        if KERNEL_WIDTH >= 4:
            col2 = tl.load(
                prior + 2 * stride_proj_tok, mask_d, 0.0, cache_modifier=".ca"
            )

    # STEP 3: roll the depthwise conv over this chunk's tokens. Taps stay in the
    # input dtype and the accumulator is fp32, associated left to right, so each
    # product is rounded exactly where ``_causal_conv1d_fwd_kernel`` rounds it.
    w_base = w_ptr + dst_ch * stride_w_dim
    w_col0 = tl.load(w_base, mask_d, other=0.0)
    w_col1 = tl.load(w_base + stride_w_width, mask_d, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_d, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_d, other=0.0)

    # The token loop carries a dependency chain through the tap window, and an
    # ncu report puts ``long_scoreboard`` at 4.01 warps stalled per issue-active
    # while DRAM sits at 17% of peak -- the loop is short of memory-level
    # parallelism, not of bandwidth, and no ``num_stages`` value made Triton
    # pipeline the load by itself. So issue the next row's load above the current
    # row's arithmetic and keep one load always outstanding. The read past the
    # end of the chunk is masked off; it is issued and discarded.
    x_chunk_base = x_seq_base + token_offset * stride_proj_tok
    xt = tl.load(x_chunk_base, mask=mask_d)
    for idx_token in range(segment_len):
        xt_next = tl.load(
            x_chunk_base + (idx_token + 1) * stride_proj_tok,
            mask=mask_d & (idx_token + 1 < segment_len),
            other=0.0,
        )
        acc = tl.zeros((BD,), dtype=tl.float32)
        if MUT_CONV_REASSOC:
            # Pairwise instead of left-to-right, and in fp32 rather than rounding
            # each product to the input dtype: more accurate, and not what
            # ``_causal_conv1d_fwd_kernel`` computes.
            f0 = col0.to(tl.float32) * w_col0.to(tl.float32)
            f3 = xt.to(tl.float32) * (
                w_col1.to(tl.float32) if KERNEL_WIDTH == 2
                else (w_col2.to(tl.float32) if KERNEL_WIDTH == 3
                      else w_col3.to(tl.float32))
            )
            if KERNEL_WIDTH == 2:
                acc = f0 + f3
            elif KERNEL_WIDTH == 3:
                acc = (f0 + col1.to(tl.float32) * w_col1.to(tl.float32)) + f3
            else:
                acc = (f0 + col1.to(tl.float32) * w_col1.to(tl.float32)) + (
                    col2.to(tl.float32) * w_col2.to(tl.float32) + f3
                )
        elif KERNEL_WIDTH == 2:
            acc += col0 * w_col0
            acc += xt * w_col1
        elif KERNEL_WIDTH == 3:
            acc += col0 * w_col0
            acc += col1 * w_col1
            acc += xt * w_col2
        else:
            acc += col0 * w_col0
            acc += col1 * w_col1
            acc += col2 * w_col2
            acc += xt * w_col3
        acc = acc / (1 + tl.exp(-acc))

        # The baseline stores the conv output as bf16 and reloads it in
        # ``_fused_post_conv_kernel``. Reproduce that seam explicitly: staying in
        # fp32 across it would be *more* accurate but would drift from the
        # reference the benchmark compares against.
        conv_out = acc.to(proj_ptr.dtype.element_ty)
        row = seq_start + token_offset + idx_token

        if IS_QK:
            if MUT_NO_SEAM:
                # Normalise the fp32 SiLU result directly, skipping the bf16
                # store/reload the baseline performs between the conv and the
                # post-conv kernel. Strictly more accurate, and a drift from the
                # reference the benchmark compares against.
                f = acc
            else:
                f = conv_out.to(tl.float32)
            sq = f * f
            if MUT_SHARED_QK_NORM and PAIRED:
                # One norm over both halves: the mistake the paired tile invites,
                # and wrong because q and k are different heads.
                inv = 1.0 / tl.sqrt(tl.sum(sq, axis=0) + L2_EPS)
            elif PAIRED:
                # One reduction per 128-wide half: the halves are different heads
                # and must not share a norm.
                inv_q = 1.0 / tl.sqrt(
                    tl.sum(tl.where(d < K, sq, 0.0), axis=0) + L2_EPS
                )
                inv_k = 1.0 / tl.sqrt(
                    tl.sum(tl.where(d >= K, sq, 0.0), axis=0) + L2_EPS
                )
                inv = tl.where(d < K, inv_q, inv_k)
            else:
                inv = 1.0 / tl.sqrt(tl.sum(sq, axis=0) + L2_EPS)
            tl.store(
                qk_ptr + qk_mat * stride_qk_mat + row * stride_qk_tok + dst_col,
                (f * inv).to(qk_ptr.dtype.element_ty),
                mask=mask_d,
            )
        else:
            tl.store(v_ptr + row * stride_v_tok + dst_col, conv_out, mask=mask_d)

        if KERNEL_WIDTH == 2:
            col0 = xt
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = xt
        else:
            col0 = col1
            col1 = col2
            col2 = xt
        xt = xt_next

    # STEP 4: the V-head calls also own this head's gating scalars and, when the
    # fused output gate is not in use, its slice of ``z``. Both are plain reads of
    # ``proj`` over the token block, so they cost no extra launch.
    if not IS_QK:
        t = tl.arange(0, BLOCK_T)
        mask_t = t < segment_len
        rows = seq_start + token_offset + t
        row_base = proj_ptr + rows * stride_proj_tok

        b_val = tl.load(
            row_base + (QKVZ + i_h * 2 * VP + i_p), mask=mask_t, other=0.0
        ).to(tl.float32)
        a_val = tl.load(
            row_base + (QKVZ + i_h * 2 * VP + VP + i_p), mask=mask_t, other=0.0
        ).to(tl.float32)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        x = a_val + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= SP_THRESH, sp, x)
        g_val = -tl.exp(A_log_val) * sp
        if OUTPUT_G_EXP:
            g_val = tl.exp(g_val)
        tl.store(g_ptr + rows * HV + i_hv, g_val, mask=mask_t)
        tl.store(beta_ptr + rows * HV + i_hv, tl.sigmoid(b_val), mask=mask_t)

        if WRITE_Z:
            m2 = mask_t[:, None] & mask_d[None, :]
            tl.store(
                z_ptr + (rows * stride_z_tok)[:, None] + dst_col[None, :],
                tl.load(
                    proj_ptr
                    + (rows * stride_proj_tok)[:, None]
                    + (i_h * GROUP + 2 * K + VP * V + i_p * V + d)[None, :],
                    m2,
                    0.0,
                ),
                m2,
            )


@triton.jit
def _gdn_prologue_kernel(
    proj_ptr,  # [N, qkvz_dim + 2*HV] joint in_proj output, row-contiguous
    w_ptr,  # [conv_dim, KERNEL_WIDTH] conv taps, indexed by conv channel
    cstate_ptr,  # [num_cache_lines, conv_dim, state_len], NOT contiguous
    cache_idx_ptr,  # [batch] cache slot per sequence
    has_init_ptr,  # [batch] bool
    qsl_ptr,  # [batch + 1] cu_seqlens
    batch_ptr,  # [num_programs] sequence id per program, -1 where unused
    tco_ptr,  # [num_programs] chunk index within that sequence
    A_log_ptr,  # [HV] fp32
    dt_bias_ptr,  # [HV]
    qk_ptr,  # [2, N, H, K] -- q is matrix 0, k is matrix 1
    v_ptr,  # [N, HV, V]
    g_ptr,  # [N, HV] fp32
    beta_ptr,  # [N, HV] fp32
    z_ptr,  # [N, HV, V], unused when WRITE_Z is false
    num_cache_lines,
    stride_proj_tok,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_w_dim,
    stride_w_width,
    stride_qk_mat,
    stride_qk_tok,
    stride_v_tok,
    stride_z_tok,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    VP: tl.constexpr,
    GROUP: tl.constexpr,
    QKVZ: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    PAD_SLOT: tl.constexpr,
    NULL_BLOCK: tl.constexpr,
    L2_EPS: tl.constexpr,
    SP_THRESH: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    WRITE_Z: tl.constexpr,
    PAIRED: tl.constexpr,
    MUT_LOCAL_TAIL: tl.constexpr,
    MUT_NO_SEAM: tl.constexpr,
    MUT_CONTIG_STATE: tl.constexpr,
    MUT_SWAP_SENTINELS: tl.constexpr,
    MUT_CONV_REASSOC: tl.constexpr,
    MUT_SHARED_QK_NORM: tl.constexpr,
    HPP: tl.constexpr,
    N_QK_COLS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BD_QK: tl.constexpr,
    BD_V: tl.constexpr,
):
    """One chunk of one sequence, ``HPP`` logical heads: ``proj`` -> q/k/v/g/beta.

    Grid is ``(nums_dict[8]["tot"], cdiv(N_COLS, HPP))``. ``program_id(0)``
    selects an 8-token chunk of one sequence through ``batch_ptr`` /
    ``token_chunk_offset_ptr`` exactly as ``_causal_conv1d_fwd_kernel`` does,
    which is what makes the "prior taps come from ``proj`` when
    ``chunk_offset > 0`` and from the conv state when it is 0" rule transfer
    unchanged.

    Program column ``c < N_QK_COLS`` is a Q/K unit -- one K head's Q and K halves
    when ``PAIRED``, otherwise one of them -- and ``c >= N_QK_COLS`` is a V head,
    which also owns that head's gating. Every column covers a disjoint set of
    conv channels and their union is the whole ``conv_dim``, so the conv-state
    update is complete with no cross-program communication.
    """
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Two early returns against two different quantities, as in the reference:
    # ``batch_ptr`` carries the sequence id and is filled with ``PAD_SLOT_ID``
    # beyond the live program count, while the *resolved cache slot* is what is
    # compared against the null block.
    idx_seq = tl.load(batch_ptr + pid).to(tl.int64)
    if MUT_SWAP_SENTINELS:
        # Compare each quantity against the *other* sentinel: the resolved cache
        # slot against pad_slot_id and the sequence id against null_block_id.
        # This is the confusion the two constants invite, and it is wrong in both
        # directions -- sequence 0 is a live sequence, and slot -1 never occurs.
        if idx_seq == NULL_BLOCK:
            return
        slot = tl.load(cache_idx_ptr + idx_seq).to(tl.int64)
        if slot == PAD_SLOT:
            return
    else:
        if idx_seq == PAD_SLOT:
            return
        slot = tl.load(cache_idx_ptr + idx_seq).to(tl.int64)
        if slot == NULL_BLOCK:
            return

    chunk_offset = tl.load(tco_ptr + pid)
    seq_start = tl.load(qsl_ptr + idx_seq).to(tl.int64)
    seqlen = tl.load(qsl_ptr + idx_seq + 1).to(tl.int64) - seq_start
    token_offset = BLOCK_T * chunk_offset
    segment_len = tl.minimum(BLOCK_T, seqlen - token_offset)
    load_init_state = tl.load(has_init_ptr + idx_seq).to(tl.int1)

    for i in tl.static_range(HPP):
        col = pid_h * HPP + i
        if col < N_COLS:
            if col < N_QK_COLS:
                if PAIRED:
                    i_h = col
                    mat = 0
                else:
                    i_h = col % H
                    mat = col // H
                _prologue_head(
                    proj_ptr, w_ptr, cstate_ptr, A_log_ptr, dt_bias_ptr,
                    qk_ptr, v_ptr, g_ptr, beta_ptr, z_ptr,
                    slot, chunk_offset, seq_start, seqlen, segment_len,
                    token_offset, load_init_state, num_cache_lines,
                    i_h, 0, 0, mat,
                    stride_proj_tok, stride_cs_seq, stride_cs_dim, stride_cs_tok,
                    stride_w_dim, stride_w_width, stride_qk_mat, stride_qk_tok,
                    stride_v_tok, stride_z_tok,
                    H=H, HV=HV, K=K, V=V, VP=VP, GROUP=GROUP, QKVZ=QKVZ,
                    KERNEL_WIDTH=KERNEL_WIDTH, NP2_STATELEN=NP2_STATELEN,
                    L2_EPS=L2_EPS, SP_THRESH=SP_THRESH,
                    OUTPUT_G_EXP=OUTPUT_G_EXP, WRITE_Z=WRITE_Z,
                    IS_QK=True, PAIRED=PAIRED,
                    MUT_LOCAL_TAIL=MUT_LOCAL_TAIL, MUT_NO_SEAM=MUT_NO_SEAM,
                    MUT_CONTIG_STATE=MUT_CONTIG_STATE,
                    MUT_CONV_REASSOC=MUT_CONV_REASSOC,
                    MUT_SHARED_QK_NORM=MUT_SHARED_QK_NORM,
                    BLOCK_T=BLOCK_T, BD=BD_QK,
                )
            else:
                i_hv = col - N_QK_COLS
                _prologue_head(
                    proj_ptr, w_ptr, cstate_ptr, A_log_ptr, dt_bias_ptr,
                    qk_ptr, v_ptr, g_ptr, beta_ptr, z_ptr,
                    slot, chunk_offset, seq_start, seqlen, segment_len,
                    token_offset, load_init_state, num_cache_lines,
                    i_hv // VP, i_hv, i_hv % VP, 0,
                    stride_proj_tok, stride_cs_seq, stride_cs_dim, stride_cs_tok,
                    stride_w_dim, stride_w_width, stride_qk_mat, stride_qk_tok,
                    stride_v_tok, stride_z_tok,
                    H=H, HV=HV, K=K, V=V, VP=VP, GROUP=GROUP, QKVZ=QKVZ,
                    KERNEL_WIDTH=KERNEL_WIDTH, NP2_STATELEN=NP2_STATELEN,
                    L2_EPS=L2_EPS, SP_THRESH=SP_THRESH,
                    OUTPUT_G_EXP=OUTPUT_G_EXP, WRITE_Z=WRITE_Z,
                    IS_QK=False, PAIRED=PAIRED,
                    MUT_LOCAL_TAIL=MUT_LOCAL_TAIL, MUT_NO_SEAM=MUT_NO_SEAM,
                    MUT_CONTIG_STATE=MUT_CONTIG_STATE,
                    MUT_CONV_REASSOC=MUT_CONV_REASSOC,
                    MUT_SHARED_QK_NORM=MUT_SHARED_QK_NORM,
                    BLOCK_T=BLOCK_T, BD=BD_V,
                )


def _gdn_prologue(
    proj: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    plan: dict,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    v_per_k: int,
    qkvz_dim: int,
    conv_kernel_size: int,
    output_g_exp: bool,
    want_z: bool,
    paired: bool | None = None,
    heads_per_program: int | None = None,
    mutate: dict | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Deinterleave, convolve, normalise and gate the joint projection in one pass.

    Returns ``(q, k, v, g, beta, z)`` in the layouts the recurrence and the output
    gate want. ``z`` is ``None`` when *want_z* is false, in which case the caller
    reads it out of ``proj`` itself.

    ``q`` and ``k`` are two contiguous slices of one ``[2, N, H, K]`` buffer, so a
    single output pointer plus a matrix stride serves both and the kernel needs
    one store site rather than two.
    """
    n = proj.shape[0]
    H, HV = num_k_heads, num_v_heads
    K, V = head_k_dim, head_v_dim
    dev, dt = proj.device, proj.dtype
    paired = _QK_PAIRED if paired is None else paired
    hpp = max(1, _HEADS_PER_PROGRAM if heads_per_program is None else heads_per_program)
    mut = {
        "MUT_LOCAL_TAIL": _MUT_LOCAL_TAIL,
        "MUT_NO_SEAM": _MUT_NO_SEAM,
        "MUT_CONTIG_STATE": _MUT_CONTIG_STATE,
        "MUT_SWAP_SENTINELS": _MUT_SWAP_SENTINELS,
        "MUT_CONV_REASSOC": _MUT_CONV_REASSOC,
        "MUT_SHARED_QK_NORM": _MUT_SHARED_QK_NORM,
    }
    if mutate:
        mut.update(mutate)

    qk = torch.empty(2, n, H, K, dtype=dt, device=dev)
    v = torch.empty(n, HV, V, dtype=dt, device=dev)
    g = torch.empty(n, HV, dtype=torch.float32, device=dev)
    beta = torch.empty(n, HV, dtype=torch.float32, device=dev)
    z = torch.empty(n, HV, V, dtype=dt, device=dev) if want_z else None

    tot = plan["tot"]
    if n == 0 or tot == 0:
        return qk[0], qk[1], v, g, beta, z

    n_qk_cols = H if paired else 2 * H
    n_cols = n_qk_cols + HV

    _gdn_prologue_kernel[(tot, triton.cdiv(n_cols, hpp))](
        proj,
        conv_weight,
        conv_state,
        cache_indices,
        has_initial_state,
        cu_seqlens,
        plan["batch_ptr"],
        plan["token_chunk_offset_ptr"],
        A_log,
        dt_bias,
        qk,
        v,
        g,
        beta,
        z,
        conv_state.shape[0],
        proj.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        qk.stride(0),
        qk.stride(1),
        v.stride(0),
        z.stride(0) if want_z else 0,
        H=H,
        HV=HV,
        K=K,
        V=V,
        VP=v_per_k,
        GROUP=2 * K + 2 * v_per_k * V,
        QKVZ=qkvz_dim,
        KERNEL_WIDTH=conv_kernel_size,
        NP2_STATELEN=triton.next_power_of_2(conv_kernel_size - 1),
        PAD_SLOT=_PAD_SLOT_ID,
        NULL_BLOCK=_NULL_BLOCK_ID,
        L2_EPS=_L2NORM_EPS,
        SP_THRESH=_SOFTPLUS_THRESHOLD,
        OUTPUT_G_EXP=output_g_exp,
        WRITE_Z=want_z,
        PAIRED=paired,
        **mut,
        HPP=hpp,
        N_QK_COLS=n_qk_cols,
        N_COLS=n_cols,
        BLOCK_T=_BLOCK_T,
        BD_QK=triton.next_power_of_2(2 * K if paired else K),
        BD_V=triton.next_power_of_2(V),
        num_warps=_PROLOGUE_WARPS,
        num_stages=_PROLOGUE_STAGES,
    )
    return qk[0], qk[1], v, g, beta, z


@triton.jit
def _gated_rmsnorm_z_kernel(
    o_ptr,  # [N, HV, V] recurrence output, contiguous
    proj_ptr,  # [N, qkvz_dim + 2*HV] joint projection; ``z`` lives here, strided
    w_ptr,  # [V] norm weight
    out_ptr,  # [N, HV*V] what out_proj consumes
    n_tokens,
    stride_o_tok,
    stride_proj_tok,
    stride_out_tok,
    K: tl.constexpr,
    V: tl.constexpr,
    VP: tl.constexpr,
    GROUP: tl.constexpr,
    Z_ROW: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BV: tl.constexpr,
):
    """``RMSNorm(o, weight) * silu(z)`` with ``z`` read in place from ``proj``.

    One program per (token block, V head). Every intermediate is fp32 and the
    result is rounded once on store, matching the reference gated norm; the only
    difference from it is where ``z`` comes from.

    ``Z_ROW`` is the per-K-head-group stride of the ``z`` block inside ``proj``.
    It is a parameter rather than a literal so the wrong-stride negative test can
    supply ``VP*V`` -- the value the layout invites and which is wrong here,
    because the group stride is ``2*K + 2*VP*V``.
    """
    pid_t = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // VP
    i_p = i_hv % VP

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, BV)
    mask_d = d < V
    mask = (t < n_tokens)[:, None] & mask_d[None, :]

    o = tl.load(
        o_ptr + (t * stride_o_tok)[:, None] + (i_hv * V + d)[None, :], mask, 0.0
    ).to(tl.float32)
    z = tl.load(
        proj_ptr
        + (t * stride_proj_tok)[:, None]
        + (i_h * Z_ROW + 2 * K + VP * V + i_p * V + d)[None, :],
        mask,
        0.0,
    ).to(tl.float32)
    w = tl.load(w_ptr + d, mask_d, other=0.0).to(tl.float32)

    ob = tl.where(mask, o, 0.0)
    var = tl.sum(ob * ob, axis=1) / V
    y = o * tl.rsqrt(var + EPS)[:, None] * w[None, :]
    y = y * (z * tl.sigmoid(z))

    tl.store(
        out_ptr + (t * stride_out_tok)[:, None] + (i_hv * V + d)[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask,
    )


def _gated_rmsnorm_from_proj(
    o: torch.Tensor,
    proj: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    v_per_k: int,
    z_row: int | None = None,
) -> torch.Tensor:
    """``[N, HV*V]`` gated norm, reading ``z`` out of ``proj`` in place."""
    n = o.shape[0]
    group = 2 * head_k_dim + 2 * v_per_k * head_v_dim
    out = torch.empty(
        n, num_v_heads * head_v_dim, dtype=o.dtype, device=o.device
    )
    if n == 0:
        return out
    _gated_rmsnorm_z_kernel[(triton.cdiv(n, _GATE_BLOCK_T), num_v_heads)](
        o,
        proj,
        weight,
        out,
        n,
        o.stride(0),
        proj.stride(0),
        out.stride(0),
        K=head_k_dim,
        V=head_v_dim,
        VP=v_per_k,
        GROUP=group,
        Z_ROW=group if z_row is None else z_row,
        EPS=eps,
        BLOCK_T=_GATE_BLOCK_T,
        BV=triton.next_power_of_2(head_v_dim),
        num_warps=_GATE_WARPS,
        num_stages=2,
    )
    return out


class Qwen3NextGDNAttention(_BaselineGDN):
    """Gated Delta Net linear attention for Qwen3-Next, prefill-fused."""

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        layer_idx: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__(
            hidden_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            layer_idx,
            conv_kernel_size=conv_kernel_size,
            rms_norm_eps=rms_norm_eps,
            reduce_output=reduce_output,
        )
        # The baseline's ``__init__`` builds its own L1 gated norm; rebind so the
        # frozen L1 winner serves the fallback path. The parameter name is
        # unchanged, so weight loading and weight sharing are unaffected.
        self.norm = RMSNormGated(
            head_v_dim,
            eps=rms_norm_eps,
            norm_before_gate=True,
            activation="swish",
        )
        self._rms_norm_eps = rms_norm_eps
        self._conv_dim = (
            2 * self.local_k_heads * self.head_k_dim
            + self.local_v_heads * self.head_v_dim
        )
        # Whether the bf16 final state is available is a property of the device
        # the forward actually runs on, so it is decided in ``forward_impl`` from
        # ``proj.device`` rather than here: ``__init__`` runs before the module is
        # moved, and ``get_device_capability()`` with no argument answers for the
        # *default* device, which in a multi-GPU process need not be the one this
        # layer ends up on.
        self._bf16_state_by_device: dict[torch.device, bool] = {}
        # Scratch reuse for the two internal recurrence buffers, **off by
        # default**. It measures 5-12 us faster of ~190 at N=1 and N=60, but the
        # plan's DEC-2 ("may internal scratch buffers assume single-stream,
        # non-reentrant execution?") is still PENDING, and enabling it by default
        # would decide that question by narrowing the operator's contract. The
        # baseline allocates per call and so does this, unless a caller opts in
        # with ``GDN_CACHE_BUFFERS=1``.
        #
        # When it *is* enabled the key includes the current CUDA stream, so two
        # concurrent forwards of the same layer on different streams get
        # different buffers rather than racing on one -- which the first version
        # of this cache did. Only internal intermediates are ever cached
        # (``o_buf`` is consumed by the output gate, ``st_buf`` by
        # ``index_copy_``), and the returned tensor is always freshly produced by
        # ``out_proj``, so nothing the caller keeps aliases the cache. Re-entrant
        # calls on the *same* stream would still share a buffer; that is the part
        # DEC-2 has to answer before this can become the default.
        self._scratch: dict[tuple, torch.Tensor] = {}

    def _buffer(self, key, shape, dtype, device) -> torch.Tensor:
        """A scratch buffer for one internal intermediate.

        Per-call ``torch.empty`` unless caching is opted into. The cache key
        carries the current CUDA stream, because two forwards of this layer
        enqueued on different streams may be in flight at once and must not be
        handed the same buffer.
        """
        if not _CACHE_BUFFERS:
            return torch.empty(shape, dtype=dtype, device=device)
        stream = torch.cuda.current_stream(device).cuda_stream
        full = (key, shape, dtype, device, stream)
        buf = self._scratch.get(full)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._scratch[full] = buf
        return buf

    def _bf16_state_ok(self, device: torch.device) -> bool:
        """Does FlashInfer accept a bf16 final state on *this* device?

        Only the SM100 path does. Cached per device rather than per module,
        because the answer belongs to the device and the query is not free.
        """
        if not _BF16_FINAL_STATE:
            return False
        cached = self._bf16_state_by_device.get(device)
        if cached is None:
            cached = (
                device.type == "cuda"
                and torch.cuda.get_device_capability(device)[0] == 10
            )
            self._bf16_state_by_device[device] = cached
        return cached

    def _prologue_eligible(
        self, md, proj: torch.Tensor, conv_state, cu_seqlens=None
    ) -> bool:
        """Every condition the fused prologue needs, as cheap host-side checks.

        Anything false here routes the call to the baseline chain, which is always
        correct -- so a new metadata shape, dtype, device or layout is slower,
        never wrong. Each condition is covered independently by
        ``tools/correctness.py::section_fallback``.
        """
        if not _FUSE_PROLOGUE:
            return False
        if self.conv_kernel_size not in _SUPPORTED_CONV_WIDTHS:
            return False
        if _HEADS_PER_PROGRAM < 1:
            return False
        # The reference conv reads ``has_initial_state`` unconditionally in its
        # chunk-0 branch, so an absent mask is not a supported input there either.
        if md.has_initial_state is None:
            return False
        plan = md.nums_dict
        if not isinstance(plan, dict) or _BLOCK_T not in plan:
            return False
        entry = plan[_BLOCK_T]
        if not isinstance(entry, dict):
            return False
        batch_ptr = entry.get("batch_ptr")
        tco_ptr = entry.get("token_chunk_offset_ptr")
        if batch_ptr is None or tco_ptr is None:
            return False
        # The plan is published twice -- on the metadata and inside the
        # per-tile entry -- and the kernel reads the entry's copy. A producer that
        # fills only one, or lets the two disagree, is not a shape this kernel has
        # been validated against: silently preferring one copy would make the
        # grid and the chunk map come from different plans.
        if md.batch_ptr is None or md.token_chunk_offset_ptr is None:
            return False
        if (
            md.batch_ptr.data_ptr() != batch_ptr.data_ptr()
            or md.token_chunk_offset_ptr.data_ptr() != tco_ptr.data_ptr()
        ):
            return False
        # The prefix-caching branch of the reference conv writes extra cache
        # blocks at chunk boundaries; it is deliberately out of scope.
        for name in (
            "block_idx_first_scheduled_token",
            "block_idx_last_scheduled_token",
            "initial_state_idx",
            "num_computed_tokens",
        ):
            if getattr(md, name, None) is not None:
                return False

        idx = md.non_spec_state_indices_tensor
        if idx is None or idx.dim() != 1 or idx.stride(0) != 1:
            return False
        if cu_seqlens is None or cu_seqlens.dim() != 1:
            return False

        # Head geometry, pinned to the validated family rather than to whatever
        # happens to be self-consistent. ``head_k_dim == head_v_dim`` is what lets
        # one conv-channel map serve q/k and v; the 128 and the ``v_per_k == 2``
        # are what the tile widths and the paired variant were measured and
        # validated against.
        if self.local_k_heads * self.v_per_k != self.local_v_heads:
            return False
        if self.head_k_dim != self.head_v_dim:
            return False
        if self.head_k_dim != _SUPPORTED_HEAD_DIM:
            return False
        if self.v_per_k != _SUPPORTED_V_PER_K:
            return False
        if self.local_k_heads <= 0:
            return False
        if self.A_log.numel() != self.local_v_heads:
            return False
        if self.dt_bias.numel() != self.local_v_heads:
            return False
        if not self.A_log.is_contiguous() or not self.dt_bias.is_contiguous():
            return False

        # Layouts and dtype. The bf16 store/reload seam the kernel reproduces is
        # specific to a 2-byte activation dtype, so the dtype is part of the
        # validated family and not merely required to be self-consistent.
        if proj.dtype not in _SUPPORTED_DTYPES:
            return False
        # The kernel indexes ``proj`` by row stride and unit column stride, and
        # reads the conv state's three strides off the tensor.
        if proj.dim() != 2 or proj.stride(1) != 1:
            return False
        if proj.shape[1] != self._qkvz_dim + 2 * self.local_v_heads:
            return False
        w = self.conv1d.weight
        if w.dim() != 2 or w.stride(1) != 1:
            return False
        if tuple(w.shape) != (self._conv_dim, self.conv_kernel_size):
            return False
        if w.dtype is not proj.dtype:
            return False
        if (
            conv_state.dim() != 3
            or conv_state.shape[1] != self._conv_dim
            or conv_state.shape[2] < self.conv_kernel_size - 1
            or conv_state.dtype is not proj.dtype
        ):
            return False

        # Everything the kernel dereferences must live on the projection's
        # device; a mismatch would otherwise fault at launch instead of falling
        # back.
        dev = proj.device
        for t in (
            batch_ptr,
            tco_ptr,
            md.batch_ptr,
            md.token_chunk_offset_ptr,
            idx,
            md.has_initial_state,
            cu_seqlens,
            conv_state,
            w,
            self.A_log,
            self.dt_bias,
        ):
            if t.device != dev:
                return False
        return True

    def forward_impl(
        self, hidden_states: torch.Tensor, state_manager=None
    ) -> torch.Tensor:
        if _ROOT_ONLY:
            return super().forward_impl(hidden_states, state_manager)

        md = get_context().kda_metadata
        if state_manager is None:
            state_manager = get_context().kda_state
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextGDNAttention requires engine-managed recurrent state "
                "and metadata",
            )
        # Decode, a missing joint projection, the non-FlashInfer recurrence, and
        # metadata without an initial-state mask are all carried by the baseline
        # chain rather than reimplemented: none is on the scored path, and the
        # baseline is already correct on each.
        if (
            md.num_prefills <= 0
            or self._in_proj_w is None
            or not self._use_flashinfer_prefill
            or md.has_initial_state is None
        ):
            return super().forward_impl(hidden_states, state_manager)

        from flashinfer.gdn_prefill import (
            chunk_gated_delta_rule as _fi_chunk_gated_delta_rule,
        )

        self._ensure_triton_allocator(hidden_states.device)

        x_flat = hidden_states.reshape(-1, self.hidden_size)
        n_tokens = x_flat.shape[0]
        proj = torch.nn.functional.linear(x_flat, self._in_proj_w)

        cu_seqlens = md.query_start_loc_int32
        if cu_seqlens is None:
            cu_seqlens = md.non_spec_query_start_loc.to(torch.int32)
        conv_state = state_manager.gdn_conv[self.layer_idx]
        recurrent_full = state_manager.recurrent[self.layer_idx]

        fuse_gate = _FUSE_GATE
        if self._prologue_eligible(md, proj, conv_state, cu_seqlens):
            q_c, k_c, v_c, g, beta, z = _gdn_prologue(
                proj,
                self.conv1d.weight,
                conv_state,
                cu_seqlens,
                md.non_spec_state_indices_tensor,
                md.has_initial_state,
                md.nums_dict[_BLOCK_T],
                self.A_log,
                self.dt_bias,
                num_k_heads=self.local_k_heads,
                num_v_heads=self.local_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                v_per_k=self.v_per_k,
                qkvz_dim=self._qkvz_dim,
                conv_kernel_size=self.conv_kernel_size,
                output_g_exp=_G_EXP_IN_KERNEL,
                want_z=not fuse_gate,
            )
        else:
            mixed_qkv, z, b, a = _unpack_qkvz_ba(
                proj[:, : self._qkvz_dim],
                proj[:, self._qkvz_dim :],
                self.local_k_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.v_per_k,
            )
            mixed_qkv = _vllm_causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                self.conv1d.weight,
                None,
                conv_state,
                cu_seqlens,
                cache_indices=md.non_spec_state_indices_tensor,
                has_initial_state=md.has_initial_state,
                activation="silu",
                metadata=md,
                validate_data=True,
            ).transpose(0, 1)
            q_c, k_c, v_c, g, beta = _vllm_fused_post_conv_prep(
                conv_output=mixed_qkv,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.local_k_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                # ``exp(g)`` in the same fp32 registers that produced ``g``,
                # rather than a separate elementwise launch at the call site.
                output_g_exp=_G_EXP_IN_KERNEL,
            )
            fuse_gate = False
        if not _G_EXP_IN_KERNEL:
            g = torch.exp(g)

        state_idx = md.state_indices_long
        if state_idx is None:
            state_idx = md.non_spec_state_indices_tensor.long()

        # Which slots carry state is decided host-side when the chunk plan is
        # built, so the mask never comes back from the device. When *no* slot
        # does, the recurrence starts from its own zero state and the gather, the
        # ``fill_`` and the bf16->fp32 copy all disappear. ``has_initial_state is
        # None`` is *not* that case -- the baseline uses the gathered state
        # unchanged there -- which is why the shortcut tests it explicitly.
        if (
            _ZERO_STATE_SHORTCUT
            and md.has_initial_state is not None
            and not md.any_have_initial_state
        ):
            init_state = None
        else:
            init_state = recurrent_full.index_select(0, state_idx)
            if md.has_initial_state is None or md.all_have_initial_state:
                pass
            elif md.any_have_initial_state:
                keep = md.has_initial_state.view(
                    -1, *([1] * (init_state.dim() - 1)),
                )
                init_state.masked_fill_(~keep, 0)
            else:
                init_state.zero_()
            init_state = init_state.to(torch.float32)

        o_buf = st_buf = None
        if _PREALLOC_OUTPUTS:
            o_buf = self._buffer(
                "o",
                (n_tokens, self.local_v_heads, self.head_v_dim),
                proj.dtype,
                proj.device,
            )
            # FlashInfer's SM100 kernel compiles *one* state dtype and uses it for
            # both the initial and the final state, taking it from
            # ``initial_state`` when that is given -- so the bf16 final state is
            # available exactly on the zero-initial-state path, which is the one
            # the benchmark scores.
            st_dtype = torch.float32
            if init_state is not None:
                st_dtype = init_state.dtype
            elif self._bf16_state_ok(proj.device):
                st_dtype = recurrent_full.dtype
            st_buf = self._buffer(
                "state",
                (
                    cu_seqlens.numel() - 1,
                    self.local_v_heads,
                    self.head_k_dim,
                    self.head_v_dim,
                ),
                st_dtype,
                proj.device,
            )
        o, final_state = _fi_chunk_gated_delta_rule(
            q=q_c,
            k=k_c,
            v=v_c,
            g=g,
            beta=beta,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            output=o_buf,
            output_state=st_buf,
        )
        recurrent_full.index_copy_(
            0,
            state_idx,
            final_state
            if final_state.dtype is recurrent_full.dtype
            else final_state.to(recurrent_full.dtype),
        )

        if fuse_gate:
            gated = _gated_rmsnorm_from_proj(
                o,
                proj,
                self.norm.weight,
                self._rms_norm_eps,
                num_v_heads=self.local_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                v_per_k=self.v_per_k,
            )
        else:
            gated = self.norm(
                o.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim),
            ).view(n_tokens, self.local_v_heads * self.head_v_dim)

        return self.out_proj(gated)
