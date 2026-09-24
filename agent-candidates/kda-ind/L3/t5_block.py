"""Fused T5 encoder block for B200 / sm_100.

The block is ~202 GFLOP of GEMM against ~386 MB of weights, so the four
projections are the floor and cuBLAS keeps them.  What this file removes is the
~340 us of elementwise and memory-movement traffic around them: the two
``T5LayerNorm``s, the attention epilogue (bias add, fp32 softmax, output
shuffle), the gated ``gelu_new``, and the two residual adds.

Numerics are the binding constraint, not speed.  ``T5SelfAttention`` rounds its
logits to bfloat16 **twice** -- once as the bf16 output of ``matmul(q, k^T)``,
and again through the in-place bf16 ``scores += position_bias``.  With the
benchmark's random weights the logits have std ~13, so that rounding decides
which key wins the softmax; every stock attention backend (which keeps the
logits in fp32) reproduces only ~30% of the block's outputs inside tolerance.
The attention kernel below therefore reproduces both roundings explicitly, and
each fused stage mirrors its eager counterpart's rounding chain step by step
rather than computing the whole expression in fp32.

Layout note: the captured ``position_bias`` is ``[q, k, h]``-contiguous -- head
is the innermost dimension, stride 1 -- so a ``[BLOCK_M, BLOCK_N]`` tile for one
head touches elements 128 B apart and 16 consecutive heads share every 32 B
sector.  The attention kernel reads the bias through runtime strides, so one
kernel serves both the bias in place and a head-major copy.  Reading in place
measured 185 us against 47 us head-major (whole block 562 us against 253 us), so
``_bias_to_head_major`` transposes first and ``_BIAS_HEAD_MAJOR`` records that
choice.  A bias that is *already* head-major skips the transpose rather than
undoing it -- that is what a preceding block in a real encoder stack would hand
over.  Case A does **not** take that shortcut: its returned leaf reproduces the
baseline's observable ``[q, k, h]`` stride, so Case A pays the relayout too.
``_CASE_A_HEAD_MAJOR_LEAF`` flips that, at the cost of changing an observable
property of the operator's output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import T5Config
from triton.language.extra import libdevice

from fastkernels.infra.tp import _tp_size
from fastkernels.tasks.baseline.L3.t5_block import (
    T5Block as _BaselineT5Block,
    T5LayerFF as _BaselineT5LayerFF,
    T5LayerSelfAttention as _BaselineT5LayerSelfAttention,
)


__targets__ = ["T5Block"]

# The one configuration the Triton kernels are written for.  The guards below
# require it exactly, because the attention and relative-position-bias kernels
# index without edge masks and the norm kernel builds a full-width `tl.arange`:
# a mismatch would read and write out of bounds rather than run slowly.
_FAST_SEQ = 512
_FAST_D_MODEL = 4096
_FAST_D_KV = 64
_FAST_N_HEADS = 64
_FAST_D_FF = 10240

# sqrt(2/pi), the constant inside HuggingFace's NewGELUActivation.
_GELU_COEF = tl.constexpr(0.7978845608028654)

# Tile parameters, chosen by measurement on the assembled block (see
# benchmark.csv).  Kept as module constants so the sweep in profile/ can call the
# launch wrappers with overrides without touching the shipped defaults.
_ATTN_BLOCK_M = 64
_ATTN_BLOCK_N = 64
_ATTN_NUM_WARPS = 4
_ATTN_NUM_STAGES = 2
_BIAS_HEAD_MAJOR = True
_RELATIVE_BIAS_FUSED = True
_CASE_A_HEAD_MAJOR_LEAF = False
_CACHE_LOGITS = False
_RELAYOUT_BLOCK_T = 128
_NORM_NUM_WARPS = 8
_ACT_BLOCK = 1024
_ACT_NUM_WARPS = 4

# Per-fusion switches. All on. They exist so the three low-risk fusions can be
# measured one at a time on the assembled block -- the plan gates each of them on
# an integrated whole-block measurement, not on a stage delta, and stage timings
# are not additive (twelve isolated stages sum to 707 us against a 565 us block).
# profile/dev_bench.py staged toggles them; nothing reads them at runtime.
_FUSE_NORM = True
_FUSE_ACT = True
_FUSE_ADDMM = True
# _FUSE_ATTENTION=False keeps the norm and addmm switches independently testable
# by computing the attention context the eager way (cuBLAS QK^T, in-place bf16
# bias add, fp32 softmax, cuBLAS PV, transpose) and skipping the bias relayout, so
# a staged run can start from genuine full-eager behaviour and add one fusion at a
# time. It is not a shipping path.
_FUSE_ATTENTION = True


# ---------------------------------------------------------------------------
# T5LayerNorm: fp32 variance, then two bf16 roundings.
#
#   variance = mean(x.float() ** 2)            (fp32)
#   y        = round_bf16(x.float() * rsqrt(variance + eps))
#   out      = round_bf16(w * y)
#
# The intermediate rounding of ``y`` is the ``hidden_states.to(weight.dtype)``
# branch in the eager module.  It is a numerical no-op for the benchmark, whose
# ``weight`` is all ones, but it is part of the contract.
# ---------------------------------------------------------------------------
@triton.jit
def _rms_norm_kernel(X, W, Y, stride_x, stride_y, eps, N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, N)
    x = tl.load(X + row * stride_x + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    y = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    w = tl.load(W + cols)
    out = (w.to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out)


def _rms_norm(x2d: torch.Tensor, weight: torch.Tensor, eps: float,
              num_warps: int = _NORM_NUM_WARPS) -> torch.Tensor:
    rows, n = x2d.shape
    out = torch.empty_like(x2d)
    _rms_norm_kernel[(rows,)](
        x2d, weight, out, x2d.stride(0), out.stride(0), eps, N=n, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Gated activation: chunk(2) + gelu_new(gate) * up in one pass.
#
# NewGELUActivation runs on a bf16 tensor, so every intermediate is rounded to
# bf16.  Computing the polynomial in fp32 and rounding once is a *different*
# chain, and the deviation it introduces is amplified by the 10240-term ``wo``
# reduction that follows, so the eager chain is reproduced step by step.  Each
# ``.to(tl.bfloat16)`` below is one eager op boundary.
# ---------------------------------------------------------------------------
@triton.jit
def _gated_gelu_kernel(GateUp, Out, stride_gu, stride_out, D: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    offs = col * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    base = GateUp + row * stride_gu + offs
    g = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(base + D, mask=mask, other=0.0).to(tl.float32)

    # ``torch.pow(x, 3.0)`` on a bf16 tensor rounds after *each* multiply -- it is
    # round(round(x*x) * x), not one fp32 expression rounded once.  Collapsing it
    # into fp32 reproduces only 78% of the eager values bit for bit.
    square = (g * g).to(tl.bfloat16).to(tl.float32)
    cube = (square * g).to(tl.bfloat16).to(tl.float32)             # pow(x, 3.0)
    inner = (0.044715 * cube).to(tl.bfloat16).to(tl.float32)       # 0.044715 * cube
    inner = (g + inner).to(tl.bfloat16).to(tl.float32)             # x + ...
    inner = (_GELU_COEF * inner).to(tl.bfloat16).to(tl.float32)    # sqrt(2/pi) * ...
    tanh = libdevice.tanh(inner).to(tl.bfloat16).to(tl.float32)    # tanh(...)
    tanh = (1.0 + tanh).to(tl.bfloat16).to(tl.float32)             # 1.0 + tanh
    half = (0.5 * g).to(tl.bfloat16).to(tl.float32)                # 0.5 * x
    act = (half * tanh).to(tl.bfloat16).to(tl.float32)             # (0.5x) * (1+tanh)

    tl.store(Out + row * stride_out + offs, (act * u).to(tl.bfloat16), mask=mask)


def _gated_gelu(gate_up: torch.Tensor, block: int = _ACT_BLOCK,
                num_warps: int = _ACT_NUM_WARPS) -> torch.Tensor:
    rows, two_d = gate_up.shape
    d = two_d // 2
    out = torch.empty((rows, d), dtype=gate_up.dtype, device=gate_up.device)
    _gated_gelu_kernel[(rows, triton.cdiv(d, block))](
        gate_up, out, gate_up.stride(0), out.stride(0),
        D=d, BLOCK=block, num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Bias relayout: [q, k, h]-contiguous -> head-major [h, q, k].
#
# Treated as a plain 2D transpose of [S*S, H].  Reads whole 128 B source rows and
# writes BLOCK_T-long contiguous runs, so both sides are coalesced; the
# alternative -- letting the attention kernel read the bias in place -- costs a
# 16x sector amplification because 16 consecutive heads share one 32 B sector.
# ---------------------------------------------------------------------------
@triton.jit
def _bias_relayout_kernel(Src, Dst, n_rows, H: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, H)
    keep = offs_t < n_rows
    tile = tl.load(Src + offs_t[:, None] * H + offs_h[None, :], mask=keep[:, None], other=0.0)
    tl.store(Dst + offs_h[:, None] * n_rows + offs_t[None, :], tl.trans(tile),
             mask=keep[None, :])


def _bias_to_head_major(bias: torch.Tensor, block_t: int = _RELAYOUT_BLOCK_T) -> torch.Tensor:
    """``bias`` is ``[1, H, S, S]`` with ``[q, k, h]``-contiguous storage."""
    _, h, s, _ = bias.shape
    n_rows = s * s
    out = torch.empty((h, s, s), dtype=bias.dtype, device=bias.device)
    _bias_relayout_kernel[(triton.cdiv(n_rows, block_t),)](
        bias, out, n_rows, H=h, BLOCK_T=block_t,
    )
    return out


# ---------------------------------------------------------------------------
# Relative-position bias, produced head-major in one pass.
#
# The eager ``compute_bias`` costs 332 us -- more than half of Case A -- because
# ``_relative_position_bucket`` runs a dozen elementwise ops over a 512x512 int64
# grid and the embedding gather then writes 33.6 MB in the wrong layout for the
# attention kernel.  The bucket index depends only on ``key - query``, so it is
# recomputed here per call from that difference and the embedding row is gathered
# straight into head-major order.  Nothing is cached across calls.
#
# The arithmetic mirrors the eager version op for op.  ``profile/dev_bench.py casea``
# is the bitwise gate: it compares this kernel's output against the eager
# ``compute_bias`` and does so for both spellings of the log-branch scalar
# division, since ATen turns a tensor-by-python-scalar divide on CUDA into a
# multiply by the reciprocal.  Both spellings measure bitwise equal here.
# ``_RELATIVE_BIAS_FUSED`` switches the path back to ``compute_bias`` if that ever
# stops holding.
# ---------------------------------------------------------------------------
@triton.jit
def _relative_bias_kernel(Emb, Out, stride_h, stride_q, stride_k,
                          H: tl.constexpr,
                          HALF: tl.constexpr, MAX_EXACT: tl.constexpr,
                          INV_LOG_RATIO, RECIPROCAL_DIV: tl.constexpr,
                          BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr):
    head = tl.program_id(0)
    offs_q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)

    relative = offs_k[None, :] - offs_q[:, None]
    bucket = tl.where(relative > 0, HALF, 0)
    distance = tl.abs(relative)

    # log(distance / MAX_EXACT) / log(MAX_DISTANCE / MAX_EXACT) * (HALF - MAX_EXACT)
    ratio = libdevice.log(tl.maximum(distance, 1).to(tl.float32) / MAX_EXACT)
    if RECIPROCAL_DIV:
        scaled = ratio * INV_LOG_RATIO
    else:
        scaled = ratio / (1.0 / INV_LOG_RATIO)
    far = MAX_EXACT + (scaled * (HALF - MAX_EXACT)).to(tl.int32)
    far = tl.minimum(far, HALF - 1)

    bucket += tl.where(distance < MAX_EXACT, distance, far)
    value = tl.load(Emb + bucket * H + head)
    tl.store(Out + head * stride_h + offs_q[:, None] * stride_q
             + offs_k[None, :] * stride_k, value)


def _relative_bias(emb_weight: torch.Tensor, seq: int, n_heads: int,
                   num_buckets: int, max_distance: int,
                   block_q: int = 16, block_k: int = 128,
                   head_major: bool = True) -> torch.Tensor:
    """``[1, n_heads, seq, seq]`` with ``compute_bias``'s values.

    ``head_major`` picks the storage: head-major contiguous, which is what the
    attention kernel wants, or the baseline's ``[q, k, h]`` storage with stride
    ``(H, 1, S*H, H)``, which is what ``compute_bias`` returns and therefore what
    the operator's caller observes.  Only the strides differ; the same kernel
    writes both, addressed through the destination strides.
    """
    half = num_buckets // 2
    max_exact = half // 2
    if head_major:
        out = torch.empty((1, n_heads, seq, seq), dtype=emb_weight.dtype,
                          device=emb_weight.device)
    else:
        out = torch.empty_strided((1, n_heads, seq, seq),
                                  (n_heads, 1, seq * n_heads, n_heads),
                                  dtype=emb_weight.dtype, device=emb_weight.device)
    _, stride_h, stride_q, stride_k = out.stride()
    _relative_bias_kernel[(n_heads, triton.cdiv(seq, block_q), triton.cdiv(seq, block_k))](
        emb_weight, out, stride_h, stride_q, stride_k,
        H=n_heads, HALF=half, MAX_EXACT=max_exact,
        INV_LOG_RATIO=1.0 / math.log(max_distance / max_exact),
        RECIPROCAL_DIV=True, BLOCK_Q=block_q, BLOCK_K=block_k,
    )
    return out


# ---------------------------------------------------------------------------
# Fused attention: one CTA per (head, query block), d = 64 in a single tile.
#
# There is no 1/sqrt(d) scale -- T5 folds it into initialization.  The two
# mandatory bf16 roundings are the two ``.to(tl.bfloat16)`` calls in
# ``_block_logits``.  The output is stored directly at [token, head * 64 + d],
# which is the layout the ``o`` projection wants, so the eager path's transpose +
# contiguous copy disappears.
#
# The loop runs **twice** rather than once, and that is a correctness requirement
# rather than an oversight.  A single-pass flash form rounds *unnormalized*
# exponentials to bf16 and divides at the end; the eager path rounds
# *normalized* probabilities, so it needs the row sum before the rounding.  The
# two disagree on ~31% of context elements by one bf16 ulp, and the block output
# turns out to be brutally sensitive to that: measured on the assembled block,
# a context that is 99.5% bit-exact still yields only a 0.961 match, against a
# 0.99 requirement.  So the first pass establishes the row max and row sum, and
# the second recomputes the logits to form normalized probabilities.
#
# The second pass does not have to recompute: every (head, q, k) logit belongs to
# exactly one CTA, so pass 1 can overwrite each bias tile it has just consumed
# with the rounded logit, and pass 2 can read that back instead of redoing QK^T
# and rereading the bias.  ``_CACHE_LOGITS`` implements that and is off, because
# it measured 248.9 us against 236.7 us for recomputing, reproducibly.  The
# reason is worth keeping: the head-major bias is 33.6 MB against ~126 MB of L2,
# so pass 2's reread is mostly an L2 hit, while the cache adds 33.6 MB of stores
# that have to be written back -- and the second QK^T is tensor-core work that
# overlaps the memory traffic anyway.  When enabled it is only ever pointed at a
# buffer this block allocated, never at a caller's ``position_bias`` and never at
# the Case A leaf that gets returned.
#
# Precisely what the first pass does and does not give: the row max is exact,
# because a max over tiles is order-independent.  The row sum is *not* bitwise
# the eager one -- it comes from an online rescaled recurrence over eight tiles,
# where the eager softmax reduces all 512 terms in its own order.  The two agree
# to a few fp32 ulp, which is ~1e-7 relative against the ~4e-3 bf16 rounding
# step, so the bf16 probabilities land on the same value except within about
# 1e-4 of a rounding boundary.  ``profile/probe_softmax_pv.py`` measures that
# directly rather than inferring it from the block result.
#
# q/k/v are read as strided views into the fused qkv projection output, with no
# pre-contiguous copy.  The bias is addressed through runtime strides so the same
# kernel serves the captured [q, k, h] layout and a head-major copy.
# ---------------------------------------------------------------------------
@triton.jit
def _block_logits(QKV, Bias, q, bias_row, offs_n, lane, stride_qkv,
                  stride_bias_k, K_OFFSET: tl.constexpr):
    """The baseline's logits for one key block, including both bf16 roundings.

    K is loaded as [BLOCK_N, D] and transposed for the dot.  Nsight Compute flags
    an average 17.9-way bank conflict on the shared loads that staging costs, and
    estimates 37.6% for removing it -- but loading K directly as a [D, BLOCK_N]
    tile instead puts the stride-12288 key axis innermost, which uncoalesces the
    global load and measured *worse* on the assembled block (66.5 us against
    46.1 us).  The bank conflict is the cheaper of the two, so it stays.
    """
    k = tl.load(QKV + K_OFFSET + offs_n[:, None] * stride_qkv + lane[None, :])
    logits = tl.dot(q, tl.trans(k)).to(tl.bfloat16)                    # rounding #1
    bias = tl.load(bias_row + offs_n[None, :] * stride_bias_k)
    logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)  # rounding #2
    return logits.to(tl.float32)


@triton.jit
def _attn_fwd_kernel(
    QKV, Bias, Logits, Out,
    stride_qkv, stride_bias_h, stride_bias_q, stride_bias_k, stride_out,
    S: tl.constexpr, D: tl.constexpr, K_OFFSET: tl.constexpr, V_OFFSET: tl.constexpr,
    CACHE_LOGITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    head = tl.program_id(0)
    offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    lane = head * D + offs_d

    q = tl.load(QKV + offs_m[:, None] * stride_qkv + lane[None, :])
    bias_row = Bias + head * stride_bias_h + offs_m[:, None] * stride_bias_q
    # Only ever a buffer this block owns; may alias Bias, which is safe because a
    # tile is read before it is overwritten and no other CTA touches it.
    logit_row = Logits + head * stride_bias_h + offs_m[:, None] * stride_bias_q

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        scores = _block_logits(QKV, Bias, q, bias_row, offs_n,
                               lane, stride_qkv, stride_bias_k, K_OFFSET)
        if CACHE_LOGITS:
            tl.store(logit_row + offs_n[None, :] * stride_bias_k, scores.to(tl.bfloat16))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        l_i = l_i * libdevice.exp(m_i - m_new) + tl.sum(
            libdevice.exp(scores - m_new[:, None]), axis=1)
        m_i = m_new

    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        if CACHE_LOGITS:
            scores = tl.load(logit_row + offs_n[None, :] * stride_bias_k).to(tl.float32)
        else:
            scores = _block_logits(QKV, Bias, q, bias_row, offs_n, lane, stride_qkv,
                                   stride_bias_k, K_OFFSET)
        p = (libdevice.exp(scores - m_i[:, None]) / l_i[:, None]).to(tl.bfloat16)
        v = tl.load(QKV + V_OFFSET + offs_n[:, None] * stride_qkv + lane[None, :])
        acc += tl.dot(p, v)

    tl.store(Out + offs_m[:, None] * stride_out + lane[None, :], acc.to(tl.bfloat16))


def _fused_attention(qkv: torch.Tensor, bias: torch.Tensor, n_heads: int, d_kv: int,
                     block_m: int = _ATTN_BLOCK_M, block_n: int = _ATTN_BLOCK_N,
                     num_warps: int = _ATTN_NUM_WARPS,
                     num_stages: int = _ATTN_NUM_STAGES,
                     logit_cache: torch.Tensor | None = None) -> torch.Tensor:
    """``qkv`` is ``[S, 3 * n_heads * d_kv]``; returns ``[S, n_heads * d_kv]``.

    ``logit_cache`` may be ``bias`` itself when the caller owns that buffer, in
    which case the second pass reads back the logits the first pass stored instead
    of recomputing them.  Pass ``None`` for a caller-owned or returned bias.
    """
    seq, _ = qkv.shape
    inner = n_heads * d_kv
    out = torch.empty((seq, inner), dtype=qkv.dtype, device=qkv.device)
    if bias.dim() == 4:
        _, sbh, sbq, sbk = bias.stride()
    else:
        sbh, sbq, sbk = bias.stride()
    _attn_fwd_kernel[(n_heads, triton.cdiv(seq, block_m))](
        qkv, bias, bias if logit_cache is None else logit_cache, out,
        qkv.stride(0), sbh, sbq, sbk, out.stride(0),
        S=seq, D=d_kv, K_OFFSET=inner, V_OFFSET=2 * inner,
        CACHE_LOGITS=logit_cache is not None,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def _eager_attention_context(qkv: torch.Tensor, bias: torch.Tensor, n_heads: int,
                             d_kv: int, seq: int) -> torch.Tensor:
    """The baseline's attention epilogue, for staged measurement only.

    Mirrors ``T5SelfAttention.forward`` between the qkv projection and the ``o``
    projection: bf16 ``QK^T``, the in-place bf16 bias add, an fp32 softmax cast
    back to bf16, ``PV``, then the transpose and contiguous copy the fused kernel
    exists to remove.
    """
    q, k, v = qkv.split([n_heads * d_kv] * 3, dim=-1)
    shape = (1, seq, n_heads, d_kv)
    q = q.view(shape).transpose(1, 2)
    k = k.view(shape).transpose(1, 2)
    v = v.view(shape).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(3, 2))
    scores += bias
    weights = F.softmax(scores.float(), dim=-1).type_as(scores)
    ctx = torch.matmul(weights, v).transpose(1, 2).contiguous()
    return ctx.view(seq, n_heads * d_kv)


# ---------------------------------------------------------------------------
# Fast-path guard.  Anything outside it delegates to the baseline forward, so the
# fallback cannot drift from baseline.py.
# ---------------------------------------------------------------------------
def _bias_layout(bias: torch.Tensor, n_heads: int, seq: int) -> str:
    """Classify the bias for dispatch: ``"qkh"``, ``"head_major"``, or ``"other"``.

    Shape and strides are not sufficient. The kernels read this pointer as bf16 on
    the device, so dtype and placement are part of the contract: a CUDA fp32
    tensor with the captured ``(H, 1, S*H, H)`` stride would otherwise be
    reinterpreted as bf16, and a CPU tensor would be handed to a CUDA launch.
    Both are classified ``"other"`` here so the caller falls back.

    ``"qkh"`` is the captured hot-path layout and needs a relayout.
    ``"head_major"`` is what this block produces for itself and what a preceding
    block in a real encoder stack would hand over, so it skips the relayout
    instead of undoing it.
    """
    if (bias.dtype != torch.bfloat16 or not bias.is_cuda
            or tuple(bias.shape) != (1, n_heads, seq, seq)):
        return "other"
    if bias.stride() == (n_heads, 1, seq * n_heads, n_heads):
        return "qkh"
    if bias.stride() == (n_heads * seq * seq, seq * seq, seq, 1):
        return "head_major"
    return "other"


class T5LayerSelfAttention(_BaselineT5LayerSelfAttention):
    """Fused self-attention sublayer, restricted to the captured configuration.

    ``_supported`` is decided once at construction from the config, and the guard
    is deliberately exact rather than permissive: the attention and
    relative-position-bias kernels index ``offs_m``/``offs_n`` with no edge masks
    and ``_rms_norm_kernel`` builds ``tl.arange(0, N)``, so a sequence length or
    model width outside the captured contract is an out-of-bounds access, not a
    slow path.  Everything else delegates to the baseline forward.
    """

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__(config, has_relative_attention_bias)
        attn = self.SelfAttention
        self._supported = (
            attn.d_model == _FAST_D_MODEL
            and attn.d_kv == _FAST_D_KV
            and attn.n_heads == _FAST_N_HEADS
            and attn.n_heads_per_partition == _FAST_N_HEADS
            and attn.inner_dim == _FAST_N_HEADS * _FAST_D_KV
            and _tp_size() == 1
            and tuple(attn.qkv_proj.weight.shape) == (3 * _FAST_D_MODEL, _FAST_D_MODEL)
            and tuple(attn.o.weight.shape) == (_FAST_D_MODEL, _FAST_D_MODEL)
        )

    def _use_fast_path(self, hidden_states: torch.Tensor,
                       mask: torch.Tensor | None,
                       position_bias: torch.Tensor | None = None) -> bool:
        """The complete dispatch decision, including the bias.

        ``position_bias`` is part of the contract because the kernels dereference
        it as bf16 device memory. When it is ``None`` the internal producer runs
        instead, and that path reads ``relative_attention_bias.emb.weight``, so
        that tensor is checked here too.
        """
        attn = self.SelfAttention
        if position_bias is not None:
            if _bias_layout(position_bias, attn.n_heads, _FAST_SEQ) == "other":
                return False
        elif attn.has_relative_attention_bias and _RELATIVE_BIAS_FUSED:
            emb = getattr(getattr(attn, "relative_attention_bias", None), "emb", None)
            w = getattr(emb, "weight", None)
            if (w is None or w.dtype != torch.bfloat16 or not w.is_cuda
                    or not w.is_contiguous()
                    or tuple(w.shape) != (attn.relative_attention_num_buckets,
                                          _FAST_N_HEADS)):
                return False
        return (self._supported
                and not torch.is_grad_enabled()
                and mask is None
                and _tp_size() == 1
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                and tuple(hidden_states.shape) == (1, _FAST_SEQ, _FAST_D_MODEL)
                and self.layer_norm.weight.dtype == torch.bfloat16
                and self.layer_norm.weight.is_cuda
                and attn.qkv_proj.weight.dtype == torch.bfloat16
                and attn.qkv_proj.weight.is_cuda
                and attn.qkv_proj.weight.is_contiguous()
                and attn.o.weight.dtype == torch.bfloat16
                and attn.o.weight.is_cuda
                and attn.o.weight.is_contiguous()
                and attn.qkv_proj.bias is None
                and attn.o.bias is None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn = self.SelfAttention
        seq = hidden_states.shape[1] if hidden_states.dim() == 3 else 0

        if not self._use_fast_path(hidden_states, mask, position_bias):
            return super().forward(hidden_states, mask=mask, position_bias=position_bias)

        if position_bias is None:
            if not attn.has_relative_attention_bias:
                return super().forward(hidden_states, mask=mask, position_bias=position_bias)
            if _RELATIVE_BIAS_FUSED:
                # The returned leaf reproduces compute_bias's observable [q, k, h]
                # stride; DEC-3 has not been decided, so the operator's output
                # layout is left exactly as the baseline's.  The head-major copy
                # the kernel wants is produced separately below.
                position_bias = _relative_bias(
                    attn.relative_attention_bias.emb.weight, seq, attn.n_heads,
                    attn.relative_attention_num_buckets,
                    attn.relative_attention_max_distance,
                    head_major=_CASE_A_HEAD_MAJOR_LEAF)
            else:
                # The baseline's own construction: bitwise identical values, and
                # already in the [q, k, h] layout the hot path expects.
                position_bias = attn.compute_bias(seq, seq, device=hidden_states.device)

        # Re-classify: for Case A this is the tensor we just produced, and the
        # guard above validated only a caller-supplied bias.
        layout = _bias_layout(position_bias, attn.n_heads, seq)
        if layout == "other":
            return super().forward(hidden_states, mask=mask, position_bias=position_bias)

        flat = hidden_states.view(seq, -1)
        normed = (_rms_norm(flat, self.layer_norm.weight,
                            self.layer_norm.variance_epsilon)
                  if _FUSE_NORM else self.layer_norm(flat))
        qkv = F.linear(normed, attn.qkv_proj.weight)

        if not _FUSE_ATTENTION:
            context = _eager_attention_context(qkv, position_bias, attn.n_heads,
                                               attn.d_kv, seq)
            out = (torch.addmm(flat, context, attn.o.weight.t()) if _FUSE_ADDMM
                   else flat + F.linear(context, attn.o.weight))
            return out.view_as(hidden_states), position_bias

        if layout == "qkh" and _BIAS_HEAD_MAJOR:
            # This buffer is ours, so the attention kernel may reuse it to carry
            # the logits from its first pass to its second.
            bias = _bias_to_head_major(position_bias)
            cache = bias if _CACHE_LOGITS else None
        else:
            bias = position_bias
            cache = None
        context = _fused_attention(qkv, bias, attn.n_heads, attn.d_kv, logit_cache=cache)

        # Residual rides along in the GEMM epilogue (beta = 1) instead of a
        # separate elementwise pass over 4 MB.
        out = (torch.addmm(flat, context, attn.o.weight.t()) if _FUSE_ADDMM
               else flat + F.linear(context, attn.o.weight))
        return out.view_as(hidden_states), position_bias


class T5LayerFF(_BaselineT5LayerFF):
    """Fused gated-FFN sublayer, restricted to the captured configuration.

    The same exactness argument as the attention sublayer applies to the width,
    and there is a second reason to guard TP: the fast path calls ``F.linear`` and
    ``torch.addmm`` directly, which would bypass ``RowParallelLinear``'s
    all-reduce on ``wo`` when TP > 1 and silently return a partial sum.
    """

    def __init__(self, config: T5Config):
        super().__init__(config)
        dense = self.DenseReluDense
        self._supported = (
            getattr(config, "is_gated_act", False)
            and getattr(config, "dense_act_fn", None) == "gelu_new"
            and type(dense).__name__ == "T5DenseGatedActDense"
            and type(dense.act).__name__ == "NewGELUActivation"
            and _tp_size() == 1
            and tuple(dense.wi.weight.shape) == (2 * _FAST_D_FF, _FAST_D_MODEL)
            and tuple(dense.wo.weight.shape) == (_FAST_D_MODEL, _FAST_D_FF)
        )

    def _use_fast_path(self, hidden_states: torch.Tensor) -> bool:
        dense = self.DenseReluDense
        return (self._supported
                and not torch.is_grad_enabled()
                and _tp_size() == 1
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.is_contiguous()
                and tuple(hidden_states.shape) == (1, _FAST_SEQ, _FAST_D_MODEL)
                and self.layer_norm.weight.dtype == torch.bfloat16
                and self.layer_norm.weight.is_cuda
                and dense.wi.weight.dtype == torch.bfloat16
                and dense.wi.weight.is_cuda
                and dense.wi.weight.is_contiguous()
                and dense.wo.weight.dtype == torch.bfloat16
                and dense.wo.weight.is_cuda
                and dense.wo.weight.is_contiguous()
                and dense.wi.bias is None
                and dense.wo.bias is None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._use_fast_path(hidden_states):
            return super().forward(hidden_states)

        dense = self.DenseReluDense
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        normed = (_rms_norm(flat, self.layer_norm.weight,
                            self.layer_norm.variance_epsilon)
                  if _FUSE_NORM else self.layer_norm(flat))
        gate_up = F.linear(normed, dense.wi.weight)
        if _FUSE_ACT:
            act = _gated_gelu(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            act = dense.act(gate) * up
        out = (torch.addmm(flat, act, dense.wo.weight.t()) if _FUSE_ADDMM
               else flat + F.linear(act, dense.wo.weight))
        return out.view_as(hidden_states)


class T5Block(_BaselineT5Block):
    """Same module tree as the baseline, built from the fused sublayers.

    ``forward`` is inherited unchanged: it only sequences the two sublayers.
    """

    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        nn.Module.__init__(self)
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])
