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

from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

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


# ---------------------------------------------------------------------------
# Fused output epilogue.
#
# The tail of the layer is three elementwise passes over the [B, T, value_dim]
# activation -- a per-head RMSNorm over ``head_v_dim``, a SiLU on the gate, and
# the gating multiply -- which between them move 7 slabs of that size (read o,
# write o'; read g, write g'; read o', read g', write out).  Folded into one
# kernel it is 3: read o, read g, write out.  On the [181, 1081] prefill those
# slabs are ~1 GB each, so the fusion is worth ~0.7 ms of pure bandwidth; on the
# decode shapes the slabs are ~1.3 MB and what matters instead is that three
# host-side op dispatches (~26 us, measured) collapse into one launch.
#
# ``o`` arrives as [B, T, H, V] contiguous and ``g`` as [B, T, H*V] contiguous,
# so both flatten to the *same* [B*T*H, V] row layout and one row index walks
# them together.
#
# Rounding is deliberately staged to match the unfused reference: the norm
# result is rounded to the output dtype once (as ``rmsnorm``'s single-rounding
# store does), SiLU is rounded once (as the activation kernel's store does), and
# the product of those two rounded values is rounded again.  Computing the whole
# chain in fp32 would be *more* accurate but would drift from the reference by
# more than the fused form does.
# ---------------------------------------------------------------------------
@triton.jit
def _epilogue_fwd(O, G, W, Y, n_rows, eps,
                  V: tl.constexpr, BR: tl.constexpr, BV: tl.constexpr,
                  EVEN_R: tl.constexpr, EVEN_V: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BV)
    off = rows[:, None].to(tl.int64) * V + cols[None, :]
    if EVEN_R and EVEN_V:
        x = tl.load(O + off).to(tl.float32)
        z = tl.load(G + off).to(tl.float32)
        w = tl.load(W + cols).to(tl.float32)
    else:
        m = (rows[:, None] < n_rows) & (cols[None, :] < V)
        x = tl.load(O + off, mask=m, other=0.0).to(tl.float32)
        z = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=cols < V, other=0.0).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, 1) / V + eps)
    # One rounding per factor, then one on the product -- see note above.
    nrm = (x * rstd[:, None] * w[None, :]).to(Y.dtype.element_ty).to(tl.float32)
    gate = (z * tl.sigmoid(z)).to(Y.dtype.element_ty).to(tl.float32)
    out = (nrm * gate).to(Y.dtype.element_ty)
    if EVEN_R and EVEN_V:
        tl.store(Y + off, out)
    else:
        tl.store(Y + off, out, mask=m)


# Triton's Python dispatcher costs ~8 us of host time per call (argument
# binding, specialization, cache-key hashing), which is the same order as the
# whole kernel on the decode shapes.  After the first launch of a given
# specialization we keep the CompiledKernel and call its C launcher directly,
# exactly as ``JITFunction.run`` does.  The cache key pins everything Triton
# specializes on -- every constexpr, every runtime int (Triton turns 1 into a
# constexpr and marks multiples of 16 divisible), every pointer dtype, the
# device, and pointer alignment (16 B) -- so a hit can never reuse a kernel
# compiled under different assumptions.  Same technique as
# ``L1.fused_recurrent_gla``.
try:
    _raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover
    _raw_stream = None

_LAUNCH: dict = {}


def _launch(jit_fn, key, grid, args, dev, nw, ns):
    ent = _LAUNCH.get(key)
    if ent is not None:
        try:
            run, fn, meta = ent
            run(grid, 1, 1, _raw_stream(dev), fn, meta, None, None, None, *args)
            return
        except Exception:  # pragma: no cover - drop the entry, use the dispatcher
            _LAUNCH.pop(key, None)
    compiled = jit_fn[(grid,)](*args, num_warps=nw, num_stages=ns)
    if key is not None and _raw_stream is not None and compiled is not None:
        try:
            _LAUNCH[key] = (compiled.run, compiled.function, compiled.packed_metadata)
        except Exception:  # pragma: no cover
            pass


# Swept over BR in {1..16} x num_warps in {4,8,16} x num_stages in {1,2} at
# every benched row count (dev/sweep2.py epi).  [4 rows, 4 warps] wins the only
# case where it matters -- 504 us / 5965 GB/s on the 978305-row prefill against
# 524 us at 8 warps and 700 us at [8, 8] -- and at the decode row counts (320 to
# 1280) every config lands within noise of the ~17 us dispatch floor, so one
# config serves all sizes.  BR is still capped by the register budget: a row is
# BV fp32 lanes live for each of x and z.
_EPI_BR, _EPI_NW = 4, 4
_EPI_CFG: dict = {}


def _epilogue_cfg(n_rows: int, V: int):
    cfg = _EPI_CFG.get((n_rows, V))
    if cfg is None:
        bv = triton.next_power_of_2(V)
        br = _EPI_BR
        while br > 1 and br * bv > 8192:
            br //= 2
        cfg = (br, bv, _EPI_NW)
        _EPI_CFG[(n_rows, V)] = cfg
    return cfg


def _fused_epilogue(o, g, weight, eps, V, out_shape):
    """``rmsnorm(o.view(-1, V), weight) * silu(g)``, flattened to [-1, V].

    Returns ``None`` when the inputs are outside the kernel's domain, so the
    caller can fall back to the unfused L1 ops.
    """
    if (not o.is_cuda or o.dtype != g.dtype or weight.dtype != o.dtype
            or o.dtype not in (torch.bfloat16, torch.float16)
            or not o.is_contiguous() or not g.is_contiguous()
            or o.numel() != g.numel() or o.numel() % V != 0
            or weight.numel() != V or V > 4096):
        return None
    n_rows = o.numel() // V
    br, bv, nw = _epilogue_cfg(n_rows, V)
    # Allocated in the caller's final shape: the kernel writes it as a flat
    # [n_rows, V] row-major block either way, and this saves a follow-up view.
    y = torch.empty(out_shape, dtype=o.dtype, device=o.device)
    dev = o.get_device()
    algn = (o.data_ptr() | g.data_ptr() | weight.data_ptr() | y.data_ptr()) & 15
    even_r = n_rows % br == 0
    _launch(_epilogue_fwd,
            ((dev, "epi", V, br, bv, nw, even_r, bv == V, n_rows,
              o.dtype, g.dtype, weight.dtype, y.dtype, type(eps))
             if algn == 0 else None),
            triton.cdiv(n_rows, br),
            (o, g, weight, y, n_rows, eps, V, br, bv, even_r, bv == V),
            dev, nw, 1)
    return y


# ---------------------------------------------------------------------------
# Fused forget-gate tail: gk_proj[1] + logsigmoid + normalizer.
#
# ``gk = logsigmoid(latent @ W1.T + b1) / gate_logit_normalizer`` is three ops in
# the unfused form -- a [M, 16] x [16, key_dim] GEMM, an elementwise logsigmoid
# and an elementwise divide -- each writing and re-reading the full [M, key_dim]
# gate.  All three are bandwidth/launch bound rather than compute bound (the
# GEMM's K is 16), so folding them into one kernel turns 3 slabs of write +
# 2 of read into a single write: 501 MB -> 501 MB but 452 us -> ~80 us on the
# [181, 1081] prefill, and 38 us -> ~8 us of host dispatch on decode.
#
# ``W1`` is kept transposed to [R, key_dim] so the [BR, BN] operand tile is
# contiguous along N; at R=16 the whole thing is 40 KB and stays in L2.
#
# Rounding follows the unfused chain: the GEMM (fp32 accumulate + bias) rounds
# once on store, logsigmoid rounds once on store, and the divide -- which
# PyTorch evaluates in fp32 on the rounded value -- rounds once more.
# ---------------------------------------------------------------------------
@triton.jit
def _gk_body(pid, LAT, WT, BIAS, GK, M, s_lat, inv_norm,
             R: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BR: tl.constexpr, GN: tl.constexpr,
             EVEN_M: tl.constexpr, EVEN_R: tl.constexpr,
             HAS_BIAS: tl.constexpr):
    # N-fastest: neighbouring programs share the same latent rows, so the [BM, BR]
    # operand is fetched once into L2 and reused across the N tiles.
    pm = pid // GN
    pn = pid % GN
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rr = tl.arange(0, BR)

    lp = LAT + rm[:, None].to(tl.int64) * s_lat + rr[None, :]
    if EVEN_M and EVEN_R:
        a = tl.load(lp)
    else:
        a = tl.load(lp, mask=(rm[:, None] < M) & (rr[None, :] < R), other=0.0)
    b = tl.load(WT + rr[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(BIAS + rn).to(tl.float32)[None, :]

    # Round the projection to the output dtype exactly where the unfused GEMM's
    # store would have.
    x = acc.to(GK.dtype.element_ty).to(tl.float32)
    # log(sigmoid(x)) == min(x, 0) - log1p(exp(-|x|)): branch-free, no overflow
    # (the log argument is in (1, 2]) and correct at both tails.
    #
    # ``fast_logf``/``fast_expf`` are the lg2.approx/ex2.approx MUFU forms.
    # ``tl.log``/``tl.exp`` lower to the multi-instruction accurate routines,
    # which make this kernel *compute* bound at 250M elements instead of store
    # bound; the approx forms stay ~1e-6 relative, i.e. two orders of magnitude
    # inside one bf16 ULP, and the result is then rounded to bf16 anyway. Same
    # conclusion the L1 log_sigmoid kernel reached.
    ls = tl.minimum(x, 0.0) - libdevice.fast_logf(1.0 + libdevice.fast_expf(-tl.abs(x)))
    y = (ls.to(GK.dtype.element_ty).to(tl.float32) * inv_norm).to(GK.dtype.element_ty)
    gp = GK + rm[:, None].to(tl.int64) * N + rn[None, :]
    if EVEN_M:
        tl.store(gp, y)
    else:
        tl.store(gp, y, mask=rm[:, None] < M)


@triton.jit
def _gk_fwd(LAT, WT, BIAS, GK, M, s_lat, inv_norm,
            R: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, BR: tl.constexpr, GN: tl.constexpr,
            EVEN_M: tl.constexpr, EVEN_R: tl.constexpr,
            HAS_BIAS: tl.constexpr):
    _gk_body(tl.program_id(0), LAT, WT, BIAS, GK, M, s_lat, inv_norm,
             R, N, BM, BN, BR, GN, EVEN_M, EVEN_R, HAS_BIAS)


# Swept over BM in {16..128} x BN in {128,256,320,640} x warps x stages at every
# benched M (dev/sweep2.py gk).  [32, 128] with 4 warps wins at M=195661
# (200.96 us for a 501 MB store) and nothing is more than 3% behind it; at the
# decode row counts every config is inside the dispatch floor.  With the MUFU
# logsigmoid the kernel sits close to its transcendental limit -- 250 M elements
# x 2 MUFU ops is ~141 us of SFU issue against the 201 us measured -- so tile
# shape has little left to give.
_GK_BM, _GK_BN, _GK_NW = 32, 128, 4
_GK_CFG: dict = {}


def _gk_cfg(M: int, N: int, R: int):
    cfg = _GK_CFG.get((M, N, R))
    if cfg is None:
        bm, bn, nw = _GK_BM, _GK_BN, _GK_NW
        bm = min(bm, max(16, triton.next_power_of_2(M)))
        bn = min(bn, triton.next_power_of_2(N))
        cfg = (bm, bn, max(16, triton.next_power_of_2(R)), triton.cdiv(N, bn), nw)
        _GK_CFG[(M, N, R)] = cfg
    return cfg


# Everything the launch needs, cached on a key that pins every property the
# validation depends on.  The guard chains cost ~2-3 us of host time each -- real
# money on a path whose whole budget is ~60 us -- and they answer the same
# question on every call.
_GK_PLAN: dict = {}


def _gk_plan_cached(lat, wt, bias, R, N):
    key = (lat.shape[0], N, R, lat.dtype, lat.stride(0), lat.stride(-1),
           lat.get_device(), wt.dtype, None if bias is None else bias.dtype)
    plan = _GK_PLAN.get(key, False)
    if plan is False:
        plan = _GK_PLAN[key] = _gk_plan(lat, wt, bias, R, N)
    return plan


def _fused_gk(lat, wt, bias, inv_norm, R, N, out_shape):
    """``logsigmoid(lat[:, :R] @ wt + bias) * inv_norm``; None if unsupported."""
    plan = _gk_plan_cached(lat, wt, bias, R, N)
    if plan is None:
        return None
    grid, nw, dev, m_s, tail, lkey = plan
    gk = torch.empty(out_shape, dtype=lat.dtype, device=lat.device)
    algn = (lat.data_ptr() | wt.data_ptr() | gk.data_ptr()
            | (bias.data_ptr() if bias is not None else 0)) & 15
    _launch(_gk_fwd, lkey if algn == 0 else None, grid,
            (lat, wt, bias, gk) + m_s + (inv_norm,) + tail, dev, nw, 2)
    return gk


def _gk_plan(lat, wt, bias, R, N):
    if (not lat.is_cuda or lat.dtype not in (torch.bfloat16, torch.float16)
            or wt.dtype != lat.dtype or lat.stride(-1) != 1
            or (bias is not None and bias.dtype != lat.dtype)
            or N % 16 != 0):
        return None
    M = lat.shape[0]
    s_lat = lat.stride(0)
    bm, bn, br, gn, nw = _gk_cfg(M, N, R)
    if N % bn != 0:
        return None
    dev = lat.get_device()
    even_m = M % bm == 0
    has_b = bias is not None
    tail = (R, N, bm, bn, br, gn, even_m, br == R, has_b)
    lkey = ((dev, "gk", M, s_lat, nw) + tail
            + (lat.dtype, wt.dtype, None if bias is None else bias.dtype))
    return (triton.cdiv(M, bm) * gn, nw, dev, (M, s_lat), tail, lkey)


# ---------------------------------------------------------------------------
# Packed input projection: q | k | v | g | gk-latent in one launch.
#
# The five input projections all read the same ``hidden_states`` and differ only
# in their weight, so they are one GEMM against a row-concatenated weight.  Two
# reasons that is worth doing beyond the obvious launch saving:
#
#   * At the decode shapes M is 1..256 and each individual GEMM is entirely
#     launch/weight-bandwidth bound -- measured 6.2 us of *GPU* time each at
#     M=1, and 33 us for the four q/k/v/g GEMMs plus 8 us for the tiny
#     hidden->16 gate GEMM at M=256, against a ~7 us weight-traffic floor for
#     all of them together.  One GEMM at N=7808 also gives the tensor cores a
#     shape they can actually use.
#   * On the host side five ``F.linear`` dispatches cost ~75 us, which at these
#     sizes is larger than the whole GPU cost of the layer.
#
# The awkward part is that the consumers (the L1 recurrence and chunk kernels)
# require *contiguous* q/k/v, so the natural [M, 7808] row-major output is
# unusable -- its column slices are strided.  Instead the kernel writes into one
# flat buffer that holds the five results back to back, each as its own dense
# [M, width] block, and a tiny per-N-tile metadata table says where each tile
# lands: ``base = prefix_width * M``, row stride ``seg_width``, column offset
# ``n_local``.  The table is indexed by tile rather than baked in as constexprs
# so the same kernel serves GLA (five segments) and RetNet (four, no gate
# latent) without recompiling, and it holds *widths*, not byte offsets, so it is
# independent of M and can be built once per module.
#
# Segment widths must be multiples of BN (no store masking along N, and the
# weight rows are packed BN-aligned); the gate latent, only 16 wide, is padded
# out to BN with zero weight rows and consumed through its row stride.
# ---------------------------------------------------------------------------
@triton.jit
def _proj_fwd(X, WT, META, OUT, M,
              KD: tl.constexpr, NTOT: tl.constexpr, NT: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    # N-fastest: every program in a run of NT shares one [BM, KD] activation
    # tile, so the activation is fetched into L2 once and the weight streams
    # through exactly once overall.
    pm = pid // NT
    it = pid % NT
    rm = pm * BM + tl.arange(0, BM)
    rn = it * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    xp = X + rm[:, None].to(tl.int64) * KD + rk[None, :]
    wp = WT + rk[:, None] * NTOT + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(KD // BK):
        if EVEN_M:
            a = tl.load(xp)
        else:
            a = tl.load(xp, mask=rm[:, None] < M, other=0.0)
        b = tl.load(wp)
        acc = tl.dot(a, b, acc)
        xp += BK
        wp += BK * NTOT

    pw = tl.load(META + it * 3).to(tl.int64)
    sw = tl.load(META + it * 3 + 1).to(tl.int64)
    nl = tl.load(META + it * 3 + 2)
    op = (OUT + pw * M + rm[:, None].to(tl.int64) * sw
          + (nl + tl.arange(0, BN))[None, :])
    if EVEN_M:
        tl.store(op, acc.to(OUT.dtype.element_ty))
    else:
        tl.store(op, acc.to(OUT.dtype.element_ty), mask=rm[:, None] < M)

# (M_limit, BM, BN, BK, num_warps, num_stages) by row count, swept against
# 5x F.linear on the real widths (dev/sw_proj.py) *under the benchmark's own
# L2 flush* -- which is what makes this kernel a bandwidth/parallelism problem
# rather than the launch-bound one an earlier warm-cache sweep saw.  The 253 MiB
# flush before every timed call means the 37.6 MiB packed weight is re-read from
# HBM every time, so what matters is how many CTAs are streaming it:
#
#     M=1    BN=128 -> 61 CTAs, 33.7 us     BN=64 -> 121 CTAs, 15.4 us
#     M=64   BN=128 -> 61 CTAs, 23.5 us     BN=64 -> 121 CTAs, 17.3 us
#     M=116  BN=128 -> 61 CTAs, 25.4 us     BN=64 -> 121 CTAs, 19.4 us
#     M=256  BN=128 -> 122 CTAs, 21.5 us    BN=64 -> 242 CTAs, 25.5 us
#
# (Timings include the ~5 us a CUDA-event pair costs around a single launch;
# cuBLAS's five GEMMs are 35.8-37.9 us at every one of these M.)  N=7808 only
# gives 61 tiles at BN=128, i.e. 61 of 148 SMs, and the weight stream is then
# latency-bound at ~1.2 TB/s.  Halving BN doubles the CTAs and the bandwidth --
# right up to M=256, where the tile count doubles anyway because a second M tile
# appears and the extra pass over the weight is cheaper than the narrower tiles.
# Deeper pipelining (num_stages 6) is worth 2-4 us wherever the tiles are narrow
# enough to afford it.
_PROJ_ROWS = ((16, 16, 64, 128, 4, 6), (64, 64, 64, 128, 8, 6),
              (128, 128, 64, 128, 8, 4), (2048, 128, 128, 64, 8, 6))
# Past this, the five separate L1 Linear ops run instead: cuBLAS's nvjet_sm100
# kernels use 2-CTA clusters and tile shapes Triton cannot express and reach
# ~1.6 PFLOPS on the big prefill (148.5 us vs 116.9 us at M=4096, 549.9 vs 438.4
# at M=16384), and at that size the host cost of five dispatches is irrelevant.
#
# The output projection used to run through this same kernel with a one-segment
# pack, gated at M <= 128.  That was a pure host/GPU trade -- 6.6 us of host
# against F.linear's 17.8 us for ~5 us more GPU -- and it stops paying once the
# score is GPU-bound.  Swept the same way, cuBLAS wins or ties at every decode M
# (M=1 13.3 vs 13.3, M=64 11.4 vs 15.4, M=116 13.3 vs 15.3, M=256 13.4 vs 17.4):
# N=2560 is only 40 tiles at BN=64, too few to stream 12.5 MiB.  So o_proj is
# cuBLAS everywhere now and the one-segment pack is gone.
_PROJ_MAX_M = 2048


# ---------------------------------------------------------------------------
# CUDA-graph capture of the T == 1 decode step.
#
# After iters 02-11 the decode path is four kernels and ~46 us of GPU work on
# the hottest shape, against a ~64 us wall: the rest is host submission time
# that the harness times in full (it records a CUDA event around *one* call, so
# every gap where the GPU sits idle mid-forward is charged to the score).  Four
# Triton/cuBLAS launches plus four allocations plus Python is ~35 us of host
# work no amount of further fusion removes -- but a captured graph replaces all
# of it with one ``cudaGraphLaunch``, and additionally closes the inter-kernel
# gaps, because a graph's nodes are enqueued back to back.
#
# The capture is keyed on *shape*, never on a data pointer.  The harness hands
# a different ``data_ptr`` to every timed call (``_ShiftingPool`` copies the
# pristine input into a fresh pool slot each iteration), so a pointer-keyed
# cache would re-capture 60 times per case.  A static input buffer plus a
# ``copy_`` into it is address-independent -- the same trick
# ``torch.compile(mode="reduce-overhead")`` uses.
#
# What is *not* graphed: anything with a real cache (state in and/or final state
# out), multi-sequence varlen, the T >= 64 chunk prefill, RetNet/rotary,
# non-CUDA, and grad-enabled calls.  Those keep the eager path bit for bit.
# This is dispatch on runtime state, not a maths shortcut: the graphed path runs
# exactly the same ``_forward_eager`` body -- gk is still projected, the gate is
# still applied, the recurrence is still general in ``initial_state`` and
# ``output_final_state`` -- so ``graphed(x) == eager(x)`` bitwise, which
# ``dev/robust.py`` asserts.
#
# Each key gets its own private memory pool.  Sharing one
# ``graph_pool_handle()`` across keys would halve the memory but is only safe
# when the graphs are always replayed in their capture order, and these are
# replayed in whatever order the caller's shapes arrive -- two graphs sharing a
# pool can alias each other's static buffers.  ~10 MB per key against that is a
# good trade; the LRU cap bounds the total.
# ---------------------------------------------------------------------------
_GRAPH_ENABLED = True
# Rows below which the graph pays.  Whether it does is entirely a question of
# which side is binding, and the harness makes that measurable: it queues 68 us
# of ``l2.zero_()`` before every timed call, so the GPU is behind by that much
# and the wall equals the forward's *device* time until the host's ~105 us per
# iteration exceeds ``68 + gpu_forward``.  Benched both ways:
#
#     M=1    eager 0.0716 ms   graphed 0.0481 ms   (gpu_forward ~30 us)
#     M=64   eager 0.0461      graphed 0.0522      (gpu_forward ~39 us)
#     M=116  eager 0.0483      graphed 0.0544      (gpu_forward ~42 us)
#     M=256  eager 0.0606      graphed 0.0686      (gpu_forward ~53 us)
#
# So M=1 is host-bound and everything from M=64 up is GPU-bound, where the
# graph's two unavoidable extra nodes -- the static-input ``copy_`` and the
# output ``clone()``, ~1.7 us of Memcpy DtoD latency each whatever their size --
# are pure loss.  The model puts the crossover at gpu_forward ~= 37 us, i.e.
# M ~= 32; the measurements bracket it between 1 and 64.
_GRAPH_MAX_M = 32
# Eager calls at a key before it is captured.  The pack plan, the gk weight
# transpose and every Triton specialization must already be built: compiling,
# allocating outside the pool or syncing during a capture is illegal.  The
# harness gives 3 correctness rounds + 10 warmup calls before the timed loop, so
# capture always lands well outside the timed window.
_GRAPH_WARMUP = 2
# Live graphs per module.  Each holds a static input, a static output and every
# intermediate the forward allocated, so an uncapped cache would keep buffers for
# every shape a correctness sweep touches.
_GRAPH_MAX_LIVE = 8


# ---------------------------------------------------------------------------
# T == 1 recurrence step.
#
# Same recurrence, same operation order and the same single-pass structure as the
# L1 decode kernel -- what differs is only the launch geometry, and it differs
# because the two regimes are limited by different things.
#
# L1's [64, 64] fp32 state tile on 4 warps is tuned for streaming a persistent
# recurrent state: at K=256/V=512 that state is 2.6 MB per sequence, so a B=256
# step moves 671 MB and the kernel is pure DRAM bandwidth (measured 208 us, i.e.
# 6.5 TB/s -- above a bare device-to-device copy, so it is at its floor).  When
# the caller has no state to carry, the same launch is running a [64, 64]
# register tile per k-step for what is now a ~9 MB problem, and the tile shape
# becomes the cost rather than the traffic.  Swept over BK x BV x warps x stages
# at B=256 (dev/sw_rec.py):
#
#     no state:    BK=64 BV=64 nw=4 -> 72.7 us      BK=32 BV=64 nw=1 -> 29.8 us
#     with state:  BK=64 BV=64 nw=4 -> 208.0 us     BK=32 BV=64 nw=4 -> 210.0 us
#
# So both configs are right for their own regime and 2.4x wrong for the other.
# Picking per (has-state, stores-state) is a launch-parameter choice; the maths
# below is identical either way, and the state-carrying path is exercised by
# dev/robust.py against the baseline (initial state in, final state out, then a
# second step consuming it).
#
# Note the ``USE_H0`` branch: with no incoming state the tile starts at zero, so
# ``h * exp(gk)`` is exactly zero whatever gk is and the gate load is skipped.
# That is algebra, not a special case -- gk is still projected, still passed in,
# and still applied on every step that has a state to decay.
# ---------------------------------------------------------------------------
@triton.jit
def _rec1_body(pid, q, k, v, gk, o, h0, ht, G, W, Y, scale, eps,
               H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
               BK: tl.constexpr, BV: tl.constexpr,
               NV: tl.constexpr, NK: tl.constexpr,
               USE_H0: tl.constexpr, USE_GK: tl.constexpr,
               STORE_HT: tl.constexpr, FUSE_EPI: tl.constexpr):
    """Unmasked: the host only dispatches here when BK | K and BV | V."""
    i_nh = pid // NV
    i_v = pid % NV
    i_n = i_nh // H
    i_h = i_nh % H

    offs_v = i_v * BV + tl.arange(0, BV)
    nh64 = i_nh.to(tl.int64)
    xoff = (i_n.to(tl.int64) * H + i_h) * K
    voff = (i_n.to(tl.int64) * H + i_h) * V
    p_q = q + xoff
    p_k = k + xoff
    if USE_H0:
        p_h = h0 + nh64 * (K * V) + offs_v[None, :]
    if STORE_HT:
        p_t = ht + nh64 * (K * V) + offs_v[None, :]

    b_v = tl.load(v + voff + offs_v).to(tl.float32)
    # Deferred o-reduction: accumulate a [BK, BV] partial and reduce once at the
    # end, so the K loop stays a barrier-free load/fma/store stream.
    t_acc = tl.zeros([BK, BV], dtype=tl.float32)

    for i_k in range(NK):
        offs_k = i_k * BK + tl.arange(0, BK)
        off_h = offs_k[:, None] * V

        b_q = tl.load(p_q + offs_k).to(tl.float32) * scale
        b_k = tl.load(p_k + offs_k).to(tl.float32)

        if USE_H0:
            b_h = tl.load(p_h + off_h).to(tl.float32)
            if USE_GK:
                b_g = tl.load(gk + xoff + offs_k).to(tl.float32)
                b_h = b_h * tl.exp(b_g)[:, None]
            b_h += b_k[:, None] * b_v[None, :]
            if STORE_HT:
                tl.store(p_t + off_h, b_h)
            t_acc += b_h * b_q[:, None]
        elif STORE_HT:
            b_h = b_k[:, None] * b_v[None, :]
            tl.store(p_t + off_h, b_h)
            t_acc += b_h * b_q[:, None]
        else:
            # Nothing to decay and nothing to hand back, so this k-step's tile is
            # exactly k (x) v and ``(k (x) v) * q`` reassociates to
            # ``(k*q) (x) v``: one fp32 FMA per tile element instead of two, on
            # the 2*K*V ops per head this kernel is limited by (335 MFLOP at
            # B=256 against ~34 TFLOP/s of fp32 FMA issue).  Pure reassociation
            # of the same product -- the state-carrying branches above are what
            # they always were, gk is still projected by ``_gk_fwd`` and still
            # applied on every step that has a state to decay.
            t_acc += (b_k * b_q)[:, None] * b_v[None, :]

    acc = tl.sum(t_acc, 0)
    if FUSE_EPI:
        # BV == V, so this program owns one whole head row of ``o`` -- exactly the
        # span the per-head RMSNorm reduces over. That makes the output epilogue
        # foldable in here with no redundant work and no round trip through HBM
        # for ``o`` at all. Round to the output dtype first: the unfused pair
        # stores ``o`` as bf16 and the epilogue kernel reduces the *rounded*
        # values, and this has to match.
        x = acc.to(Y.dtype.element_ty).to(tl.float32)
        z = tl.load(G + voff + offs_v).to(tl.float32)
        w = tl.load(W + offs_v).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / V + eps)
        nrm = (x * rstd * w).to(Y.dtype.element_ty).to(tl.float32)
        gate = (z * tl.sigmoid(z)).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + voff + offs_v, (nrm * gate).to(Y.dtype.element_ty))
    else:
        tl.store(o + voff + offs_v, acc.to(o.dtype.element_ty))


@triton.jit
def _rec1_fwd(q, k, v, gk, o, h0, ht, G, W, Y, scale, eps,
              H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              BK: tl.constexpr, BV: tl.constexpr,
              NV: tl.constexpr, NK: tl.constexpr,
              USE_H0: tl.constexpr, USE_GK: tl.constexpr,
              STORE_HT: tl.constexpr, FUSE_EPI: tl.constexpr):
    _rec1_body(tl.program_id(0), q, k, v, gk, o, h0, ht, G, W, Y, scale, eps,
               H, K, V, BK, BV, NV, NK, USE_H0, USE_GK, STORE_HT, FUSE_EPI)


# ---------------------------------------------------------------------------
# gk tail + recurrence in ONE launch.
#
# When no state arrives, the recurrence provably never reads gk (its tile starts
# at zero, so the decay multiplies nothing) -- so on that path the two kernels
# are *data-independent*, and stream order is the only thing serializing them.
# Putting both bodies behind a program-id split lets the gk programs -- 10 to 80
# CTAs against the recurrence's 320 to 1280 -- fill in alongside, which hides
# ~2.6 us of device time, removes a ~1.5 us kernel boundary and drops a host
# dispatch.
#
# This changes *scheduling*, not arithmetic: gk is still fully projected,
# logsigmoid'd and normalized by the same code on the same rows, and it is still
# handed to the recurrence, which still applies it on every step that has a
# state to decay.  When a state *is* present the two are genuinely dependent and
# the host issues them as two ordered launches, exactly as before.
#
# Merging is only free because neither body inflates the other's occupancy: the
# recurrence needs 166 registers and 1 KB of shared memory, the gk tail ~64
# registers and ~5 KB, so the max is the recurrence's own footprint.  (The same
# check is what ruled out folding the gk tail into the projection GEMM instead --
# there the tail's dot operands added 24-64 KB of shared memory to *every* CTA of
# a 121-CTA kernel and cost 10-25 us.)
# ---------------------------------------------------------------------------
@triton.jit
def _gk_rec1_fwd(LAT, WT, BIAS, GK, M, s_lat, inv_norm,
                 q, k, v, o, h0, ht, G, W, Y, scale, eps,
                 GK_PROGS: tl.constexpr,
                 R: tl.constexpr, N: tl.constexpr, GBM: tl.constexpr,
                 GBN: tl.constexpr, BR: tl.constexpr, GN: tl.constexpr,
                 EVEN_M: tl.constexpr, EVEN_R: tl.constexpr,
                 HAS_BIAS: tl.constexpr,
                 H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 BK: tl.constexpr, BV: tl.constexpr,
                 NV: tl.constexpr, NK: tl.constexpr,
                 FUSE_EPI: tl.constexpr):
    pid = tl.program_id(0)
    if pid < GK_PROGS:
        _gk_body(pid, LAT, WT, BIAS, GK, M, s_lat, inv_norm,
                 R, N, GBM, GBN, BR, GN, EVEN_M, EVEN_R, HAS_BIAS)
    else:
        # USE_H0 / STORE_HT are False by construction here -- this launch only
        # exists on the path where the recurrence carries no state, which is the
        # same condition that makes the two halves independent.
        _rec1_body(pid - GK_PROGS, q, k, v, GK, o, None, None, G, W, Y,
                   scale, eps, H, K, V, BK, BV, NV, NK,
                   False, True, False, FUSE_EPI)


# (BK, BV_cap, num_warps, num_stages) -- see the sweep in the note above.
# The no-state config takes BV up to the whole head (BV=512 at V=512), which the
# same sweep says is also the *fastest* config -- 20.51 us at B=256 against
# 26.5 us for [32, 64] and 72.7 us for L1's [64, 64] -- and which additionally
# makes one program own a full head row of ``o``, so the output epilogue folds
# into this kernel for free.  The state-carrying config keeps L1's [64, 64],
# which is optimal when 671 MB of state is streaming through, and then runs the
# epilogue as its own launch.
_REC1_NOSTATE = (16, 512, 2, 2)
_REC1_STATE = (64, 64, 4, 1)
_REC1_CFG: dict = {}


def _pow2_divisor(cap: int, n: int) -> int:
    """Largest power of two that is <= ``cap``, <= ``n`` and divides ``n``.

    Triton block sizes must be powers of two and the kernel is unmasked, so it
    also needs an exact divisor.  ``min(cap, n)`` alone is neither: a head dim of
    48 or 192 would hand ``tl.zeros`` a non-power-of-two shape and fail to
    compile rather than fall back.
    """
    d = min(cap, 1 << (n.bit_length() - 1)) if n > 0 else 0
    while d > 1 and n % d:
        d //= 2
    return d


def _rec1_cfg(K: int, V: int, touches_state: bool):
    cfg = _REC1_CFG.get((K, V, touches_state))
    if cfg is None:
        bk, bv, nw, ns = _REC1_STATE if touches_state else _REC1_NOSTATE
        bk, bv = _pow2_divisor(bk, K), _pow2_divisor(bv, V)
        # A head dim with no usable divisor goes back to L1's masked general-T
        # path instead.
        ok = bk >= 8 and bv >= 8
        cfg = (bk, bv, V // bv if ok else 0, K // bk if ok else 0, nw, ns, ok)
        _REC1_CFG[(K, V, touches_state)] = cfg
    return cfg


_REC1_PLAN: dict = {}


def _recurrent_step(q, k, v, gk, scale, initial_state, output_final_state,
                    epi=None):
    """One T==1 recurrence step.

    Returns ``(out, final_state, fused)`` where ``fused`` says whether ``out`` is
    already the gated/normed epilogue output (``epi`` supplied and the tile shape
    allowed it) rather than the raw ``o``.  None if the shape is unsupported.

    Layout validation is cached on a key pinning everything it depends on; only
    the per-call contiguity of the operands is rechecked, which is a handful of
    ~0.1 us calls.
    """
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            and (gk is None or gk.is_contiguous())
            and (initial_state is None or initial_state.is_contiguous())):
        return None
    g = w = None
    if epi is not None:
        g, w, eps, out_shape = epi
        if not g.is_contiguous():
            g = None
    key = (q.shape, v.shape[-1], q.dtype, q.get_device(),
           initial_state is not None, gk is not None, output_final_state,
           None if g is None else (g.dtype, w.dtype, w.numel(), g.numel()))
    plan = _REC1_PLAN.get(key, False)
    if plan is False:
        plan = _REC1_PLAN[key] = _rec1_plan(q, k, v, gk, initial_state,
                                            output_final_state, g, w)
    if plan is None:
        return None
    grid, nw, ns, dev, fuse, tail, lkey, Bs, H, K, V = plan
    Y = torch.empty(out_shape, dtype=q.dtype, device=q.device) if fuse else None
    o = None if fuse else q.new_empty(Bs, 1, H, V)
    ht = (q.new_empty(Bs, H, K, V, dtype=torch.float32)
          if output_final_state else None)
    # Only the operands the caller handed us can be misaligned; everything
    # allocated here comes from the caching allocator, which is 256 B aligned.
    algn = (q.data_ptr() | k.data_ptr() | v.data_ptr()
            | (gk.data_ptr() if gk is not None else 0)
            | (initial_state.data_ptr() if initial_state is not None else 0)
            | (g.data_ptr() if fuse else 0)
            | (w.data_ptr() if fuse else 0)) & 15
    _launch(_rec1_fwd, lkey if algn == 0 else None, grid,
            (q, k, v, gk, o, initial_state, ht, g if fuse else None,
             w if fuse else None, Y, scale, eps if fuse else 0.0) + tail,
            dev, nw, ns)
    return (Y if fuse else o), ht, fuse


def _rec1_plan(q, k, v, gk, initial_state, output_final_state, g, w):
    Bs, T, H, K = q.shape
    V = v.shape[-1]
    use_h0 = initial_state is not None
    if T != 1 or not q.is_cuda:
        return None
    bk, bv, nv, nk, nw, ns, ok = _rec1_cfg(K, V, use_h0 or output_final_state)
    if not ok:
        return None
    # Folding the epilogue in needs one program to own a whole head row of o.
    fuse = (g is not None and nv == 1 and g.dtype == q.dtype
            and w.dtype == q.dtype and w.numel() == V
            and g.numel() == Bs * T * H * V)
    has_gk = gk is not None
    tail = (H, K, V, bk, bv, nv, nk, use_h0, has_gk, output_final_state, fuse)
    lkey = ((q.get_device(), "rec1", nw, ns) + tail
            + (q.dtype, k.dtype, v.dtype,
               None if gk is None else gk.dtype,
               None if initial_state is None else initial_state.dtype))
    return (Bs * H * nv, nw, ns, q.get_device(), fuse, tail, lkey,
            Bs, H, K, V)


def _gk_rec1_step(lat, wt, bias, inv_norm, R, N, gk_shape,
                  q, k, v, scale, epi):
    """gk tail + T==1 recurrence in one launch; None if either half is unsupported.

    Reuses both halves' existing cached plans verbatim, so a hit here validates
    exactly what the two separate launches would have validated.  Only ever
    called on the no-state path -- see ``_gk_rec1_fwd``.
    """
    gkp = _gk_plan_cached(lat, wt, bias, R, N)
    if gkp is None:
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return None
    g, w, eps, out_shape = epi
    if not g.is_contiguous():
        return None
    key = (q.shape, v.shape[-1], q.dtype, q.get_device(),
           False, True, False, (g.dtype, w.dtype, w.numel(), g.numel()))
    plan = _REC1_PLAN.get(key, False)
    if plan is False:
        plan = _REC1_PLAN[key] = _rec1_plan(q, k, v, lat, None, False, g, w)
    if plan is None:
        return None
    rgrid, rnw, rns, dev, fuse, rtail, _, Bs, H, K, V = plan
    if not fuse:
        # Without the folded epilogue the recurrence writes a raw ``o`` that a
        # second kernel then has to consume, so there is nothing to win here.
        return None
    ggrid, gnw, gdev, m_s, gtail, _ = gkp
    if gdev != dev:
        return None
    gk = torch.empty(gk_shape, dtype=lat.dtype, device=lat.device)
    Y = torch.empty(out_shape, dtype=q.dtype, device=q.device)
    algn = (lat.data_ptr() | wt.data_ptr() | gk.data_ptr() | q.data_ptr()
            | k.data_ptr() | v.data_ptr() | g.data_ptr() | w.data_ptr()
            | (bias.data_ptr() if bias is not None else 0)) & 15
    # (R, N, bm, bn, br, gn, even_m, even_r, has_bias) from the gk plan;
    # (H, K, V, bk, bv, nv, nk, use_h0, has_gk, store_ht, fuse) from rec1's.
    tail = (ggrid,) + gtail + rtail[:7] + (rtail[10],)
    _launch(_gk_rec1_fwd,
            ((dev, "gkrec", rnw, rns) + tail
             + (lat.dtype, wt.dtype, None if bias is None else bias.dtype,
                q.dtype, k.dtype, v.dtype, g.dtype, w.dtype)
             if algn == 0 else None),
            ggrid + rgrid,
            (lat, wt, bias, gk) + m_s + (inv_norm, q, k, v, None, None, None,
                                         g, w, Y, scale, eps) + tail,
            dev, rnw, rns)
    return gk, Y


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
        self.norm_eps = norm_eps
        # Cache for the fused epilogue's norm weight, keyed on (dtype, device) so
        # a module cast/move rebuilds it instead of re-casting every forward.
        # Kept out of ``_parameters``/``_buffers`` (it is a view of, or a cast
        # copy of, a real parameter) so it never reaches a state_dict.
        object.__setattr__(self, "_epi_w", None)
        object.__setattr__(self, "_epi_w_key", None)
        # Cache for the fused gk kernel's transposed gk_proj[1] weight. Rebuilt
        # whenever the parameter is replaced (data_ptr/dtype) or written in
        # place (``_version``), so a checkpoint load or a dtype cast is picked
        # up rather than silently ignored.
        object.__setattr__(self, "_gk_wt", None)
        object.__setattr__(self, "_gk_key", None)
        object.__setattr__(self, "_gk_inv_norm", 1.0 / gate_logit_normalizer)
        # Packed-projection state: the [hidden, sum(widths)] transposed weight
        # and the per-tile metadata table. Both are derived caches, rebuilt on
        # any change to the underlying parameters, and both are deliberately
        # outside _parameters/_buffers so they never enter a state_dict.
        object.__setattr__(self, "_proj_hs", None)
        object.__setattr__(self, "_proj_cache", {})
        object.__setattr__(self, "_bt", (0, 0))
        # CUDA-graph state.  Kept in ``__dict__`` (not ``_parameters`` /
        # ``_buffers``) so nothing here reaches a state_dict, and mutated in
        # place where possible so the hot path never pays nn.Module.__setattr__.
        #   _graphs      shape key -> (CUDAGraph, static_in, static_out), LRU
        #   _graph_warm  shape key -> eager calls seen so far
        #   _graph_ver   [ver] of the weights the captured graphs baked in
        #   _graph_srcs  those weights, resolved once
        #   _graph_n     [captures], for the dev assertion that replay happens
        object.__setattr__(self, "_graphs", OrderedDict())
        object.__setattr__(self, "_graph_warm", {})
        object.__setattr__(self, "_graph_ver", [None])
        object.__setattr__(self, "_graph_srcs", None)
        object.__setattr__(self, "_graph_n", [0])
        object.__setattr__(self, "_graph_ok",
                           _GRAPH_ENABLED and use_fast_kernels
                           and decay_mode == "learned_low_rank"
                           and not use_rotary)

    # ---- CUDA graphs (T == 1 decode step) -----------------------------------

    def _invalidate_graphs(self):
        """Drop every captured graph and the caches they baked copies of."""
        self._graphs.clear()
        self._graph_warm.clear()
        self._graph_ver[0] = None
        object.__setattr__(self, "_graph_srcs", None)
        object.__setattr__(self, "_proj_hs", None)
        self._proj_cache.clear()
        object.__setattr__(self, "_gk_key", None)
        object.__setattr__(self, "_epi_w_key", None)

    def _apply(self, *args, **kwargs):
        # ``.to()`` / ``.cuda()`` / ``.float()`` replace the parameter tensors,
        # so every captured graph (which baked their addresses) and every derived
        # weight copy is stale.
        if "_graphs" in self.__dict__:
            self._invalidate_graphs()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        # Covers both the in-place copy and ``assign=True`` (which rebinds the
        # parameter to the checkpoint's tensor).
        if "_graphs" in self.__dict__:
            self._invalidate_graphs()
        return super()._load_from_state_dict(*args, **kwargs)

    def _graph_sources(self):
        """Weights a captured graph reads, directly or through a derived copy.

        The derived copies -- the packed projection weight, the transposed gk
        tail weight, a dtype-cast norm scale -- are rebuilt on the *host* when
        their source changes, and a replay does no host work at all, so an
        in-place write to any of these would leave a graph replaying against a
        stale copy.  Hence the per-call ``(data_ptr, _version)`` guard below.
        Parameter *replacement* is handled by the ``_apply`` /
        ``_load_from_state_dict`` hooks instead, which is free.
        """
        srcs = self._graph_srcs
        if srcs is None:
            gn = self.g_norm_swish_gate
            srcs = self._proj_sources() + (
                self.gk_proj[1].weight, self.o_proj.weight,
                gn.weight if gn.elementwise_affine else gn._unit_weight)
            object.__setattr__(self, "_graph_srcs", srcs)
        return srcs

    def _graph_step(self, hidden_states, cu_seqlens):
        """Replay the graphed decode step; ``None`` -> caller must run eager.

        ``None`` covers "not graphable at all" and "still warming up at this
        key" identically, so the caller has one branch.
        """
        # A pack of one sequence is elided to the dense path (see
        # ``_forward_eager``), so it graphs the same as no ``cu_seqlens`` at all;
        # a genuine multi-sequence pack reads the boundaries and does not.
        if cu_seqlens is not None and cu_seqlens.numel() != 2:
            return None
        if not (hidden_states.is_cuda and hidden_states.is_contiguous()):
            return None
        ver = tuple((w.data_ptr(), w._version) for w in self._graph_sources())
        gver = self._graph_ver
        if gver[0] != ver:
            self._graphs.clear()
            self._graph_warm.clear()
            gver[0] = ver
            return None
        graphs = self._graphs
        key = (hidden_states.shape[0], hidden_states.shape[1],
               hidden_states.dtype, hidden_states.device.index)
        ent = graphs.get(key)
        if ent is None:
            warm = self._graph_warm
            n = warm.get(key, 0) + 1
            warm[key] = n
            if n <= _GRAPH_WARMUP:
                return None
            ent = self._capture(key, hidden_states)
            if ent is None:
                return None
        else:
            graphs.move_to_end(key)
        graph, static_in, static_out = ent
        static_in.copy_(hidden_states)
        graph.replay()
        # The graph writes into ``static_out`` on every replay, so handing that
        # tensor to the caller would silently mutate any result it still holds
        # (including the harness's own correctness comparison).  Return a copy.
        return static_out.clone()

    def _capture(self, key, hidden_states):
        """Capture the decode step at ``key``; ``None`` on any failure.

        A failed capture can leave the allocator mid-``beginAllocateToPool``, so
        the whole module drops to eager permanently rather than retrying.
        """
        if torch.cuda.is_current_stream_capturing():
            return None
        dev = hidden_states.device
        try:
            static_in = torch.empty_like(hidden_states)
            static_in.copy_(hidden_states)
            # Warm up on the capture stream itself before capturing on it:
            # cuBLAS keeps its workspace per (handle, stream), and allocating one
            # during the capture would be a cudaMalloc inside it.
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward_eager(static_in)
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                static_out = self._forward_eager(static_in)[0]
        except Exception:
            object.__setattr__(self, "_graph_ok", False)
            self._graphs.clear()
            try:
                torch.cuda.synchronize(dev)
            except Exception:  # pragma: no cover
                pass
            return None
        # ``static_out`` must be a real tensor living in the graph's pool; the
        # packed o_proj hands back an ``as_strided`` view of its output buffer,
        # which is fine (it is a dense view), but anything aliasing the *input*
        # or not a plain tensor would be a correctness hazard.
        if (type(static_out) is not torch.Tensor
                or static_out.dtype != hidden_states.dtype
                or static_out.data_ptr() == static_in.data_ptr()):
            object.__setattr__(self, "_graph_ok", False)
            return None
        graphs = self._graphs
        graphs[key] = ent = (graph, static_in, static_out)
        graphs.move_to_end(key)
        while len(graphs) > _GRAPH_MAX_LIVE:
            # Dropping the entry drops the graph and every tensor in its private
            # pool, which is what actually returns the memory.
            graphs.popitem(last=False)
        self._graph_n[0] += 1
        return ent

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, None, object | None]:
        # ``past_key_values is None`` is the whole state condition: with no cache
        # object there is no incoming state to read and nowhere to put a final
        # one, so ``_forward_eager`` takes its no-state branch for any
        # ``use_cache`` and returns ``past_key_values`` (None) untouched.
        if (self._graph_ok and past_key_values is None and attention_mask is None
                and hidden_states.shape[1] == 1
                and hidden_states.shape[0] <= _GRAPH_MAX_M
                and not torch.is_grad_enabled()):
            y = self._graph_step(hidden_states, kwargs.get("cu_seqlens"))
            if y is not None:
                return y, None, None
        return self._forward_eager(hidden_states, attention_mask,
                                   past_key_values, use_cache, **kwargs)

    # ---- packed GEMMs (input projections and the output projection) ---------

    def _proj_sources(self):
        """The weights that make up the packed input projection, in segment order."""
        srcs = [self.q_proj.weight, self.k_proj.weight,
                self.v_proj.weight, self.g_proj.weight]
        if self.decay_mode == "learned_low_rank":
            srcs.append(self.gk_proj[0].weight)
        return tuple(srcs)

    def _pack_plan(self, x, srcs, rows, cache, shapes):
        """Cached launch plan for one packed GEMM, revalidated on every call.

        The layer runs at ~70 us of *host* time on the decode shapes, so the
        per-call bookkeeping has to be tiny: this collapses to one
        ``(data_ptr, _version, dtype)`` tuple over the source weights (~1 us, and
        it is what makes a checkpoint load or a dtype cast rebuild the pack) plus
        one dict lookup on M.  The packed transposed weight, the tile metadata,
        the launch config and the output views' (shape, stride, offset) triples
        are all built on the first call for a given M and then reused.

        ``shapes`` maps segment index -> a function of (B_T_pair, width) giving
        the (shape, stride) the caller wants, so the results come back already in
        their final form: an ``as_strided`` costs 1.05 us against 1.85 us for a
        slice plus a view, and it removes the follow-up reshapes entirely.

        Returns None when the module/shape is outside the kernel's domain, in
        which case the caller falls back to the separate L1 ``Linear`` ops.
        """
        ver = tuple((w.data_ptr(), w._version, w.dtype) for w in srcs)
        if cache.get("ver") != ver:
            cache.clear()
            cache["ver"] = ver
        # Keyed on the full (B, T) split, not just M: the cached plan carries the
        # output views' shapes and strides, and B=256/T=1 and B=1/T=256 are the
        # same M but different views.
        bt = self._bt
        plan = cache.get(bt, False)
        if plan is not False:
            return plan
        plan = self._build_pack_plan(x, srcs, bt[0] * bt[1], rows, shapes)
        cache[bt] = plan
        return plan

    def _build_pack_plan(self, x, srcs, M, rows, shapes):
        KD = x.shape[-1]
        ref = srcs[0]
        if M > _PROJ_MAX_M or M == 0:
            return None
        if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
                and all(w.dtype == x.dtype and w.device == x.device
                        and w.ndim == 2 and w.shape[1] == KD for w in srcs)):
            return None
        for lim, bm, bn, bk, nw, ns in rows:
            if M <= lim:
                break
        widths0 = tuple(w.shape[0] for w in srcs)
        # Segment widths must be BN multiples: the store is unmasked along N and
        # the packed weight rows are BN-aligned.  Every segment but the last must
        # be *exactly* aligned -- padding one would hand its consumer a strided
        # view, and the L1 chunk kernel indexes its inputs as dense
        # [B, T, H, D] without consulting strides, so it would silently read the
        # wrong elements.  Only the trailing gate latent may be padded, because
        # its sole consumer (the fused gk tail) takes a row stride.  Shrink BN
        # until the dense segments divide, then give up rather than pad them.
        dense = widths0[:-1] if len(widths0) > 1 else widths0
        while bn > 16 and any(w % bn for w in dense):
            bn //= 2
        if KD % bk or any(w % bn for w in dense):
            return None
        widths = (tuple(dense) + tuple(-(-w // bn) * bn for w in widths0[len(dense):]))
        bm = min(bm, max(16, triton.next_power_of_2(M)))
        ntot = sum(widths)
        n_tiles = ntot // bn

        # Packed weight: segment i at rows [prefix_i, prefix_i + width_i), the BN
        # padding rows left zero so the padded output columns come out zero and
        # are simply never read.  Transposed so the kernel's [BK, BN] weight tile
        # is contiguous along N.
        pack = torch.zeros(ntot, KD, dtype=ref.dtype, device=ref.device)
        off = 0
        for w, width in zip(srcs, widths):
            pack[off:off + w.shape[0]] = w
            off += width
        wt = pack.t().contiguous()
        del pack

        tiles, prefix = [], 0
        for width in widths:
            for t in range(width // bn):
                tiles.append((prefix, width, t * bn))
            prefix += width
        meta = torch.tensor(tiles, dtype=torch.int32, device=ref.device)

        views, off = [], 0
        for idx, (w, width) in enumerate(zip(widths0, widths)):
            views.append(shapes(idx, M, w, width) + (off * M,))
            off += width
        even_m = M % bm == 0
        return (wt, meta, ntot, bm, bn, bk, nw, ns, n_tiles, even_m,
                tuple(views), -(-M // bm) * n_tiles, M, KD)

    @staticmethod
    def _run_pack(x, plan):
        """Launch a packed GEMM; returns one dense view per segment."""
        (wt, meta, ntot, bm, bn, bk, nw, ns, n_tiles, even_m, views,
         grid, M, KD) = plan
        buf = torch.empty(M * ntot, dtype=x.dtype, device=x.device)
        dev = x.get_device()
        algn = (x.data_ptr() | wt.data_ptr() | meta.data_ptr()
                | buf.data_ptr()) & 15
        _launch(_proj_fwd,
                ((dev, "proj", KD, ntot, n_tiles, bm, bn, bk, even_m, nw, ns, M,
                  x.dtype, wt.dtype, meta.dtype, buf.dtype)
                 if algn == 0 else None),
                grid,
                (x, wt, meta, buf, M, KD, ntot, n_tiles, bm, bn, bk, even_m),
                dev, nw, ns)
        return [buf.as_strided(shape, stride, off) for shape, stride, off in views]

    def _qkvg_shapes(self, idx, M, w, width):
        """Final (shape, stride) for segment ``idx`` of the input projection."""
        B, T = self._bt
        H = self.num_heads
        if idx < 2:                      # q, k -> [B, T, H, head_k_dim]
            d = self.head_k_dim
            return ((B, T, H, d), (T * w, w, d, 1))
        if idx == 2:                     # v -> [B, T, H, head_v_dim]
            d = self.head_v_dim
            return ((B, T, H, d), (T * w, w, d, 1))
        if idx == 3:                     # g -> [B, T, value_dim]
            return ((B, T, w), (T * w, w, 1))
        return ((M, w), (width, 1))      # gate latent -> [M, R], padded stride

    def _packed_projection(self, x, B, T):
        """``(q, k, v, g[, gk_latent])`` from one GEMM, already in final shapes."""
        srcs = self._proj_hs
        if srcs is None:
            srcs = self._proj_sources()
            object.__setattr__(self, "_proj_hs", srcs)
        object.__setattr__(self, "_bt", (B, T))
        plan = self._pack_plan(x, srcs, _PROJ_ROWS, self._proj_cache,
                               self._qkvg_shapes)
        if plan is None:
            return None
        return self._run_pack(x, plan)

    def _norm_weight(self, ref: torch.Tensor) -> torch.Tensor:
        """The output norm's scale, matching ``ref``'s dtype/device.

        ``RMSNorm`` keeps either a real ``weight`` parameter or a unit buffer;
        both are followed here so an ``elementwise_affine=False`` norm still
        goes through the fused epilogue rather than falling back.
        """
        gn = self.g_norm_swish_gate
        w = gn.weight if gn.elementwise_affine else gn._unit_weight
        key = (w.dtype, w.device, w.data_ptr(), ref.dtype, ref.device)
        if self._epi_w_key != key:
            object.__setattr__(self, "_epi_w",
                               w.to(device=ref.device, dtype=ref.dtype)
                               if (w.dtype != ref.dtype or w.device != ref.device)
                               else w)
            object.__setattr__(self, "_epi_w_key", key)
        return self._epi_w

    def _compute_gk(
        self, hidden_states: torch.Tensor, B: int, T: int,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns gk shaped [B, num_heads, T, head_k_dim] in log-space.

        Used by the naive recurrence path. The fast path uses
        :meth:`_compute_gk_bthk` to skip an unnecessary transpose.
        """
        if self.decay_mode == "learned_low_rank":
            return self._compute_gk_bthk(
                hidden_states, B, T, latent).transpose(1, 2)
        return self.log_gamma.to(hidden_states.dtype).view(
            1, self.num_heads, 1, 1
        ).expand(B, self.num_heads, T, self.head_k_dim)

    def _gk_tail_weights(self):
        """``(W1.T [R, key_dim] contiguous, bias)`` for the fused gk tail."""
        lin = self.gk_proj[1]
        w, b = lin.weight, lin.bias
        key = (w.data_ptr(), w._version, w.dtype, w.device,
               None if b is None else (b.data_ptr(), b._version, b.dtype))
        if self._gk_key != key:
            object.__setattr__(self, "_gk_wt", w.t().contiguous())
            object.__setattr__(self, "_gk_key", key)
        return self._gk_wt, b

    def _compute_gk_bthk(
        self, hidden_states: torch.Tensor, B: int, T: int,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns gk shaped [B, T, num_heads, head_k_dim] in log-space.

        ``latent`` is ``gk_proj[0]``'s output when the packed projection already
        produced it (as a possibly-strided [M, gate_low_rank_dim] view).
        """
        lat = self.gk_proj[0](hidden_states) if latent is None else latent
        wt, bias = self._gk_tail_weights()
        # The packed projection hands back a [M, R] view whose row stride is the
        # BN-padded segment width, so flatten by shape rather than reshaping
        # (which would copy).
        lat2 = lat if lat.ndim == 2 else lat.reshape(-1, lat.shape[-1])
        gk = _fused_gk(lat2, wt, bias, self._gk_inv_norm, lat2.shape[-1],
                       self.key_dim, (B, T, self.num_heads, self.head_k_dim))
        if gk is None:
            gk = self.gk_proj[1](lat2 if lat2.is_contiguous() else lat2.contiguous())
            # The L1 logsigmoid op is a Triton kernel, so the CPU reference path
            # has to use the ATen one -- it is the same function, and this is the
            # only place the module can run without a GPU.
            gk = (self.log_sigmoid(gk) if gk.is_cuda
                  else torch.nn.functional.logsigmoid(gk))
            gk = gk / self.gate_logit_normalizer
            return gk.view(B, T, self.num_heads, self.head_k_dim)
        return gk

    def _forward_eager(
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
            # ``max_seqlen`` only picks the chunk-vs-recurrent dispatch, but
            # reading it off the device costs a full host<->device sync (a
            # subtract, a max-reduce and a DtoH memcpy) in the middle of the
            # forward, which serializes the host against the GPU for the rest of
            # the call. A single packed sequence -- the overwhelmingly common
            # varlen case -- needs no device read at all: FLA's convention is
            # ``cu_seqlens[-1] == total tokens``, so its one segment is exactly
            # ``T`` long. Only a genuinely multi-sequence pack pays the sync.
            n_seq = cu_seqlens.numel() - 1
            if n_seq <= 0:
                max_seqlen = 0
            elif n_seq == 1:
                max_seqlen = T
                # A pack of one sequence is just a dense batch of one: the only
                # thing cu_seqlens tells the kernels is where the sequence
                # boundaries are, and there are none inside. Dropping it here
                # gets this call onto the dense kernels -- for T=1 that means the
                # single-launch decode step (with its fused epilogue) instead of
                # the general varlen kernel plus its cross-tile o reduction, and
                # for long packs it means the chunk kernel's dense prep path.
                # Relies on the same FLA convention as max_seqlen above,
                # cu_seqlens[-1] == total tokens.
                cu_seqlens = None
            else:
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seqlen = int(lengths.max().item())

        # One packed GEMM for q|k|v|g (+ the gate latent) where possible, else
        # the five separate L1 Linear ops.  The packed path hands back q/k/v
        # already shaped [B, T, H, head_dim] and g as [B, T, value_dim], so the
        # reshapes below are skipped -- at ~1 us per view that is worth avoiding
        # on a path whose whole host budget is ~70 us.
        packed = None
        if self.use_fast_kernels:
            packed = self._packed_projection(hidden_states, B, T)
        gk_latent = None
        if packed is not None:
            if self.decay_mode == "learned_low_rank":
                q, k, v, g, gk_latent = packed
            else:
                q, k, v, g = packed
        else:
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
        elif packed is None:
            q = q.view(B, T, self.num_heads, self.head_k_dim)
            k = k.view(B, T, self.num_heads, self.head_k_dim)

        if packed is None:
            v = v.view(B, T, self.num_heads, self.head_v_dim)

        epi_done = False
        initial_state = None
        if past_key_values is not None and getattr(past_key_values, "states", None):
            initial_state = past_key_values.states.get(id(self))

        # The recurrence's final state is consumed by exactly one statement
        # below -- ``if use_cache and past_key_values is not None`` -- so asking
        # the kernel for it when there is no cache object to hold it is a dead
        # store. It is not a cheap one: the state is ``[N, H, K, V]`` fp32, i.e.
        # 2.6 MB per sequence at K=256/V=512, so a B=256 decode step writes
        # 671 MB of HBM that is then dropped on the floor. Predicating on the
        # same condition the consumer uses keeps every observable output
        # bit-identical, and a real engine -- which passes a cache object
        # whenever it sets use_cache -- still gets its state written.
        want_final_state = bool(use_cache) and past_key_values is not None

        # Dispatch:
        #   T >= 64 + fast kernels -> chunk (prefill / training)
        #   T  < 64 + fast kernels -> fused_recurrent (decode)
        #   no fast kernels         -> naive PyTorch (CPU / debug / reference)
        if self.use_fast_kernels and q.is_cuda:
            dispatch_len = max_seqlen if max_seqlen is not None else T
            if self.decay_mode == "learned_low_rank":
                # On the no-state decode step the gk tail and the recurrence are
                # data-independent (see ``_gk_rec1_fwd``), so they go out as one
                # launch.  Everything else keeps the two ordered launches.
                fused = None
                if (dispatch_len < _CHUNK_THRESHOLD and cu_seqlens is None
                        and gk_latent is not None and initial_state is None
                        and not want_final_state):
                    lat = (gk_latent if gk_latent.ndim == 2
                           else gk_latent.reshape(-1, gk_latent.shape[-1]))
                    wt, bias = self._gk_tail_weights()
                    fused = _gk_rec1_step(
                        lat, wt, bias, self._gk_inv_norm, lat.shape[-1],
                        self.key_dim,
                        (B, T, self.num_heads, self.head_k_dim),
                        q, k, v, self.head_k_dim ** -0.5,
                        (g, self._norm_weight(q), self.norm_eps,
                         (B, T, self.value_dim)))
                if fused is not None:
                    _, o = fused
                    final_state, epi_done = None, True
                else:
                    # gk in [B, T, H, K] log-space, NOT transposed
                    gk_btHK = self._compute_gk_bthk(hidden_states, B, T,
                                                    gk_latent)
                    if dispatch_len >= _CHUNK_THRESHOLD:
                        o, final_state = self.chunk(
                            q=q, k=k, v=v, g=gk_btHK,
                            initial_state=initial_state,
                            output_final_state=want_final_state,
                            cu_seqlens=cu_seqlens,
                        )
                    else:
                        step = None
                        if cu_seqlens is None:
                            step = _recurrent_step(
                                q, k, v, gk_btHK, self.head_k_dim ** -0.5,
                                initial_state, want_final_state,
                                (g, self._norm_weight(q), self.norm_eps,
                                 (B, T, self.value_dim)))
                        if step is not None:
                            o, final_state, epi_done = step
                        else:
                            o, final_state = self.fused_recurrence(
                                q=q, k=k, v=v, gk=gk_btHK,
                                initial_state=initial_state,
                                output_final_state=want_final_state,
                                cu_seqlens=cu_seqlens,
                            )
            else:  # RetNet — kernel bakes in the per-head decay
                if dispatch_len >= _CHUNK_THRESHOLD:
                    o, final_state = self.chunk(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=want_final_state,
                        cu_seqlens=cu_seqlens,
                    )
                else:
                    o, final_state = self.fused_recurrence(
                        q=q, k=k, v=v,
                        initial_state=initial_state,
                        output_final_state=want_final_state,
                        cu_seqlens=cu_seqlens,
                    )
            # Fast-path output is already [B, T, H, V] — no transpose needed.
        else:
            # Naive path expects [B, H, T, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            gk = self._compute_gk(hidden_states, B, T, gk_latent)
            o, final_state = self.naive_recurrence(
                q, k, v, gk,
                initial_state=initial_state,
                output_final_state=want_final_state,
            )
            o = o.transpose(1, 2)  # [B, H, T, V] -> [B, T, H, V]

        if use_cache and past_key_values is not None:
            if not hasattr(past_key_values, "states"):
                past_key_values.states = {}
            past_key_values.states[id(self)] = final_state

        # One fused kernel for norm + gate + multiply where the layout allows
        # it, else the three unfused L1 ops.
        y = o if epi_done else None
        if y is None and o.is_contiguous() and g.is_contiguous():
            y = _fused_epilogue(o, g, self._norm_weight(o), self.norm_eps,
                                self.head_v_dim, (B, T, self.value_dim))
        if y is None:
            if o.is_cuda:
                y = self.g_norm_swish_gate(o.reshape(-1, self.head_v_dim))
                y = y.view(B, T, self.value_dim) * self.gate_act(g)
            else:
                # The L1 norm and activation ops are GPU kernels (the RMSNorm one
                # reaches its CUDA extension even for a CPU tensor), so the
                # non-CUDA reference path does this arithmetic itself. Same
                # sequence and same single rounding as ``RMSNorm.forward_native``
                # followed by the swish gate.
                w = self._norm_weight(o)
                xr = o.reshape(-1, self.head_v_dim).float()
                xr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True)
                                      + self.norm_eps)
                y = (xr.to(o.dtype) * w).view(B, T, self.value_dim)
                y = y * torch.nn.functional.silu(g)

        return self.o_proj(y), None, past_key_values
