"""PairFormer stack for AlphaFold3 -- launch-overhead-collapsed rewrite.

The captured shape is tiny (N_token=16, so z is [1,16,16,128] = 64 KB and s is
[1,16,384] = 12 KB) while the stack is 48 blocks deep.  Every GEMM is a
~256x128x128 matmul that costs microseconds of arithmetic, so the eager
baseline is essentially pure per-op dispatch cost: ~10k kernel launches per
forward at ~9 us of launch overhead each (86 ms measured).  Total weight
traffic is only ~294 MB, i.e. tens of microseconds of real work.

Three levers, in order of size:

1. Capture the entire 48-block forward into a CUDA graph and replay it.  The
   captured shape is single and static, and the harness hands us fresh input
   tensor objects at new addresses on every call, so instead of keying a graph
   cache on ``data_ptr()`` we copy the four inputs into *static* buffers on each
   call and replay one graph.  Four small copies replace ~10k launches.

2. Collapse the kernel count *inside* the graph.  Graph-replay node dispatch is
   ~0.75-1.2 us on B200, so with ~170 eager ops per block a graph alone would
   still cost ~10 ms.  Each block is rewritten as ~24 Triton kernels: the four
   TriangleMultiplication projections become one packed GEMM, q/k/v/g/triangle
   projections become one packed GEMM, every weight is pre-transposed and packed
   at first forward so no runtime permute/copy survives, and the
   layernorm/sigmoid-gate/mask/residual chains are fused into the neighbouring
   GEMM kernels.  Programmatic Dependent Launch (PDL) is wired through every
   kernel so each one prefetches its weights while its producer drains.

3. Run the two tracks concurrently *inside that one graph*.  A block is
   ``z_i = pair_stack(z_{i-1})`` and then ``s_i = s_{i-1} + attn_pair_bias(
   s_{i-1}, z_i) + transition(...)``: the pair track never reads s, so the
   48-block z chain is the only real critical path and each block's s update is
   a tap off it.  ``_run`` forks the s track onto a side stream with a per-block
   event, which makes it a parallel *branch of the same graph*, and joins once
   at the end.  Only ``ZSB`` (= layer_norm_z(z_i)) crosses from the z branch to
   the s branch, so only it needs a per-block slot.  Nothing crosses the other
   way, so there is no per-block join.  That halves the critical path; what
   is left is the z chain: 17 nodes per block whose cost is set mostly by CTA
   count, hence the tile choices at the bottom of the kernel section.

bf16 rounding is emulated at every point the reference materializes a bf16
tensor (``_rb`` below), and every layernorm runs its reduction in fp32 exactly
like the reference, so numerics track the baseline through all 48 blocks.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.language.extra import libdevice

try:  # PDL intrinsics (Triton >= 3.5); harmless no-ops without launch_pdl=True
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # pragma: no cover
    _HAS_PDL = False

    @triton.jit
    def gdc_wait():
        pass

    @triton.jit
    def gdc_launch_dependents():
        pass


__targets__ = ["PairFormerStack"]

_EPS = 1e-5

# Debug switches, read once at import (all default to the fast path).  These
# exist because the failure mode of this operator is a handful of one-ULP
# elements, and being able to bisect graph-vs-eager / PDL-vs-not without editing
# the kernel is what made it tractable.
_NO_PDL = os.environ.get("FK_NO_PDL") == "1"
_NO_GRAPH = os.environ.get("FK_NO_GRAPH") == "1"
_FORCE_EAGER = os.environ.get("FK_FORCE_EAGER") == "1"
# Run the single (s) track serially after the pair (z) track instead of on a
# forked branch of the graph.  Same numerics, ~2x slower; kept as a bisect
# switch because a scheduling bug and a numerics bug look identical here.
_NO_OVERLAP = os.environ.get("FK_NO_OVERLAP") == "1"


# ---------------------------------------------------------------------------
# Parameter containers.  Names/shapes must match the baseline module tree
# exactly: the harness builds baseline + candidate independently and then does
# ``candidate.load_state_dict(baseline.state_dict(), strict=False)`` to share
# weights, so a renamed or reshaped parameter silently leaves us with different
# random weights than the reference.
# ---------------------------------------------------------------------------
class _Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class _LayerNorm(nn.Module):
    def __init__(self, n: int, create_scale: bool = True, create_offset: bool = True):
        super().__init__()
        self.normalized_shape = (n,)
        if create_scale:
            self.weight = nn.Parameter(torch.ones(n))
        else:
            self.register_parameter("weight", None)
        if create_offset:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, w, b, _EPS).to(x.dtype)


class _TriMul(nn.Module):
    def __init__(self, c_z: int, c_hidden: int, outgoing: bool):
        super().__init__()
        self._outgoing = outgoing
        self.linear_a_p = _Linear(c_z, c_hidden, bias=False)
        self.linear_a_g = _Linear(c_z, c_hidden, bias=False)
        self.linear_b_p = _Linear(c_z, c_hidden, bias=False)
        self.linear_b_g = _Linear(c_z, c_hidden, bias=False)
        self.linear_g = _Linear(c_z, c_z, bias=False)
        self.linear_z = _Linear(c_hidden, c_z, bias=False)
        self.layer_norm_in = _LayerNorm(c_z)
        self.layer_norm_out = _LayerNorm(c_hidden)


class _MHA(nn.Module):
    def __init__(self, c_q, c_k, c_v, c_hidden, no_heads, gating=True, q_bias=False):
        super().__init__()
        self.linear_q = _Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = _Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = _Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = _Linear(c_hidden * no_heads, c_q, bias=False)
        self.linear_g = _Linear(c_q, c_hidden * no_heads, bias=False) if gating else None


class _TriAtt(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting):
        super().__init__()
        self.starting = starting
        self.layer_norm = _LayerNorm(c_in)
        self.linear_z = _Linear(c_in, no_heads, bias=False)
        self.mha = _MHA(c_in, c_in, c_in, c_hidden, no_heads)


class _SwiGLU(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.linear_a = _Linear(c_in, c_out, bias=False)
        self.linear_b = _Linear(c_in, c_out, bias=False)


class _Transition(nn.Module):
    def __init__(self, c_in, n):
        super().__init__()
        self.layer_norm = _LayerNorm(c_in)
        self.swiglu = _SwiGLU(c_in, n * c_in)
        self.linear_out = _Linear(n * c_in, c_in, bias=False)


class _PairBlock(nn.Module):
    def __init__(self, c_z, c_hidden_mul, c_hidden_pair_att, no_heads_pair, transition_n):
        super().__init__()
        self.tri_mul_out = _TriMul(c_z, c_hidden_mul, True)
        self.tri_mul_in = _TriMul(c_z, c_hidden_mul, False)
        self.tri_att_start = _TriAtt(c_z, c_hidden_pair_att, no_heads_pair, True)
        self.tri_att_end = _TriAtt(c_z, c_hidden_pair_att, no_heads_pair, False)
        self.pair_transition = _Transition(c_z, transition_n)


class _AttnPairBias(nn.Module):
    def __init__(self, c_q, c_z, c_hidden, no_heads):
        super().__init__()
        self.layer_norm_a = _LayerNorm(c_q)
        self.layer_norm_z = _LayerNorm(c_z)
        self.linear_z = _Linear(c_z, no_heads, bias=False)
        self.mha = _MHA(c_q, c_q, c_q, c_hidden, no_heads, gating=True, q_bias=True)


class PairFormerBlock(nn.Module):
    def __init__(self, c_s, c_z, c_hidden_pair_bias, no_heads_pair_bias, c_hidden_mul,
                 c_hidden_pair_att, no_heads_pair, transition_n, pair_dropout=0.0,
                 fuse_projection_weights=False, inf=1e9):
        super().__init__()
        self.inf = inf
        self.pair_stack = _PairBlock(c_z, c_hidden_mul, c_hidden_pair_att,
                                     no_heads_pair, transition_n)
        self.attn_pair_bias = _AttnPairBias(c_s, c_z, c_hidden_pair_bias,
                                            no_heads_pair_bias)
        self.single_transition = _Transition(c_s, transition_n)


# ---------------------------------------------------------------------------
# Triton kernels.
#
# The 48-block stack is chaotic: a *single* one-ULP perturbation of one input
# element moves the final output far outside the harness tolerance (measured:
# matched_ratio 0.41 on z).  So every kernel here must be **bit-exact** with the
# reference op sequence, not merely accurate.  What that costs:
#   * `_rb` rounds through bf16 at every point the reference materializes a bf16
#     tensor.
#   * `_exp` is libdevice `__nv_expf`; Triton's `tl.exp` is `ex2.approx` and
#     differs from ATen's `expf` on ~50% of inputs in the last fp32 bit.
#   * `_silu` is ATen's `x / (1 + exp(-x))`, not `x * sigmoid(x)`.
#   * `_lnstats` replicates ATen `vectorized_layer_norm_kernel` exactly: a
#     4-element sequential Welford per emulated lane, then the 5-level intra-warp
#     shuffle butterfly.  A plain two-pass mean/variance disagrees with it on
#     ~8e-6 of elements, which is enough to fail.  (Verified bit-exact for
#     C=128; C=384 has a residual 1-ULP rstd disagreement, so the two 384-wide
#     layernorms on the single track are left to ATen inside the graph.)
#   * softmax sums with ATen's XOR-butterfly order (offsets 8,4,2,1) -- verified
#     against ATen by inverting an fp32 softmax and recovering its divisor.
#   * `tl.reshape`/`tl.split` on a *dot-derived* (MMA) layout permutes values
#     within the row, which silently reorders the Welford lanes and shifts the
#     mean by one ULP on every row.  So no kernel here has a layernorm whose
#     input or output is on the same dataflow as a `tl.dot` operand: the
#     layernorms live in dot-free kernels or read their input back from memory
#     (store, `debug_barrier`, reload) so it is load-derived and their output
#     terminates in a store.
#   * the triangle einsum is a *batched* `tl.dot` over the channel axis, matching
#     the reference's bmm; an accumulate loop over the contracted index sums the
#     16 terms in a different order and disagrees on ~3e-4 of elements.
#   * `F.linear` with a bias picks a different cuBLAS kernel than without one, so
#     the one biased projection in the stack (AttentionPairBias q) stays an ATen
#     call; the unbiased ones are bit-exact as `tl.dot`.
#   * K dimensions are only ever *trailing*-zero-padded to a power of two.  Zeros
#     appended after the real K leave every partial sum of the real terms
#     untouched, but zeros *interleaved* into K (e.g. padding each 24-wide head
#     to 32) regroup the MMA's K steps and change the rounding.
# `tl.dot` at these shapes was verified bit-exact against cuBLAS, so the GEMMs
# are free to be packed/fused.
#
# One more access-pattern note: k_ta_att's triangle bias is
# ``linear_z(zln)[lrow, h]``, i.e. 256 elements at stride NTA -- 256 cache lines
# touched for 512 bytes of data, on 64 CTAs per launch.  k_ta_proj therefore
# mirrors those HP columns, transposed, into a compact ``[HP, R]`` buffer, and
# k_ta_att reads that instead.  Same values, so bit-exactness is structural.
#
# Layout: z lives as ``[R, CZ]`` with ``R = N*N`` and row ``i*N + j``.  Every
# kernel loads its *weights* before ``gdc_wait()`` so the weight fetch overlaps
# the producer's tail under PDL, and never stores before ``gdc_wait()`` (which
# would race the producer's own reads of the buffer being overwritten).
# ---------------------------------------------------------------------------
@triton.jit
def _rb(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _exp(x):
    return libdevice.exp(x)


@triton.jit
def _div(a, b):
    """Correctly-rounded fp32 division.  Triton's `/` lowers to an
    approximate reciprocal-multiply that disagrees with nvcc's `div.rn.f32` on
    ~28% of inputs in the last fp32 bit -- harmless on its own, but it flips the
    bf16 rounding whenever the quotient lands on a tie, which is exactly what
    softmax normalization does often enough to matter here."""
    return libdevice.div_rn(a, b)


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + libdevice.exp(-x))


@triton.jit
def _silu(x):
    return x / (1.0 + libdevice.exp(-x))


# --- ATen-exact Welford layernorm statistics (C == 128) ---------------------
@triton.jit
def _wf_on(m, s2, c, v):
    d = v - m
    cn = c + 1.0
    mn = m + d * (1.0 / cn)
    return mn, s2 + d * (v - mn), cn


@triton.jit
def _wf_comb(am, as2, ac, bm, bs2, bc):
    cnt = ac + bc
    coef = 1.0 / cnt
    nA = ac * coef
    nB = bc * coef
    d = am - bm
    return nA * am + nB * bm, as2 + bs2 + d * d * nA * nB * cnt, cnt


@triton.jit
def _wf_lvl(m, s2, c, BM: tl.constexpr, H: tl.constexpr):
    am, bm = tl.split(tl.permute(tl.reshape(m, (BM, 2, H)), (0, 2, 1)))
    a2, b2 = tl.split(tl.permute(tl.reshape(s2, (BM, 2, H)), (0, 2, 1)))
    ac, bc = tl.split(tl.permute(tl.reshape(c, (BM, 2, H)), (0, 2, 1)))
    return _wf_comb(am, a2, ac, bm, b2, bc)


@triton.jit
def _lnstats(x, BM: tl.constexpr, C: tl.constexpr, EPS: tl.constexpr):
    """(mean, rstd) for a [BM, C] fp32 tile, bit-identical to ATen.  C == 128."""
    x4 = tl.reshape(x, (BM, C // 4, 2, 2))
    ev, od = tl.split(x4)
    v0, v2 = tl.split(ev)
    v1, v3 = tl.split(od)
    m = tl.zeros((BM, C // 4), dtype=tl.float32)
    s = tl.zeros((BM, C // 4), dtype=tl.float32)
    c = tl.zeros((BM, C // 4), dtype=tl.float32)
    m, s, c = _wf_on(m, s, c, v0)
    m, s, c = _wf_on(m, s, c, v1)
    m, s, c = _wf_on(m, s, c, v2)
    m, s, c = _wf_on(m, s, c, v3)
    m, s, c = _wf_lvl(m, s, c, BM, 16)
    m, s, c = _wf_lvl(m, s, c, BM, 8)
    m, s, c = _wf_lvl(m, s, c, BM, 4)
    m, s, c = _wf_lvl(m, s, c, BM, 2)
    m, s, c = _wf_lvl(m, s, c, BM, 1)
    mu = tl.reshape(m, (BM,))
    return mu, tl.rsqrt(tl.reshape(s, (BM,)) / C + EPS)


@triton.jit
def _ln(x, w, b, BM: tl.constexpr, C: tl.constexpr, EPS: tl.constexpr):
    mu, rstd = _lnstats(x, BM, C, EPS)
    return w[None, :] * (rstd[:, None] * (x - mu[:, None])) + b[None, :]


# --- softmax reductions in ATen's warp_reduce (XOR butterfly) order --------
# The 16 exponentials are pulled out as single-element masked reductions and then
# combined by an explicit XOR tree (offsets 8,4,2,1).  Building the tree with
# reshape/split instead is *not* safe here: `sc` is dot-derived, and on an MMA
# layout Triton regroups values within the row, which silently changes the
# summation order.  Being layout-proof is also what lets k_ta_att run on a
# single warp, which is worth 2.8%.  A masked reduction has one non-zero summand, so its result is
# exact whatever the layout.  ATen's divisor was confirmed to be exactly this
# tree by inverting an fp32 softmax (probes/p25_smid.py).
@triton.jit
def _xtree(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):
    a0 = x0 + x8
    a1 = x1 + x9
    a2 = x2 + x10
    a3 = x3 + x11
    a4 = x4 + x12
    a5 = x5 + x13
    a6 = x6 + x14
    a7 = x7 + x15
    return ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7))


@triton.jit
def _rowsum16_2(e, idx):
    x0 = tl.sum(tl.where(idx[None, :] == 0, e, 0.0), 1)
    x1 = tl.sum(tl.where(idx[None, :] == 1, e, 0.0), 1)
    x2 = tl.sum(tl.where(idx[None, :] == 2, e, 0.0), 1)
    x3 = tl.sum(tl.where(idx[None, :] == 3, e, 0.0), 1)
    x4 = tl.sum(tl.where(idx[None, :] == 4, e, 0.0), 1)
    x5 = tl.sum(tl.where(idx[None, :] == 5, e, 0.0), 1)
    x6 = tl.sum(tl.where(idx[None, :] == 6, e, 0.0), 1)
    x7 = tl.sum(tl.where(idx[None, :] == 7, e, 0.0), 1)
    x8 = tl.sum(tl.where(idx[None, :] == 8, e, 0.0), 1)
    x9 = tl.sum(tl.where(idx[None, :] == 9, e, 0.0), 1)
    x10 = tl.sum(tl.where(idx[None, :] == 10, e, 0.0), 1)
    x11 = tl.sum(tl.where(idx[None, :] == 11, e, 0.0), 1)
    x12 = tl.sum(tl.where(idx[None, :] == 12, e, 0.0), 1)
    x13 = tl.sum(tl.where(idx[None, :] == 13, e, 0.0), 1)
    x14 = tl.sum(tl.where(idx[None, :] == 14, e, 0.0), 1)
    x15 = tl.sum(tl.where(idx[None, :] == 15, e, 0.0), 1)
    return _xtree(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)


@triton.jit
def k_ln(X, W, B, Y, R, C: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr):
    """Prologue: Y = LayerNorm(X) for the first block's tri_mul_out."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    c = tl.arange(0, C)
    w = tl.load(W + c)
    b = tl.load(B + c)
    gdc_wait()
    x = tl.load(X + rows[:, None] * C + c[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    y = _rb(_ln(x, w, b, BM, C, EPS))
    tl.store(Y + rows[:, None] * C + c[None, :], y.to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_tm_proj(ZLN, WTM, PM, A, B, GZ, R,
              CZ: tl.constexpr, CM: tl.constexpr, NW: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    """TriangleMultiplication input projections as one packed GEMM.

    grp 0 -> a = mask*sigmoid(a_g)*a_p, grp 1 -> b, grp 2 -> sigmoid(linear_g).
    WTM is [CZ, 4*CM + CZ] = [a_p | a_g | b_p | b_g | g], pre-transposed at
    __init__ so no runtime permute/copy survives in the forward path.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    grp = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < R
    kk = tl.arange(0, CZ)
    n = pid_n * BN + tl.arange(0, BN)
    if grp == 2:
        nm = n < CZ
        wg = tl.load(WTM + kk[:, None] * NW + (4 * CM + n)[None, :],
                     mask=nm[None, :], other=0.0)
        gdc_wait()
        zln = tl.load(ZLN + rows[:, None] * CZ + kk[None, :], mask=rm[:, None], other=0.0)
        out = _rb(_sig(_rb(tl.dot(zln, wg))))
        tl.store(GZ + rows[:, None] * CZ + n[None, :], out.to(tl.bfloat16),
                 mask=rm[:, None] & nm[None, :])
    else:
        nm = n < CM
        base = grp * (2 * CM)
        wp = tl.load(WTM + kk[:, None] * NW + (base + n)[None, :], mask=nm[None, :], other=0.0)
        wg = tl.load(WTM + kk[:, None] * NW + (base + CM + n)[None, :],
                     mask=nm[None, :], other=0.0)
        gdc_wait()
        zln = tl.load(ZLN + rows[:, None] * CZ + kk[None, :], mask=rm[:, None], other=0.0)
        m = tl.load(PM + rows, mask=rm, other=0.0).to(tl.float32)
        p = _rb(tl.dot(zln, wp))
        g = _rb(tl.dot(zln, wg))
        v = _rb(_rb(m[:, None] * _rb(_sig(g))) * p)
        if grp == 0:
            tl.store(A + rows[:, None] * CM + n[None, :], v.to(tl.bfloat16),
                     mask=rm[:, None] & nm[None, :])
        else:
            tl.store(B + rows[:, None] * CM + n[None, :], v.to(tl.bfloat16),
                     mask=rm[:, None] & nm[None, :])
    gdc_launch_dependents()


@triton.jit
def k_tm_ein(A, B, XS, N: tl.constexpr, CM: tl.constexpr, BC: tl.constexpr,
             OUTGOING: tl.constexpr):
    """Triangle einsum as a channel-batched matmul (the reference's bmm).

    outgoing: x[i,k,c] = sum_j a[i,j,c] b[k,j,c]
    incoming: x[i,k,c] = sum_m a[m,i,c] b[m,k,c]
    The reference reaches this through four permutes into [C,I,J] and a bmm; the
    permutes are folded into these load indices, so no copy kernels remain.
    """
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    gdc_wait()
    if OUTGOING:
        a3 = tl.load(A + (i[None, :, None] * N + j[None, None, :]) * CM + c[:, None, None])
        b3 = tl.load(B + (i[None, None, :] * N + j[None, :, None]) * CM + c[:, None, None])
    else:
        a3 = tl.load(A + (j[None, None, :] * N + i[None, :, None]) * CM + c[:, None, None])
        b3 = tl.load(B + (j[None, :, None] * N + i[None, None, :]) * CM + c[:, None, None])
    tl.store(XS + (i[None, :, None] * N + j[None, None, :]) * CM + c[:, None, None],
             _rb(tl.dot(a3, b3)).to(tl.bfloat16))
    gdc_launch_dependents()


@triton.jit
def k_tm_lno(XS, LOW, LOB, XL, R, CM: tl.constexpr, BM: tl.constexpr,
             EPS: tl.constexpr):
    """layer_norm_out.  Deliberately dot-free (see the layout note above)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    cm = tl.arange(0, CM)
    lw = tl.load(LOW + cm)
    lb = tl.load(LOB + cm)
    gdc_wait()
    x = tl.load(XS + rows[:, None] * CM + cm[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(XL + rows[:, None] * CM + cm[None, :],
             _rb(_ln(x, lw, lb, BM, CM, EPS)).to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_tm_fin(Z, XL, GZ, WZ, LNW, LNB, ZLN, R, CZ: tl.constexpr, CM: tl.constexpr,
             BM: tl.constexpr, EPS: tl.constexpr, HAS_NEXT: tl.constexpr):
    """linear_z + output gate + residual, then the next sublayer's layernorm
    (read back from memory so its dataflow never touches the dot)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    cm = tl.arange(0, CM)
    cz = tl.arange(0, CZ)
    wz = tl.load(WZ + cm[:, None] * CZ + cz[None, :])
    nw = tl.load(LNW + cz)
    nb = tl.load(LNB + cz)
    gdc_wait()
    xl = tl.load(XL + rows[:, None] * CM + cm[None, :], mask=rm[:, None], other=0.0)
    y = _rb(tl.dot(xl, wz))
    g = tl.load(GZ + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    zo = tl.load(Z + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(Z + rows[:, None] * CZ + cz[None, :], _rb(zo + _rb(y * g)).to(tl.bfloat16),
             mask=rm[:, None])
    if HAS_NEXT:
        tl.debug_barrier()
        zn = tl.load(Z + rows[:, None] * CZ + cz[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32)
        zl = _rb(_ln(zn, nw, nb, BM, CZ, EPS))
        tl.store(ZLN + rows[:, None] * CZ + cz[None, :], zl.to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_ta_proj(ZLN, WTA, QK, TB, R, CZ: tl.constexpr, NTA: tl.constexpr,
              HP: tl.constexpr, NB: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    """TriangleAttention q/k/v/g and the triangle-bias projection, one GEMM.

    WTA is [CZ, NTA] = [q | k | v | g | linear_z | zero-pad], pre-transposed.
    The HP linear_z columns are additionally mirrored, transposed, into TB so
    that k_ta_att's bias read is contiguous instead of a 256-way gather.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < R
    kk = tl.arange(0, CZ)
    n = pid_n * BN + tl.arange(0, BN)
    w = tl.load(WTA + kk[:, None] * NTA + n[None, :])
    gdc_wait()
    zln = tl.load(ZLN + rows[:, None] * CZ + kk[None, :], mask=rm[:, None], other=0.0)
    v = _rb(tl.dot(zln, w)).to(tl.bfloat16)
    tl.store(QK + rows[:, None] * NTA + n[None, :], v, mask=rm[:, None])
    if pid_n == NB // BN:
        h = n - NB
        tl.store(TB + h[None, :] * R + rows[:, None], v,
                 mask=rm[:, None] & (h < HP)[None, :])
    gdc_launch_dependents()


@triton.jit
def k_ta_att(QK, TB, PM, OSA, R, N: tl.constexpr, HP: tl.constexpr, DP: tl.constexpr,
             NTA: tl.constexpr, INF: tl.constexpr, SQD: tl.constexpr,
             STARTING: tl.constexpr):
    """One CTA per (row group, head): pair-biased softmax attention + gate.

    The head axis is independent all the way to linear_o, so this split changes
    no accumulation order; linear_o itself stays whole-K in k_ta_out.
    """
    pid = tl.program_id(0)
    hh = tl.program_id(1)
    t = tl.arange(0, N)
    d = tl.arange(0, DP)
    HD: tl.constexpr = HP * DP
    if STARTING:
        rows = pid * N + t
        lrow = t[:, None] * N + t[None, :]
    else:
        rows = t * N + pid
        lrow = t[None, :] * N + t[:, None]
    gdc_wait()
    off = rows[:, None] * NTA + (hh * DP + d)[None, :]
    q = tl.load(QK + off)
    k = tl.load(QK + off + HD)
    v = tl.load(QK + off + 2 * HD)
    qs = _rb(q.to(tl.float32) / SQD).to(tl.bfloat16)
    sc = _rb(tl.dot(qs, tl.trans(k)))
    m = tl.load(PM + rows).to(tl.float32)
    sc = _rb(sc + _rb(INF * (m - 1.0))[None, :])
    lb = tl.load(TB + hh * R + lrow)
    sc = _rb(sc + lb.to(tl.float32))
    e = _exp(sc - tl.max(sc, 1)[:, None])
    p = _rb(_div(e, _rowsum16_2(e, t)[:, None]))
    o = _rb(tl.dot(p.to(tl.bfloat16), v))
    g = tl.load(QK + rows[:, None] * NTA + (3 * HD + hh * DP + d)[None, :]).to(tl.float32)
    o = _rb(o * _rb(_sig(g)))
    tl.store(OSA + rows[:, None] * HD + (hh * DP + d)[None, :], o.to(tl.bfloat16))
    gdc_launch_dependents()


@triton.jit
def k_ta_out(Z, OSA, WO, LNW, LNB, ZLN, R, CZ: tl.constexpr, HD: tl.constexpr,
             BM: tl.constexpr, EPS: tl.constexpr, HAS_NEXT: tl.constexpr):
    """linear_o (whole K), residual, and the next sublayer's layernorm."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    hd = tl.arange(0, HD)
    cz = tl.arange(0, CZ)
    wo = tl.load(WO + hd[:, None] * CZ + cz[None, :])
    nw = tl.load(LNW + cz)
    nb = tl.load(LNB + cz)
    gdc_wait()
    o2 = tl.load(OSA + rows[:, None] * HD + hd[None, :], mask=rm[:, None], other=0.0)
    y = _rb(tl.dot(o2, wo))
    zo = tl.load(Z + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    tl.store(Z + rows[:, None] * CZ + cz[None, :], _rb(zo + y).to(tl.bfloat16),
             mask=rm[:, None])
    if HAS_NEXT:
        tl.debug_barrier()
        zn = tl.load(Z + rows[:, None] * CZ + cz[None, :], mask=rm[:, None],
                     other=0.0).to(tl.float32)
        zl = _rb(_ln(zn, nw, nb, BM, CZ, EPS))
        tl.store(ZLN + rows[:, None] * CZ + cz[None, :], zl.to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_swiglu(XLN, WA, WB, H, R, K: tl.constexpr, KP: tl.constexpr,
             NH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """h = silu(xln @ Wa) * (xln @ Wb); WA/WB are [KP, NH] pre-transposed."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < R
    kk = tl.arange(0, KP)
    n = pid_n * BN + tl.arange(0, BN)
    wa = tl.load(WA + kk[:, None] * NH + n[None, :])
    wb = tl.load(WB + kk[:, None] * NH + n[None, :])
    gdc_wait()
    x = tl.load(XLN + rows[:, None] * K + kk[None, :],
                mask=rm[:, None] & (kk < K)[None, :], other=0.0)
    a = _rb(tl.dot(x, wa))
    b = _rb(tl.dot(x, wb))
    tl.store(H + rows[:, None] * NH + n[None, :],
             _rb(_rb(_silu(a)) * b).to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_trans_out(H, WO, MASK, D, R, KH: tl.constexpr, CO: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """delta = (h @ Wout) * mask, K-chunked so the dot operands stay small."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < R
    n = pid_n * BN + tl.arange(0, BN)
    kb = tl.arange(0, BK)
    gdc_wait()
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kc in tl.static_range(KH // BK):
        w = tl.load(WO + (kc * BK + kb)[:, None] * CO + n[None, :])
        hh = tl.load(H + rows[:, None] * KH + (kc * BK + kb)[None, :],
                     mask=rm[:, None], other=0.0)
        acc += tl.dot(hh, w)
    m = tl.load(MASK + rows, mask=rm, other=0.0).to(tl.float32)
    tl.store(D + rows[:, None] * CO + n[None, :],
             _rb(_rb(acc) * m[:, None]).to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_pt_tail(Z, DZ, LNW, LNB, LZW, LZB, ZLN, ZSB, R,
              CZ: tl.constexpr, BM: tl.constexpr, EPS: tl.constexpr,
              HAS_NEXT: tl.constexpr):
    """pair-transition residual, the next block's layer_norm_in, and
    AttentionPairBias's layer_norm_z.  Dot-free, so both layernorm inputs are
    load-derived and no memory round-trip is needed to fix their layout."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    cz = tl.arange(0, CZ)
    nw = tl.load(LNW + cz)
    nb = tl.load(LNB + cz)
    zw = tl.load(LZW + cz)
    zb = tl.load(LZB + cz)
    gdc_wait()
    zo = tl.load(Z + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    dz = tl.load(DZ + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    zn = _rb(zo + dz)
    tl.store(Z + rows[:, None] * CZ + cz[None, :], zn.to(tl.bfloat16), mask=rm[:, None])
    if HAS_NEXT:
        zl = _rb(_ln(zn, nw, nb, BM, CZ, EPS))
        tl.store(ZLN + rows[:, None] * CZ + cz[None, :], zl.to(tl.bfloat16), mask=rm[:, None])
    zs = _rb(_ln(zn, zw, zb, BM, CZ, EPS))
    tl.store(ZSB + rows[:, None] * CZ + cz[None, :], zs.to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


@triton.jit
def k_pt_bias(ZSB, WZS, BIAS, R, CZ: tl.constexpr, HSP: tl.constexpr,
              BM: tl.constexpr):
    """AttentionPairBias pair bias: linear_z on the normalized pair embedding."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < R
    cz = tl.arange(0, CZ)
    hs = tl.arange(0, HSP)
    wzs = tl.load(WZS + cz[:, None] * HSP + hs[None, :])
    gdc_wait()
    zs = tl.load(ZSB + rows[:, None] * CZ + cz[None, :], mask=rm[:, None], other=0.0)
    tl.store(BIAS + rows[:, None] * HSP + hs[None, :],
             _rb(tl.dot(zs, wzs)).to(tl.bfloat16), mask=rm[:, None])
    gdc_launch_dependents()


# --- single (s) track.  The two 384-wide layernorms are ATen calls captured in
# --- the graph, so these kernels take an already-normalized fp32 input.
@triton.jit
def k_s_qkv(SLN, W, OUT, SLNB, T: tl.constexpr, CS: tl.constexpr,
            CSP: tl.constexpr, NKVG: tl.constexpr, BN: tl.constexpr):
    """AttentionPairBias packed k|v|g projection, and the bf16 layernorm output
    that the (biased, hence ATen) q projection consumes."""
    pid = tl.program_id(0)
    t = tl.arange(0, T)
    kk = tl.arange(0, CSP)
    km = kk < CS
    n = pid * BN + tl.arange(0, BN)
    w = tl.load(W + kk[:, None] * NKVG + n[None, :])
    gdc_wait()
    x = tl.load(SLN + t[:, None] * CS + kk[None, :], mask=km[None, :], other=0.0)
    xb = _rb(x).to(tl.bfloat16)
    if pid == 0:
        tl.store(SLNB + t[:, None] * CS + kk[None, :], xb, mask=km[None, :])
    tl.store(OUT + t[:, None] * NKVG + n[None, :], _rb(tl.dot(xb, w)).to(tl.bfloat16))
    gdc_launch_dependents()


@triton.jit
def k_s_attn(Q, KVG, BIAS, SM, OS, T: tl.constexpr, HDS: tl.constexpr,
             HS: tl.constexpr, DS: tl.constexpr, DSP: tl.constexpr,
             NKVG: tl.constexpr, HSP: tl.constexpr,
             INF: tl.constexpr, SQD: tl.constexpr):
    """One CTA per head: pair-biased attention over the N_token axis + gate."""
    hh = tl.program_id(0)
    t = tl.arange(0, T)
    d = tl.arange(0, DSP)
    dm = d < DS
    coff = hh * DS + d
    gdc_wait()
    q = tl.load(Q + t[:, None] * HDS + coff[None, :], mask=dm[None, :], other=0.0)
    k = tl.load(KVG + t[:, None] * NKVG + coff[None, :], mask=dm[None, :], other=0.0)
    v = tl.load(KVG + t[:, None] * NKVG + (HDS + coff)[None, :], mask=dm[None, :], other=0.0)
    g = tl.load(KVG + t[:, None] * NKVG + (2 * HDS + coff)[None, :], mask=dm[None, :], other=0.0)
    qs = _rb(q.to(tl.float32) / SQD).to(tl.bfloat16)
    sc = _rb(tl.dot(qs, tl.trans(k)))
    m = tl.load(SM + t).to(tl.float32)
    sc = _rb(sc + _rb(INF * (m - 1.0))[None, :])
    lb = tl.load(BIAS + (t[:, None] * T + t[None, :]) * HSP + hh)
    sc = _rb(sc + lb.to(tl.float32))
    e = _exp(sc - tl.max(sc, 1)[:, None])
    p = _rb(_div(e, _rowsum16_2(e, t)[:, None]))
    o = _rb(tl.dot(p.to(tl.bfloat16), v))
    o = _rb(o * _rb(_sig(g.to(tl.float32))))
    tl.store(OS + t[:, None] * HDS + (hh * DS + d)[None, :], o.to(tl.bfloat16),
             mask=dm[None, :])
    gdc_launch_dependents()


@triton.jit
def k_s_o(OS, WO, S, S1, S132, T: tl.constexpr, HDS: tl.constexpr,
          HDSP: tl.constexpr, CS: tl.constexpr, BN: tl.constexpr):
    """linear_o, then the attention residual s1 = s + out, emitted in both bf16
    (for the transition residual) and fp32 (for the ATen layernorm)."""
    pid = tl.program_id(0)
    t = tl.arange(0, T)
    kk = tl.arange(0, HDSP)
    km = kk < HDS
    n = pid * BN + tl.arange(0, BN)
    w = tl.load(WO + kk[:, None] * CS + n[None, :])
    gdc_wait()
    o = tl.load(OS + t[:, None] * HDS + kk[None, :], mask=km[None, :], other=0.0)
    d = _rb(tl.dot(o, w))
    s0 = tl.load(S + t[:, None] * CS + n[None, :]).to(tl.float32)
    v = _rb(s0 + d)
    tl.store(S1 + t[:, None] * CS + n[None, :], v.to(tl.bfloat16))
    tl.store(S132 + t[:, None] * CS + n[None, :], v)
    gdc_launch_dependents()


@triton.jit
def k_s_swiglu(XLN, WA, WB, H, T: tl.constexpr, CS: tl.constexpr,
               CSP: tl.constexpr, NH: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    t = tl.arange(0, T)
    kk = tl.arange(0, CSP)
    km = kk < CS
    n = pid * BN + tl.arange(0, BN)
    wa = tl.load(WA + kk[:, None] * NH + n[None, :])
    wb = tl.load(WB + kk[:, None] * NH + n[None, :])
    gdc_wait()
    x = tl.load(XLN + t[:, None] * CS + kk[None, :], mask=km[None, :], other=0.0)
    xb = _rb(x).to(tl.bfloat16)
    a = _rb(tl.dot(xb, wa))
    b = _rb(tl.dot(xb, wb))
    tl.store(H + t[:, None] * NH + n[None, :], _rb(_rb(_silu(a)) * b).to(tl.bfloat16))
    gdc_launch_dependents()


@triton.jit
def k_s_tail(S1, D2, S, S32, T, CS: tl.constexpr, CSP: tl.constexpr,
             BM: tl.constexpr):
    """s <- s1 + transition_out, in bf16 and in fp32 (next block's ATen LN)."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    rm = rows < T
    kk = tl.arange(0, CSP)
    msk = rm[:, None] & (kk < CS)[None, :]
    gdc_wait()
    s1 = tl.load(S1 + rows[:, None] * CS + kk[None, :], mask=msk, other=0.0).to(tl.float32)
    d2 = tl.load(D2 + rows[:, None] * CS + kk[None, :], mask=msk, other=0.0).to(tl.float32)
    out = _rb(s1 + d2)
    tl.store(S + rows[:, None] * CS + kk[None, :], out.to(tl.bfloat16), mask=msk)
    tl.store(S32 + rows[:, None] * CS + kk[None, :], out, mask=msk)
    gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Tile sizes.  The GEMMs are all bandwidth/latency bound at this shape, so the
# tiles are chosen to keep per-CTA *weight* bytes near 8-24 KB (a single SM only
# sustains ~46 GB/s, so a 128 KB slab on one CTA costs ~3 us) while keeping the
# CTA count high enough to hide memory latency.
#
# Once the single track runs concurrently (see _run) the z chain is contended
# rather than latency-bound, and CTA *count on the z side* became the dominant
# knob: _BM_Z 32 -> 16 is -8.7% and _BC 16 -> 8 a further -3.4%, while the same
# move on the s side is a large loss (_BN_S 16 -> 8 is +25%) because extra s
# CTAs are taken straight off the critical path.  Measured: _BM_Z 8 (+4%),
# _BN_Z 16 (+7%), _BN_PTO 8 (+1%), _BN_S 32 (+2%) are all worse.
# ---------------------------------------------------------------------------
_BM_Z = 16      # z rows per CTA in the z-side projection GEMMs
_BN_Z = 32      # output columns per CTA, z-side
_BN_PTO = 16    # output columns per CTA for the transition output GEMM
_BK_CHUNK = 512
_BN_S = 16      # output columns per CTA, s-side
_BC = 4         # channel block of the triangle einsum (-> CM/_BC CTAs)
_BM_LN = 4      # dot-free kernels: no MMA minimum tile, so go wide on CTAs
_BM_S = 16


class _NoGraph:
    def replay(self):
        pass


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack (48 blocks), launch-collapsed."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        blocks_per_ckpt: int | None = None,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden_pair_bias = c_hidden_pair_bias
        self.no_heads_pair_bias = no_heads_pair_bias
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_pair = no_heads_pair
        self.transition_n = transition_n
        self.inf = inf
        self.blocks = nn.ModuleList([
            PairFormerBlock(
                c_s=c_s, c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                inf=inf,
            )
            for _ in range(no_blocks)
        ])
        self._graph = None
        self._key = None
        self._probe = None

    # -- fast-path eligibility -------------------------------------------------
    def _eligible(self, s, z, single_mask, pair_mask, mask_trans) -> bool:
        if _FORCE_EAGER:
            return False
        if not (mask_trans and torch.cuda.is_available()):
            return False
        for t in (s, z, single_mask, pair_mask):
            if not (isinstance(t, torch.Tensor) and t.is_cuda
                    and t.dtype == torch.bfloat16 and t.is_contiguous()):
                return False
        if z.dim() != 4 or s.dim() != 3 or z.shape[0] != 1 or s.shape[0] != 1:
            return False
        n = z.shape[1]
        if z.shape[2] != n or s.shape[1] != n or z.shape[3] != self.c_z:
            return False
        if s.shape[2] != self.c_s or n != 16:
            return False
        # _lnstats replicates ATen's vectorized layernorm for exactly one warp
        # of 4-element Welford lanes, and the softmax butterfly is unrolled for
        # N_token == 16; anything else takes the reference path.
        if self.c_z != 128 or self.c_hidden_mul != 128:
            return False
        if tuple(single_mask.shape) != (1, n) or tuple(pair_mask.shape) != (1, n, n):
            return False
        p = next(self.parameters())
        if p.dtype != torch.bfloat16 or p.device != z.device:
            return False
        # every packed GEMM tile must divide evenly
        hdp = self.c_hidden_pair_att * self.no_heads_pair
        hds = self.c_hidden_pair_bias * self.no_heads_pair_bias
        checks = [
            self.c_z % _BN_Z, self.c_hidden_mul % _BN_Z, self.c_z % _BN_PTO,
            (self.transition_n * self.c_z) % _BN_Z,
            (self.transition_n * self.c_z) % _BK_CHUNK,
            (self.transition_n * self.c_s) % _BN_S,
            (self.transition_n * self.c_s) % _BK_CHUNK,
            self.c_s % _BN_S, (4 * hds) % _BN_S,
            _next_pow2(self.c_z) - self.c_z, _next_pow2(hdp) - hdp,
            (4 * hdp) % _BN_Z, max(0, self.no_heads_pair - _BN_Z),
        ]
        return not any(checks)

    # -- weight packing + buffers --------------------------------------------
    def _pack(self, n: int, dev):
        bf = torch.bfloat16
        cz, cs, cm = self.c_z, self.c_s, self.c_hidden_mul
        hp, dp = self.no_heads_pair, self.c_hidden_pair_att
        hs, ds = self.no_heads_pair_bias, self.c_hidden_pair_bias
        hdp, hds = hp * dp, hs * ds
        dsp = max(16, _next_pow2(ds))
        hsp = max(16, _next_pow2(hs))
        csp = _next_pow2(cs)
        npt, nst = self.transition_n * cz, self.transition_n * cs
        nta = -(-(4 * hdp + hp) // _BN_Z) * _BN_Z
        nsq = 4 * hds

        self.dims = dict(N=n, R=n * n, CZ=cz, CS=cs, CSP=csp, CM=cm, HP=hp, DP=dp,
                         HDP=hdp, NTA=nta, HS=hs, DS=ds, DSP=dsp, HSP=hsp,
                         HDS=hds, HDSP=_next_pow2(hds), NSQ=nsq, NPT=npt, NST=nst)

        def T(w):  # [out, in] -> contiguous [in, out]
            return w.detach().t().contiguous()

        def f32(p):
            return p.detach().float().contiguous()

        packed = []
        for blk in self.blocks:
            ps = blk.pair_stack
            d = {}
            for tag, tm in (("tmo", ps.tri_mul_out), ("tmi", ps.tri_mul_in)):
                W = torch.zeros(cz, 4 * cm + cz, dtype=bf, device=dev)
                W[:, 0:cm] = T(tm.linear_a_p.weight)
                W[:, cm:2 * cm] = T(tm.linear_a_g.weight)
                W[:, 2 * cm:3 * cm] = T(tm.linear_b_p.weight)
                W[:, 3 * cm:4 * cm] = T(tm.linear_b_g.weight)
                W[:, 4 * cm:4 * cm + cz] = T(tm.linear_g.weight)
                d[tag + "_W"] = W
                d[tag + "_Wz"] = T(tm.linear_z.weight)
                d[tag + "_low"] = f32(tm.layer_norm_out.weight)
                d[tag + "_lob"] = f32(tm.layer_norm_out.bias)
                d[tag + "_lnw"] = f32(tm.layer_norm_in.weight)
                d[tag + "_lnb"] = f32(tm.layer_norm_in.bias)
            for tag, ta in (("tas", ps.tri_att_start), ("tae", ps.tri_att_end)):
                W = torch.zeros(cz, nta, dtype=bf, device=dev)
                W[:, 0:hdp] = T(ta.mha.linear_q.weight)
                W[:, hdp:2 * hdp] = T(ta.mha.linear_k.weight)
                W[:, 2 * hdp:3 * hdp] = T(ta.mha.linear_v.weight)
                W[:, 3 * hdp:4 * hdp] = T(ta.mha.linear_g.weight)
                W[:, 4 * hdp:4 * hdp + hp] = T(ta.linear_z.weight)
                d[tag + "_W"] = W
                d[tag + "_Wo"] = T(ta.mha.linear_o.weight)
                d[tag + "_lnw"] = f32(ta.layer_norm.weight)
                d[tag + "_lnb"] = f32(ta.layer_norm.bias)
            pt = ps.pair_transition
            d["pt_Wa"] = T(pt.swiglu.linear_a.weight)
            d["pt_Wb"] = T(pt.swiglu.linear_b.weight)
            d["pt_Wo"] = T(pt.linear_out.weight)
            d["pt_lnw"] = f32(pt.layer_norm.weight)
            d["pt_lnb"] = f32(pt.layer_norm.bias)
            ab = blk.attn_pair_bias
            d["apb_law"] = f32(ab.layer_norm_a.weight)
            d["apb_lab"] = f32(ab.layer_norm_a.bias)
            d["apb_lzw"] = f32(ab.layer_norm_z.weight)
            d["apb_lzb"] = f32(ab.layer_norm_z.bias)
            Wzs = torch.zeros(cz, hsp, dtype=bf, device=dev)
            Wzs[:, :hs] = T(ab.linear_z.weight)
            d["apb_Wzs"] = Wzs
            Wkvg = torch.zeros(csp, 3 * hds, dtype=bf, device=dev)
            for i, lin in enumerate((ab.mha.linear_k, ab.mha.linear_v,
                                     ab.mha.linear_g)):
                Wkvg[:cs, i * hds:(i + 1) * hds] = T(lin.weight)
            d["apb_Wkvg"] = Wkvg
            d["apb_q_w"] = ab.mha.linear_q.weight
            d["apb_q_b"] = ab.mha.linear_q.bias
            Wo = torch.zeros(_next_pow2(hds), cs, dtype=bf, device=dev)
            Wo[:hds] = T(ab.mha.linear_o.weight)
            d["apb_Wo"] = Wo
            st = blk.single_transition
            Wa = torch.zeros(csp, nst, dtype=bf, device=dev)
            Wa[:cs] = T(st.swiglu.linear_a.weight)
            Wb = torch.zeros(csp, nst, dtype=bf, device=dev)
            Wb[:cs] = T(st.swiglu.linear_b.weight)
            d["st_Wa"], d["st_Wb"] = Wa, Wb
            d["st_Wo"] = T(st.linear_out.weight)
            d["st_lnw"] = f32(st.layer_norm.weight)
            d["st_lnb"] = f32(st.layer_norm.bias)
            packed.append(d)
        self._bw = packed

        dm = self.dims
        R = dm["R"]
        e = lambda *sh: torch.zeros(*sh, dtype=bf, device=dev)
        self._zbuf = e(1, n, n, cz)
        self._sbuf = e(1, n, cs)
        self._pmbuf = e(1, n, n)
        self._smbuf = e(1, n)
        self.Z = self._zbuf.view(R, cz)
        self.S = self._sbuf.view(n, cs)
        self.PM = self._pmbuf.view(R)
        self.SM = self._smbuf.view(n)
        self.ZLN = e(R, cz)
        self.A = e(R, cm)
        self.B = e(R, cm)
        self.GZ = e(R, cz)
        self.QK = e(R, nta)
        self.HZ = e(R, npt)
        self.DZ = e(R, cz)
        self.XL = e(R, cm)
        self.XS = e(R, cm)
        self.ZSB = e(len(self.blocks), R, cz)
        self.OSA = e(R, hdp)
        self.TB = e(hp, R)
        # BIAS is the *only* tensor that crosses from the pair branch to the
        # single branch, so it is the only one that needs a per-block slot:
        # block bi+1's k_pt_bias would otherwise overwrite what block bi's
        # k_s_attn is still reading on the other branch.  48 x 256 x 16 bf16 is
        # 384 KB, cheap enough that indexing by block beats parity
        # double-buffering plus 48 write-after-read joins on the critical path.
        self.BIAS = e(len(self.blocks), R, hsp)
        self.SKVG = e(n, 3 * hds)
        self.SLNB = e(n, cs)
        self.OS = e(n, hds)
        self.S1B = e(n, cs)
        self.D2 = e(n, cs)
        self.HSB = e(n, nst)
        self.S32 = torch.zeros(n, cs, dtype=torch.float32, device=dev)
        self.S132 = torch.zeros(n, cs, dtype=torch.float32, device=dev)
        self._aten = []
        # Second branch of the captured graph (see _run).  Created once and
        # reused so that the cuBLAS workspace for it is allocated by the
        # warm-up rather than from inside the capture region.
        self._two = not _NO_OVERLAP
        self._sstream = torch.cuda.Stream(device=dev)
        self._ev_fork = [torch.cuda.Event() for _ in self.blocks]
        self._ev_join = torch.cuda.Event()

    # -- one graph-captured pass over the whole stack -------------------------
    def _run(self):
        dm = self.dims
        N, R, CZ, CS, CSP, CM = dm["N"], dm["R"], dm["CZ"], dm["CS"], dm["CSP"], dm["CM"]
        HP, DP, NTA, HDP = dm["HP"], dm["DP"], dm["NTA"], dm["HDP"]
        HS, DS, DSP, HSP, HDS, HDSP = (dm["HS"], dm["DS"], dm["DSP"], dm["HSP"],
                                       dm["HDS"], dm["HDSP"])  # HDSP: K padding
        NPT, NST = dm["NPT"], dm["NST"]
        NWTM = 4 * CM + CZ
        mz = triton.cdiv(R, _BM_Z)
        mln = triton.cdiv(R, _BM_LN)
        gzn = triton.cdiv(max(CM, CZ), _BN_Z)
        pdl = _HAS_PDL and not _NO_PDL
        inf = float(self.inf)
        sqdp, sqds = math.sqrt(DP), math.sqrt(DS)
        nb = len(self._bw)
        w0 = self._bw[0]
        two = self._two
        zs = torch.cuda.current_stream()
        ss = self._sstream

        def s_block(w, bi):
            """AttentionPairBias + SwiGLUTransition on the single embedding.

            Everything here runs on the side stream.  ``ZSB[bi]``
            (= layer_norm_z(z_bi), written by the z branch's k_pt_tail) is the
            only tensor it reads from the pair track, which is why ZSB is the
            one that carries a per-block slot; every scratch buffer it writes is
            its own, so nothing has to be duplicated and nothing needs a join.
            """
            # The two 384-wide layernorms go through ATen inside the graph:
            # the Welford replica above is bit-exact only for C=128, and this
            # stack is chaotic enough that a 1-ULP rstd disagreement fails.
            k_pt_bias[(mz,)](
                self.ZSB[bi], w["apb_Wzs"], self.BIAS[bi], R,
                CZ=CZ, HSP=HSP, BM=_BM_Z, launch_pdl=pdl)
            sln_a = F.layer_norm(self.S32, (CS,), w["apb_law"], w["apb_lab"], _EPS)
            k_s_qkv[((3 * HDS) // _BN_S,)](
                sln_a, w["apb_Wkvg"], self.SKVG, self.SLNB,
                T=N, CS=CS, CSP=CSP, NKVG=3 * HDS, BN=_BN_S, launch_pdl=pdl)
            # linear_q carries a bias, and cuBLAS's bias epilogue is a different
            # (non-reproducible-by-tl.dot) accumulation, so this one stays ATen.
            sq = F.linear(self.SLNB.view(1, N, CS), w["apb_q_w"], w["apb_q_b"])
            k_s_attn[(HS,)](
                sq, self.SKVG, self.BIAS[bi], self.SM, self.OS,
                T=N, HDS=HDS, HS=HS, DS=DS, DSP=DSP, NKVG=3 * HDS, HSP=HSP,
                INF=inf, SQD=sqds, launch_pdl=pdl)
            k_s_o[(CS // _BN_S,)](
                self.OS, w["apb_Wo"], self.S, self.S1B, self.S132,
                T=N, HDS=HDS, HDSP=HDSP, CS=CS, BN=_BN_S, launch_pdl=pdl)
            sln_t = F.layer_norm(self.S132, (CS,), w["st_lnw"], w["st_lnb"], _EPS)
            k_s_swiglu[(NST // _BN_S,)](
                sln_t, w["st_Wa"], w["st_Wb"], self.HSB,
                T=N, CS=CS, CSP=CSP, NH=NST, BN=_BN_S, launch_pdl=pdl)
            k_trans_out[(1, CS // _BN_S)](
                self.HSB, w["st_Wo"], self.SM, self.D2, N,
                KH=NST, CO=CS, BM=_BM_S, BN=_BN_S, BK=_BK_CHUNK, launch_pdl=pdl)
            k_s_tail[(triton.cdiv(N, _BM_S),)](
                self.S1B, self.D2, self.S, self.S32, N,
                CS=CS, CSP=CSP, BM=_BM_S, launch_pdl=pdl)
            self._aten.append((sln_a, sln_t, sq))

        k_ln[(mz,)](self.Z, w0["tmo_lnw"], w0["tmo_lnb"], self.ZLN, R,
                    C=CZ, BM=_BM_Z, EPS=_EPS, launch_pdl=pdl)
        for bi in range(nb):
            w = self._bw[bi]
            has_next = bi + 1 < nb
            nxt = self._bw[bi + 1] if has_next else w0
            for tag, out in (("tmo", True), ("tmi", False)):
                nl = "tmi" if out else "tas"
                k_tm_proj[(mz, gzn, 3)](
                    self.ZLN, w[tag + "_W"], self.PM, self.A, self.B, self.GZ, R,
                    CZ=CZ, CM=CM, NW=NWTM, BM=_BM_Z, BN=_BN_Z, launch_pdl=pdl)
                k_tm_ein[(CM // _BC,)](
                    self.A, self.B, self.XS, N=N, CM=CM, BC=_BC, OUTGOING=out,
                    launch_pdl=pdl)
                k_tm_lno[(mln,)](
                    self.XS, w[tag + "_low"], w[tag + "_lob"], self.XL, R,
                    CM=CM, BM=_BM_LN, EPS=_EPS, launch_pdl=pdl)
                k_tm_fin[(mz,)](
                    self.Z, self.XL, self.GZ, w[tag + "_Wz"],
                    w[nl + "_lnw"], w[nl + "_lnb"], self.ZLN, R,
                    CZ=CZ, CM=CM, BM=_BM_Z, EPS=_EPS, HAS_NEXT=True, launch_pdl=pdl)
            for tag, start in (("tas", True), ("tae", False)):
                nl = "tae" if start else "pt"
                k_ta_proj[(mz, NTA // _BN_Z)](
                    self.ZLN, w[tag + "_W"], self.QK, self.TB, R,
                    CZ=CZ, NTA=NTA, HP=HP, NB=4 * HDP, BM=_BM_Z, BN=_BN_Z,
                    launch_pdl=pdl)
                k_ta_att[(N, HP)](
                    self.QK, self.TB, self.PM, self.OSA, R,
                    N=N, HP=HP, DP=DP, NTA=NTA,
                    INF=inf, SQD=sqdp, STARTING=start, launch_pdl=pdl, num_warps=1)
                k_ta_out[(mz,)](
                    self.Z, self.OSA, w[tag + "_Wo"], w[nl + "_lnw"],
                    w[nl + "_lnb"], self.ZLN, R,
                    CZ=CZ, HD=HDP, BM=_BM_Z, EPS=_EPS, HAS_NEXT=True, launch_pdl=pdl)
            k_swiglu[(mz, NPT // _BN_Z)](
                self.ZLN, w["pt_Wa"], w["pt_Wb"], self.HZ, R,
                K=CZ, KP=CZ, NH=NPT, BM=_BM_Z, BN=_BN_Z, launch_pdl=pdl)
            k_trans_out[(mz, CZ // _BN_PTO)](
                self.HZ, w["pt_Wo"], self.PM, self.DZ, R,
                KH=NPT, CO=CZ, BM=_BM_Z, BN=_BN_PTO, BK=_BK_CHUNK, launch_pdl=pdl)
            k_pt_tail[(mln,)](
                self.Z, self.DZ, nxt["tmo_lnw"], nxt["tmo_lnb"],
                w["apb_lzw"], w["apb_lzb"], self.ZLN, self.ZSB[bi], R,
                CZ=CZ, BM=_BM_LN, EPS=_EPS, HAS_NEXT=has_next, launch_pdl=pdl)
            # ---- fork the single track onto a second graph branch -----------
            # The pair stack never reads s (baseline: z = pair_stack(z) first,
            # then s += attn_pair_bias(s, z)), so the 48-block z-chain is the
            # only true critical path and each block's s-update is a tap off it
            # that needs nothing but BIAS[bi] and s_{bi-1}.  Recording an event
            # on the capturing stream and waiting on it from the side stream
            # makes the side stream part of the *same* graph, as a parallel
            # branch: the 26.5 us/block single track then runs under the next
            # block's 43 us/block pair track instead of after it.
            if two:
                ev = self._ev_fork[bi]
                ev.record(zs)
                ss.wait_event(ev)
                with torch.cuda.stream(ss):
                    s_block(w, bi)
            else:
                s_block(w, bi)
        # Single join, at the end of the stack: nothing on the pair branch ever
        # reads a single-track buffer, so per-block joins would only add nodes
        # to the critical path.
        if two:
            self._ev_join.record(ss)
            zs.wait_event(self._ev_join)

    # -- graph build / replay -------------------------------------------------
    def _copy_in(self, s, z, single_mask, pair_mask):
        self._zbuf.copy_(z)
        self._sbuf.copy_(s)
        self._pmbuf.copy_(pair_mask)
        self._smbuf.copy_(single_mask)
        self.S32.copy_(self._sbuf.view(-1, self.dims["CS"]))

    def _probes(self):
        b0, bl = self.blocks[0], self.blocks[-1]
        return (b0.attn_pair_bias.mha.linear_q.weight,
                b0.pair_stack.tri_mul_out.linear_a_p.weight,
                bl.single_transition.linear_out.weight,
                bl.pair_stack.tri_att_end.mha.linear_o.weight)

    def _probe_state(self):
        return tuple((t.data_ptr(), t._version) for t in self._probes())

    def _build(self, s, z, single_mask, pair_mask):
        self._pack(z.shape[1], z.device)
        self._copy_in(s, z, single_mask, pair_mask)
        cur = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(cur)
        self._aten = []
        with torch.cuda.stream(side):
            for _ in range(3):
                self._run()
        cur.wait_stream(side)
        torch.cuda.synchronize()
        # Release the warm-up runs' ATen outputs *before* capture starts.  The
        # two layernorms and the q projection are ATen calls, so each _run
        # allocates; freeing those blocks from inside the capture region (which
        # is what clearing this list inside _run did) hands the capture's own
        # allocator blocks that the warm-up still owned on another stream, and
        # the replayed graph then reads the wrong buffers.
        self._aten = []
        torch.cuda.synchronize()
        if _NO_GRAPH:
            self._graph = _NoGraph()
            self._probe = self._probe_state()
            return
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._run()
        torch.cuda.synchronize()
        self._graph = g
        self._probe = self._probe_state()

    # -- reference (fallback / oracle) ---------------------------------------
    @staticmethod
    def _pfd(t, inds):
        zi = -len(inds)
        first = list(range(len(t.shape[:zi])))
        return t.permute(first + [zi + i for i in inds])

    def _tm(self, tm, z, mask):
        m = mask.unsqueeze(-1)
        zl = tm.layer_norm_in(z)
        a = m * torch.sigmoid(tm.linear_a_g(zl)) * tm.linear_a_p(zl)
        b = m * torch.sigmoid(tm.linear_b_g(zl)) * tm.linear_b_p(zl)
        if tm._outgoing:
            aa, bb = self._pfd(a, (2, 0, 1)), self._pfd(b, (2, 1, 0))
        else:
            aa, bb = self._pfd(a, (2, 1, 0)), self._pfd(b, (2, 0, 1))
        x = self._pfd(torch.einsum("...ij,...jk->...ik", aa, bb), (1, 2, 0))
        x = tm.layer_norm_out(x)
        x = tm.linear_z(x)
        return x * torch.sigmoid(tm.linear_g(zl))

    def _mha(self, mha, q_x, kv_x, biases, no_heads):
        q = mha.linear_q(q_x)
        k = mha.linear_k(kv_x)
        v = mha.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (no_heads, -1)).transpose(-2, -3)
        k = k.view(k.shape[:-1] + (no_heads, -1)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (no_heads, -1)).transpose(-2, -3)
        q = q / math.sqrt(q.shape[-1])
        sc = torch.einsum("...qc,...kc->...qk", q, k)
        for b in biases:
            sc = sc + b
        sc = F.softmax(sc, dim=-1)
        o = torch.einsum("...qk,...kc->...qc", sc.to(dtype=v.dtype), v)
        o = o.transpose(-2, -3)
        if mha.linear_g is not None:
            g = torch.sigmoid(mha.linear_g(q_x))
            o = o * g.view(g.shape[:-1] + (no_heads, -1))
        return mha.linear_o(o.reshape(o.shape[:-2] + (-1,)))

    def _ta(self, ta, x, mask):
        if not ta.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        x = ta.layer_norm(x)
        mb = (self.inf * (mask - 1))[..., :, None, None, :]
        tb = self._pfd(ta.linear_z(x), (2, 0, 1)).unsqueeze(-4)
        out = self._mha(ta.mha, x, x, [mb, tb], self.no_heads_pair)
        return out.transpose(-2, -3) if not ta.starting else out

    def _apb(self, ab, a, z, mask):
        mb = (self.inf * (mask.expand(a.shape[:-2] + (-1,)) - 1))[..., None, None, :]
        zb = self._pfd(ab.linear_z(ab.layer_norm_z(z)), (2, 0, 1))
        an = ab.layer_norm_a(a)
        return self._mha(ab.mha, an, an, [mb, zb], self.no_heads_pair_bias)

    @staticmethod
    def _trans(tr, x, mask):
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        m = mask.unsqueeze(-1)
        y = tr.layer_norm(x)
        h = F.silu(tr.swiglu.linear_a(y)) * tr.swiglu.linear_b(y)
        return tr.linear_out(h) * m

    def _forward_eager(self, s, z, single_mask, pair_mask, mask_trans):
        for blk in self.blocks:
            ps = blk.pair_stack
            z = z + self._tm(ps.tri_mul_out, z, pair_mask)
            z = z + self._tm(ps.tri_mul_in, z, pair_mask)
            z = z + self._ta(ps.tri_att_start, z, pair_mask)
            z = z + self._ta(ps.tri_att_end, z, pair_mask)
            z = z + self._trans(ps.pair_transition, z,
                                pair_mask if mask_trans else None)
            s = s + self._apb(blk.attn_pair_bias, s, z, single_mask)
            s = s + self._trans(blk.single_transition, s,
                                single_mask if mask_trans else None)
        return s, z

    # -- public ---------------------------------------------------------------
    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:           [*, N_token, C_s] single embedding
            z:           [*, N_token, N_token, C_z] pair embedding
            single_mask: [*, N_token] single mask
            pair_mask:   [*, N_token, N_token] pair mask

        Returns:
            (s, z): updated single and pair embeddings
        """
        if (not torch.is_grad_enabled()
                and self._eligible(s, z, single_mask, pair_mask, _mask_trans)):
            key = (tuple(z.shape), tuple(s.shape))
            if (self._graph is None or self._key != key
                    or self._probe != self._probe_state()):
                self._graph = None
                self._key = key
                self._build(s, z, single_mask, pair_mask)
            # Always (re-)stage the inputs: capture-time warmup runs the stack
            # in place, so the static buffers hold spent state afterwards.  This
            # is also what makes the replay read *this* call's inputs rather
            # than silently returning the previous invocation's outputs.
            self._copy_in(s, z, single_mask, pair_mask)
            if _NO_GRAPH:
                self._run()
            else:
                self._graph.replay()
            return self._sbuf.clone(), self._zbuf.clone()
        return self._forward_eager(s, z, single_mask, pair_mask, _mask_trans)
