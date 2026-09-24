"""PairFormer stack for AlphaFold3 -- the whole 48-block stack as one CUDA graph.

48-block PairFormer: each block runs a PairBlock on pair (z) then
AttentionPairBias + SwiGLUTransition on single (s).

Reference: openfold3/core/model/latent/pairformer.py PairFormerStack

What this operator actually costs
--------------------------------
Captured init is ``c_s=384, c_z=128, c_hidden_pair_bias=24, no_heads_pair_bias=16,
c_hidden_mul=128, c_hidden_pair_att=32, no_heads_pair=4, transition_n=4,
no_blocks in {48, 4}``; forward is ``s:bf16[1,16,384] z:bf16[1,16,16,128]``.
That is 256 pair rows and 16 tokens -- a few hundred microseconds of *arithmetic*
spread over 48 blocks, against ~6600 eager kernel launches and ~90 ms of host
Python.  Nothing here is bandwidth- or FLOP-bound; the whole operator is launch
and dispatch bound, and the only two levers are (a) how many kernels a block
needs and (b) whether the host has to enqueue them at all.

Why numerics decide the design
------------------------------
This is the part that is not obvious from the shapes, and it is what the round-1
measurements went into (see ITERATIONS.md).  A 48-block residual stack amplifies
rounding disagreement: perturbing the *input* ``z`` by one bf16 ulp moves the
reference's own output by 2.8e-1 max / 4.5e-2 rms, which is only 37% of elements
inside the bench's ``atol=rtol=1e-2`` band.  Error injected per block accumulates
roughly linearly in the block count, and the measured curve is

    z rms err   1.8e-4  1.7e-3  4.0e-3  7.7e-3  1.4e-2  2.7e-2  4.0e-2
    blocks           1       2       4       8      16      32      48
    matched     1.0000  1.0000  0.9994  0.9823  0.8311  0.5571  0.4162

so passing at 48 blocks needs the *per-block* deviation from the reference held
under ~2 single-element bf16 flips.  Two consequences:

* The frozen L2 ``PairBlock`` winner cannot be used for the z track here.  It is
  a genuinely excellent standalone block (33.7x, 1-2 ulp per call) but it keeps
  its intermediates in fp32 across points where the reference rounds to bf16.
  Benched unchanged inside this stack it scores **INCORRECT_NUMERICAL, matched
  0.8690** -- the import is kept (it owns the pair parameters and their
  ``state_dict`` names) but the compute is re-derived here with the reference's
  rounding points reproduced exactly.
* Every rounding point of the reference is reproduced: bf16 after each
  ``F.linear``, after each ``sigmoid``/``silu``, after each softmax, after each
  gated product, and after each residual add.  Measured against the eager
  reference this is what turns 0.42 matched into 1.0000.

The three primitives that are not what they look like
----------------------------------------------------
``tl.dot`` is **bit-exact** against cuBLAS bf16 GEMM at every shape this operator
uses -- 0 flips over 5.2M elements, at every reduction chunk size from 16 to K --
so chunking a contraction is numerically free. Three other things are not:

* **``tl.exp`` is not ``expf``.** It lowers to ``ex2.approx.f32`` and disagrees
  with ``torch.exp`` on 73% of fp32 results (rel err up to 8.4e-7).
  ``libdevice.exp`` is bit-exact over 4.2M samples. Invisible in one op; worth
  ~0.6 bf16 flips per block here.
* **Triton's fp32 ``/`` is not ``div.rn``.** It disagrees with torch's divide on
  30% of fp32 results. Both sigmoids, the softmax normalisation,
  ``q / sqrt(c_hidden)`` and every divide inside the layer norm need ``div_rn``.
* **ATen's softmax reduces with ``WARP_SHFL_XOR`` at offsets N/2..1** -- a
  balanced halving tree, not the order ``tl.sum`` picks for a 3-D tile.
  ``_rowsum`` reproduces it with ``tl.reduce`` over successive size-2 axes.

With all three, softmax goes from a 2.0e-5 flip rate to 0.

Reproducing ATen's layer norm
-----------------------------
``F.layer_norm``'s moments come from ATen's Welford tree, and a plain two-pass
fp32 norm is worth matched **0.4326** at 48 blocks on its own -- with every other
op exact. ``_wmom`` reproduces ``vectorized_layer_norm_kernel`` instead: four
consecutive elements per lane folded in with ``cuWelfordOnlineSum``, then the same
halving tree across lanes. Two details decide it, and neither is visible in a
small sample:

* every divide is ``div_rn`` (the reason earlier attempts looked hopeless);
* ``cuWelfordOnlineSum``'s ``mean + delta * (1/new_count)`` is **contracted into
  an FMA** by nvcc. Reproducing that is worth going from a 3e-7 flip rate to 0
  (measured over 13.1M bf16 outputs).

Claimed only for ``c <= 128``: four elements per lane means one active warp there,
so the cross-warp combine never runs with unequal counts, which is where nvcc's
contraction diverges again (``c_s = 384`` gives three active warps and a 5e-7
residual). So the seven norms of the *pair* track -- all over ``c_z`` or
``c_hidden_mul``, both 128 -- are fused into their consumers at zero launch cost,
and the single track's two ``c_s``-wide norms keep their ATen call. They sit on
the graph branch that has slack, so they cost nothing.

The shape of a block
--------------------
17 launches per block, 10 of them on the critical path:

  z track (10)      tm_proj . tm_einsum . tm_epi+next_proj  (x2: outgoing, incoming)
                    ta_attn                                 (x2: starting, ending)
                    ta_proj  (ending node only)
                    sg_hidden . sg_out
  s track (7)       ln_a . s_proj . s_gate . s_out . ln_t . sg_hidden . sg_out

Every one of those carries its own layer norm. ``s_proj`` runs the z->pair-bias
projection and the q/k/v/g projection as two task ranges of one grid.
``sg_hidden``/``sg_out`` are shared between the pair transition and the single
transition -- same math, different constexprs.

Each triangle multiplication's epilogue *can* be folded into the projection that
consumes it (``_tm_epi_proj`` / ``_tm_epi_taproj``, gated by ``_FUSE_EPI``): the
epilogue's program owns the whole row, so the residual reaches the next stage's
LayerNorm in registers rather than through L2 and a second dispatch, and the
critical path becomes 10 nodes per block instead of 12. Measured interleaved on
graph replay that is worth only **-0.76%** -- two dispatches and two dependent L2
round trips came out of the chain and almost nothing happened, because PDL already
overlaps a node's dispatch with its predecessor's execution. Worth having, but the
number is the useful part: it says the *count* of serialization points is not what
this operator is paying for (see ITERATIONS.md).

Two branches, one graph
-----------------------
816 launches for 48 blocks. Enqueued from Python that is ~11 ms of host time with
the GPU idle through it; as CUDA-graph nodes the host cost is one
``cudaGraphLaunch`` (0.040 ms, flat in block count) and what is left is
device-side work. Both numbers are in ITERATIONS.md per block count.

The graph is **two branches**, not one chain: the z track never reads ``s``, and
the s track of block ``i`` reads only ``ZST[i+1]``, which the z track writes once
and never touches again. So the s track of block ``i`` runs concurrently with the
z track of block ``i+1``, with one event per block as the only cross edge. Worth
4.12 -> 2.82 ms, both because the critical path drops from 19 nodes per block to
12 and because these kernels use 8-144 programs on 148 SMs and leave the memory
system idle on their own.

The caller's four inputs are staged into fixed buffers by one ``_ingest`` kernel
before the replay and the two outputs are copied out by one ``_egress`` kernel
after, which is what lets every node in between keep a constant argument list
(the harness hands a different ``data_ptr`` every iteration). Because capture
happens once, the launch path inside the graph does not need the frozen winners'
hand-rolled C-entry-point launcher -- ``kernel[grid](...)`` is free when it runs
once. ``launch_pdl=True`` is still set so the captured nodes carry the
programmatic-dependent-launch attribute (measured inside a graph on trivial nodes:
0.62 us/node against 0.87 without), and every kernel issues its weight and affine
loads above ``gdc_wait()``. That prefetch is not a small effect and it is already
complete: replacing every weight load in ``_tm_proj`` with a constant changes its
node cost by 0.07 us out of 4.5, so there is nothing left for a deeper pipeline
(TMA, shared-memory double buffering, warp specialisation) to hide.

Not the bottleneck, checked
---------------------------
The weight arena is 300 MiB, so the harness's L2 flush is irrelevant: timed with
and without it, 2.806 vs 2.802 ms. Nothing here is bandwidth- or miss-bound, and
an L2 prefetch prologue (which the frozen L2 winner needed) would buy nothing.

Neither is the *number* of serialization points, which is the surprise of round 2.
A device-side grid barrier -- what a persistent megakernel would use in place of a
graph node -- costs 0.94 us at 16 CTAs and 2.6 us at 148, against 0.62 us for the
PDL node it would replace, so collapsing the chain into one launch would be slower
at every grid size this operator uses. And fusing two dependent stages into one
kernel so the residual never leaves registers (``_FUSE_EPI``, 12 critical nodes
per block down to 10) is worth 0.76%, because PDL already overlaps a node's dispatch
with its predecessor's execution. What is left after that is just the sum of the
dependent in-kernel work, which is why round 2's gains all came from making that
work cheaper: the Welford tree and the warp count below.

Numerics detail that is easy to get wrong: the 16 heads of the single track's
attention have ``c_hidden=24``, which is not a power of two. q/k/v/g are stored
pre-transposed into a padded ``[c_s, 4*16*32]`` head layout with zero columns at
the padding lanes, which is exactly neutral for the score dot and ``p @ v``
(adding zeros inside an MMA is exact) -- so the attention needs no masks at all.
It is **not** neutral for ``linear_o``, which contracts the head axis: a padded
contraction groups its fp32 partials differently from cuBLAS on the packed axis,
measured at a 6.5e-5 flip rate and a failing run on its own. ``_s_gate`` therefore
stores the gated output with the padding lanes dropped and ``_s_out`` contracts
the packed axis.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

try:  # Triton 3.6+; probed, not assumed
    from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False

from ..L2.alphafold3_attention_pair_bias import AttentionPairBias
from ..L2.alphafold3_pair_block import PairBlock
from ..L2.alphafold3_swiglu_transition import SwiGLUTransition


__targets__ = ["PairFormerStack"]

# Input dtypes the fused path handles. fp32 is excluded: the reference's F.linear
# runs an exact-fp32 cuBLAS kernel there while tl.dot would drop to tf32, and the
# fp32 tolerance (atol 1e-5) does not cover that gap.
_FUSED_DTYPES = (torch.bfloat16, torch.float16)

_PDL = _HAS_PDL

# Tile shapes. The pattern from the frozen L2 winners holds here too: the column
# block wants to be 32-64 (smaller stops feeding the MMA, larger concentrates
# weight bytes into fewer programs) and the row block wants the grid at >= ~32
# programs. Swept in ITERATIONS.md.
_BR = 16      # pair rows per program (projections / epilogues)
_CB = 64      # output columns per program (triangle multiplication, transition)
_CQ = 64      # output columns per program (q/k/v/g/bias projections)
_CA = 32      # output columns per program (attention out-projection)
_EB = 8       # einsum channels per program (batch dim of its 3-D dot)
_SB = 8       # transition out-projection: pair rows per program
_ST = 32      # transition out-projection: columns per program
_TK = 256     # transition out-projection: reduction chunk
_KB = 128     # reduction chunk for non-power-of-2 contractions (c_s = 384)
_BRZ = 32     # pair rows per program for the z->pair-bias projection
_IBL = 1024   # _ingest / _egress elements per program

# Widest layer norm the Welford path claims bit-exactness for: four elements per
# lane and at most one active warp (see ``_wmom``). Anything wider keeps its ATen
# call. Set to 0 to route every norm through ATen.
_LN_FUSE_MAX = 128

# Fold each ``_tm_epi`` into the projection that consumes it (see ``_epi_row``),
# turning the z track's 12 critical nodes per block into 10. Worth -0.76% at 48
# blocks and -0.78% at 4, measured interleaved in one process -- far less than the
# two dispatches it removes would suggest, because PDL already overlaps a node's
# dispatch with its predecessor's execution. Set to 0 for r1's unfused chain.
_FUSE_EPI = 1

# How a bit-exact-able norm is spent: 1 = recompute it inside every consumer task
# (no launch, redundant moments), 2 = one ``_lnorm`` launch feeding all consumers.
_LN_MODE = 1

_BRL = 16     # rows per program in the standalone norm kernel

_WARPS = 8    # the single track
# The z track wants 4. r1 measured 8 there, against the old `tl.reduce` Welford
# tree; the tree is now a permute + split over registers whose cost falls with
# elements per thread (16x32 over 128 threads is 4 each, over 256 threads only 2),
# so halving the warps makes more of every level intra-thread. Re-measured per
# kernel as 200-node chains: `_tm_epi` 3.28 -> 2.74, `_sg_hidden` 3.75 -> 3.35,
# `_ta_proj` 3.01 -> 2.73, `_tm_proj` 3.68 -> 3.49, `_tm_einsum` neutral; 16 warps
# is far worse everywhere (57.3 us/block against 33.0). The single track keeps 8:
# it sits on the branch with slack, and its `_dot_k` loops contract 384 and 1536,
# where the warp count can change how Triton splits a dot.
_WARPSZ = 4
_WARPS4 = 4   # the attentions and both transition out-projections


# ---------------------------------------------------------------------------
# Shared device helpers
# ---------------------------------------------------------------------------
@triton.jit
def _exp(x):
    """``expf``. NOT ``tl.exp``.

    ``tl.exp`` lowers to ``ex2.approx.f32``, which differs from CUDA's ``expf`` --
    and therefore from every ``std::exp`` inside an ATen kernel -- in 73% of fp32
    results (relative error up to 8.4e-7). That is far below bf16's ulp, so it is
    invisible in a single op, but at 48 residual blocks it is worth ~0.6 bf16
    flips per block, which is most of a failing run. libdevice's ``exp`` is
    bit-exact against ``torch.exp`` over 4.2M samples.
    """
    return tl.extra.cuda.libdevice.exp(x)


@triton.jit
def _div(a, b):
    """Correctly-rounded fp32 divide. NOT ``a / b``.

    Triton's ``/`` on fp32 is not ``div.rn``: it disagrees with torch's divide on
    30% of fp32 results. Every divide the reference performs -- the softmax
    normalisation, both sigmoids, ``q / sqrt(c_hidden)`` -- has to be ``div_rn``
    or the stack drifts out of tolerance by block 48.
    """
    return tl.math.div_rn(a, b)


@triton.jit
def _sig(x):
    """sigmoid in fp32, bit-exact against torch.sigmoid on bf16 in/out."""
    return _div(1.0, 1.0 + _exp(-x))


@triton.jit
def _add2(a, b):
    return a + b


@triton.jit
def _lvl(x, LEAD: tl.constexpr, QN: tl.constexpr, L: tl.constexpr):
    """One level of a halving reduction tree over the last axis (pairs k, k+L/2)."""
    return tl.reduce(tl.reshape(x, (LEAD, QN, 2, L // 2)), 2, _add2)


@triton.jit
def _rowsum(p, LEAD: tl.constexpr, QN: tl.constexpr, N: tl.constexpr):
    """Sum over the key axis in ATen's warp-softmax order.

    ``softmax_warp_forward`` reduces with ``WARP_SHFL_XOR`` at offsets N/2..1,
    which is a balanced halving tree -- not the order ``tl.sum`` happens to pick
    for a 3-D tile (its per-thread partials come first). Reproducing the tree is
    worth another ~2x on the flip rate on top of ``expf``.
    """
    if N == 16:
        q = _lvl(_lvl(_lvl(_lvl(p, LEAD, QN, 16), LEAD, QN, 8), LEAD, QN, 4),
                 LEAD, QN, 2)
    elif N == 32:
        q = _lvl(_lvl(_lvl(_lvl(_lvl(p, LEAD, QN, 32), LEAD, QN, 16), LEAD, QN, 8),
                      LEAD, QN, 4), LEAD, QN, 2)
    else:
        q = _lvl(_lvl(_lvl(_lvl(_lvl(_lvl(p, LEAD, QN, 64), LEAD, QN, 32),
                               LEAD, QN, 16), LEAD, QN, 8), LEAD, QN, 4),
                 LEAD, QN, 2)
    return tl.reshape(q, (LEAD, QN))


@triton.jit
def _rnd(x, DT: tl.constexpr):
    """Round an fp32 tile to the storage dtype and back -- one reference rounding."""
    return x.to(DT).to(tl.float32)


@triton.jit
def _cw(mA, sA, mB, sB, CA: tl.constexpr, CB: tl.constexpr):
    """``cuWelfordCombine`` from ATen's layer_norm_kernel.cu, op for op.

    Symmetric in fp32 (`delta` only enters squared, and the adds commute), so the
    A/B assignment does not matter -- only the *tree* does.

    The counts are ``tl.constexpr`` rather than tensors, which is what makes the
    tree cheap. Inside the tree every count is a power of two and the two sides
    are always equal, so ``coef = 1/(cA+cB)`` is exactly representable and folding
    it at compile time changes no bit -- the ``div.rn`` it replaces was only ever
    needed because Triton's ``/`` is not correctly rounded, and there is nothing
    to round here. For the same reason the ``count > 0`` selects can never fire,
    and the count does not have to be carried through the tree as a third tile at
    all. (The *online* sums below keep their ``div_rn``: their counts are 1, 2, 3,
    4 and 1/3 is not exact.)
    """
    delta = mB - mA
    COEF: tl.constexpr = 1.0 / (CA + CB)
    NA: tl.constexpr = CA * COEF
    NB: tl.constexpr = CB * COEF
    return (NA * mA + NB * mB, sA + sB + delta * delta * NA * NB * (CA + CB))


@triton.jit
def _lvlw(m, s, BR: tl.constexpr, L: tl.constexpr, C: tl.constexpr):
    """One halving level of the Welford tree over the lane axis (pairs l, l+L/2).

    ``tl.reduce`` over the *middle* axis of ``[BR, 2, L/2]`` has to bring the two
    partners into the same lane, which Triton does with a layout conversion
    through shared memory -- and five levels of it on three tiles is most of what
    the fused norm costs (measured: 2.08 us of ``_tm_proj``'s 4.47 us node; ncu
    charges 9% of stalls to ``barrier`` and reports 45504 excessive shared
    wavefronts). Transposing the halves into a size-2 *last* axis and taking them
    apart with ``tl.split`` is a register-level extract, and the pairs -- hence
    every add -- are identical. Measured: 4.468 -> 3.985 us/node, 0 flips in 8.4M
    bf16 outputs.

    Reversing the leaf order instead, so the pairs are already adjacent and the
    transpose is unnecessary, was also tried: it makes the four leaf loads a
    permutation within the row and costs more than it saves (4.658 us/node).
    """
    mA, mB = tl.split(tl.permute(tl.reshape(m, (BR, 2, L // 2)), (0, 2, 1)))
    sA, sB = tl.split(tl.permute(tl.reshape(s, (BR, 2, L // 2)), (0, 2, 1)))
    return _cw(mA, sA, mB, sB, C, C)


@triton.jit
def _wstep(mean, sig, cnt, val):
    """One ``cuWelfordOnlineSum``.

    ``mean + delta * (1/new_count)`` and ``sigma2 + delta * (val - new_mean)``;
    nvcc contracts both into FMAs, and reproducing that is not optional -- with
    plain mul+add the moments are off by a fraction of an fp32 ulp, which is a
    3e-7 bf16 flip rate and a failing 48-block run. Measured: 0 flips in 13.1M
    outputs with the FMAs, 4-5 without. The reciprocal keeps its ``div_rn``: the
    counts here are 1, 2, 3, 4 and 1/3 is not exact.
    """
    delta = val - mean
    nc = cnt + 1.0
    nm = tl.math.fma(delta, _div(1.0, nc), mean)
    return nm, tl.math.fma(delta, val - nm, sig), nc


@triton.jit
def _wmom(x, C: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr):
    """ATen's layer-norm moments, bit-exact, from a tile already in registers.

    ``vectorized_layer_norm_kernel`` gives lane ``t`` the four consecutive
    elements ``4t..4t+3``, folds them in with ``cuWelfordOnlineSum``, then reduces
    across lanes with ``WARP_SHFL_XOR`` at offsets 16..1 -- a balanced halving
    tree. Both halves are reproduced here. Only ``C/4 <= 32`` is claimed exact
    (one active warp, so the cross-warp combine never runs with unequal counts,
    where nvcc's FMA contraction diverges); the plan sends anything wider to ATen.

    The four leaf tiles come out of the consumer's own ``[BR, C]`` tile rather
    than from four more strided loads of the same row: splitting ``[BR, T, 2, 2]``
    on its last axis gives ``(e0, e2)`` and ``(e1, e3)``, and splitting those
    gives the four in order. Same values, same folding order -- and the row is
    read once instead of five times (measured 3.98 -> 3.65 us/node on ``_tm_proj``,
    0 flips in 5.5M bf16 outputs at C = 32 / 64 / 128).
    """
    T: tl.constexpr = C // 4
    a, b = tl.split(tl.reshape(x, (BR, T, 2, 2)))
    e0, e2 = tl.split(a)
    e1, e3 = tl.split(b)
    z = tl.zeros([BR, T], tl.float32)
    m, sg, c = _wstep(z, z, z, e0)
    m, sg, c = _wstep(m, sg, c, e1)
    m, sg, c = _wstep(m, sg, c, e2)
    m, sg, c = _wstep(m, sg, c, e3)
    # Counts entering each level: 4, 8, 16, ... one per lane pair.
    if T == 32:
        m, sg = _lvlw(m, sg, BR, 32, 4)
        m, sg = _lvlw(m, sg, BR, 16, 8)
        m, sg = _lvlw(m, sg, BR, 8, 16)
        m, sg = _lvlw(m, sg, BR, 4, 32)
        m, sg = _lvlw(m, sg, BR, 2, 64)
    elif T == 16:
        m, sg = _lvlw(m, sg, BR, 16, 4)
        m, sg = _lvlw(m, sg, BR, 8, 8)
        m, sg = _lvlw(m, sg, BR, 4, 16)
        m, sg = _lvlw(m, sg, BR, 2, 32)
    elif T == 8:
        m, sg = _lvlw(m, sg, BR, 8, 4)
        m, sg = _lvlw(m, sg, BR, 4, 8)
        m, sg = _lvlw(m, sg, BR, 2, 16)
    elif T == 4:
        m, sg = _lvlw(m, sg, BR, 4, 4)
        m, sg = _lvlw(m, sg, BR, 2, 8)
    elif T == 2:
        m, sg = _lvlw(m, sg, BR, 2, 4)
    mu = tl.reshape(m, (BR,))
    return mu, tl.rsqrt(_div(tl.reshape(sg, (BR,)), C) + EPS)


@triton.jit
def _lnrow(X, LNW, LNB, rows, C: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr,
           LNF: tl.constexpr, DT: tl.constexpr):
    """A layer-normed row tile, rounded to bf16.

    ``LNF = 1`` normalizes here; ``LNF = 0`` means ``X`` already holds an ATen
    norm's fp32 output (affine applied) and only the reference's
    ``.to(orig_dtype)`` is left. One code path, one constexpr.
    """
    c = tl.arange(0, C)
    x = tl.load(X + rows[:, None] * C + c[None, :])
    if LNF:
        mu, rs = _wmom(x, C, BR, EPS)
        x = rs[:, None] * (x - mu[:, None]) * tl.load(LNW + c)[None, :] \
            + tl.load(LNB + c)[None, :]
    return x.to(DT)


@triton.jit
def _wt(WT, col, K: tl.constexpr, NW: tl.constexpr, LD: tl.constexpr):
    """``[K, NW]`` slice of a pre-transposed weight starting at column ``col``."""
    return tl.load(WT + tl.arange(0, K)[:, None] * LD + (col + tl.arange(0, NW))[None, :])


@triton.jit
def _dot_k(X, W, rows, col, K: tl.constexpr, KB: tl.constexpr, NW: tl.constexpr,
           LDX: tl.constexpr, LDW: tl.constexpr, BR: tl.constexpr, DT: tl.constexpr):
    """``X[rows, :K] @ W[:K, col:col+NW]`` walking the reduction in ``KB`` chunks.

    Used where the contraction is not a power of two (``c_s = 384``,
    ``transition_n * c_s = 1536``): chunking keeps every tile a legal Triton
    shape without padding either operand. Measured bit-exact against cuBLAS at
    every chunk size from 16 to K, so the chunking is free numerically.
    ``X`` is fp32 holding bf16-rounded values (a layer-norm output).
    """
    acc = tl.zeros([BR, NW], tl.float32)
    for k0 in range(0, K, KB):
        kk = k0 + tl.arange(0, KB)
        x = tl.load(X + rows[:, None] * LDX + kk[None, :]).to(DT)
        w = tl.load(W + kk[:, None] * LDW + (col + tl.arange(0, NW))[None, :])
        acc += tl.dot(x, w)
    return acc


@triton.jit
def _lnorm(X, LNW, LNB, Y, C: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr,
           DT: tl.constexpr, PDL: tl.constexpr):
    """One layer norm as its own PDL kernel, writing bf16.

    The alternative to normalizing inside every consumer: a column-split consumer
    has 2-9 tasks per row block and would recompute the row's moments in each of
    them. Which of the two wins is measured, not assumed (see ITERATIONS.md) --
    and both beat the ATen call, whose 3.07 us node also breaks the PDL chain.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    lw = tl.load(LNW + c)[None, :]
    lb = tl.load(LNB + c)[None, :]
    if PDL:
        gdc_wait()
    x = tl.load(X + rows[:, None] * C + c[None, :])
    mu, rs = _wmom(x, C, BR, EPS)
    tl.store(Y + rows[:, None] * C + c[None, :],
             (rs[:, None] * (x - mu[:, None]) * lw + lb).to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Input staging / output copy-out (the two launches outside the graph)
# ---------------------------------------------------------------------------
@triton.jit
def _ingest(S, SD, Z, ZD, MS, MSD, MP, MPD,
           NS: tl.constexpr, NZ: tl.constexpr, NMS: tl.constexpr, NMP: tl.constexpr,
           BL: tl.constexpr, PDL: tl.constexpr):
    """Copy the caller's four inputs into the graph's fixed buffers.

    ``s`` and ``z`` land in fp32 buffers: every fused kernel that consumes them
    treats fp32-holding-bf16 as the canonical residual-stream layout, because the
    reference's own ``x.float()`` in front of each layer norm produces exactly
    that. The masks stay in their input dtype -- they are consumed as bf16.
    """
    o = tl.program_id(0) * BL + tl.arange(0, BL)
    if PDL:
        gdc_wait()
    m = o < NS
    tl.store(SD + o, tl.load(S + o, mask=m, other=0).to(tl.float32), mask=m)
    m = o < NZ
    tl.store(ZD + o, tl.load(Z + o, mask=m, other=0).to(tl.float32), mask=m)
    m = o < NMS
    tl.store(MSD + o, tl.load(MS + o, mask=m, other=0), mask=m)
    m = o < NMP
    tl.store(MPD + o, tl.load(MP + o, mask=m, other=0), mask=m)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _egress(SS, SO, ZS, ZO, NS: tl.constexpr, NZ: tl.constexpr,
            BL: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """Round the final fp32 residual streams into the caller's output tensors."""
    o = tl.program_id(0) * BL + tl.arange(0, BL)
    if PDL:
        gdc_wait()
    m = o < NS
    tl.store(SO + o, tl.load(SS + o, mask=m, other=0.0).to(DT), mask=m)
    m = o < NZ
    tl.store(ZO + o, tl.load(ZS + o, mask=m, other=0.0).to(DT), mask=m)
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Triangle multiplication (AF3 Algorithms 12 / 13)
# ---------------------------------------------------------------------------
@triton.jit
def _tm_task(t, CH: tl.constexpr, CB: tl.constexpr, NCA: tl.constexpr):
    """Decode a triangle-multiplication input-projection column task.

    Tasks ``[0, 2*NCA)`` produce a column block of ``a`` or ``b`` -- each needs the
    *same* column range of two different projections (the value and its sigmoid
    gate). The remaining tasks produce the output gate, which needs only one; it
    aims both weight offsets at the same slice and discards the second dot rather
    than branching, so the kernel stays divergence-free and every load can be
    issued before ``gdc_wait()``.

    Returns (is_ab, value-weight column, gate-weight column, output column).
    """
    ab = t < 2 * NCA
    cbi = tl.where(ab, t % NCA, t - 2 * NCA) * CB
    cp = tl.where(ab, (t // NCA) * 2 * CH + cbi, 4 * CH + cbi)
    return (ab, cp, cp + tl.where(ab, CH, 0),
            tl.where(ab, (t // NCA) * CH + cbi, 2 * CH + cbi))


@triton.jit
def _tm_proj(LNY, LNW, LNB, M, WT, ABG,
             C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
             NCA: tl.constexpr, EPS: tl.constexpr, LNF: tl.constexpr,
             DT: tl.constexpr, PDL: tl.constexpr):
    """``a``, ``b`` and the output gate from a layer-normed row, one column slice each.

    ``a = mask * sigmoid(linear_a_g(zln)) * linear_a_p(zln)`` is three bf16
    roundings in the reference (the two projections, the sigmoid, then each
    product); all of them are reproduced, which is what keeps 48 blocks inside
    tolerance. ``LNY`` is the ATen norm's fp32 output; rounding it to bf16 here is
    the reference's own ``.to(orig_dtype)``.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, C)
    ab, cp, cg, col = _tm_task(tl.program_id(1), CH, CB, NCA)
    wp = _wt(WT, cp, C, CB, 4 * CH + C)
    wg = _wt(WT, cg, C, CB, 4 * CH + C)
    if PDL:
        gdc_wait()
    m = tl.load(M + rows)[:, None].to(tl.float32)
    xl = _lnrow(LNY, LNW, LNB, rows, C, BR, EPS, LNF, DT)
    pv = _rnd(tl.dot(xl, wp), DT)
    sg = _rnd(_sig(_rnd(tl.dot(xl, wg), DT)), DT)
    val = tl.where(ab, _rnd(_rnd(m * sg, DT) * pv, DT), _rnd(_sig(pv), DT))
    tl.store(ABG + rows[:, None] * (2 * CH + C) + (col + tl.arange(0, CB))[None, :],
             val.to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _tm_einsum(ABG, X, N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
               EB: tl.constexpr, SAI: tl.constexpr, SAJ: tl.constexpr,
               SBK: tl.constexpr, SBJ: tl.constexpr, DT: tl.constexpr,
               PDL: tl.constexpr):
    """``x[i,k,c] = sum_j a[..,c] * b[..,c]`` as a batched dot over the channels.

    The contraction is not a matmul -- ``c`` is a free index shared by both
    operands -- but it *is* ``EB`` independent NxNxN matmuls, which is exactly a
    3-D ``tl.dot``. Splitting over channels makes every program's footprint two
    ``[EB, N, N]`` tiles and needs no weights at all.

    Outgoing vs incoming is entirely in the strides: outgoing contracts ``a[i,j]``
    with ``b[k,j]``, incoming ``a[j,i]`` with ``b[j,k]``. ``X`` is written fp32
    holding the bf16-rounded einsum result, ready for the ATen norm.
    """
    cb = tl.program_id(0) * EB
    bb = tl.program_id(1) * (N * N)
    c = cb + tl.arange(0, EB)
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    k = tl.arange(0, N)
    W: tl.constexpr = 2 * CH + C
    if PDL:
        gdc_wait()
    a3 = tl.load(ABG + (bb + i[None, :, None] * SAI + j[None, None, :] * SAJ) * W
                 + c[:, None, None])
    b3 = tl.load(ABG + (bb + k[None, None, :] * SBK + j[None, :, None] * SBJ) * W
                 + CH + c[:, None, None])
    tl.store(X + (bb + i[None, :, None] * N + k[None, None, :]) * CH + c[:, None, None],
             _rnd(tl.dot(a3, b3), DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _tm_epi(LNX, LNW, LNB, ABG, ZIN, ZOUT, WZT,
            C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
            EPS: tl.constexpr, LNF: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``linear_z(layer_norm_out(x)) * output_gate + z``, one column slice per program.

    Purely column-local: the reduction is over ``CH`` (which every program holds
    whole) and the gate, the residual and the store all live in the same column
    range, so unlike the frozen L2 block's epilogue there is nothing to
    redundantly recompute here -- extracting the norm removed that seam.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CB
    col = cb + tl.arange(0, CB)
    wz = _wt(WZT, cb, CH, CB, C)
    # ABG's gate columns were written two grids back (the einsum in between wrote
    # only X) and ZIN two grids back or earlier, so both are final already: their
    # loads belong above gdc_wait(), inside the PDL overlap window.
    g = tl.load(ABG + rows[:, None] * (2 * CH + C) + (2 * CH + col)[None, :]).to(tl.float32)
    zv = tl.load(ZIN + rows[:, None] * C + col[None, :])
    if PDL:
        gdc_wait()
    xl = _lnrow(LNX, LNW, LNB, rows, CH, BR, EPS, LNF, DT)
    upd = _rnd(_rnd(tl.dot(xl, wz), DT) * g, DT)
    tl.store(ZOUT + rows[:, None] * C + col[None, :], _rnd(zv + upd, DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _epi_row(LNX, LNW, LNB, ZOUT, wz, g, zv, rows, store,
             C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
             EPS: tl.constexpr, LNF: tl.constexpr, DT: tl.constexpr):
    """``_tm_epi`` over a program's *whole* row, returned in registers.

    Every residual write on the z track is immediately followed by a LayerNorm of
    exactly the rows it wrote, so a program that owns the full row can hand the
    result to the next stage in registers instead of storing it and having the
    next kernel dispatch, read it back through L2 and normalise it. Two of the
    twelve critical nodes per block go away this way, and it is worth **0.76%** --
    a lot less than the 0.62 us/node a trivial node costs, because PDL already
    overlaps node i+1's dispatch with node i's execution, so most of what a
    boundary appears to cost is not actually on the critical path.

    The row is still stored, because later stages read it, but only by the caller's
    task 0; the other tasks recompute the same bit-identical value, which is
    redundant arithmetic spread over parallel SMs rather than extra latency (all of
    these grids are one wave on 148 SMs). Nothing about the arithmetic changes: the
    ``[BR, C]`` tile here is what ``_tm_epi`` would have stored, and an fp32 store
    and reload is exact.
    """
    xl = _lnrow(LNX, LNW, LNB, rows, CH, BR, EPS, LNF, DT)
    znew = _rnd(zv + _rnd(_rnd(tl.dot(xl, wz), DT) * g, DT), DT)
    if store:
        tl.store(ZOUT + rows[:, None] * C + tl.arange(0, C)[None, :], znew)
    return znew


@triton.jit
def _ln_reg(x, LNW, LNB, C: tl.constexpr, BR: tl.constexpr, EPS: tl.constexpr,
            DT: tl.constexpr):
    """The reference's ``LayerNorm(x.float())`` on a tile already in registers."""
    c = tl.arange(0, C)
    mu, rs = _wmom(x, C, BR, EPS)
    return (rs[:, None] * (x - mu[:, None]) * tl.load(LNW + c)[None, :]
            + tl.load(LNB + c)[None, :]).to(DT)


@triton.jit
def _tm_epi_proj(LNX, LNW, LNB, ABG, ZIN, ZOUT, WZT,
                 L2W, L2B, M, WT2, ABG2,
                 C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
                 CB: tl.constexpr, NCA: tl.constexpr, EPS: tl.constexpr,
                 EPS2: tl.constexpr, LNF: tl.constexpr, DT: tl.constexpr,
                 PDL: tl.constexpr):
    """``_tm_epi`` of one triangle multiplication + ``_tm_proj`` of the next, one node.

    The grid is the *consumer's* (one program per row block x column task), so the
    parallelism is unchanged; what the producer costs is paid once per column task
    instead of once per launch. ``ABG2`` must not be ``ABG``: the gate columns this
    kernel reads are in the range its own gate tasks write, and there is no
    ordering between tasks of one launch.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    t = tl.program_id(1)
    c = tl.arange(0, C)
    ab, cp, cg, col = _tm_task(t, CH, CB, NCA)
    wz = _wt(WZT, 0, CH, C, C)
    wp = _wt(WT2, cp, C, CB, 4 * CH + C)
    wg = _wt(WT2, cg, C, CB, 4 * CH + C)
    g = tl.load(ABG + rows[:, None] * (2 * CH + C) + (2 * CH + c)[None, :]).to(tl.float32)
    zv = tl.load(ZIN + rows[:, None] * C + c[None, :])
    if PDL:
        gdc_wait()
    m = tl.load(M + rows)[:, None].to(tl.float32)
    znew = _epi_row(LNX, LNW, LNB, ZOUT, wz, g, zv, rows, t == 0,
                    C, CH, BR, EPS, LNF, DT)
    xl = _ln_reg(znew, L2W, L2B, C, BR, EPS2, DT)
    pv = _rnd(tl.dot(xl, wp), DT)
    sg = _rnd(_sig(_rnd(tl.dot(xl, wg), DT)), DT)
    val = tl.where(ab, _rnd(_rnd(m * sg, DT) * pv, DT), _rnd(_sig(pv), DT))
    tl.store(ABG2 + rows[:, None] * (2 * CH + C) + (col + tl.arange(0, CB))[None, :],
             val.to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _tm_epi_taproj(LNX, LNW, LNB, ABG, ZIN, ZOUT, WZT,
                   L2W, L2B, WT2, QKVG,
                   C: tl.constexpr, CH: tl.constexpr, BR: tl.constexpr,
                   LD: tl.constexpr, HD: tl.constexpr, CQ: tl.constexpr,
                   SQD: tl.constexpr, EPS: tl.constexpr, EPS2: tl.constexpr,
                   LNF: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``_tm_epi`` of the incoming triangle multiplication + ``_ta_proj``, one node.

    Only used for the *starting* triangle-attention node, whose row map
    (``SZI = N, SZJ = 1``) is the identity, so the rows this program projects are
    exactly the rows it just wrote. The ending node reads the transpose, where one
    program's output rows come from N different producer programs, and cannot fuse.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CQ
    c = tl.arange(0, C)
    wz = _wt(WZT, 0, CH, C, C)
    w = _wt(WT2, cb, C, CQ, LD)
    g = tl.load(ABG + rows[:, None] * (2 * CH + C) + (2 * CH + c)[None, :]).to(tl.float32)
    zv = tl.load(ZIN + rows[:, None] * C + c[None, :])
    if PDL:
        gdc_wait()
    znew = _epi_row(LNX, LNW, LNB, ZOUT, wz, g, zv, rows, cb == 0,
                    C, CH, BR, EPS, LNF, DT)
    xl = _ln_reg(znew, L2W, L2B, C, BR, EPS2, DT)
    v = _rnd(tl.dot(xl, w), DT)
    if cb < HD:
        v = _rnd(_div(v, SQD), DT)
    tl.store(QKVG + rows[:, None] * LD + (cb + tl.arange(0, CQ))[None, :], v.to(DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Attention: shared body for the triangle attentions and the single track
# ---------------------------------------------------------------------------
@triton.jit
def _gated_heads(QKVG, ZB, mb, qbase, kbase, zbase, hgi,
                 LD: tl.constexpr, HD: tl.constexpr, N: tl.constexpr,
                 D: tl.constexpr, HG: tl.constexpr, ZBLD: tl.constexpr,
                 DT: tl.constexpr):
    """One head group's biased, gated attention output as a ``[HG, N, D]`` tile.

    At ``N = 16`` the score matrix for one head is a single MMA tile, so there is
    no flash-style KV loop and no ``[H, Q, K]`` tensor in HBM -- the heads are
    independent NxN problems, which is exactly a 3-D ``tl.dot``. Working a group
    at a time keeps the live q/k/v/g tiles at ``[HG, N, D]`` instead of
    ``[H, N, D]``.

    Every rounding the reference performs is reproduced: bf16 after the score
    einsum, after each bias add, after the softmax, after ``p @ v``, after the
    sigmoid gate and after the gated product.
    """
    n = tl.arange(0, N)
    kk = tl.arange(0, N)
    h = hgi * HG + tl.arange(0, HG)
    hd = h[:, None, None] * D + tl.arange(0, D)[None, None, :]
    q = tl.load(QKVG + (qbase + n[None, :, None]) * LD + hd)
    kt = tl.load(QKVG + (kbase + kk[None, :, None]) * LD + (HD + hd))
    v = tl.load(QKVG + (kbase + kk[None, :, None]) * LD + (2 * HD + hd))
    g = tl.load(QKVG + (qbase + n[None, :, None]) * LD + (3 * HD + hd))
    tb = tl.load(ZB + (zbase + n[None, :, None] * N + kk[None, None, :]) * ZBLD
                 + h[:, None, None])
    sc = _rnd(tl.dot(q, tl.permute(kt, (0, 2, 1))), DT)
    sc = _rnd(sc + mb[None, None, :], DT)
    sc = _rnd(sc + tb.to(tl.float32), DT)
    p = _exp(sc - tl.max(sc, axis=2)[:, :, None])
    p = _div(p, _rowsum(p, HG, N, N)[:, :, None]).to(DT)
    o = _rnd(tl.dot(p, v), DT)
    return _rnd(o * _rnd(_sig(g.to(tl.float32)), DT), DT).to(DT)


@triton.jit
def _ta_proj(LNY, LNW, LNB, WT, QKVG,
             N: tl.constexpr, C: tl.constexpr, LD: tl.constexpr, HD: tl.constexpr,
             BR: tl.constexpr, CQ: tl.constexpr, SQD: tl.constexpr,
             SZI: tl.constexpr, SZJ: tl.constexpr, EPS: tl.constexpr,
             LNF: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``q / k / v / g`` and the triangle bias, one column slice per program.

    All five share one output buffer and one concatenated weight of width ``LD``,
    zero-padded so ``LD`` is a whole number of column blocks; every program is
    then the same shape -- one ``[C, CQ]`` weight slice and one store -- with no
    branch for the (only ``H`` wide) bias task. The ``1/sqrt(c_hidden)`` scale is
    applied here, to the *rounded* projection, because that is where the reference
    applies it (``q = q / sqrt(c_hidden)`` after ``F.linear`` has already rounded).

    ``SZI``/``SZJ`` are the pair matrix's row strides: swapping them turns the
    ending node's ``x = x.transpose(-2, -3)`` into the index map
    ``logical (i, j) -> actual (j, i)``, at no cost. The layer norm upstream is
    row-wise, so it does not care which row is which and runs on the untransposed
    tensor.
    """
    lr = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CQ
    c = tl.arange(0, C)
    nn2: tl.constexpr = N * N
    rows = (lr // nn2) * nn2 + ((lr % nn2) // N) * SZI + (lr % N) * SZJ
    w = _wt(WT, cb, C, CQ, LD)
    if PDL:
        gdc_wait()
    xl = _lnrow(LNY, LNW, LNB, rows, C, BR, EPS, LNF, DT)
    v = _rnd(tl.dot(xl, w), DT)
    if cb < HD:
        v = _rnd(_div(v, SQD), DT)
    tl.store(QKVG + lr[:, None] * LD + (cb + tl.arange(0, CQ))[None, :], v.to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _ta_attn(QKVG, M, WOT, ZIN, ZOUT,
             N: tl.constexpr, C: tl.constexpr, LD: tl.constexpr, HD: tl.constexpr,
             H: tl.constexpr, D: tl.constexpr, HG: tl.constexpr, NHG: tl.constexpr,
             CA: tl.constexpr, SMI: tl.constexpr, SMJ: tl.constexpr,
             SZI: tl.constexpr, SZJ: tl.constexpr, INF: tl.constexpr,
             DT: tl.constexpr, ZE: tl.constexpr, PDL: tl.constexpr):
    """One triangle attention: attention + gate + ``linear_o`` + the residual add.

    One program per (outer index, output column block). ``linear_o`` contracts the
    head axis, so splitting the output columns across programs re-runs the (tiny)
    attention per block and in exchange cuts each program's ``linear_o`` slice to
    ``[H*D, CA]``. The mask bias and the triangle bias are consumed inline and
    never materialized; starting vs ending node is index math only.
    """
    i = tl.program_id(0)
    cb = tl.program_id(1) * CA
    bb = tl.program_id(2) * (N * N)
    col = cb + tl.arange(0, CA)
    zo = (bb + i * SZI + tl.arange(0, N) * SZJ)[:, None] * C + col[None, :]
    # The mask is the caller's own tensor. ``ZE`` marks the ending node, whose
    # residual was written two grids back and so is final already; the starting
    # node's comes from its immediate predecessor and has to wait.
    mb = _rnd(INF * (tl.load(M + bb + i * SMI + tl.arange(0, N) * SMJ).to(tl.float32)
                     - 1.0), DT)
    # ``linear_o``'s slice is a fixed slice of the arena, so it belongs above the
    # wait like every other weight -- measured 3.43 -> 3.27 us/node. Only the
    # single-group case is hoisted: with NHG > 1 the groups' slices would have to
    # be indexed out of one tile, which Triton has no operation for, and NHG is 1
    # for every configuration that reaches this kernel (the head axis is split
    # only in ``_s_gate``, which contracts it in a separate launch).
    if NHG == 1:
        wo = tl.load(WOT + tl.arange(0, HG * D)[:, None] * C
                     + (cb + tl.arange(0, CA))[None, :])
    if ZE:
        zres = tl.load(ZIN + zo)
    if PDL:
        gdc_wait()
    if not ZE:
        zres = tl.load(ZIN + zo)
    acc = tl.zeros([N, CA], tl.float32)
    for hgi in tl.static_range(NHG):
        oc = tl.reshape(tl.permute(_gated_heads(QKVG, QKVG + 4 * HD, mb, bb + i * N,
                                                bb + i * N, bb, hgi, LD, HD, N, D, HG,
                                                LD, DT), (1, 0, 2)), (N, HG * D))
        if NHG == 1:
            w = wo
        else:
            w = tl.load(WOT + (hgi * HG * D + tl.arange(0, HG * D))[:, None] * C
                        + (cb + tl.arange(0, CA))[None, :])
        acc += tl.dot(oc, w)
    tl.store(ZOUT + zo, _rnd(zres + _rnd(acc, DT), DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# SwiGLU transition (AF3 Algorithm 11) -- shared by the pair and single tracks
# ---------------------------------------------------------------------------
@triton.jit
def _sg_hidden(LNY, LNW, LNB, WT, HID,
               C: tl.constexpr, CT: tl.constexpr, KB: tl.constexpr,
               BR: tl.constexpr, CB: tl.constexpr, EPS: tl.constexpr,
               LNF: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``SiLU(linear_a(xln)) * linear_b(xln)``, one column slice of the hidden state.

    The reduction is walked in ``KB`` chunks so a non-power-of-two ``c_in`` (384
    for the single track) needs no padding; both projections share the chunk's
    activation tile.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cb = tl.program_id(1) * CB
    col = cb + tl.arange(0, CB)
    # One reduction chunk means both projection slices are fixed arena addresses,
    # so they belong above the wait like every other weight -- measured 3.75 ->
    # 3.35 us/node. With several chunks (c_in = 384 on the single track, which is
    # on the branch with slack) they stay in the loop so Triton can pipeline them.
    if KB == C:
        k1 = tl.arange(0, KB)
        wa = tl.load(WT + k1[:, None] * (2 * CT) + col[None, :])
        wb = tl.load(WT + k1[:, None] * (2 * CT) + (CT + col)[None, :])
    if PDL:
        gdc_wait()
    acca = tl.zeros([BR, CB], tl.float32)
    accb = tl.zeros([BR, CB], tl.float32)
    # LNF == 1 implies KB == C (only c_in <= 128 is fused), so the normalised row
    # is exactly one chunk and the moments are computed once.
    for k0 in range(0, C, KB):
        kk = k0 + tl.arange(0, KB)
        if LNF:
            x = _lnrow(LNY, LNW, LNB, rows, C, BR, EPS, 1, DT)
        else:
            x = tl.load(LNY + rows[:, None] * C + kk[None, :]).to(DT)
        if KB == C:
            acca += tl.dot(x, wa)
            accb += tl.dot(x, wb)
        else:
            acca += tl.dot(x, tl.load(WT + kk[:, None] * (2 * CT) + col[None, :]))
            accb += tl.dot(x, tl.load(WT + kk[:, None] * (2 * CT) + (CT + col)[None, :]))
    a = _rnd(acca, DT)
    tl.store(HID + rows[:, None] * CT + col[None, :],
             _rnd(_rnd(a * _sig(a), DT) * _rnd(accb, DT), DT).to(DT))
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _sg_out(HID, M, WOT, XIN, XOUT,
            C: tl.constexpr, CT: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr,
            TB: tl.constexpr, MS: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``linear_out`` -> mask -> the residual add that closes the transition.

    ``linear_out`` contracts the whole ``CT``-wide hidden state, so the reduction
    is walked in ``TB``-wide chunks; the loop is a plain ``range`` so Triton
    software-pipelines the weight slices rather than holding all of ``[CT, CB]``.
    ``MS`` is the mask's row stride -- 1 for the pair track (one mask entry per
    pair row) and it indexes the token axis for the single track.
    """
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.program_id(1) * CB + tl.arange(0, CB)
    o = rows[:, None] * C + col[None, :]
    # mask: the caller's tensor; XIN: written two grids back (the _sg_hidden in
    # between writes only HID). Both are final, so both loads sit above the wait.
    m = tl.load(M + rows * MS)[:, None].to(tl.float32)
    xv = tl.load(XIN + o)
    if PDL:
        gdc_wait()
    acc = tl.zeros([BR, CB], tl.float32)
    for t in range(0, CT, TB):
        hh = tl.load(HID + rows[:, None] * CT + (t + tl.arange(0, TB))[None, :])
        acc += tl.dot(hh, tl.load(WOT + (t + tl.arange(0, TB))[:, None] * C + col[None, :]))
    tl.store(XOUT + o, _rnd(xv + _rnd(_rnd(acc, DT) * m, DT), DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Single track: AttentionPairBias (AF3 Algorithm 24)
# ---------------------------------------------------------------------------
@triton.jit
def _zb_task(LNZ, LNW, LNB, WZ, ZB, pid, CZ: tl.constexpr, HZ: tl.constexpr,
             BRZ: tl.constexpr, EPS: tl.constexpr, LNF: tl.constexpr,
             DT: tl.constexpr, PDL: tl.constexpr):
    """``linear_z(layer_norm_z(z))`` -> the single track's per-head pair bias."""
    rows = pid * BRZ + tl.arange(0, BRZ)
    w = _wt(WZ, 0, CZ, HZ, HZ)
    if PDL:
        gdc_wait()
    xl = _lnrow(LNZ, LNW, LNB, rows, CZ, BRZ, EPS, LNF, DT)
    tl.store(ZB + rows[:, None] * HZ + tl.arange(0, HZ)[None, :],
             _rnd(tl.dot(xl, w), DT).to(DT))


@triton.jit
def _qkvg_task(LNA, WQ, BQ, QKVG, cb, CS: tl.constexpr, KB: tl.constexpr,
               T: tl.constexpr, HD: tl.constexpr, LD: tl.constexpr,
               CQ: tl.constexpr, SQD: tl.constexpr, DT: tl.constexpr,
               PDL: tl.constexpr):
    """One column slice of the fused q/k/v/g projection of the single track.

    The q columns get ``linear_q``'s bias and the ``1/sqrt(c_hidden)`` scale, both
    applied where the reference applies them (bias inside ``F.linear``'s rounding,
    then a second rounding after the divide). ``BQ`` is zero over the k/v/g and
    padding columns so there is no branch for them.
    """
    col = cb + tl.arange(0, CQ)
    rows = tl.arange(0, T)
    bq = tl.load(BQ + col)[None, :]
    if PDL:
        gdc_wait()
    v = _rnd(_dot_k(LNA, WQ, rows, cb, CS, KB, CQ, CS, LD, T, DT) + bq, DT)
    if cb < HD:
        v = _rnd(_div(v, SQD), DT)
    tl.store(QKVG + rows[:, None] * LD + col[None, :], v.to(DT))


@triton.jit
def _s_proj(LNZ, LNZW, LNZB, LNA, WZ, WQ, BQ, ZB, QKVG,
            CZ: tl.constexpr, HZ: tl.constexpr, BRZ: tl.constexpr, NZB: tl.constexpr,
            EPSZ: tl.constexpr, LNFZ: tl.constexpr,
            CS: tl.constexpr, KB: tl.constexpr, T: tl.constexpr, HD: tl.constexpr,
            LD: tl.constexpr, CQ: tl.constexpr, SQD: tl.constexpr,
            DT: tl.constexpr, PDL: tl.constexpr):
    """The two independent projections of the single track, as two task ranges.

    Tasks ``[0, NZB)`` are the z->pair-bias path (``linear_z`` over the pair rows'
    normed ``c_z`` channels); the rest are the fused q/k/v/g projection over the
    ``T`` token rows. They read different norms of different tensors and have
    nothing in common but being launch-bound, which is exactly why they share a
    grid: one launch per block instead of two. The two bodies live in separate
    device functions so no tile shape escapes the branch.
    """
    pid = tl.program_id(0)
    if pid < NZB:
        _zb_task(LNZ, LNZW, LNZB, WZ, ZB, pid, CZ, HZ, BRZ, EPSZ, LNFZ, DT, PDL)
    else:
        _qkvg_task(LNA, WQ, BQ, QKVG, (pid - NZB) * CQ, CS, KB, T, HD, LD, CQ,
                   SQD, DT, PDL)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _s_gate(QKVG, ZB, MS, OC,
            N: tl.constexpr, LD: tl.constexpr, HD: tl.constexpr, HZ: tl.constexpr,
            D: tl.constexpr, DR: tl.constexpr, HR: tl.constexpr, HG: tl.constexpr,
            HDR: tl.constexpr, INF: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """The single track's attention through the output gate, stored **packed**.

    ``c_hidden = 24`` is not a power of two, so q/k/v/g live in a head-padded
    ``[.., HZ, 32]`` layout (zero weight columns -> exact zeros in the padding
    lanes, so the score dot and ``p @ v`` need no masks). ``linear_o`` cannot be
    fused on top of that: it contracts the head axis, and a padded contraction
    groups its fp32 partial sums differently from cuBLAS's on the packed axis --
    measured, that alone is a 6.5e-5 flip rate, which at 48 residual blocks is a
    failing run. So the gated output is written back with the padding lanes
    dropped (``d < DR``) and ``_s_out`` contracts the packed axis.
    """
    b = tl.program_id(0)
    hgi = tl.program_id(1)
    h = hgi * HG + tl.arange(0, HG)
    n = tl.arange(0, N)
    d = tl.arange(0, D)
    mb = _rnd(INF * (tl.load(MS + b * N + tl.arange(0, N)).to(tl.float32) - 1.0), DT)
    if PDL:
        gdc_wait()
    g3 = _gated_heads(QKVG, ZB, mb, b * N, b * N, b * N * N, hgi,
                      LD, HD, N, D, HG, HZ, DT)
    # Both axes are padded: ``d >= DR`` are the head-dim padding lanes and
    # ``h >= HR`` whole padding heads (present when no_heads is not already a
    # power of two >= 16). Their output is exactly zero and the packed buffer has
    # no column for them, so the store has to drop both.
    tl.store(OC + (b * N + n[None, :, None]) * HDR
             + (h[:, None, None] * DR + d[None, None, :]),
             g3, mask=(h < HR)[:, None, None] & (d < DR)[None, None, :])
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _s_out(OC, WOT, SIN, SOUT,
           N: tl.constexpr, CS: tl.constexpr, HDR: tl.constexpr, KB: tl.constexpr,
           CA: tl.constexpr, DT: tl.constexpr, PDL: tl.constexpr):
    """``linear_o`` over the packed head axis + the single track's residual add."""
    b = tl.program_id(0)
    cb = tl.program_id(1) * CA
    rows = b * N + tl.arange(0, N)
    so = rows[:, None] * CS + (cb + tl.arange(0, CA))[None, :]
    sres = tl.load(SIN + so)
    if PDL:
        gdc_wait()
    acc = _dot_k(OC, WOT, rows, cb, HDR, KB, CA, HDR, CS, N, DT)
    tl.store(SOUT + so, _rnd(sres + _rnd(acc, DT), DT))
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class _Unsupported(Exception):
    """This (shape, dtype, init) combination is not covered by the fused path."""


def _ln(x, w, b, eps, n):
    """One reference layer norm, bit-exact: ATen's own kernel on the fp32 buffer.

    The buffer already holds bf16-rounded values (that is what every fused kernel
    stores), so handing it straight to ``native_layer_norm`` is exactly the
    reference's ``F.layer_norm(x.float(), ...)`` with no cast launch. The fp32
    result is rounded to bf16 by whichever kernel consumes it.
    """
    return torch.native_layer_norm(x, (n,), w, b, eps)[0]


class _StackPlan:
    """Every block's weights, buffers and launches for one (shape, dtype).

    Built on the first forward rather than in ``__init__`` so externally loaded
    weights are what gets folded into the concatenated / pre-transposed /
    head-padded buffers, and invalidated by ``_apply`` / ``_load_from_state_dict``
    on the owning module so a later ``.to()`` or checkpoint load rebuilds it.

    The whole stack is then captured as a single CUDA graph. ``_run`` is the
    capture body *and* the fallback launch path, so there is exactly one
    description of the stack.
    """

    def __init__(self, mod, s, z, sm, pm):
        dt, dev = z.dtype, z.device
        blocks = list(mod.blocks)
        L = len(blocks)
        B = 1
        for d in z.shape[:-3]:
            B *= d
        N, C = z.shape[-2], z.shape[-1]
        CS = s.shape[-1]
        R, T = B * N * N, B * N
        self.n, self.c, self.cs, self.dtype, self.device = N, C, CS, dt, dev
        self.znumel, self.snumel = z.numel(), s.numel()
        self.zshape, self.sshape = tuple(z.shape), tuple(s.shape)
        self.L = L

        b0 = blocks[0]
        tmo, tmi = b0.pair_stack.tri_mul_out, b0.pair_stack.tri_mul_in
        tas, tae = b0.pair_stack.tri_att_start, b0.pair_stack.tri_att_end
        pt = b0.pair_stack.pair_transition
        apb, st = b0.attn_pair_bias, b0.single_transition
        CH = tmo.c_hidden
        H4, D4 = tas.no_heads, tas.c_hidden
        HD4 = H4 * D4
        CTP = pt.n * pt.c_in
        H, D = apb.mha.no_heads, apb.mha.c_hidden
        DP = triton.next_power_of_2(D)
        HZ = max(16, triton.next_power_of_2(H))
        HDP = HZ * DP
        CTS = st.n * st.c_in

        BR = min(_BR, R)
        CB = min(_CB, C, CH)
        EB = min(_EB, CH)
        SB, ST = min(_SB, R), min(_ST, C)
        CA = min(_CA, C)
        CQ = max(16, min(_CQ, 4 * HD4))
        TKP = min(_TK, CTP)
        CAS = min(_CA, CS)
        CQS = min(_CQ, 4 * HDP)
        CBS = min(_CB, CTS)
        TKS = min(_TK, CTS)
        STS = min(_ST, CS)
        KB = min(_KB, CS)
        BRZ = min(_BRZ, R)
        LDA = 4 * HD4 + CQ
        LDS = 4 * HDP
        HG = min(4, HZ)
        HG4 = H4

        if not (_pow2(N) and 16 <= N <= 64 and _pow2(C) and _pow2(CH) and _pow2(T)
                and _pow2(D4) and _pow2(H4) and _pow2(HD4) and _pow2(DP) and _pow2(HZ)
                and _pow2(CB) and _pow2(EB) and _pow2(ST) and _pow2(CA) and _pow2(CQ)
                and _pow2(TKP) and _pow2(CAS) and _pow2(CQS) and _pow2(CBS)
                and _pow2(TKS) and _pow2(KB) and _pow2(BRZ) and _pow2(HG)
                and C % CB == 0 and CH % CB == 0 and CH % EB == 0 and C % CA == 0
                and C % ST == 0 and R % BR == 0 and R % SB == 0 and R % BRZ == 0
                and LDA % CQ == 0 and CQ >= H4 and CTP % CB == 0 and CTP % TKP == 0
                # The triangle attentions pass c_z where they want the width of one
                # of the four projections (that is the stride between q/k/v/g in the
                # shared buffer, and the column range the 1/sqrt(c_hidden) scale
                # applies to). They coincide at the captured init; anywhere else the
                # fused path would be silently wrong, so it declines instead.
                and HD4 == C
                and CS % CAS == 0 and CS % KB == 0 and LDS % CQS == 0
                and _pow2(STS) and CS % STS == 0
                and _pow2(min(_KB, H * D)) and (H * D) % min(_KB, H * D) == 0
                and CTS % CBS == 0 and CTS % TKS == 0 and HZ % HG == 0 and H4 % HG4 == 0
                and tmi.c_hidden == CH and tae.no_heads == H4 and tae.c_hidden == D4
                and pt.c_in == C and st.c_in == CS and apb.c_z == C and apb.c_q == CS
                and H * D == CS and apb.mha.gating and apb.mha.linear_g is not None
                and tas.mha.gating and apb.mha.linear_q.bias is not None):
            raise _Unsupported
        self.dims = dict(N=N, C=C, CH=CH, CS=CS, R=R, T=T, B=B, CTP=CTP, CTS=CTS,
                         LDA=LDA, LDS=LDS, HZ=HZ, DP=DP, HD4=HD4, D4=D4, H4=H4)

        # ---- weights: one contiguous arena, block-major ----
        pend = []

        def wt(*ws):
            t = torch.cat(ws, 0).t().contiguous()
            pend.append(t)
            return t

        def raw(t):
            """Arena-resident weight that is already in the layout the kernel wants."""
            t = t.contiguous()
            pend.append(t)
            return t

        def pad_heads(w, nh, dh):
            """``[nh*dh, K]`` -> pre-transposed ``[K, HZ*DP]`` with zero pad lanes."""
            v = w.detach().reshape(nh, dh, -1)
            out = torch.zeros(HZ, DP, v.shape[-1], dtype=v.dtype, device=v.device)
            out[:nh, :dh] = v
            return out.reshape(HZ * DP, -1)

        def f32(t):
            return t.detach().float().contiguous()

        def lnaff(m):
            """(weight32, bias32, eps, width, fused?) for one reference LayerNorm.

            ``fused`` is the width test from ``_wmom``: four elements per lane and
            at most one active warp, which is where the Welford tree is proven
            bit-exact. Everything wider keeps its ATen call.
            """
            if m.weight is None or m.bias is None:
                raise _Unsupported
            n = m.normalized_shape[0]
            fused = (n <= _LN_FUSE_MAX and n % 4 == 0 and _pow2(n // 4)
                     and n // 4 <= 32)
            return (f32(m.weight), f32(m.bias), float(m.eps), n, fused)

        zpad4 = torch.zeros(CQ - H4, C, dtype=dt, device=dev)
        W = []
        for blk in blocks:
            ps, apb, st = blk.pair_stack, blk.attn_pair_bias, blk.single_transition
            tmo, tmi = ps.tri_mul_out, ps.tri_mul_in
            tas, tae = ps.tri_att_start, ps.tri_att_end
            pt = ps.pair_transition
            d = {}
            for tag, m in (("o", tmo), ("i", tmi)):
                d["ln" + tag] = lnaff(m.layer_norm_in)
                d["lo" + tag] = lnaff(m.layer_norm_out)
                d["w1" + tag] = wt(m.linear_a_p.weight, m.linear_a_g.weight,
                                   m.linear_b_p.weight, m.linear_b_g.weight,
                                   m.linear_g.weight)
                d["wz" + tag] = wt(m.linear_z.weight)
            for tag, m in (("s", tas), ("e", tae)):
                d["ln" + tag] = lnaff(m.layer_norm)
                d["w2" + tag] = wt(m.mha.linear_q.weight, m.mha.linear_k.weight,
                                   m.mha.linear_v.weight, m.mha.linear_g.weight,
                                   m.linear_z.weight, zpad4)
                d["wo" + tag] = wt(m.mha.linear_o.weight)
            d["lnp"] = lnaff(pt.layer_norm)
            d["w3p"] = wt(pt.swiglu.linear_a.weight, pt.swiglu.linear_b.weight)
            d["wop"] = wt(pt.linear_out.weight)
            # single track
            d["lnz"] = lnaff(apb.layer_norm_z)
            d["lna"] = lnaff(apb.layer_norm_a)
            zp = torch.zeros(HZ - apb.linear_z.weight.shape[0], C, dtype=dt, device=dev)
            d["wzs"] = wt(apb.linear_z.weight, zp)
            mh = apb.mha
            d["wqs"] = wt(*[pad_heads(w.weight, H, D) for w in
                            (mh.linear_q, mh.linear_k, mh.linear_v, mh.linear_g)])
            bq = torch.zeros(4 * HDP, dtype=torch.float32, device=dev)
            bqv = mh.linear_q.bias.detach().float().reshape(H, D)
            bq.view(4, HZ, DP)[0, :H, :D] = bqv
            d["bqs"] = bq
            # linear_o contracts the *packed* head axis (see _s_gate), so it is the
            # plain pre-transposed [H*D, c_s] -- no head padding here.
            d["wo1"] = wt(mh.linear_o.weight)
            d["lnt"] = lnaff(st.layer_norm)
            d["w3s"] = wt(st.swiglu.linear_a.weight, st.swiglu.linear_b.weight)
            d["wot"] = wt(st.linear_out.weight)
            W.append(d)

        arena = torch.empty(sum(t.numel() for t in pend), dtype=dt, device=dev)
        off = 0
        remap = {}
        for t in pend:
            v = arena[off:off + t.numel()].view(t.shape)
            v.copy_(t)
            remap[id(t)] = v
            off += t.numel()
        for d in W:
            for k, val in list(d.items()):
                if torch.is_tensor(val):
                    d[k] = remap.get(id(val), val)
        self.arena = arena
        self.W = W

        # ---- buffers ----
        def f(*shape):
            return torch.zeros(shape, dtype=torch.float32, device=dev)

        def h(*shape):
            return torch.zeros(shape, dtype=dt, device=dev)

        self.ZST = [f(R, C) for _ in range(L + 1)]
        self.SST = [f(T, CS) for _ in range(L + 1)]
        self.MP = torch.empty(R, dtype=dt, device=dev)
        self.MS = torch.empty(T, dtype=dt, device=dev)
        self.XP = f(R, CH)
        self.ZW = [f(R, C) for _ in range(4)]
        # Two, so the fused epilogue can read one tri-mul's gate columns while
        # writing the next tri-mul's a/b/gate (see _tm_epi_proj).
        self.ABG = [h(R, 2 * CH + C) for _ in range(2)]
        self.QP = h(R, LDA)
        self.HP = h(R, CTP)
        self.ZBS = h(R, HZ)
        self.QS = h(T, LDS)
        self.OCS = h(T, H * D)
        self.S1 = f(T, CS)
        self.HS = h(T, CTS)
        # Two alternating destinations per track so a norm three grids downstream
        # can never overwrite a tile an earlier consumer is still reading.
        self.LNP = [h(R, max(C, CH)) for _ in range(2)]
        self.LNZ = h(R, C)

        self.k = dict(
            BR=BR, CB=CB, EB=EB, SB=SB, ST=ST, CA=CA, CQ=CQ, TKP=TKP, CAS=CAS,
            CQS=CQS, CBS=CBS, TKS=TKS, STS=STS, KB=KB, BRZ=BRZ, HG=HG, HG4=HG4,
            DTt=(tl.bfloat16 if dt is torch.bfloat16 else tl.float16),
            NCA=CH // CB, NCG=C // CB, NZB=R // BRZ, NQ=LDS // CQS,
            SQD4=float(D4) ** 0.5, SQD=float(D) ** 0.5, HDR=H * D, DR=D, HR=H,
            KBO=min(_KB, H * D),
            INF=float(tas.inf), INFS=float(apb.inf),
            NPF=(max(z.numel(), s.numel()) + _IBL - 1) // _IBL,
        )
        self.graph = None
        self.mode = "eager"
        self._keep = None
        self._retain = None
        self._bind()

    # -- the stack, once: capture body and eager fallback are the same code --
    def _bind(self):
        """Hoist every per-launch constant out of the block loop into one tuple.

        Resolved once per plan, so a block's cost on the fallback (no-graph) path
        is one tuple unpack plus the launches -- no attribute chains, no per-call
        grid arithmetic, no broadcast or mask plumbing. (The per-block weights stay
        in a dict; there are 48 of those and they are indexed once each.) On the
        graph path this runs once, at capture.
        """
        k, d = self.k, self.dims
        N, C, CH, CS = d["N"], d["C"], d["CH"], d["CS"]
        R, T, B, CTP, CTS = d["R"], d["T"], d["B"], d["CTP"], d["CTS"]
        LDA, LDS, HZ, DP = d["LDA"], d["LDS"], d["HZ"], d["DP"]
        BR, CB, ST, SB, CA, CQ = k["BR"], k["CB"], k["ST"], k["SB"], k["CA"], k["CQ"]
        self._bp = (
            N, C, CH, k["DTt"], BR, CB, k["EB"], SB, ST, CA, CQ, k["NCA"],
            k["NCG"], k["TKP"], LDA, CTP, d["D4"], k["HG4"], k["SQD4"], k["INF"],
            self.MP, self.XP, self.ABG, self.QP, self.HP, self.ZW,
            (R // BR, 2 * k["NCA"] + k["NCG"]), (R // BR, C // CB), (CH // k["EB"], B),
            (R // BR, LDA // CQ), (N, C // CA, B), (R // BR, CTP // CB),
            (R // SB, C // ST),
        )
        self._bs = (
            N, C, CS, T, k["DTt"], LDS, HZ, DP, k["CAS"], k["CQS"], k["CBS"],
            k["TKS"], k["STS"], k["KB"], k["BRZ"], k["HG"], k["NZB"], CTS,
            k["SQD"], k["INFS"], k["DR"], k["HR"], k["HDR"], k["KBO"],
            self.MS, self.ZBS, self.QS, self.OCS, self.S1, self.HS,
            (k["NZB"] + k["NQ"],), (B, HZ // k["HG"]), (B, CS // k["CAS"]),
            (1, CTS // k["CBS"]), (1, CS // k["STS"]),
        )

    def _lnk(self, y):
        """A layer-norm output; retained for the graph's lifetime during capture.

        ATen allocates each norm's output itself (there is no ``out=`` overload).
        Holding every one alive across a capture stops the caching allocator from
        recycling a block between the two capture streams, where a reuse would be
        a cross-branch race rather than the harmless in-order reuse it is on a
        single stream.
        """
        if self._keep is not None:
            self._keep.append(y)
        return y

    def _pre(self, x, aff, dst=None):
        """Resolve one reference LayerNorm into what its consumer kernel needs.

        Three ways to spend a norm, in order of preference:
        ``_LN_MODE == 2`` and bit-exact-able -- one ``_lnorm`` launch into ``dst``,
        consumers read the bf16 tile (``LNF = 0``);
        ``_LN_MODE == 1`` and bit-exact-able -- no launch, the consumer normalizes
        the raw fp32 residual buffer itself (``LNF = 1``);
        otherwise -- the ATen call, consumer reads its fp32 output (``LNF = 0``).
        """
        lw, lb, eps, n, fused = aff
        if fused and _LN_MODE == 2 and dst is not None:
            _lnorm[(x.shape[0] // min(_BRL, x.shape[0]),)](
                x, lw, lb, dst, n, min(_BRL, x.shape[0]), eps, self.k["DTt"], _PDL,
                num_warps=_WARPS, launch_pdl=_PDL)
            return dst, lw, lb, eps, 0
        if fused and _LN_MODE == 1:
            return x, lw, lb, eps, 1
        return self._lnk(_ln(x, lw, lb, eps, n)), lw, lb, eps, 0

    def _run_pair(self, i):
        """The z track of one block: 10 launches, or 12 with ``_FUSE_EPI = 0``.

        Written out linearly rather than as a loop over the two triangle
        multiplications, because each one's epilogue is fused into a *different*
        successor -- the outgoing one into the incoming one's input projection, the
        incoming one into the starting triangle attention's q/k/v/g projection.
        """
        (N, C, CH, DT, BR, CB, EB, SB, ST, CA, CQ, NCA, NCG, TKP, LDA, CTP,
         D4, HG4, SQD4, INF, MP, XP, ABG, QP, HP, ZW,
         gtm, gep, gei, gtp, gat, ghp, gop) = self._bp
        ZA, ZB2, ZC, ZD = ZW
        # Two a/b/gate buffers: a fused epilogue reads one triangle
        # multiplication's gate columns while writing the next one's a/b/gate into
        # the same column range, and tasks of one launch have no ordering.
        AG0, AG1 = ABG
        NW, NW4, P = _WARPSZ, _WARPS4, _PDL
        w = self.W[i]
        zin, znx = self.ZST[i], self.ZST[i + 1]
        F = _FUSE_EPI and _LN_MODE == 1

        # ---- outgoing triangle multiplication ----
        x, lw, lb, eps, lnf = self._pre(zin, w["lno"], self.LNP[0])
        _tm_proj[gtm](x, lw, lb, MP, w["w1o"], AG0, C, CH, BR, CB, NCA,
                      eps, lnf, DT, P, num_warps=NW, launch_pdl=P)
        _tm_einsum[gei](AG0, XP, N, C, CH, EB, N, 1, N, 1, DT, P,
                        num_warps=NW, launch_pdl=P)
        xo, low, lob, epso, lnfo = self._pre(XP, w["loo"], self.LNP[1])
        n2w, n2b, n2e, _, n2f = w["lni"]
        if F and lnfo and n2f:
            _tm_epi_proj[gtm](xo, low, lob, AG0, zin, ZA, w["wzo"],
                              n2w, n2b, MP, w["w1i"], AG1,
                              C, CH, BR, CB, NCA, epso, n2e, lnfo, DT, P,
                              num_warps=NW, launch_pdl=P)
        else:
            _tm_epi[gep](xo, low, lob, AG0, zin, ZA, w["wzo"], C, CH, BR, CB,
                         epso, lnfo, DT, P, num_warps=NW, launch_pdl=P)
            x, lw, lb, eps, lnf = self._pre(ZA, w["lni"], self.LNP[0])
            _tm_proj[gtm](x, lw, lb, MP, w["w1i"], AG1, C, CH, BR, CB, NCA,
                          eps, lnf, DT, P, num_warps=NW, launch_pdl=P)

        # ---- incoming triangle multiplication ----
        _tm_einsum[gei](AG1, XP, N, C, CH, EB, 1, N, 1, N, DT, P,
                        num_warps=NW, launch_pdl=P)
        xi, liw, lib, epsi, lnfi = self._pre(XP, w["loi"], self.LNP[1])
        n2w, n2b, n2e, _, n2f = w["lns"]
        if F and lnfi and n2f:
            _tm_epi_taproj[gtp](xi, liw, lib, AG1, ZA, ZB2, w["wzi"],
                                n2w, n2b, w["w2s"], QP,
                                C, CH, BR, LDA, C, CQ, SQD4, epsi, n2e, lnfi,
                                DT, P, num_warps=NW, launch_pdl=P)
        else:
            _tm_epi[gep](xi, liw, lib, AG1, ZA, ZB2, w["wzi"], C, CH, BR, CB,
                         epsi, lnfi, DT, P, num_warps=NW, launch_pdl=P)
            x, lw, lb, eps, lnf = self._pre(ZB2, w["lns"], self.LNP[0])
            _ta_proj[gtp](x, lw, lb, w["w2s"], QP, N, C, LDA, C, BR, CQ,
                          SQD4, N, 1, eps, lnf, DT, P, num_warps=NW, launch_pdl=P)

        # ---- the two triangle attentions ----
        _ta_attn[gat](QP, MP, w["wos"], ZB2, ZC, N, C, LDA, C, HG4, D4,
                      HG4, 1, CA, N, 1, N, 1, INF, DT, 0, P,
                      num_warps=NW4, launch_pdl=P)
        # The ending node reads the transpose of what the starting one wrote -- one
        # of its programs' rows come from N different producer programs -- so this
        # projection cannot be fused into its producer the way the two above are.
        x, lw, lb, eps, lnf = self._pre(ZC, w["lne"], self.LNP[0])
        _ta_proj[gtp](x, lw, lb, w["w2e"], QP, N, C, LDA, C, BR, CQ,
                      SQD4, 1, N, eps, lnf, DT, P, num_warps=NW, launch_pdl=P)
        _ta_attn[gat](QP, MP, w["woe"], ZC, ZD, N, C, LDA, C, HG4, D4,
                      HG4, 1, CA, 1, N, 1, N, INF, DT, 1, P,
                      num_warps=NW4, launch_pdl=P)

        # ---- pair transition ----
        x, lw, lb, eps, lnf = self._pre(ZD, w["lnp"], self.LNP[1])
        _sg_hidden[ghp](x, lw, lb, w["w3p"], HP, C, CTP, C, BR, CB, eps, lnf,
                        DT, P, num_warps=NW, launch_pdl=P)
        _sg_out[gop](HP, MP, w["wop"], ZD, znx, C, CTP, SB, ST, TKP, 1, DT, P,
                     num_warps=NW4, launch_pdl=P)

    def _run_single(self, i):
        """The s track of one block: 3 ATen norms + 5 fused kernels.

        Reads only ``ZST[i+1]`` from the z track, which is why it can be a second
        graph branch: nothing the z track does later touches it.
        """
        (N, C, CS, T, DT, LDS, HZ, DP, CAS, CQS, CBS, TKS, STS, KB, BRZ, HG,
         NZB, CTS, SQD, INFS, DR, HR, HDR, KBO, MS, ZBS, QS, OCS, S1, HS,
         gsp, gsg, gsa, ghs, gos) = self._bs
        NW, NW4, P = _WARPS, _WARPS4, _PDL
        w = self.W[i]
        sin, snx = self.SST[i], self.SST[i + 1]
        xz, lzw, lzb, epsz, lnfz = self._pre(self.ZST[i + 1], w["lnz"], self.LNZ)
        ya, _, _, _, _ = self._pre(sin, w["lna"])
        _s_proj[gsp](xz, lzw, lzb, ya, w["wzs"], w["wqs"], w["bqs"], ZBS, QS,
                     C, HZ, BRZ, NZB, epsz, lnfz, CS, KB, T, LDS // 4, LDS, CQS,
                     SQD, DT, P, num_warps=NW, launch_pdl=P)
        _s_gate[gsg](QS, ZBS, MS, OCS, N, LDS, LDS // 4, HZ, DP, DR, HR, HG,
                     HDR, INFS, DT, P, num_warps=NW4, launch_pdl=P)
        _s_out[gsa](OCS, w["wo1"], sin, S1, N, CS, HDR, KBO, CAS, DT, P,
                    num_warps=NW, launch_pdl=P)
        x, lw, lb, eps, lnf = self._pre(S1, w["lnt"])
        _sg_hidden[ghs](x, lw, lb, w["w3s"], HS, CS, CTS, KB, T, CBS, eps, lnf,
                        DT, P, num_warps=NW, launch_pdl=P)
        _sg_out[gos](HS, MS, w["wot"], S1, snx, CS, CTS, T, STS, TKS, 1, DT, P,
                     num_warps=NW4, launch_pdl=P)

    def _run(self):
        """Both tracks in program order -- the fallback path and the linear graph."""
        for i in range(self.L):
            self._run_pair(i)
            self._run_single(i)

    def _capture(self):
        """Capture the stack, preferring a two-branch graph over a linear one.

        The two tracks are almost independent: the z track never reads ``s``, and
        the s track of block ``i`` reads only ``ZST[i+1]``, which the z track
        writes once and never touches again. So the natural graph is two chains
        with one cross edge per block -- 10 z-track nodes and 7 s-track nodes per
        block running concurrently instead of 17 in series. Each kernel here uses
        16-144 programs on 148 SMs, i.e. one CTA per SM with nothing else resident
        to hide its dependent loads (ncu: 12.6% achieved occupancy, 79% of cycles
        with no eligible warp, 36-44% of stalls in ``long_scoreboard``), so the
        overlap buys both a shorter critical path and something for the memory
        system to work on.

        Both branches' scratch is private to their track and the residual streams
        are per block, so the only ordering that has to be expressed is the cross
        edge. Falls back to the linear capture, then to eager launches.
        """
        for mode in ("branched", "linear"):
            self._keep = []
            try:
                g = torch.cuda.CUDAGraph()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._run()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                keep = []
                self._keep = keep
                with torch.cuda.graph(g):
                    if mode == "linear":
                        self._run()
                    else:
                        main = torch.cuda.current_stream()
                        alt = torch.cuda.Stream()
                        evz = [torch.cuda.Event() for _ in range(self.L)]
                        for i in range(self.L):
                            self._run_pair(i)
                            evz[i].record(main)
                        with torch.cuda.stream(alt):
                            for i in range(self.L):
                                alt.wait_event(evz[i])
                                self._run_single(i)
                            done = torch.cuda.Event()
                            done.record(alt)
                        main.wait_event(done)
                        # The events are baked into the graph as node edges; hold
                        # them anyway so nothing about their lifetime is implicit.
                        self._events = (evz, done)
                torch.cuda.synchronize()
            except Exception:
                self.graph = None
                self._keep = None
                continue
            self.graph = g
            self.mode = mode
            self._keep = None
            self._retain = keep
            return
        self._keep = None

    def matches(self, s, z, sm, pm) -> bool:
        # The alignment test is not optional: Triton specialized _ingest/_egress on
        # the 16-byte alignment of the pointers they were built with.
        d = self.dims
        return (z.dtype is self.dtype and s.dtype is self.dtype
                and z.device == self.device and tuple(z.shape) == self.zshape
                and tuple(s.shape) == self.sshape
                and z.is_contiguous() and s.is_contiguous()
                and sm is not None and pm is not None
                and sm.dtype is self.dtype and pm.dtype is self.dtype
                and sm.is_contiguous() and pm.is_contiguous()
                and sm.numel() == d["T"] and pm.numel() == d["R"]
                and not (z.data_ptr() & 15) and not (s.data_ptr() & 15)
                and not (sm.data_ptr() & 15) and not (pm.data_ptr() & 15))

    def run(self, s, z, sm, pm):
        k, d = self.k, self.dims
        DT, P = k["DTt"], _PDL
        _ingest[(k["NPF"],)](s, self.SST[0], z, self.ZST[0], sm, self.MS, pm, self.MP,
                             s.numel(), z.numel(), sm.numel(), pm.numel(), _IBL, P,
                             num_warps=4, launch_pdl=P)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._run()
        so = torch.empty(self.sshape, dtype=self.dtype, device=self.device)
        zo = torch.empty(self.zshape, dtype=self.dtype, device=self.device)
        _egress[(k["NPF"],)](self.SST[self.L], so, self.ZST[self.L], zo,
                             so.numel(), zo.numel(), _IBL, DT, P,
                             num_warps=4, launch_pdl=P)
        return so, zo


class PairFormerBlock(nn.Module):
    """Single block of AF3 Algorithm 17.

    Kept as the parameter owner so ``state_dict`` keys match the reference
    exactly; the fused path reads its weights and never calls its submodules.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s, c_k=c_s, c_v=c_s,
            c_s=c_s, c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
        )

        self.single_transition = SwiGLUTransition(c_in=c_s, n=transition_n)

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
        _attn_chunk_size: int | None = None,
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
        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(z=z, pair_mask=pair_mask)

        s = s + self.attn_pair_bias(a=s, z=z, s=None, mask=single_mask)

        s = s + self.single_transition(s, mask=single_trans_mask)

        return s, z


class PairFormerStack(nn.Module):
    """AF3 Algorithm 17: PairFormer stack.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden_pair_bias: Hidden dim for AttentionPairBias
        no_heads_pair_bias: Heads for AttentionPairBias
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Hidden dim for triangle attention
        no_heads_pair: Heads for triangle attention
        no_blocks: Number of PairFormer blocks
        transition_n: Scale for transition hidden dim
        pair_dropout: Dropout rate
        inf: Large masking constant
    """

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
        self._plan: _StackPlan | None = None
        self._no_fuse = False
        self._rejected: set = set()

    # The plan folds every block's parameters into concatenated / pre-transposed /
    # head-padded buffers and then captures a graph over them, so any wholesale
    # change to the parameters has to invalidate it. Both hooks fire before the
    # first forward in normal use (``.to(device)``, then a checkpoint load), which
    # is exactly why the plan is built lazily.
    def _apply(self, *args, **kwargs):
        self._plan = None
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._plan = None
        return super()._load_from_state_dict(*args, **kwargs)

    def _reference(self, s, z, single_mask, pair_mask, _mask_trans):
        for block in self.blocks:
            s, z = block(s=s, z=z, single_mask=single_mask, pair_mask=pair_mask,
                         _mask_trans=_mask_trans)
        return s, z

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
        plan = self._plan
        if plan is not None and _mask_trans and plan.matches(s, z, single_mask, pair_mask):
            return plan.run(s, z, single_mask, pair_mask)

        key = (s.dtype, z.dtype, z.device, tuple(s.shape), tuple(z.shape))
        if (not self._no_fuse and key not in self._rejected and _mask_trans
                and len(self.blocks) > 0 and z.is_cuda and z.dtype in _FUSED_DTYPES
                and s.dtype is z.dtype and z.ndim >= 3 and s.ndim >= 2
                and z.shape[-2] == z.shape[-3] and z.is_contiguous() and s.is_contiguous()
                and single_mask is not None and pair_mask is not None
                and single_mask.dtype is z.dtype and pair_mask.dtype is z.dtype
                and single_mask.is_contiguous() and pair_mask.is_contiguous()
                and pair_mask.shape == z.shape[:-1] and single_mask.shape == s.shape[:-1]
                and not torch.is_grad_enabled()):
            try:
                plan = _StackPlan(self, s, z, single_mask, pair_mask)
                plan._capture()
            except _Unsupported:
                # A shape / init the kernels do not cover. Remembered per key rather
                # than latched for the module, so one odd shape does not lose the
                # fused path for every later one.
                self._rejected.add(key)
            except Exception:  # a Triton/driver surprise must not lose the op
                self._no_fuse = True
            else:
                self._plan = plan
                if plan.matches(s, z, single_mask, pair_mask):
                    return plan.run(s, z, single_mask, pair_mask)

        return self._reference(s, z, single_mask, pair_mask, _mask_trans)
