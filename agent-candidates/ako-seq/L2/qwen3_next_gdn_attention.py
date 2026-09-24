"""Qwen3-Next Gated Delta Net (GDN) linear attention (L2) -- two fused paths.

Two regimes, two kernels, one dispatch:

  T <= 64      one program per V head, the whole post-projection pipeline in a
               single chunk of the big tile.  The call is *host*-bound here, so
               what matters is that there are exactly three launches.
  64 < T <= 2048
               three launches instead of one, buying chunk-level parallelism: the
               elementwise stage runs once at full occupancy and the recurrence
               is split into G segments solved in parallel and stitched by a
               scalar scan.  The call is device-bound here, and a single-program-
               per-head kernel leaves 116 of the 148 SMs idle.
  T > 2048     the baseline's own FlashInfer chunk kernel; see _LONG_MAX_T.


The baseline is entirely *host*-bound below T~1000: a T=60 call spends 386 us of
Python dispatch while the GPU sits idle (measured -- host enqueue 403 us, CUDA
event span 405 us, so every launch gap is exposed).  The work itself is one
[T,2048]x[2048,12352] GEMM, ~67 MB of projection-weight traffic, and a recurrence
over 32 heads of a [128,128] state; at T=60 that is a couple of microseconds.
What costs 386 us is *eighteen* kernels, each paying a Triton/cuBLAS Python
dispatch of 20-50 us:

    in_proj GEMM               23 us
    _unpack_qkvz_ba            40
    causal_conv1d_fn           51
    fused_post_conv_prep       49
    recurrent gather + zero_    20
    FlashInfer chunk GDR        80   (6 internal launches)
    final-state writeback       19
    RMSNormGated                75
    out_proj GEMM              28

Everything between the two GEMMs -- deinterleave, depthwise causal conv1d + SiLU,
q/k L2-norm, the g/beta gating, the chunked gated-delta-rule solve against the
recurrent state, the final-state write-back and the swish-gated RMSNorm -- is one
Triton kernel here, so a short-T call is exactly three launches.  The 334 us of
dispatch for the middle seven stages becomes a single ~30 us launch.

Above 64 tokens that same kernel runs out of programs -- the recurrence is
sequential in the chunk index and the RMSNorm reduction spans the full
head_v_dim, so the grid is 32 and 116 SMs idle (measured: running it at 4x its
real grid, with every program doing real work, takes the same wall time).  The
second path below fixes that.  Past ~2400 tokens even a fully parallel
hand-written chunk pass cannot match the single CUTLASS kernel the baseline
already calls, so the baseline keeps the tail.

Math reproduced (an independent fp32 PyTorch reimplementation was validated to
matched=1.0 against the baseline *before* any of this was written; that is also
what pinned the final-state buffer layout as ``[hv, v, k]``, transposed from the
chunk path's ``[k, v]``):

  in_proj_qkvz emits one group per K head, ``[q(K) k(K) v(VP*V) z(VP*V)]``, and
  in_proj_ba one ``[b(VP) a(VP)]`` group per K head, so the deinterleave is pure
  index arithmetic inside the kernel and ``_unpack_qkvz_ba`` disappears.

  conv:   y[t,c] = silu( sum_{j<4} w[c,j] * x[t-3+j, c] )   (fp32, then bf16)
  q,k:    L2-normalised in fp32 *from the bf16 conv output* (the reference
          rounds through bf16 there; skipping that rounding drifts the compare)
  g       = -exp(A_log) * softplus(a + dt_bias)     fp32, softplus threshold 20
  beta    = sigmoid(b)                              fp32
  state:  S[k,v], alpha = exp(g), scale = K**-0.5
            S <- alpha_t * S
            u_t = beta_t * (v_t - k_t^T S)
            S  <- S + k_t u_t^T
            o_t = scale * q_t^T S
  norm:   out = RMSNorm(o) * swish(z), reduction over the full head_v_dim

The recurrence is evaluated chunk-parallel (WY / UT transform).  With
A_t = prod_{s<=t} alpha_s and gc = cumsum(g) (so A_t = exp(gc_t), gc non-increasing
because g <= 0, hence every exponent below is <= 0 and nothing can overflow):

    M[t,r]  = -beta_t * exp(gc_t - gc_r) * (k_t . k_r)      r < t
    w[t]    =  beta_t * (v_t - exp(gc_t) * (k_t^T S_init))
    u       = (I - M)^{-1} w
    o[t]    =  scale * ( exp(gc_t) * q_t^T S_init
                         + sum_{r<=t} exp(gc_t - gc_r) (k_r . q_t) u_r )
    S_new   =  exp(gc_last) * S_init + sum_r exp(gc_last - gc_r) k_r u_r^T

``(I - M)^{-1} w`` is solved by ``_block_solve`` -- roughly a dozen ``tl.dot``
pairs in place of FlashInfer's separate scaled_dot_kkt / solve_tril /
recompute_w_u launches.  See that function for why the obvious closed form
(nilpotent doubling over the whole chunk) is numerically unusable here.

This subclasses the baseline module: the harness locates the GDN layer with
``isinstance`` against the *baseline* class (bench.py ``_locate_recurrent_attn``),
so a candidate that only redefines the class name is invisible to the recurrent-
state prep and every case is reported SKIPPED.  Inheriting also keeps the
``__init__`` / weight-loader / ``forward`` contracts exactly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ....infra.context import get_context as _get_context
from ...baseline.L2.qwen3_next_gdn_attention import (
    Qwen3NextGDNAttention as _BaselineGDN,
)

# T <= _CHUNK1 runs as one chunk at the larger tile; longer sequences loop the
# smaller one (the big tile does not fit Blackwell tensor memory inside a loop).
# The looped form is now only reachable when the segment-parallel path declines a
# call, since _LONG_MIN_T == _CHUNK1; it is kept because it is what makes the
# fused kernel correct for any T, and the dispatch is a threshold, not a rewrite.
_CHUNK1 = 64
_CHUNKN = 32
_NUM_WARPS = 8
# Precision of the fp32 solve/state dots: 0 = ieee (exact fp32), 1 = tf32,
# 2 = bf16 tensor cores with fp32 accumulate (what FLA's own chunk kernels use).
_PREC = 2
# Precision of the triangular-solve dots specifically. bf16 measured identical to
# tf32 on the real data (max_abs 2.9e-3 either way) and 5% better geomean, so bf16
# it is; tf32 is the safer choice if the keys ever correlate more than the 0.40
# max |k_t . k_r| measured here, since _block_solve's worst case goes 1.5e-2
# (fp32/tf32) -> 6.2e-2 (bf16). ieee is a 5x slowdown and never worth it.
_PSOLVE = 2
# Block size for the triangular solve. See _block_solve.
_BS = 8
# Finite floor for the log-decay. C * |_GFLOOR| must stay inside fp32 range so
# that cumulative sums, and differences of them, are always finite.
_GFLOOR = -1e30

# --- long-T (segment-parallel) path -----------------------------------------
# Above _LONG_MIN_T the segment-parallel path replaces the single fused kernel:
# two extra launches buy chunk-level parallelism (G*HV programs instead of HV)
# and move the elementwise stage out of the recurrence loop. The measured
# crossover is exactly the fused kernel's own single-chunk boundary -- T <= 64 is
# one chunk of the big tile and unbeatable at 53-62 us, while at T=65 the fused
# path starts looping C=32 chunks over 32 programs and the long path already
# wins (64.0 us vs 70.5, then 64.5 vs 83.0 at T=128 and 79.6 vs 130.1 at T=256).
# So every shape at or below 64 tokens -- which is all three short benched cases
# (T=1, 26, 60) and every short shape in the capture set -- keeps the r1
# three-launch path bit-identical.
_LONG_MIN_T = _CHUNK1
# Above this the baseline's single fused FlashInfer chunk kernel wins again: the
# segment passes cost a fixed ~0.16 us per (chunk, head) of SM time and the
# machine only has 128 program slots, so their cost is linear in T with a worse
# constant than the CUTLASS kernel's. Measured candidate/baseline: T=1024 2.17x,
# 2048 1.26x, 2304 1.11x, 2560 0.94x, 4096 0.83x, 16384 0.66x. 2048 keeps a
# margin rather than sitting on the ~2400 break-even.
_LONG_MAX_T = 2048
_LC = 32          # chunk size in the segment passes
_LBT = 32         # token tile of the prep kernel
# Cap on the segment count: G*HV is the program count of passes 2 and 3.
# Occupancy is one program per SM (the [K,V] fp32 state alone is 64 KB), so the
# sweet spot is one wave -- G*HV just under the 148 SMs -- with the chunks of a
# segment looped inside the program: measured at T=445, G=4 (128 programs, ~3.5
# chunks each) is 128 us of device time against 158 us for G=14 (448 programs,
# one chunk each), because per-step latency does not overlap and each extra
# program repays its 64 KB state load only once.
_LMAX_G = 4
_LWARPS = (8, 8, 4)      # seg, out, prep
_LSTAGES = (1, 1, 2)


@triton.jit
def _fdot(a, b, PREC: tl.constexpr):
    """``tl.dot`` at a selectable input precision (see ``_PREC``)."""
    if PREC == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif PREC == 1:
        return tl.dot(a, b, input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _fadd_dot(acc, a, b, PREC: tl.constexpr):
    """``acc + a @ b`` as one accumulating MMA.

    Written this way rather than as ``acc + _fdot(a, b)`` because Triton gives
    the un-accumulated form its own tensor-memory region: on sm100 each live fp32
    [*, 128] accumulator costs 128 of the 512 available columns, and the three
    ``u``/state updates below were 384 of them on their own -- enough to push
    C = 64 over the limit (768 required).  Accumulating in place also keeps the
    products inside the MMA's fp32 accumulator instead of rounding a separate
    result first.
    """
    if PREC == 0:
        return tl.dot(a, b, acc=acc, input_precision="ieee")
    elif PREC == 1:
        return tl.dot(a, b, acc=acc, input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc=acc)


@triton.jit
def _conv_silu(
    proj_ptr,       # *bf16 [T, S_PROJ]  merged in_proj output
    cw_ptr,         # *fp32/bf16 [conv_dim, KC]
    cs_ptr,         # conv_state
    t,              # [C] int32  global token index
    tmask,          # [C] bool
    col,            # [D] int32  column in proj
    ch,             # [D] int32  conv channel
    T,
    cs_base,        # int32  cache_index * stride_seq
    cs_dim,
    cs_tok,
    USE_INIT: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    KC: tl.constexpr,
    S_PROJ: tl.constexpr,
):
    """Depthwise causal conv1d + SiLU over one [C, D] slice, returned as bf16.

    ``w[c,0]`` multiplies x[t-3] and ``w[c,3]`` x[t] (cross-correlation with a
    left pad of KC-1, matching vLLM's kernel).  Taps come from ``proj`` for
    s >= 0 and from ``conv_state`` for s in [-3,-1] when the sequence carries
    one; the reference stores the conv output as bf16, so the cast back here is
    load-bearing for the L2-norm that follows.
    """
    acc = tl.zeros([C, D], dtype=tl.float32)
    for j in tl.static_range(KC):
        s = t - (KC - 1) + j
        m = tmask[:, None] & (s >= 0)[:, None] & (s < T)[:, None]
        val = tl.load(proj_ptr + s[:, None] * S_PROJ + col[None, :],
                      mask=m, other=0.0).to(tl.float32)
        if USE_INIT:
            ms = tmask[:, None] & (s < 0)[:, None]
            sv = tl.load(cs_ptr + cs_base + ch[None, :] * cs_dim
                         + (s + (KC - 1))[:, None] * cs_tok,
                         mask=ms, other=0.0).to(tl.float32)
            val = tl.where(ms, sv, val)
        w = tl.load(cw_ptr + ch * KC + j).to(tl.float32)
        acc += w[None, :] * val
    y = acc * tl.sigmoid(acc)
    return y.to(tl.bfloat16)


@triton.jit
def _l2norm_bf16(x_bf, EPS: tl.constexpr):
    """Row-wise L2 normalise a bf16 tile in fp32, back to bf16 (reference order)."""
    xf = x_bf.to(tl.float32)
    return (xf * tl.rsqrt(tl.sum(xf * xf, axis=1) + EPS)[:, None]).to(tl.bfloat16)


@triton.jit
def _store_conv_state(
    proj_ptr, cs_ptr, col, ch, T, cs_base, cs_dim, cs_tok,
    USE_INIT: tl.constexpr, KC: tl.constexpr, NP2: tl.constexpr,
    S_PROJ: tl.constexpr,
):
    """Leave ``conv_state`` holding the last KC-1 *pre-conv* tokens.

    Matches vLLM's kernel: the tail of ``x`` for T >= KC-1, and for a shorter
    sequence the surviving tail of the incoming state shifted left (zeros when
    the sequence carries no initial state).  ``NP2`` is KC-1 rounded up to a
    power of two because ``tl.arange`` needs one.
    """
    js = tl.arange(0, NP2)
    keep = js < KC - 1
    sj = T - (KC - 1) + js
    ok = (sj >= 0) & keep
    src = tl.load(proj_ptr + sj[:, None] * S_PROJ + col[None, :],
                  mask=ok[:, None], other=0.0)
    if USE_INIT:
        old = tl.load(cs_ptr + cs_base + ch[None, :] * cs_dim
                      + (sj + (KC - 1))[:, None] * cs_tok,
                      mask=(~ok)[:, None] & keep[:, None], other=0.0)
        tl.debug_barrier()   # vLLM's kernel needs this for the same
                             # load -> where -> store on one buffer
        src = tl.where(ok[:, None], src, old)
    tl.store(cs_ptr + cs_base + ch[None, :] * cs_dim + js[:, None] * cs_tok,
             src, mask=keep[:, None])


@triton.jit
def _block_solve(M, w, tt, eye, BS: tl.constexpr, LOG_BS: tl.constexpr,
                 LOG_NB: tl.constexpr, PREC: tl.constexpr):
    """u = (I - M)^{-1} w for strictly lower triangular M, stably.

    M^C = 0, so ``(I+M)(I+M^2)(I+M^4)...`` is *exactly* (I-M)^{-1} in log2(C)
    steps -- but it gets there by explicitly forming M^32, whose entries grow
    like C(t-r-1, 31) even when the answer does not. On the real post-SiLU keys
    (max |k_t . k_r| = 0.40 measured) that already costs 1e-2 relative error, and
    on more correlated keys it overflows outright: at |k.k| ~ 0.9 plain doubling
    returns 1e11 where the true u is 5.5, which is where the intermittent NaN in
    the [1, 2048] case came from.

    So split I - M = D - L with D = I - blockdiag_BS(M) and L the strictly
    block-lower rest:

        (I - M)^{-1} = (I - D^{-1} L)^{-1} D^{-1}

    Mb = blockdiag(M) is strictly lower *and* block diagonal, so Mb^BS = 0 and
    its inverse needs only log2(BS) doubling steps with the powers confined to a
    BS-wide block -- amplification bounded by the size of the answer rather than
    by C(62,31).  N = D^{-1}L is strictly *block* lower, so N^(C/BS) = 0 and the
    outer solve needs log2(C/BS) steps over a recursion only C/BS blocks deep.

    Same dot count as the naive doubling (11 at C=64, 9 at C=32); BS=8 measured
    stable to 1.5e-2 relative worst case even with perfectly parallel keys, where
    BS=32 and plain doubling both diverge and BS=4 (16 blocks deep) is worse
    again.
    """
    blk = tt // BS
    same = blk[:, None] == blk[None, :]
    Mb = tl.where(same, M, 0.0)          # block diagonal, strictly lower
    Lb = tl.where(same, 0.0, M)          # strictly block lower

    # Dinv = sum_{j<BS} Mb^j
    P = eye + Mb
    Q = Mb
    for _ in tl.static_range(LOG_BS - 1):
        Q = _fdot(Q, Q, PREC)
        P = _fdot(P, eye + Q, PREC)

    Nn = _fdot(P, Lb, PREC)
    u = _fdot(P, w, PREC)
    u = _fadd_dot(u, Nn, u, PREC)
    Np = Nn
    for _ in tl.static_range(LOG_NB - 1):
        Np = _fdot(Np, Np, PREC)
        u = _fadd_dot(u, Np, u, PREC)
    return u


@triton.jit
def _gdn_chunk(
    proj_ptr, cw_ptr, cs_ptr, out_ptr,
    S, i_c, T, cs_base, cs_dim, cs_tok,
    q_col, k_col, v_col, z_col, a_col, b_col, q_ch, k_ch, v_ch,
    A_log, dt_b, nw, dk, dv, tt, lo_s, lo_i, eye, i_hv,
    USE_INIT: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    C: tl.constexpr, KC: tl.constexpr, S_PROJ: tl.constexpr, SCALE: tl.constexpr,
    NEPS: tl.constexpr, L2EPS: tl.constexpr, SPT: tl.constexpr,
    BS: tl.constexpr, LOG_BS: tl.constexpr, LOG_NB: tl.constexpr,
    PREC: tl.constexpr, PSOLVE: tl.constexpr, _GFLOOR: tl.constexpr,
):
    """One chunk of C tokens for one V head: everything from the projection
    output to the out_proj input. Returns the updated [K, V] state."""
    t = i_c * C + tt
    tmask = t < T

    # ---- conv + SiLU, then L2-norm q/k (fp32 from the bf16 conv output) ----
    qb = _l2norm_bf16(_conv_silu(proj_ptr, cw_ptr, cs_ptr, t, tmask, q_col, q_ch,
                                 T, cs_base, cs_dim, cs_tok, USE_INIT,
                                 C, K, KC, S_PROJ), L2EPS)
    kb = _l2norm_bf16(_conv_silu(proj_ptr, cw_ptr, cs_ptr, t, tmask, k_col, k_ch,
                                 T, cs_base, cs_dim, cs_tok, USE_INIT,
                                 C, K, KC, S_PROJ), L2EPS)
    vb = _conv_silu(proj_ptr, cw_ptr, cs_ptr, t, tmask, v_col, v_ch, T,
                    cs_base, cs_dim, cs_tok, USE_INIT, C, V, KC, S_PROJ)

    # ---- gating: g = -exp(A_log)*softplus(a + dt_bias), beta = sigmoid(b) ----
    av = tl.load(proj_ptr + t * S_PROJ + a_col, mask=tmask, other=0.0).to(tl.float32)
    bv = tl.load(proj_ptr + t * S_PROJ + b_col, mask=tmask, other=0.0).to(tl.float32)
    xg = av + dt_b
    # softplus in the branch-free form: identical to the reference's
    # where(x>0, x+log1p(exp(-x)), log1p(exp(x))) term for term, but the
    # exponent is never positive, so neither arm can overflow to inf. The
    # reference computes *both* arms and selects, and its discarded arm does
    # overflow -- harmless only for as long as the select is not lowered to
    # arithmetic on the inf.
    sp = tl.maximum(xg, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(xg)))
    sp = tl.where(xg <= SPT, sp, xg)
    # Masked-out tokens must not decay the state: g = 0 there, so gc stays flat
    # past T-1 and min(gc) is the true last-token cumulative decay.
    gv = tl.where(tmask, -tl.exp(A_log) * sp, 0.0)
    # Floor g at a finite value. The harness only re-initialises a parameter whose
    # amax falls outside [1e-6, 1e4] (bench.py _sanitize_float_params), so an
    # A_log left as torch.empty garbage near 1e4 makes exp(A_log) overflow: g is
    # then -inf, or NaN where softplus underflowed to 0 and the product is
    # inf * 0. Either poisons gc, and gc_t - gc_r becomes inf - inf = NaN, which
    # is the intermittent "NaN in output" this case shows. The comparison is
    # ordered so that NaN takes the floor too (NaN > x is false), and the clamp
    # is exact in every use: g is only ever consumed as exp() of a sum of
    # non-positive terms, and exp(-1e30) and exp(-inf) are both 0.
    gv = tl.where(gv > _GFLOOR, gv, _GFLOOR)
    bt = tl.sigmoid(bv)

    gc = tl.cumsum(gv, axis=0)                       # [C], non-increasing
    Ak = tl.exp(gc)
    # Only r <= t is ever used, where gc_t - gc_r <= 0. Above the diagonal the
    # difference is *positive* and exp() overflows -- exp(+70) at T=60, and inf
    # times a masked-out zero Kk entry is NaN. Those entries are masked off
    # below, but relying on tl.where to contain a NaN only holds while the
    # select is not lowered to a multiply, and it is exactly the intermittent
    # NaN seen in the [1, 2048] case. Clamping the exponent at 0 first is a
    # no-op where the value is used and keeps Dm in (0, 1] everywhere.
    Dm = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))

    # ---- UT transform: u = (I - M)^{-1} w ----
    kf = kb.to(tl.float32)
    Kk = tl.dot(kb, tl.trans(kb))                    # bf16 x bf16 -> fp32
    M = tl.where(lo_s, -bt[:, None] * Dm * Kk, 0.0)
    Sk = _fdot(kf, S, PREC)                          # k_t^T S_init  [C,V]
    w = bt[:, None] * (vb.to(tl.float32) - Ak[:, None] * Sk)
    u = _block_solve(M, w, tt, eye, BS, LOG_BS, LOG_NB, PSOLVE)

    # ---- output and state update ----
    Sq = _fdot(qb.to(tl.float32), S, PREC)           # q_t^T S_init  [C,V]
    Qk = tl.where(lo_i, tl.dot(qb, tl.trans(kb)) * Dm, 0.0)
    o = SCALE * (Ak[:, None] * Sq + _fdot(Qk, u, PREC))

    # gc is non-increasing and flat past T-1, so gc[C-1] == min(gc); Triton has
    # no scalar element extraction.
    gc_last = tl.min(gc)
    dec = tl.exp(gc_last - gc)
    S = tl.exp(gc_last) * S + _fdot(tl.trans(dec[:, None] * kf), u, PREC)

    # ---- swish-gated RMSNorm straight into the out_proj input ----
    of = o.to(tl.bfloat16).to(tl.float32)            # the reference rounds here
    zf = tl.load(proj_ptr + t[:, None] * S_PROJ + z_col[None, :],
                 mask=tmask[:, None], other=0.0).to(tl.float32)
    gate = zf * tl.sigmoid(zf)
    rstd = tl.rsqrt(tl.sum(of * of, axis=1) * (1.0 / V) + NEPS)
    y = of * rstd[:, None] * nw[None, :] * gate
    tl.store(out_ptr + t[:, None] * (HV * V) + (i_hv * V + dv)[None, :],
             y.to(tl.bfloat16), mask=tmask[:, None])
    return S


@triton.jit
def _gdn_fused_kernel(
    proj_ptr,        # *bf16 [T, S_PROJ]
    cw_ptr,          # conv1d weight [conv_dim, KC]
    A_log_ptr,       # fp32 [HV]
    dt_bias_ptr,     # [HV]
    nw_ptr,          # norm weight [V]
    cs_ptr,          # conv_state
    rec_ptr,         # recurrent state [slots, HV, V, K]
    sidx_ptr,        # int32 [n_seq] state / cache indices
    out_ptr,         # *bf16 [T, HV*V]
    T,
    NC,              # number of chunks = cdiv(T, C)
    cs_seq,
    cs_dim,
    cs_tok,
    USE_INIT: tl.constexpr,
    ONE_CHUNK: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    VP: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    KC: tl.constexpr,
    S_PROJ: tl.constexpr,
    QKVZ: tl.constexpr,
    SCALE: tl.constexpr,
    NEPS: tl.constexpr,
    L2EPS: tl.constexpr,
    SPT: tl.constexpr,
    BS: tl.constexpr,
    LOG_BS: tl.constexpr,
    LOG_NB: tl.constexpr,
    NP2: tl.constexpr,
    PREC: tl.constexpr,
    PSOLVE: tl.constexpr,
    GFLOOR: tl.constexpr,
):
    """One program per V head; the whole post-projection pipeline, chunks looped.

    The recurrence is sequential across chunks so a head cannot be split over
    programs, and the RMSNorm reduction spans the full head_v_dim so the V axis
    cannot be split either. 32 programs underfills B200, but at short T the
    device is idle anyway: the point is that there is *one* launch.

    ``ONE_CHUNK`` is a real compile-time branch, not a hint. With C=64 the chunk
    body needs ~1024 columns of Blackwell tensor memory when it sits in a loop --
    twice the 512 available -- so the looped form only ever compiles at C=32,
    while T <= 64 gets the cheaper single-chunk C=64 form (27 us vs 34 us at
    T=60). Leaving that to Triton's implicit ``NC == 1`` argument specialization
    happened to work, but silently OutOfResources'd the moment NC > 1.
    """
    i_hv = tl.program_id(0)
    i_h = i_hv // VP
    i_p = i_hv % VP

    G: tl.constexpr = 2 * K + 2 * VP * V          # qkvz group stride per K head

    dk = tl.arange(0, K)
    dv = tl.arange(0, V)
    tt = tl.arange(0, C)

    # proj columns: in_proj_qkvz emits [q(K) k(K) v(VP*V) z(VP*V)] per K head,
    # in_proj_ba [b(VP) a(VP)] per K head -- the deinterleave is index arithmetic.
    q_col = i_h * G + dk
    k_col = i_h * G + K + dk
    v_col = i_h * G + 2 * K + i_p * V + dv
    z_col = i_h * G + 2 * K + VP * V + i_p * V + dv
    b_col = QKVZ + i_h * (2 * VP) + i_p
    a_col = QKVZ + i_h * (2 * VP) + VP + i_p
    # conv channels: [q_all(H*K) | k_all(H*K) | v_all(HV*V)]
    q_ch = i_h * K + dk
    k_ch = H * K + i_h * K + dk
    v_ch = 2 * H * K + i_hv * V + dv

    sidx = tl.load(sidx_ptr).to(tl.int32)
    cs_base = sidx * cs_seq

    # --- initial recurrent state, S[k, v] (buffer is [slot, hv, v, k]) ---
    rec_base = rec_ptr + (sidx * HV + i_hv) * (V * K)
    if USE_INIT:
        S = tl.load(rec_base + dv[None, :] * K + dk[:, None]).to(tl.float32)
    else:
        S = tl.zeros([K, V], dtype=tl.float32)

    A_log = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_b = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    nw = tl.load(nw_ptr + dv).to(tl.float32)

    lo_s = tt[:, None] > tt[None, :]     # r <  t
    lo_i = tt[:, None] >= tt[None, :]    # r <= t
    eye = tl.where(tt[:, None] == tt[None, :], 1.0, 0.0)

    if ONE_CHUNK:
        S = _gdn_chunk(proj_ptr, cw_ptr, cs_ptr, out_ptr, S, 0, T, cs_base,
                       cs_dim, cs_tok, q_col, k_col, v_col, z_col, a_col, b_col,
                       q_ch, k_ch, v_ch, A_log, dt_b, nw, dk, dv, tt, lo_s, lo_i,
                       eye, i_hv, USE_INIT, HV, K, V, C, KC, S_PROJ, SCALE, NEPS,
                       L2EPS, SPT, BS, LOG_BS, LOG_NB, PREC, PSOLVE, GFLOOR)
    else:
        for i_c in range(NC):
            S = _gdn_chunk(proj_ptr, cw_ptr, cs_ptr, out_ptr, S, i_c, T, cs_base,
                           cs_dim, cs_tok, q_col, k_col, v_col, z_col, a_col,
                           b_col, q_ch, k_ch, v_ch, A_log, dt_b, nw, dk, dv, tt,
                           lo_s, lo_i, eye, i_hv, USE_INIT, HV, K, V, C, KC,
                           S_PROJ, SCALE, NEPS, L2EPS, SPT, BS, LOG_BS, LOG_NB,
                           PREC, PSOLVE, GFLOOR)

    # ---- final recurrent state, back to the [hv, v, k] buffer layout ----
    tl.store(rec_base + dv[None, :] * K + dk[:, None],
             S.to(rec_ptr.dtype.element_ty))

    # ---- conv state: the last KC-1 *pre-conv* tokens (zero/old-state padded) ----
    # This program owns its V channels; the Q/K channels of K head i_h are shared
    # with the i_p == 1 sibling, so only one of the pair writes them.
    _store_conv_state(proj_ptr, cs_ptr, v_col, v_ch, T, cs_base, cs_dim, cs_tok,
                      USE_INIT, KC, NP2, S_PROJ)
    if i_p == 0:
        _store_conv_state(proj_ptr, cs_ptr, q_col, q_ch, T, cs_base, cs_dim,
                          cs_tok, USE_INIT, KC, NP2, S_PROJ)
        _store_conv_state(proj_ptr, cs_ptr, k_col, k_ch, T, cs_base, cs_dim,
                          cs_tok, USE_INIT, KC, NP2, S_PROJ)


# ---------------------------------------------------------------------------
# Long-T path: prep -> segment-local states -> output (with the scan inlined).
#
# The single fused kernel above is one program per V head, so above a few
# hundred tokens it runs out of programs (32 of 148 SMs) *and* out of
# efficiency: measured on one SM a C=32 chunk step costs 10.6 us for ~5 MFLOP,
# 3% of the SM's bf16 peak, and 47% of that is the conv/SiLU/L2-norm/gating --
# embarrassingly parallel elementwise work that only sits inside the recurrence
# loop because a short call cannot afford a second launch.  A grid-scaling probe
# (dev/gridscale.py) shows 4x the programs run in the same wall time, i.e. the
# machine is idle.
#
# Above _LONG_MIN_T the extra launches are affordable, so the work is split:
#
#   1. _gdn_prep_kernel  grid (T/BT, 2H+HV)  conv1d+SiLU, q/k L2-norm, g/beta,
#                                            conv-state -- one pass over proj.
#   2. _gdn_seg_kernel   grid (G, HV)        each segment's state from a *zero*
#                                            entry state, plus its total decay.
#   3. _gdn_out_kernel   grid (G, HV)        segment entry states by replaying the
#                                            scan, then the segment's chunks again,
#                                            gated RMSNorm, final state, store.
#
# The recurrence is linear in S with a *scalar* per-token decay, so a segment is
# summarised exactly by (local state, total decay) and combining segments is an
# ordinary scalar scan over G elements -- cheap enough to replay per program
# rather than pay a launch for (see _gdn_out_kernel).
# ---------------------------------------------------------------------------


@triton.jit
def _conv_silu_rows(proj_ptr, cw_ptr, t, tmask, col, ch, T, BT: tl.constexpr,
                    D: tl.constexpr, KC: tl.constexpr, S_PROJ: tl.constexpr):
    """``_conv_silu`` without the conv-state arm: the long path is gated on
    ``has_initial_state`` all-False, so taps before token 0 are zero."""
    acc = tl.zeros([BT, D], dtype=tl.float32)
    for j in tl.static_range(KC):
        s = t - (KC - 1) + j
        val = tl.load(proj_ptr + s[:, None] * S_PROJ + col[None, :],
                      mask=tmask[:, None] & (s >= 0)[:, None], other=0.0)
        w = tl.load(cw_ptr + ch * KC + j).to(tl.float32)
        acc += w[None, :] * val.to(tl.float32)
    y = acc * tl.sigmoid(acc)
    return y.to(tl.bfloat16)


@triton.jit
def _tail(proj_ptr, cs_ptr, sj, js, keep, col, ch, cs_base, cs_dim, cs_tok,
          S_PROJ: tl.constexpr):
    """conv_state <- the KC-1 pre-conv tokens at sj, for channels ``ch``."""
    src = tl.load(proj_ptr + sj[:, None] * S_PROJ + col[None, :],
                  mask=keep[:, None], other=0.0)
    tl.store(cs_ptr + cs_base + ch[None, :] * cs_dim + js[:, None] * cs_tok,
             src, mask=keep[:, None])


@triton.jit
def _gdn_prep_kernel(
    proj_ptr,        # *bf16 [T, S_PROJ]
    cw_ptr,          # conv1d weight [conv_dim, KC]
    A_log_ptr, dt_bias_ptr,
    qkv_ptr,         # *bf16 [T, 2*H*K + HV*V]  conv'd + SiLU, q/k L2-normalised
    gb_ptr,          # *fp32 [T, 2*HV]          g | beta
    cs_ptr, sidx_ptr, T, NT, cs_seq, cs_dim, cs_tok,
    H: tl.constexpr, HV: tl.constexpr, VP: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, KC: tl.constexpr, S_PROJ: tl.constexpr,
    QKVZ: tl.constexpr, L2EPS: tl.constexpr, SPT: tl.constexpr,
    NP2: tl.constexpr, GFLOOR: tl.constexpr,
):
    """One program per (token tile, conv channel block): the whole
    post-projection elementwise stage, in the layout the segment passes read.

    The output buffer is indexed by *conv channel* -- ``[q(H*K) | k(H*K) |
    v(HV*V)]``, the same order the conv weight is in -- so the store offset is
    just the channel index and the three groups need no separate pointers.
    Channel blocks are K = V = D wide, one per (K head) for q/k and one per
    (V head) for v; the v blocks also emit their head's g/beta.  Every channel is
    written by exactly one program.
    """
    i_t = tl.program_id(0)
    i_b = tl.program_id(1)
    GRP: tl.constexpr = 2 * D + 2 * VP * D      # qkvz group stride per K head

    d = tl.arange(0, D)
    t = i_t * BT + tl.arange(0, BT)
    tmask = t < T

    # channel block -> (proj column base, conv channel base)
    if i_b < H:                                  # q of K head i_b
        col = i_b * GRP + d
    elif i_b < 2 * H:                            # k of K head i_b - H
        col = (i_b - H) * GRP + D + d
    else:                                        # v of V head i_b - 2H
        i_hv = i_b - 2 * H
        col = (i_hv // VP) * GRP + 2 * D + (i_hv % VP) * D + d
    ch = i_b * D + d

    y = _conv_silu_rows(proj_ptr, cw_ptr, t, tmask, col, ch, T, BT, D, KC,
                        S_PROJ)
    if i_b < 2 * H:
        y = _l2norm_bf16(y, L2EPS)
    tl.store(qkv_ptr + t[:, None] * ((2 * H + HV) * D) + ch[None, :], y,
             mask=tmask[:, None])

    if i_b >= 2 * H:
        # g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), fp32.
        # Same branch-free softplus and NaN-safe floor as the fused kernel --
        # see _gdn_chunk for why both are load-bearing.
        i_hv = i_b - 2 * H
        b_col = QKVZ + (i_hv // VP) * (2 * VP) + i_hv % VP
        av = tl.load(proj_ptr + t * S_PROJ + b_col + VP, mask=tmask,
                     other=0.0).to(tl.float32)
        bv = tl.load(proj_ptr + t * S_PROJ + b_col, mask=tmask,
                     other=0.0).to(tl.float32)
        xg = av + tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        sp = tl.maximum(xg, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(xg)))
        sp = tl.where(xg <= SPT, sp, xg)
        gv = -tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32)) * sp
        gv = tl.where(gv > GFLOOR, gv, GFLOOR)
        tl.store(gb_ptr + t * (2 * HV) + i_hv, gv, mask=tmask)
        tl.store(gb_ptr + t * (2 * HV) + HV + i_hv, tl.sigmoid(bv), mask=tmask)

    # conv_state: the last KC-1 *pre-conv* tokens of this program's channels.
    # T > _LONG_MIN_T > KC-1 on this path, so the short-sequence case (surviving
    # tail of the incoming state, shifted) cannot arise.
    if i_t == NT - 1:
        js = tl.arange(0, NP2)
        _tail(proj_ptr, cs_ptr, T - (KC - 1) + js, js, js < KC - 1, col, ch,
              tl.load(sidx_ptr).to(tl.int32) * cs_seq, cs_dim, cs_tok, S_PROJ)


@triton.jit
def _gdn_seg_kernel(
    qkv_ptr, gb_ptr,
    sst_ptr,         # *fp32 [G, HV, K, V]  segment-local states
    sdec_ptr,        # *fp32 [G, HV]        segment total log-decay
    T, NC, CPS,
    H: tl.constexpr, HV: tl.constexpr, VP: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, C: tl.constexpr, BS: tl.constexpr, LOG_BS: tl.constexpr,
    LOG_NB: tl.constexpr, PREC: tl.constexpr, PSOLVE: tl.constexpr,
):
    """Pass 1: segment ``i_s`` of V head ``i_hv``, run from a zero entry state.

    Emits the segment's own state contribution and ``sum(g)`` over its tokens --
    together an exact summary of the segment as an affine map on S, because the
    decay is a scalar per token.
    """
    i_s = tl.program_id(0)
    i_hv = tl.program_id(1)
    QD: tl.constexpr = (2 * H + HV) * K

    dk = tl.arange(0, K)
    dv = tl.arange(0, V)
    tt = tl.arange(0, C)
    lo_s = tt[:, None] > tt[None, :]
    eye = tl.where(tt[:, None] == tt[None, :], 1.0, 0.0)
    k_off = (H * K + (i_hv // VP) * K + dk)[None, :]
    v_off = (2 * H * K + i_hv * V + dv)[None, :]

    S = tl.zeros([K, V], dtype=tl.float32)
    gtot = 0.0
    c0 = i_s * CPS
    n = tl.minimum(CPS, NC - c0)
    for j in range(n):
        t = (c0 + j) * C + tt
        tmask = t < T
        kb = tl.load(qkv_ptr + t[:, None] * QD + k_off, mask=tmask[:, None],
                     other=0.0)
        vb = tl.load(qkv_ptr + t[:, None] * QD + v_off, mask=tmask[:, None],
                     other=0.0)
        gv = tl.load(gb_ptr + t * (2 * HV) + i_hv, mask=tmask, other=0.0)
        bt = tl.load(gb_ptr + t * (2 * HV) + HV + i_hv, mask=tmask, other=0.0)

        gc = tl.cumsum(gv, axis=0)
        Dm = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
        kf = kb.to(tl.float32)
        M = tl.where(lo_s, -bt[:, None] * Dm * tl.dot(kb, tl.trans(kb)), 0.0)
        # w = beta*(v - exp(gc)*(k^T S)) with the scaling moved onto the A
        # operand so the k^T S term needs no accumulator of its own -- the same
        # tensor-memory economy as _fadd_dot, and no precision cost, because
        # _fdot rounds both operands to bf16 anyway.
        w = _fadd_dot(bt[:, None] * vb.to(tl.float32),
                      (-bt * tl.exp(gc))[:, None] * kf, S, PREC)
        u = _block_solve(M, w, tt, eye, BS, LOG_BS, LOG_NB, PSOLVE)
        gcl = tl.min(gc)
        S = _fadd_dot(tl.exp(gcl) * S,
                      tl.trans(tl.exp(gcl - gc)[:, None] * kf), u, PREC)
        gtot += gcl

    tl.store(sst_ptr + (i_s * HV + i_hv) * (K * V) + dk[:, None] * V
             + dv[None, :], S)
    tl.store(sdec_ptr + i_s * HV + i_hv, gtot)


@triton.jit
def _gdn_out_kernel(
    qkv_ptr, gb_ptr, sst_ptr, sdec_ptr, rec_ptr, sidx_ptr, proj_ptr, nw_ptr,
    out_ptr, T, NC, CPS, G,
    H: tl.constexpr, HV: tl.constexpr, VP: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, C: tl.constexpr, S_PROJ: tl.constexpr,
    QKVZ: tl.constexpr, SCALE: tl.constexpr, NEPS: tl.constexpr,
    BS: tl.constexpr, LOG_BS: tl.constexpr, LOG_NB: tl.constexpr,
    PREC: tl.constexpr, PSOLVE: tl.constexpr,
):
    """Pass 3: segment ``i_s`` of V head ``i_hv`` again, from its corrected entry
    state, through the swish-gated RMSNorm and straight into the out_proj input.

    The entry-state scan is done here rather than in a kernel of its own.  It is
    a scalar-decay recurrence over G elements -- ``S <- exp(sum g_j) S + S_loc[j]``
    -- so program ``i_s`` can just replay the first ``i_s`` of them itself.  G is
    pinned near ``148 / HV`` (see ``_LMAX_G``), so that is at most three extra
    [K, V] loads and it buys back a whole launch (~13 us of host) plus the scan
    kernel's own 8 us.  The last segment ends holding the sequence's final state
    and writes it to the recurrent buffer, in the ``[hv, v, k]`` layout the decode
    kernel uses.
    """
    i_s = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // VP
    QD: tl.constexpr = (2 * H + HV) * K
    GRP: tl.constexpr = 2 * K + 2 * VP * V

    dk = tl.arange(0, K)
    dv = tl.arange(0, V)
    tt = tl.arange(0, C)
    lo_s = tt[:, None] > tt[None, :]
    lo_i = tt[:, None] >= tt[None, :]
    eye = tl.where(tt[:, None] == tt[None, :], 1.0, 0.0)
    q_off = (i_h * K + dk)[None, :]
    k_off = (H * K + i_h * K + dk)[None, :]
    v_off = (2 * H * K + i_hv * V + dv)[None, :]
    z_off = (i_h * GRP + 2 * K + VP * V + (i_hv % VP) * V + dv)[None, :]
    nw = tl.load(nw_ptr + dv).to(tl.float32)

    s_off = dk[:, None] * V + dv[None, :]
    S = tl.zeros([K, V], dtype=tl.float32)
    for j in range(i_s):
        S = tl.exp(tl.load(sdec_ptr + j * HV + i_hv)) * S \
            + tl.load(sst_ptr + (j * HV + i_hv) * (K * V) + s_off)
    c0 = i_s * CPS
    n = tl.minimum(CPS, NC - c0)
    for j in range(n):
        t = (c0 + j) * C + tt
        tmask = t < T
        qb = tl.load(qkv_ptr + t[:, None] * QD + q_off, mask=tmask[:, None],
                     other=0.0)
        kb = tl.load(qkv_ptr + t[:, None] * QD + k_off, mask=tmask[:, None],
                     other=0.0)
        vb = tl.load(qkv_ptr + t[:, None] * QD + v_off, mask=tmask[:, None],
                     other=0.0)
        gv = tl.load(gb_ptr + t * (2 * HV) + i_hv, mask=tmask, other=0.0)
        bt = tl.load(gb_ptr + t * (2 * HV) + HV + i_hv, mask=tmask, other=0.0)

        gc = tl.cumsum(gv, axis=0)
        Ak = tl.exp(gc)
        Dm = tl.exp(tl.minimum(gc[:, None] - gc[None, :], 0.0))
        kf = kb.to(tl.float32)
        M = tl.where(lo_s, -bt[:, None] * Dm * tl.dot(kb, tl.trans(kb)), 0.0)
        w = _fadd_dot(bt[:, None] * vb.to(tl.float32),
                      (-bt * Ak)[:, None] * kf, S, PREC)
        u = _block_solve(M, w, tt, eye, BS, LOG_BS, LOG_NB, PSOLVE)

        Qk = tl.where(lo_i, tl.dot(qb, tl.trans(kb)) * Dm, 0.0)
        o = SCALE * _fadd_dot(_fdot(Ak[:, None] * qb.to(tl.float32), S, PREC),
                              Qk, u, PREC)
        gcl = tl.min(gc)
        S = _fadd_dot(tl.exp(gcl) * S,
                      tl.trans(tl.exp(gcl - gc)[:, None] * kf), u, PREC)

        of = o.to(tl.bfloat16).to(tl.float32)     # the reference rounds here
        zf = tl.load(proj_ptr + t[:, None] * S_PROJ + z_off,
                     mask=tmask[:, None], other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(of * of, axis=1) * (1.0 / V) + NEPS)
        y = of * rstd[:, None] * nw[None, :] * (zf * tl.sigmoid(zf))
        tl.store(out_ptr + t[:, None] * (HV * V) + (i_hv * V + dv)[None, :],
                 y.to(tl.bfloat16), mask=tmask[:, None])

    if i_s == G - 1:
        sidx = tl.load(sidx_ptr).to(tl.int32)
        tl.store(rec_ptr + (sidx * HV + i_hv) * (V * K) + dv[None, :] * K
                 + dk[:, None], S.to(rec_ptr.dtype.element_ty))


class Qwen3NextGDNAttention(_BaselineGDN):
    """GDN linear attention with the post-projection pipeline fused into one kernel."""

    def _plan(self):
        """Launch-invariant state, cached in ``__dict__``.

        Every ``self.<param>`` here goes through ``nn.Module.__getattr__``, which
        walks ``_parameters`` / ``_buffers`` / ``_modules`` -- about a microsecond
        each, and this path touches six of them plus a method call. At a 60 us
        budget that overhead was measurable (70.9 us for the module's forward
        against 58.4 us for the same work called inline). Returns ``False`` when
        the layout is not one the fused kernel handles.
        """
        plan = self.__dict__.get("_fplan")
        if plan is None:
            plan = False
            if (self._in_proj_w is not None
                    and self.local_k_heads == self.num_k_heads      # tp == 1 layout
                    and self.head_k_dim == self.head_v_dim
                    and self.conv_kernel_size == 4
                    and not getattr(self.out_proj, "use_fp8", False)
                    and self.out_proj.tp_size == 1
                    and self.out_proj.bias is None):
                K, V = self.head_k_dim, self.head_v_dim
                HV = self.local_v_heads
                common = dict(
                    H=self.local_k_heads, HV=HV, VP=self.v_per_k, K=K, V=V,
                    KC=self.conv_kernel_size, S_PROJ=self._in_proj_w.shape[0],
                    QKVZ=self._qkvz_dim, SCALE=K ** -0.5, NEPS=self.norm.eps,
                    L2EPS=1e-6, SPT=20.0, BS=_BS, LOG_BS=_BS.bit_length() - 1,
                    NP2=triton.next_power_of_2(self.conv_kernel_size - 1),
                    PREC=_PREC, PSOLVE=_PSOLVE, GFLOOR=_GFLOOR, USE_INIT=False,
                    num_warps=_NUM_WARPS,
                )
                # Only two constexpr sets are ever needed: T <= _CHUNK1 is one
                # chunk of the big tile, anything longer loops the small one.
                plan = (
                    self._in_proj_w.t(), self.conv1d.weight, self.A_log,
                    self.dt_bias, self.norm.weight, self.out_proj.weight.t(),
                    self.layer_idx, HV, HV * V,
                    dict(common, C=_CHUNK1, ONE_CHUNK=True, num_stages=1,
                         LOG_NB=(_CHUNK1 // _BS).bit_length() - 1),
                    # Two stages let the next chunk's conv loads overlap the
                    # current chunk's solve: 12.9 -> 11.7 us/chunk at T=445.
                    # Three does not fit shared memory (232 KB limit).
                    dict(common, C=_CHUNKN, ONE_CHUNK=False, num_stages=2,
                         LOG_NB=(_CHUNKN // _BS).bit_length() - 1),
                )
            object.__setattr__(self, "_fplan", plan)
        return plan

    def _scratch(self, slot: str, n: int, width: int, ref: torch.Tensor):
        """Reusable internal buffer, one slot per name, re-cut when the shape changes.

        Holds the projection output and the out_proj *input* -- both consumed
        inside the same call, so recycling them is invisible. The value returned
        to the caller is always freshly allocated; handing back a recycled buffer
        would alias across calls and break any caller that keeps the result.
        """
        buf = self.__dict__.get(slot)
        if buf is None or buf[0] != n or buf[1].shape[1] != width:
            buf = (n, torch.empty(n, width, dtype=ref.dtype, device=ref.device))
            object.__setattr__(self, slot, buf)
        return buf[1]

    def forward_impl(self, hidden_states: torch.Tensor, state_manager=None) -> torch.Tensor:
        md = _get_context().kda_metadata
        if state_manager is None:
            state_manager = _get_context().kda_state
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextGDNAttention requires engine-managed recurrent state "
                "and metadata",
            )
        x_flat = hidden_states.reshape(-1, self.hidden_size)
        N = x_flat.shape[0]
        sidx = md.non_spec_state_indices_tensor
        plan = self._plan()

        # ``has_initial_state`` becomes a compile-time constant, taken from the
        # host-side summary the chunk planner already computed -- reading the
        # device mask instead would cost a stream sync per layer. It must be
        # all-*False*, not merely known: with an incoming conv state the kernel
        # would have to read it, and the q/k conv columns of K head h are read by
        # both of its V-head programs (hv = 2h and 2h+1) but written by one, with
        # no grid-wide barrier between -- a real race, seen as a wrong conv state
        # at T < 4. Continuation chunks and the ambiguous "some but not all"
        # summary therefore stay on the baseline; fixing it properly means a
        # fourth launch on the path that matters.
        if not (plan
                and md.num_prefills > 0
                and md.num_decodes == 0
                and sidx is not None
                and sidx.numel() == 1
                and md.has_initial_state is not None
                and not md.any_have_initial_state
                and 0 < N <= _LONG_MAX_T):
            return super().forward_impl(hidden_states, state_manager)

        (in_w_t, conv_w, a_log, dt_bias, norm_w, out_w_t,
         li, HV, OW, kw1, kwN) = plan

        proj = self._scratch("_fbuf_p", N, in_w_t.shape[1], x_flat)
        torch.mm(x_flat, in_w_t, out=proj)
        o = self._scratch("_fbuf_o", N, OW, x_flat)
        cs = state_manager.gdn_conv[li]
        if N > _LONG_MIN_T:
            return self._forward_long(plan, x_flat, N, proj, o, cs, sidx,
                                      state_manager)
        if N <= _CHUNK1:
            kw, nc = kw1, 1
        else:
            kw, nc = kwN, -(-N // _CHUNKN)

        _gdn_fused_kernel[(HV,)](
            proj, conv_w, a_log, dt_bias, norm_w, cs,
            state_manager.recurrent[li], sidx, o, N, nc,
            cs.stride(0), cs.stride(1), cs.stride(2), **kw,
        )
        return torch.mm(o, out_w_t)

    def _forward_long(self, plan, x_flat, N, proj, o, cs, sidx, state_manager):
        """Three Triton launches: prep, segment states, output.

        Segment length is chosen so ``G * HV`` is one wave of the machine's 148
        SMs -- occupancy is one program per SM, so a second wave costs a full
        extra pass over the chunks (measured: G=4 128 us, G=5 153 us at T=445).
        """
        (in_w_t, conv_w, a_log, dt_bias, norm_w, out_w_t,
         li, HV, OW, _kw1, _kwN) = plan
        lp = self._lplan()
        H, K, V, VP = lp["H"], lp["K"], lp["V"], lp["VP"]
        nc = -(-N // _LC)
        cps = -(-nc // _LMAX_G)
        G = -(-nc // cps)
        nt = -(-N // _LBT)

        qkv = self._scratch("_lb_x", N, (2 * H + HV) * K, x_flat)
        gb = self._fp32("_lb_g", N, 2 * HV, x_flat.device)
        sst = self._fp32("_lb_s", G * HV, K * V, x_flat.device)
        sdec = self._fp32("_lb_d", G, HV, x_flat.device)

        _gdn_prep_kernel[(nt, 2 * H + HV)](
            proj, conv_w, a_log, dt_bias, qkv, gb, cs, sidx, N, nt,
            cs.stride(0), cs.stride(1), cs.stride(2), **lp["prep"],
        )
        _gdn_seg_kernel[(G, HV)](qkv, gb, sst, sdec, N, nc, cps, **lp["seg"])
        _gdn_out_kernel[(G, HV)](
            qkv, gb, sst, sdec, state_manager.recurrent[li], sidx, proj, norm_w,
            o, N, nc, cps, G, **lp["out"],
        )
        return torch.mm(o, out_w_t)

    def _fp32(self, slot: str, n: int, width: int, device):
        buf = self.__dict__.get(slot)
        if buf is None or buf.shape != (n, width):
            buf = torch.empty(n, width, dtype=torch.float32, device=device)
            object.__setattr__(self, slot, buf)
        return buf

    def _lplan(self):
        """Constexpr sets for the four long-path kernels, cached in ``__dict__``."""
        lp = self.__dict__.get("_lp")
        if lp is None:
            K, V = self.head_k_dim, self.head_v_dim
            H, HV, VP = self.local_k_heads, self.local_v_heads, self.v_per_k
            dims = dict(H=H, HV=HV, VP=VP, K=K, V=V)
            solve = dict(BS=_BS, LOG_BS=_BS.bit_length() - 1,
                         LOG_NB=(_LC // _BS).bit_length() - 1,
                         PREC=_PREC, PSOLVE=_PSOLVE)
            lp = dict(
                dims,
                prep=dict(H=H, HV=HV, VP=VP, D=K, BT=_LBT,
                          KC=self.conv_kernel_size,
                          S_PROJ=self._in_proj_w.shape[0], QKVZ=self._qkvz_dim,
                          L2EPS=1e-6, SPT=20.0, GFLOOR=_GFLOOR,
                          NP2=triton.next_power_of_2(self.conv_kernel_size - 1),
                          num_warps=_LWARPS[2], num_stages=_LSTAGES[2]),
                seg=dict(H=H, HV=HV, VP=VP, K=K, V=V, C=_LC, **solve,
                         num_warps=_LWARPS[0], num_stages=_LSTAGES[0]),
                out=dict(H=H, HV=HV, VP=VP, K=K, V=V, C=_LC,
                         S_PROJ=self._in_proj_w.shape[0], QKVZ=self._qkvz_dim,
                         SCALE=K ** -0.5, NEPS=self.norm.eps, **solve,
                         num_warps=_LWARPS[1], num_stages=_LSTAGES[1]),
            )
            object.__setattr__(self, "_lp", lp)
        return lp
