"""Gated multi-head attention with a bias list, for B200 / sm_100.

Same contract as the baseline: openfold3's ``Attention``, four projections, an
additive bias list, a softmax, a sigmoid gate and an output projection.

The baseline is not GPU-bound. It issues 14-17 kernel launches for 1-2 us of real
work, and ``profile/01-launch-budget/`` shows why that is the whole problem: inside
the harness's timed window, a host-issued CUDA operation costs ~4.1 us of measured
latency *while its own GPU time stays under about 4 us* -- and in that regime what the
operation does is irrelevant. One elementwise add costs 4.09 us on a 12 KB tensor and
4.09 us on a 96 KB one, an already-compiled Triton launch costs the same as an eager
aten op, and the ``n_ops=0`` floor tracks the *number* of contiguous input leaves the
shifting pool copies (16.4 us at three, 12.3 us at two) rather than their size. Above
that threshold the window tracks real GPU time in 2.048 us steps, so the general model
is ``floor + sum over operations of max(~4.1 us, that operation's GPU time)``. Either
way the launch count is what there is to attack, and the count is what this file
collapses.

This file issues two: one Triton kernel per ``(batch, head)`` that projects q/k/v/g,
forms the score tile, adds the biases, takes an exact single-pass softmax, multiplies
by V and applies the sigmoid gate into a ``[B*Q, H*C]`` buffer; then the frozen L1
``Linear`` for the output projection. Measured against the same window that scores
the operator, that op set lands at 18.6-24.6 us against the baseline's 145-178 us
(``profile/01-launch-budget/REPORT.md`` section 3).

The specified pure-torch alternative -- four projections, a pre-summed bias and
``F.scaled_dot_product_attention`` -- was measured on the same five cases and is a
*regression* (0.73x-0.81x): SDPA with an additive ``attn_mask`` at these sizes
(Q <= 32, K <= 128, C <= 48) issues more host operations than the baseline's einsum
chain, and by the paragraph above that is all that matters. It is not implemented.

Two properties of the harness shape the rest:

* Weights arrive through ``load_state_dict(..., strict=False)`` *after* the module is
  moved and cast, so a renamed parameter would silently load nothing and the module
  would run on random values -- a wrong answer, not an error. The submodule tree is
  therefore the baseline's verbatim, and nothing is derived from a weight in
  ``__init__``, where it is still ``torch.empty`` garbage.
* The bench replaces every captured bias with ``torch.zeros_like``, so adding the
  biases and ignoring them score bit-identically. The bias arithmetic below is
  verified by ``profile/02-parity/parity.py`` against the baseline class on inputs the
  bench never generates, not by the bench.

Everything the dispatch predicate does not positively claim runs the baseline body
below verbatim, over the same L1 ``Linear`` submodules.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.linear import Linear
from ..L1.softmax import Softmax  # noqa: F401  -- kept: the baseline imports it too

# ---------------------------------------------------------------------------
# The kernel's written domain, and the configuration table.
# ---------------------------------------------------------------------------

# Bounds the kernel body is written for. The score tile is a single [QP, KP] block,
# which is what makes the softmax one exact pass with no online rescaling and no
# second weight read -- so K is bounded, not tiled. C is bounded because the whole
# per-head projection lives in registers.
_Q_MAX = 64
_K_MAX = 128
_C_MAX = 64
_NB_MAX = 4      # bias slots in the kernel signature; a 5th bias delegates
_MIN_TILE = 16   # tl.dot's contraction floor for bf16 on this stack

# (DP, num_warps, num_stages) keyed by (Q, K, C, c_q, c_k). DP blocks the projection
# reduction over the input width. Every entry comes from the 27-point sweep in
# profile/04-config-sweep/ and compiled with n_spills == 0; the comment records the
# kernel's profiled device time and its register count. Deliberately not
# triton.autotune: autotuning costs host time on every call, and one launch inside the
# scored window is already ~4.1 us of it.
#
# Larger DP is what mattered -- it puts more weight bytes in flight per iteration of
# the projection reduction, which is the limiter on the low-CTA cases. Where two
# configurations were within the scored window's 2.048 us quantum, the one with the
# lower register count was taken: case 1's (64, 4, 3) is 1.0 us faster than the
# (64, 8, 3) below but compiles to 255 registers, i.e. exactly at the cap and one
# compiler decision away from spilling, and that 1.0 us cannot be seen in the score.
_CONFIGS: dict[tuple, tuple] = {
    (16, 16, 24, 384, 384): (128, 4, 4),    # 10.19 us kernel, n_regs=128
    (32, 128, 32, 128, 128): (64, 8, 3),    # 18.43 us kernel, n_regs=201
    (16, 16, 32, 128, 128): (64, 4, 4),     #  7.23 us kernel, n_regs=80
    (16, 16, 48, 768, 768): (128, 4, 4),    # 17.94 us kernel, n_regs=164
}
# For a geometry the sweep never visited, eight warps rather than four: it spreads the
# same tile over twice the threads, which is the cheapest insurance against register
# pressure at the corners of the written domain (KP=128 with CP=64), where the sweep
# did see configurations spill.
_DEFAULT_CONFIG = (64, 8, 3)

# Per-tile-class defaults, for the classes the single default above does not clear the
# no-spill gate on. `profile/07-domain-spills/` compiles the enumerable part of the
# specialization space -- every (QP, KP, CP) tile class x NB 0..4 x HAS_QB, in both of
# Triton's divisibility extremes -- and found the single default spilling. Each affected
# class gets a configuration verified spill-free across all of those combinations at once,
# because none of the axes is well behaved: the spill count is not monotonic in the bias
# count (34 stores at NB=0 against 8 at NB=4 on one class), and neither divisibility form
# bounds the other (one class is clean at 128 registers unaligned and spills 2 stores at
# 64 registers aligned, the compiler having picked a different register target). Keyed by
# tile class rather than by geometry because that is what the compiler sees.
#
# That sweep is corroboration, not a domain-wide proof, and it does not claim to be one:
# the *mixed* divisibility forms of the seven geometry integers are reachable and are not
# enumerable. `_reject_if_spilled` is what makes the no-spill property hold across the
# whole admitted domain.
#
# `num_warps = 16` does most of the work: spreading the same tile over twice the threads
# halves per-thread pressure and takes 255 registers down to 60-128. It is not a universal
# answer -- on the two largest tiles, capping the budget makes the tile no longer fit and it
# spills, so those classes take fewer pipeline stages and a narrower reduction block
# instead and land at 250 registers with zero spills.
#
# Every entry is verified spill-free across NB 0..4 x HAS_QB x **both** of Triton's runtime
# divisibility forms. All three axes matter and none is well behaved: the spill count is not
# monotonic in the bias count (34 stores at NB=0 against 8 at NB=4 on one class), and
# neither divisibility form bounds the other -- one class is clean at 128 registers
# unaligned and spills 2 stores at 64 registers aligned, the compiler having chosen a
# different register target. Four of these nine classes were only discovered once the
# sweep's "aligned" form was fixed to actually align K; before that the fully aligned
# kernel was never built, and the dispatch-time guard caught one of them spilling first.
#
# Selection rule among the spill-free candidates: lowest worst-case register count, which
# is the only quantity there is evidence about for these classes -- none is a scored
# geometry, so none has a measured latency to trade against. The cost is real and worth
# naming: 16 warps on a 16x32 score tile puts 512 threads on 512 score elements and is
# very likely slower than the clean 8-warp option. These entries buy resource safety, not
# speed. `profile/07-domain-spills/repair_runs.txt` keeps the full ladder so a later round
# can re-pick on measured latency without repeating the search.
_DEFAULT_BY_TILE: dict[tuple, tuple] = {
    (16, 32, 32): (32, 16, 2),      # worst n_regs=60
    (32, 128, 16): (64, 8, 1),      # worst n_regs=255
    (32, 128, 32): (32, 16, 2),     # worst n_regs=128
    (32, 128, 64): (64, 16, 2),     # worst n_regs=128
    (64, 64, 16): (32, 16, 2),      # worst n_regs=120
    (64, 64, 64): (64, 16, 2),      # worst n_regs=112
    (64, 128, 16): (32, 16, 2),     # worst n_regs=128
    (64, 128, 32): (32, 8, 1),      # worst n_regs=250
    (64, 128, 64): (32, 8, 1),      # worst n_regs=250
}

# Resolved launch constants per geometry key: (QP, KP, CP, DP, num_warps, num_stages).
# Filled on first sight of a geometry so the repeated path is one dict lookup. Nothing
# is keyed on data_ptr -- the shifting pool hands out a fresh one every iteration.
_LAUNCH: dict[tuple, tuple] = {}

# Fast-path entries per geometry. Plain host ints, incremented on dispatch: no device
# traffic, no sync, nothing the harness's integrity guards watch. This is what
# distinguishes "the fast path ran" from "the fast path never ran and the baseline
# tied itself".
_FASTPATH_HITS: dict[tuple, int] = {}

# Geometries whose compiled kernel has been inspected, and those withdrawn because it
# spills. A spilling configuration must not be shipped, and for this kernel that cannot
# be established by enumerating the domain: Triton specializes every integer argument on
# divisibility-by-16 and on being 1, the predicate constrains almost none of them, and the
# geometry integers' divisibility turns out to be load-bearing for codegen -- denying it
# with `do_not_specialize` costs 12 to 90 spilled stores and up to 2.2x on the scored
# cases (measured in profile/09-specialization/). So the property is *enforced* instead:
# the first launch for a geometry returns its `CompiledKernel`, its `n_spills` is read
# once, and a geometry whose kernel spills is delegated from then on. That covers every
# geometry, including ones no sweep anticipated, which enumeration could not.
_VERIFIED: set[tuple] = set()
_SPILL_REJECTED: dict[tuple, int] = {}

_KERNEL = None
_KERNEL_STATUS = "uninitialized"

# Device indices whose compute capability is sm_100. The configuration table below was
# measured on B200 and nothing here has been measured anywhere else, so the fast path
# admits only those devices; widening this is a measurement, not an edit. Enumerated
# once at import because the predicate runs on every forward and must stay to host-side
# integer and attribute comparisons -- ``torch.cuda.get_device_capability`` is a runtime
# call and does not belong there.
_SM100_DEVICES: frozenset[int] = frozenset()


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _launch_constants(key: tuple) -> tuple:
    """Resolve (QP, KP, CP, DP, num_warps, num_stages) for a geometry, once.

    Three tiers, most specific first: a measured entry for this exact geometry, then a
    default for its tile class (which exists where the global default was shown to
    spill), then the global default.
    """
    nq, nk, c = key[0], key[1], key[2]
    tile = (max(_MIN_TILE, _next_pow2(nq)), max(_MIN_TILE, _next_pow2(nk)),
            max(_MIN_TILE, _next_pow2(c)))
    cfg = _CONFIGS.get(key) or _DEFAULT_BY_TILE.get(tile) or _DEFAULT_CONFIG
    cst = tile + cfg
    _LAUNCH[key] = cst
    return cst


# ---------------------------------------------------------------------------
# Kernel: one program per (batch, head).
#
# Body order is K, S+bias, softmax, V, O, G rather than all four projections up
# front. Both compute the same thing; this order roughly halves peak live state
# (Qh + Kh + S at the worst scored point instead of all of Qh/Kh/Vh/Gh plus S), which
# is what keeps the register file from spilling at KP=128, CP=64.
#
# Rounding follows the baseline's rather than improving on it: the comparison target
# is the baseline, so extra accuracy could only widen the measured deviation. Each
# projection is rounded to bf16 where its Linear would round, the scale is applied in
# bf16, the score tile is rounded to bf16 before the bias adds, the softmax runs in
# fp32 from those bf16 scores and rounds once, and O is rounded before the gate. Each
# of those roundings was checked against torch individually; see the note on the
# softmax's residual ulp below.
#
# Padding is inert for every input, not only for finite ones. Masked weight loads with
# other=0 would zero the padded lanes of Qh/Kh/Vh on their own, but only because
# 0 * x == 0 for finite x -- a single Inf in a weight would make a padded lane NaN and,
# since C and K are the two contraction axes, that NaN would reach every valid element.
# So the padded lanes of the two contraction axes are selected to zero outright. Padded
# K columns are set to -inf before the softmax (not a large negative sentinel -- on a
# fully -inf row a finite sentinel would win the softmax and return a plausible number
# where the baseline returns NaN); bias loads are masked on padded q *and* padded k, so
# a padded q row cannot issue an out-of-bounds bias read.
#
# The rounding is placed where the baseline places it, so deviation stays at the level
# of a single bf16 rounding rather than accumulating -- it is not bit-exactness, and is
# not claimed as such: Triton's exp and reduction schedule differ from PyTorch's fused
# softmax, which can move the result by one bf16 ulp (~0.4 % relative, against a 1e-2
# tolerance).
# ---------------------------------------------------------------------------
def _build_kernel():
    """Compile the fused kernel and return a launcher, or raise.

    Constructing a ``@triton.jit`` function compiles nothing; each specialization is
    compiled on its first launch. The scored geometry is fixed per case, so that
    first launch falls in the harness's correctness rounds, which precede its
    ``threading.active_count()`` snapshot, and the specialization is stable
    afterwards because the pool steps slots by 256 bytes and the shapes never change.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def _bias_add(s, P, sb, sh, sq, sk, b, h, rq, rk, ok):
        """One bias slot, added in the baseline's order: bf16 score + bf16 bias."""
        v = tl.load(P + b * sb + h * sh + rq[:, None] * sq + rk[None, :] * sk,
                    mask=ok, other=0.0)
        return (s.to(tl.float32) + v.to(tl.float32)).to(tl.bfloat16)

    @triton.jit
    def _fused_gated_attention(
        OUT, QX, KVX, WQ, WK, WV, WG, QB, B0, B1, B2, B3,
        H, Q, K, C, CQ, CK, HC, SQRT_C,
        s0b, s0h, s0q, s0k, s1b, s1h, s1q, s1k,
        s2b, s2h, s2q, s2k, s3b, s3h, s3q, s3k,
        QP: tl.constexpr, KP: tl.constexpr, CP: tl.constexpr, DP: tl.constexpr,
        NB: tl.constexpr, HAS_QB: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        rq = tl.arange(0, QP)
        rk = tl.arange(0, KP)
        rc = tl.arange(0, CP)
        q_ok = rq < Q
        k_ok = rk < K
        c_ok = rc < C

        # q_x and kv_x are contiguous, so their row strides are CQ and CK.
        qx_row = QX + (b * Q + rq[:, None]) * CQ
        kvx_row = KVX + (b * K + rk[:, None]) * CK
        # Head h reads only rows h*C .. (h+1)*C of each weight: a contiguous row
        # block, so summed over heads each matrix is read once per batch element.
        wq_row = WQ + (h * C + rc[:, None]) * CQ
        wk_row = WK + (h * C + rc[:, None]) * CK
        wv_row = WV + (h * C + rc[:, None]) * CK
        wg_row = WG + (h * C + rc[:, None]) * CQ

        acc = tl.zeros((QP, CP), dtype=tl.float32)
        for d0 in range(0, CQ, DP):
            rd = d0 + tl.arange(0, DP)
            d_ok = rd < CQ
            a = tl.load(qx_row + rd[None, :], mask=q_ok[:, None] & d_ok[None, :],
                        other=0.0)
            w = tl.load(wq_row + rd[None, :], mask=c_ok[:, None] & d_ok[None, :],
                        other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
        if HAS_QB:
            # F.linear folds the bias into the fp32 epilogue, before the bf16 round.
            acc += tl.load(QB + h * C + rc, mask=c_ok, other=0.0).to(tl.float32)[None, :]
        # Zero the padded C lanes rather than relying on their masked operands to
        # produce zero. A masked-to-zero weight gives 0 * x == 0 only for finite x; a
        # non-finite activation or weight would make 0 * x NaN and, because C is the
        # QK contraction axis, that NaN would reach every valid score. One select is
        # cheaper than the assumption that the caller's tensors are finite.
        acc = tl.where(c_ok[None, :], acc, 0.0)
        qh = (acc.to(tl.bfloat16).to(tl.float32) / SQRT_C).to(tl.bfloat16)

        acc = tl.zeros((KP, CP), dtype=tl.float32)
        for d0 in range(0, CK, DP):
            rd = d0 + tl.arange(0, DP)
            d_ok = rd < CK
            a = tl.load(kvx_row + rd[None, :], mask=k_ok[:, None] & d_ok[None, :],
                        other=0.0)
            w = tl.load(wk_row + rd[None, :], mask=c_ok[:, None] & d_ok[None, :],
                        other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
        acc = tl.where(c_ok[None, :], acc, 0.0)

        # Padded K rows of Kh need no such treatment: they only reach the padded score
        # columns, which tl.where replaces with -inf before the softmax.
        s = tl.dot(qh, tl.trans(acc.to(tl.bfloat16))).to(tl.bfloat16)
        ok = q_ok[:, None] & k_ok[None, :]
        if NB >= 1:
            s = _bias_add(s, B0, s0b, s0h, s0q, s0k, b, h, rq, rk, ok)
        if NB >= 2:
            s = _bias_add(s, B1, s1b, s1h, s1q, s1k, b, h, rq, rk, ok)
        if NB >= 3:
            s = _bias_add(s, B2, s2b, s2h, s2q, s2k, b, h, rq, rk, ok)
        if NB >= 4:
            s = _bias_add(s, B3, s3b, s3h, s3q, s3k, b, h, rq, rk, ok)

        f = tl.where(k_ok[None, :], s.to(tl.float32), float("-inf"))
        e = tl.exp(f - tl.max(f, 1)[:, None])
        p = (e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)

        acc = tl.zeros((KP, CP), dtype=tl.float32)
        for d0 in range(0, CK, DP):
            rd = d0 + tl.arange(0, DP)
            d_ok = rd < CK
            a = tl.load(kvx_row + rd[None, :], mask=k_ok[:, None] & d_ok[None, :],
                        other=0.0)
            w = tl.load(wv_row + rd[None, :], mask=c_ok[:, None] & d_ok[None, :],
                        other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
        # K is the PV contraction axis, so the same argument applies to Vh's padded
        # rows: the softmax makes those probabilities exactly zero, but 0 * NaN is NaN.
        acc = tl.where(k_ok[:, None], acc, 0.0)
        o = tl.dot(p, acc.to(tl.bfloat16)).to(tl.bfloat16)

        acc = tl.zeros((QP, CP), dtype=tl.float32)
        for d0 in range(0, CQ, DP):
            rd = d0 + tl.arange(0, DP)
            d_ok = rd < CQ
            a = tl.load(qx_row + rd[None, :], mask=q_ok[:, None] & d_ok[None, :],
                        other=0.0)
            w = tl.load(wg_row + rd[None, :], mask=c_ok[:, None] & d_ok[None, :],
                        other=0.0)
            acc = tl.dot(a, tl.trans(w), acc)
        g = acc.to(tl.bfloat16).to(tl.float32)
        g = 1.0 / (1.0 + tl.exp(-g))
        y = (o.to(tl.float32) * g.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)

        # Every (row, col) of OUT is written by exactly one (b, h) program, so the
        # torch.empty behind it leaves nothing uninitialised.
        tl.store(OUT + (b * Q + rq[:, None]) * HC + h * C + rc[None, :], y,
                 mask=q_ok[:, None] & c_ok[None, :])

    def launch(out, q_x, kv_x, weights, qb, specs, nb, nq, nk, h, c, cq, ck, cst):
        wq, wk, wv, wg = weights
        qp, kp, cp, dp, num_warps, num_stages = cst
        # Unused bias slots get a real pointer and zero strides; the constexpr NB
        # keeps the compiler from emitting their loads at all.
        b0, b1, b2, b3 = out, out, out, out
        z = (0, 0, 0, 0)
        p0, p1, p2, p3 = z, z, z, z
        n = len(specs)
        if n >= 1:
            b0, p0 = specs[0][0], specs[0][1]
        if n >= 2:
            b1, p1 = specs[1][0], specs[1][1]
        if n >= 3:
            b2, p2 = specs[2][0], specs[2][1]
        if n >= 4:
            b3, p3 = specs[3][0], specs[3][1]
        return _fused_gated_attention[(nb * h,)](
            out, q_x, kv_x, wq, wk, wv, wg,
            qb if qb is not None else out, b0, b1, b2, b3,
            h, nq, nk, c, cq, ck, h * c, math.sqrt(c),
            p0[0], p0[1], p0[2], p0[3], p1[0], p1[1], p1[2], p1[3],
            p2[0], p2[1], p2[2], p2[3], p3[0], p3[1], p3[2], p3[3],
            QP=qp, KP=kp, CP=cp, DP=dp, NB=n, HAS_QB=qb is not None,
            num_warps=num_warps, num_stages=num_stages)

    return launch


def _reject_if_spilled(key: tuple, kernel) -> None:
    """Inspect a geometry's compiled kernel once; withdraw the geometry if it spills.

    Runs on the first dispatch for a geometry and never again -- ``_VERIFIED`` is checked
    before the call, so the repeated path pays one set-membership test. Pure host work:
    ``n_spills`` is compiler metadata carried on the ``CompiledKernel`` the launch returns.

    The first call for a spilling geometry has already run that kernel. That is correct,
    only slow; every later call for the same geometry runs the reference body instead.
    """
    _VERIFIED.add(key)
    spills = getattr(kernel, "n_spills", 0) or 0
    if spills:
        _SPILL_REJECTED[key] = spills
        print(f"[candidate L2/alphafold3_of3_attention] geometry {key} compiles with "
              f"{spills} spilled store(s); delegating this geometry from now on",
              file=sys.stderr, flush=True)


def _init_kernel() -> None:
    """Resolve the kernel once, at import. Any failure degrades to the baseline."""
    global _KERNEL, _KERNEL_STATUS, _SM100_DEVICES
    if not torch.cuda.is_available():
        _KERNEL_STATUS = "disabled:no-cuda-device"
        return
    _SM100_DEVICES = frozenset(
        index for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index) == (10, 0))
    if not _SM100_DEVICES:
        _KERNEL_STATUS = "disabled:no-sm100-device"
        return
    try:
        _KERNEL = _build_kernel()
        _KERNEL_STATUS = "built"
    except Exception as exc:  # compiler, driver, or architecture rejected the kernel
        _KERNEL = None
        _KERNEL_STATUS = f"failed:{type(exc).__name__}: {exc}"
        # One line, once, at import. The bench worker forwards stderr to the
        # per-operator log, so a swallowed build failure cannot hide behind a silent
        # 1.00x. Not retried per call: a retry would pay the failure every forward.
        print(f"[candidate L2/alphafold3_of3_attention] kernel unavailable, "
              f"delegating to the reference body: {_KERNEL_STATUS}",
              file=sys.stderr, flush=True)


_init_kernel()


# ---------------------------------------------------------------------------
# The reference body, reproduced verbatim for everything the fast path does not
# claim. Not a re-derivation: an out-of-domain call must return exactly what it
# returns today, bit for bit, which an SDPA-based rewrite would not.
# ---------------------------------------------------------------------------
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Core SDPA: scores = softmax(Q K^T + biases) V.

    Args:
        query: [*, H, Q, C_hidden]
        key:   [*, H, K, C_hidden]
        value: [*, H, V, C_hidden]
        biases: list of tensors broadcastable to [*, H, Q, K]

    Returns:
        [*, H, Q, C_hidden]
    """
    scores = torch.einsum("...qc,...kc->...qk", query, key)

    for b in biases:
        scores = scores + b

    scores = F.softmax(scores, dim=-1)

    return torch.einsum("...qk,...kc->...qc", scores.to(dtype=value.dtype), value)


class OF3Attention(nn.Module):
    """Standard multi-head attention with gating and bias list support.

    Reference: openfold3/core/model/primitives/attention.py Attention

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        gating: Whether to gate output using query data
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        q_bias: bool = False,
    ):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(c_q, c_hidden * no_heads, bias=q_bias)
        self.linear_k = Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = Linear(c_hidden * no_heads, c_q, bias=False)

        self.linear_g = None
        if gating:
            self.linear_g = Linear(c_q, c_hidden * no_heads, bias=False)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = torch.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------
    def _plan(self, q_x: torch.Tensor, kv_x: torch.Tensor, biases: list):
        """Return the launch plan for this call, or None to run the reference body.

        Pure and cheap: integer, dtype and attribute comparisons only, no CUDA calls
        and no synchronization. Every clause guards something the kernel relies on,
        and the conjunction is of *sufficient* conditions -- anything it does not
        positively admit delegates.
        """
        if _KERNEL is None or torch.is_grad_enabled():
            return None          # the kernel builds no autograd graph
        linear_g = self.linear_g
        if linear_g is None:
            return None          # ungated is a two-line change, but unmeasured
        if q_x.dtype is not torch.bfloat16 or kv_x.dtype is not torch.bfloat16:
            return None
        if not q_x.is_cuda:
            return None
        device = q_x.device
        if device.index not in _SM100_DEVICES:
            return None      # the configuration table was measured on sm_100 only
        if kv_x.device != device:
            return None
        if not q_x.is_contiguous() or not kv_x.is_contiguous():
            return None          # a hidden .contiguous() would cost a whole launch
        rank = q_x.dim()
        if rank < 2 or kv_x.dim() != rank:
            return None
        batch = q_x.shape[:-2]
        if kv_x.shape[:-2] != batch:
            return None
        c_q, c_k = self.c_q, self.c_k
        if q_x.shape[-1] != c_q or kv_x.shape[-1] != c_k or self.c_v != c_k:
            return None          # V shares the kv_x tile with K
        c, h = self.c_hidden, self.no_heads
        if c <= 0 or c > _C_MAX or c % 8 or h <= 0:
            return None
        nq = q_x.shape[-2]
        nk = kv_x.shape[-2]
        if nq <= 0 or nq > _Q_MAX or nk <= 0 or nk > _K_MAX:
            return None          # K bounded: the score tile is a single block
        hc = h * c
        wq = self.linear_q.weight
        wk = self.linear_k.weight
        wv = self.linear_v.weight
        wg = linear_g.weight
        for w, c_in in ((wq, c_q), (wk, c_k), (wv, c_k), (wg, c_q)):
            if (w.dtype is not torch.bfloat16 or w.device != device or w.dim() != 2
                    or w.shape[0] != hc or w.shape[1] != c_in
                    or not w.is_contiguous()):
                return None      # row-block addressing assumes a dense [HC, c_in]
        qb = self.linear_q.bias
        if qb is not None and (qb.dtype is not torch.bfloat16 or qb.device != device
                               or qb.dim() != 1 or qb.shape[0] != hc
                               or not qb.is_contiguous()):
            return None
        if (self.linear_k.bias is not None or self.linear_v.bias is not None
                or linear_g.bias is not None):
            return None          # the kernel folds only linear_q's bias
        n_batch = 1
        for d in batch:
            n_batch *= d
        if n_batch <= 0:
            return None
        specs = self._bias_specs(biases, batch, n_batch, h, nq, nk, device)
        if specs is None:
            return None
        key = (nq, nk, c, c_q, c_k)
        if key in _SPILL_REJECTED:
            return None      # its kernel spills; see _reject_if_spilled
        return (key, _LAUNCH.get(key) or _launch_constants(key), (wq, wk, wv, wg),
                qb, specs, n_batch, nq, nk, h, c, c_q, c_k, hc, batch)

    @staticmethod
    def _bias_specs(biases: list, batch, n_batch: int, h: int, nq: int, nk: int,
                    device):
        """Reduce each bias to (tensor, (flat_batch, head, q, k) strides), or None.

        Every stride reaches the kernel as a runtime argument, so an arbitrary
        permuted or expanded view is read in place -- no bias is ever normalised by a
        copy, and no unit last stride is required.

        The kernel indexes the batch as one flat integer ``flat = sum_d i_d * acc_d``
        with ``acc_d = prod(target_batch[d+1:])``, so a bias is representable only if
        its per-axis batch strides collapse to one flat stride ``s``. A broadcast axis
        contributes nothing to the offset whatever ``stride()`` reports for it, so its
        *effective* stride is 0; the condition is then ``eff_d == acc_d * s`` for every
        batch axis whose **target** size exceeds 1. Quantifying over axes whose *bias*
        size exceeds 1 instead would wrongly admit a bias that broadcasts on a leading
        batch axis while a trailing one does not -- there ``eff_d == 0`` forces
        ``s == 0``, so no single flat stride exists and the call must delegate.
        """
        if not biases:
            return ()
        if len(biases) > _NB_MAX:
            return None
        target = tuple(batch) + (h, nq, nk)
        rank = len(target)
        n_dim = len(batch)
        specs = []
        for bias in biases:
            if not isinstance(bias, torch.Tensor):
                return None
            if bias.dtype is not torch.bfloat16 or bias.device != device:
                return None      # a wider dtype would change the rounding order
            pad = rank - bias.dim()
            if pad < 0:
                return None
            eff = []
            for axis in range(rank):
                if axis < pad:
                    eff.append(0)
                    continue
                size = bias.shape[axis - pad]
                if size == 1:
                    eff.append(0)
                elif size == target[axis]:
                    eff.append(bias.stride(axis - pad))
                else:
                    return None
            flat = 0
            last = -1
            for axis in range(n_dim):
                if target[axis] > 1:
                    last = axis
            if last >= 0:
                flat = eff[last]
                acc = 1
                for axis in range(n_dim - 1, -1, -1):
                    if target[axis] > 1 and eff[axis] != acc * flat:
                        return None
                    acc *= target[axis]
            specs.append((bias, (flat, eff[n_dim], eff[n_dim + 1], eff[n_dim + 2])))
        return tuple(specs)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:  [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]

        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        plan = self._plan(q_x, kv_x, biases)
        if plan is None:
            q, k, v = self._prep_qkv(q_x, kv_x)

            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

            return self._wrap_up(o, q_x)

        (key, cst, weights, qb, specs, n_batch, nq, nk, h, c, c_q, c_k, hc,
         batch) = plan
        _FASTPATH_HITS[key] = _FASTPATH_HITS.get(key, 0) + 1
        o_flat = torch.empty((n_batch * nq, hc), device=q_x.device, dtype=q_x.dtype)
        # No try/except around the launch: past the predicate a launch error is a bug
        # that must surface, not silently become the reference answer.
        kernel = _KERNEL(o_flat, q_x, kv_x, weights, qb, specs, n_batch, nq, nk, h, c,
                         c_q, c_k, cst)
        if key not in _VERIFIED:
            _reject_if_spilled(key, kernel)
        return self.linear_o(o_flat).view(*batch, nq, -1)
