"""Qwen3 MoE decoder layer: QK-norm attention + MoE with RMSNorm residual connections.

Three things are owned here rather than delegated: the MoE data plane, the layer's
host dispatch at small token counts, and the M-RoPE half of the post-QKV glue.

**Host dispatch (the small-token path).**  At one token this layer's wall was 97%
CPU: 665 us of Python and driver calls inside ``forward`` over 115 us of device
work, measured three ways in one process.  None of that host time depends on a
device value -- the token count fixes every tile shape, buffer size and branch on
the path -- so the whole forward is captured into a CUDA graph once per
``(token count, residual is None)`` key and replayed for 2.1 us
(``_GraphKey``).  Above a few tokens the graph buys nothing measurable and is
*wrong* (FA4's CuTe launcher does not enter the capture), so there are two gates:
a small ``_GRAPH_MAX_M``, and ``_graph_validate``, which replays every fresh
capture once and compares it against eager before trusting it.

**The MoE data plane.**  The reference spends most of its wall on plumbing, not
arithmetic: it rebuilds routing metadata from scatter_add / argsort / searchsorted
every call, zero-fills a padded ``[M_sum, K]`` activation copy, materializes the
permuted activation and then a ``[M, top_k, K]`` gather plus a same-sized product
before reducing.  At M=1000 that is ~700 us of the 1530 us layer, against 490 us
in the two expert GEMMs (which are streaming 2.4 GB of FP8 expert weights and
therefore already close to HBM bound).

What runs here instead:

  router/top-k  ->  MoeAlign (L1 winner: one CUDA kernel for all routing
                    metadata)  ->  GEMM1 gathers its A rows straight out of the
                    unpermuted activation through ``sorted_token_ids`` and writes
                    the SiLU-mul'd, re-quantized FP8 intermediate to the *flat
                    routed row* ``token * top_k + slot``  ->  GEMM2 scatters its
                    bf16 partials back to that same flat row  ->  one weighted
                    reduction over the top_k slots of each token.

So no activation permutation buffer, no zero-fill, no argsort, no gather/product
temporaries, and every buffer is sized from a host-constant worst-case bound
(``M * top_k + E * (align - 1)``) with the tile loop driven by MoeAlign's
device-side ``num_tokens_post_padded`` -- no device-to-host copy anywhere on the
path.

**The glue.**  The reference's post-QKV sequence is six launches and five
allocations for ~3 us of arithmetic.  The rope half of it -- a
``cos_sin_cache[positions]`` gather that materializes ``(3, N, 128)``, two
``.contiguous()`` copies of its halves, and ``_mrope_kernel`` -- collapses into
one kernel that reads the three T/H/W table rows directly
(``_qk_rope_kernel``), verified bit-exact.  The norm half is written, measured
faster, and **disabled**: see ``_qk_norm_one``.

Numerics follow the reference chain step for step, which the FP8 tolerances make
mandatory (a mismatched quantization scale is a ~3% error on a K=4096 dot):
per-128-group UE8M0 (power-of-two) activation scales, GEMM1's output rounded to
bf16 before SiLU, SiLU rounded to bf16 before the bf16 multiply, and -- the one
that reads like a kernel bug when it is wrong -- the top_k weights applied where
the *reference's own* ``_valid_deep_gemm`` puts them: in fp32 at the reduction
above 128 tokens, folded into GEMM2's epilogue below it.

How strict "step for step" has to be, measured here rather than assumed:
**31 flipped bf16 bits out of 8.2 million in q cost 15% of this layer's output.**
Two amplifiers in series -- the FP8 activation quantization in front of ``o_proj``,
then the router, where a perturbed logit reorders top-k and the token's whole
4096-wide row changes.  So a difference in fp32 *summation order* is not a small
error here, and anything upstream of q, k or the router must be bit-exact.  Check
that with a stage-level differing-count probe before an end-to-end bench, not
after.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from ...baseline.L1.moe_grouped_gemm import (
    _valid_deep_gemm as _ref_valid_deep_gemm,
)
from ...baseline.L1.rms_norm import RMSNorm as RefRMSNorm
from ..L1.fp8_linear import PerTokenGroupQuantFp8
from ..L1.moe_align import MoeAlign
from ..L1.rms_norm import RMSNorm
from ..L2.attention import _MROPE_CLASSES, LlamaAttention
from ..L2.qwen3_moe import Qwen3MoE

_GROUP = 128
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _iv(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


# Measured tile plans.  ``_PLAN_SMALL`` is for the mid-M shapes (M=314..1000,
# ~50-60 routed rows per expert, one row block per expert) and ``_PLAN_BIG`` for
# the prefill shape (M=16384, ~1024 rows per expert).  Weight bytes per flop go
# as 1/(RB * BLOCK_M), so what changes with M is how many row blocks one program
# folds together: at 62 rows per expert a second row block is empty and a
# 64-row tile with SKIP_EMPTY already reads each expert's weights once, while at
# 1024 rows RB=1 streams every expert's 12.6 MB w13 slice out of L2 sixteen
# times -- 26 GB, which is the wall.  Swept over BLOCK_M/RB/warps/stages/unroll
# for GEMM1 and BLOCK_M/BLOCK_N/warps/stages/KSTEP/GROUP_M for GEMM2 at both
# shapes (tools/sweep_all.py): GEMM2 at M=1000 went 370 -> 208 us on tile shape
# alone, and at M=16384 it wants the *wide* row tile instead (BLOCK_M=128 with
# GROUP_M=4 scheduling groups: 1857 us against 1911), which is the whole reason
# the plan is picked per rows-per-expert.
# RB > 1 measured *slower* at every shape (M=16384: 5292 us at RB=2 against 3888
# at RB=1, and worse still at 4 warps): the four accumulators plus the address
# math spill, and the spill traffic costs more than the halved weight traffic
# buys.  So both plans keep RB=1 and the code path stays for the next round to
# retry against a 2-CTA cluster, which is how DeepGEMM gets the same traffic
# reduction without the registers.
#   key: align, (bm1, rb1, warps1, stages1, unroll1),
#             (bm2, bn2, warps2, stages2, kstep2, group_m2)
_PLAN_SMALL = (128, (64, 1, 4, 2, 1), (64, 128, 4, 3, 2, 1))
_PLAN_BIG = (128, (64, 1, 4, 2, 1), (128, 128, 8, 3, 2, 4))
# Rows per expert at which the big plan takes over (the crossover is broad: the
# big plan needs several row blocks per expert to pay for its wider tiles).
_BIG_ROWS_PER_EXPERT = _iv("FK_L3_BIGRPE", 256)
_BN_RED = _iv("FK_L3_BNRED", 2048)  # reduce columns per program (swept)
# Smallest token count the fused path takes.  The *placement* reason for r1's 128
# is gone -- GEMM2's epilogue now switches on the reference's own
# ``_valid_deep_gemm`` (see ``_gemm2_kernel``), and at one token the fused path
# is bit-exact against the reference block, matched 1.0000 max_abs 0 -- but the
# *speed* reason turned out to be real and is measured: at one token the fused
# path costs 77 us of device against the reference block's 57.  ``_gemm1_kernel``
# alone is 52 us there, because one program owns (expert, 128-wide N group) and
# folds gate and up together, so eight fed experts x twelve groups is only 96
# programs each streaming 1 MB of w13 -- 0.65 of a wave on 148 SMs, 1.9 TB/s.
# The reference's Triton kernel splits gate and up into separate column blocks
# and gets 192 programs of 512 KB for the same bytes.  Beating it at one token
# needs split-K (or gate/up in separate programs plus the reference's extra
# intermediate buffer), which is a next-round question; the gate stays at 128
# and the epilogue switch stays, correct and dormant, for whoever does it.
_MIN_M = _iv("FK_L3_MINM", 128)
_FUSE_NORM = _iv("FK_L3_FUSENORM", 1)
# Programmatic dependent launch across the chain: each kernel waits at entry
# before it touches its producer's output and signals after its stores, so the
# next kernel's blocks are resident by the time the producer's tail drains.
_PDL = bool(_iv("FK_L3_PDL", 1))
# 1 = bypass the frozen winners' M==1 fused-GEMV specializations (see
# ``_pin_small_m_paths``).
_PIN_M1 = _iv("FK_L3_PINM1", 1)
_USE_FUSED = _iv("FK_L3_MOE", 1)   # 0 disables the fused path (A/B only)
# Whole-layer CUDA graph, keyed per (token count, residual is None).  Measured:
# at one token the layer's wall is 97% host (665 us of CPU inside ``forward``
# over 115 us of device work), so replacing the enqueue with a graph replay is
# worth ~4x there.  It is worth *nothing* above a few tokens -- at 870 and 1000
# tokens the graphed wall measured 1.3% better than eager, i.e. inside the
# noise, and at 16384 tokens it is worse (a 268 MB input copy against an 11.9 ms
# wall) -- and above one token the attention path picks up FA4's CuTe launcher,
# whose graphed replay measured *wrong* (matched 0.20 at 1000 tokens, and an
# illegal access at 870): something on that path reaches the driver without
# being recorded.  Hence a small ``_GRAPH_MAX_M`` -- and, because "capture
# succeeded" is not evidence that "replay is correct", every capture is checked
# against eager before it is used (``_graph_validate``).
_GRAPH = bool(_iv("FK_L3_GRAPH", 1))
_GRAPH_MAX_M = _iv("FK_L3_GRAPHMAXM", 64)
_GRAPH_WARM = _iv("FK_L3_GRAPHWARM", 2)
# Fused per-head QK-norm + M-RoPE in place of the reference's six launches.
_GLUE = bool(_iv("FK_L3_GLUE", 1))
_GLUE_WARPS = _iv("FK_L3_GLUEWARPS", 8)
# Off by measurement, not by caution.  ``_qk_norm_kernel`` is a real win -- it
# deletes the two ``.contiguous()`` copies of the strided q/k views and both norm
# launches, and measured 1.25x -> 1.35x at 16384 tokens, 1.99x -> 2.16x at 1000
# and 2.13x -> 2.33x at 870 -- and it cannot be used, because its fp32 variance
# does not match the vendored kernel's *bit for bit* and this layer does not
# tolerate that.  See ``_qk_norm_one`` for the numbers.  The rope half of the
# glue, which is bit-exact, stays on.
_GLUE_NORM = bool(_iv("FK_L3_GLUENORM", 0))
# ``gcd(16 / sizeof(dtype), head_dim)`` -- the vendored launcher's vector width,
# which fixes the reduction shape the norm has to reproduce.
_GLUE_VEC = _iv("FK_L3_GLUEVEC", 8)
_GLUE_FMA = _iv("FK_L3_GLUEFMA", 1)


def _never_eligible(*args, **kwargs) -> bool:
    return False


def _plan_for(rows: int, num_experts: int):
    if _iv("FK_L3_FORCE_BIG", 0) or rows >= _BIG_ROWS_PER_EXPERT * num_experts:
        return _PLAN_BIG
    return _PLAN_SMALL


# ---------------------------------------------------------------------------
# Fused residual-add RMSNorm + activation quantization (MoE prologue)
# ---------------------------------------------------------------------------
@triton.jit
def _norm_quant_kernel(x_ptr, r_ptr, w_ptr, h_ptr, q_ptr, s_ptr,
                       stride_x, stride_r, stride_h, stride_q, stride_s,
                       eps,
                       H: tl.constexpr, NUM_G: tl.constexpr, GROUP: tl.constexpr,
                       FP8_MAX: tl.constexpr):
    """``residual += x; h = rmsnorm(residual) * w; (q, s) = quant(h)`` in one pass.

    The layer's second norm boundary and the MoE's activation quantization are
    two separate passes over ``[tokens, 4096]`` in the reference: the norm writes
    bf16 ``h``, the quantizer reads it straight back, and the router reads it a
    third time.  Here the row is read once and ``h`` is written once (the router
    still needs it in bf16), so the quantizer's read disappears -- 134 MB at
    M=16384 -- along with one launch, which is what the M=1 shapes feel.

    The arithmetic is the vendored reference's, step for step, because the FP8
    quantization downstream turns a last-bit difference here into a ~2% error:
    the residual add is a *bf16* add, the variance is fp32 with ``rsqrtf``, and
    the weight multiply happens in fp32 with a *single* rounding of the whole
    product (``vllm::fused_add_rms_norm_kernel``'s
    ``Converter::convert(x * s_variance * wf)``).  Rounding the normalized value
    to bf16 first instead -- which is what the *other* vendored variant in that
    same file does (``F16Vec::operator*=(float)`` then ``operator*=(F16Vec)``) --
    changes 22.8% of h by one ULP, and the FP8 quantizer downstream turns that
    into 1.5% of the quantized bytes and a ~1.5% error on the GEMM: measured, a
    hard fail.  Only the fp32 reduction *order* differs from the reference here,
    which moves ~1e-7 and flips ~2e-5 of the bf16 roundings.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    x = tl.load(x_ptr + row * stride_x + cols)
    r = tl.load(r_ptr + row * stride_r + cols) + x
    tl.store(r_ptr + row * stride_r + cols, r)

    rf = r.to(tl.float32)
    rstd = tl.math.rsqrt(tl.sum(rf * rf, axis=0) / H + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    hb = (rf * rstd * w).to(tl.bfloat16)
    tl.store(h_ptr + row * stride_h + cols, hb)

    hg = tl.reshape(hb.to(tl.float32), (NUM_G, GROUP))
    absmax = tl.maximum(tl.max(tl.abs(hg), axis=1), 1e-10)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax * (1.0 / FP8_MAX))))
    q = tl.clamp(hg / scale[:, None], -FP8_MAX, FP8_MAX)
    tl.store(q_ptr + row * stride_q + cols,
             tl.reshape(q, (H,)).to(q_ptr.dtype.element_ty))
    tl.store(s_ptr + row * stride_s + tl.arange(0, NUM_G), scale)


# ---------------------------------------------------------------------------
# GEMM1: gate/up projection + SiLU-mul + re-quantization, one pass
# ---------------------------------------------------------------------------
@triton.jit
def _gemm1_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, o_ptr, os_ptr,
    sorted_ptr, eid_ptr, ntpp_ptr,
    n_valid,
    stride_am, stride_ask,
    stride_be, stride_bn,
    stride_bse, stride_bsn, stride_bsk,
    stride_om, stride_osm,
    NG: tl.constexpr, N_HALF: tl.constexpr, TOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr, ALIGN: tl.constexpr, SKIP_EMPTY: tl.constexpr,
    UNROLL: tl.constexpr, RB: tl.constexpr, INT64: tl.constexpr,
    FP8_MAX: tl.constexpr, PDL: tl.constexpr,
):
    """One program: ``RB`` x ``BLOCK_M`` routed rows x one 128-wide intermediate group.

    The gate and up halves of ``w13`` share the A tile, so two accumulators double
    the MMA issued per A byte moved.  Both ``tl.dot`` results are named and kept
    live across the block-scale promote: written as ``acc += dot(a, b) * s``
    Triton gives the dot a single tensor-memory buffer and the next MMA cannot
    issue until that buffer has been read back and scaled, which puts the promote
    on the critical path.

    ``RB > 1`` walks that many consecutive row blocks against *one* load of the
    two weight tiles.  Weight bytes per flop go as 1/(RB * BLOCK_M) and at 16384
    tokens each expert owns 16 64-row blocks, so RB=1 streams its 12.6 MB w13
    slice out of L2 sixteen times -- 26 GB of L2 traffic, which is the wall there.
    Two row blocks per program halve that at 4 accumulators and still 2 live dot
    results, where BLOCK_M=128 (same traffic) needs 2 accumulators *and* 2
    full-width temporaries and spills past 255 registers.

    The epilogue reproduces the reference chain exactly -- GEMM output rounded to
    bf16, SiLU evaluated in fp32 and rounded back to bf16, a bf16 product, then
    the per-128-group UE8M0 quantization of that bf16 value -- and writes the FP8
    intermediate to the flat routed row, so nothing is permuted.
    """
    pid = tl.program_id(0)
    pid_n = pid % NG
    pid_g = pid // NG
    blk0 = pid_g * RB
    if PDL:
        gdc_wait()
    ntpp = tl.load(ntpp_ptr)
    if blk0 * BLOCK_M >= ntpp:
        return

    offs_m = tl.arange(0, BLOCK_M)
    pos0 = blk0 * BLOCK_M + offs_m
    offs_t0 = tl.load(sorted_ptr + pos0, mask=pos0 < ntpp, other=n_valid)
    mask0 = offs_t0 < n_valid
    if SKIP_EMPTY:
        if RB == 1:
            if tl.min(offs_t0) >= n_valid:
                return
    offs_t0 = tl.where(mask0, offs_t0, 0)
    if INT64:
        offs_t0 = offs_t0.to(tl.int64)
    if RB > 1:
        pos1 = pos0 + BLOCK_M
        offs_t1 = tl.load(sorted_ptr + pos1, mask=pos1 < ntpp, other=n_valid)
        mask1 = offs_t1 < n_valid
        offs_t1 = tl.where(mask1, offs_t1, 0)
        if INT64:
            offs_t1 = offs_t1.to(tl.int64)

    if ALIGN == BLOCK_M:
        off_e = tl.load(eid_ptr + blk0)
    else:
        off_e = tl.load(eid_ptr + blk0 // (ALIGN // BLOCK_M))

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row0 = offs_t0 // TOP_K

    a0_ptrs = a_ptr + row0[:, None] * stride_am + offs_k[None, :]
    bg_ptrs = b_ptr + off_e * stride_be + offs_k[:, None] + offs_bn[None, :] * stride_bn
    bu_ptrs = bg_ptrs + N_HALF * stride_bn
    as0_ptrs = as_ptr + row0 * stride_ask
    bsg_ptr = bs_ptr + off_e * stride_bse + pid_n * stride_bsn
    bsu_ptr = bsg_ptr + NG * stride_bsn

    acc_g0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if RB > 1:
        row1 = offs_t1 // TOP_K
        a1_ptrs = a_ptr + row1[:, None] * stride_am + offs_k[None, :]
        as1_ptrs = as_ptr + row1 * stride_ask
        acc_g1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, NUM_K, disallow_acc_multi_buffer=True,
                      loop_unroll_factor=UNROLL):
        bg = tl.load(bg_ptrs)
        bu = tl.load(bu_ptrs)
        sg = tl.load(bsg_ptr + k * stride_bsk)
        su = tl.load(bsu_ptr + k * stride_bsk)
        a0 = tl.load(a0_ptrs, mask=mask0[:, None], other=0.0)
        as0 = tl.load(as0_ptrs + k, mask=mask0, other=0.0)
        tg0 = tl.dot(a0, bg)
        tu0 = tl.dot(a0, bu)
        acc_g0 += tg0 * (as0 * sg)[:, None]
        acc_u0 += tu0 * (as0 * su)[:, None]
        if RB > 1:
            a1 = tl.load(a1_ptrs, mask=mask1[:, None], other=0.0)
            as1 = tl.load(as1_ptrs + k, mask=mask1, other=0.0)
            tg1 = tl.dot(a1, bg)
            tu1 = tl.dot(a1, bu)
            acc_g1 += tg1 * (as1 * sg)[:, None]
            acc_u1 += tu1 * (as1 * su)[:, None]
            a1_ptrs += BLOCK_K
        a0_ptrs += BLOCK_K
        bg_ptrs += BLOCK_K
        bu_ptrs += BLOCK_K

    _gemm1_epilogue(acc_g0, acc_u0, offs_t0, mask0, o_ptr, os_ptr, stride_om,
                    stride_osm, offs_bn, pid_n, FP8_MAX)
    if RB > 1:
        _gemm1_epilogue(acc_g1, acc_u1, offs_t1, mask1, o_ptr, os_ptr, stride_om,
                        stride_osm, offs_bn, pid_n, FP8_MAX)
    if PDL:
        gdc_launch_dependents()


@triton.jit
def _gemm1_epilogue(acc_g, acc_u, offs_token, token_mask, o_ptr, os_ptr,
                    stride_om, stride_osm, offs_bn, pid_n,
                    FP8_MAX: tl.constexpr):
    """SiLU-mul then per-128-group UE8M0 quantization, rounding exactly where the
    reference rounds: bf16 GEMM output, bf16 SiLU, bf16 product, then quantize."""
    gate = acc_g.to(tl.bfloat16).to(tl.float32)
    up = acc_u.to(tl.bfloat16).to(tl.float32)
    silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    h = (silu * up).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(h), axis=1), 1e-10)
    scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax * (1.0 / FP8_MAX))))
    q = tl.clamp(h / scale[:, None], -FP8_MAX, FP8_MAX)
    tl.store(o_ptr + offs_token[:, None] * stride_om + offs_bn[None, :],
             q.to(o_ptr.dtype.element_ty), mask=token_mask[:, None])
    tl.store(os_ptr + offs_token * stride_osm + pid_n, scale, mask=token_mask)


# ---------------------------------------------------------------------------
# GEMM2: down projection, partials scattered back to the flat routed row
# ---------------------------------------------------------------------------
@triton.jit
def _gemm2_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, tw_ptr,
    sorted_ptr, eid_ptr, ntpp_ptr,
    n_valid,
    stride_am, stride_ask,
    stride_be, stride_bn,
    stride_bse, stride_bsn, stride_bsk,
    stride_cm,
    NUM_M: tl.constexpr, NUM_N: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_K: tl.constexpr, KSTEP: tl.constexpr, GROUP_N: tl.constexpr,
    ALIGN: tl.constexpr, SKIP_EMPTY: tl.constexpr, INT64: tl.constexpr,
    MUL_W: tl.constexpr, PDL: tl.constexpr,
):
    """``BLOCK_M`` routed rows x ``BLOCK_N`` hidden columns of the down projection.

    ``KSTEP`` dot results are named and kept live before their block-scale
    promotes, so the MMAs pipeline instead of each one waiting for the previous
    result to be read back out of tensor memory and scaled.

    ``GROUP_M`` row blocks x every N tile form one scheduling group, so what is
    resident is a *rectangle*: the group's A tiles (shared down the N direction)
    and one expert's ``w2`` window (shared across the row blocks of that expert,
    which is what a large M needs -- at 16384 tokens each expert owns 8 row
    blocks and its 6.3 MB slice would otherwise be streamed once per block).

    Where the top_k weight is applied follows the reference, which puts it in two
    different places depending on which expert path *its own* gate picks.  At and
    above 128 tokens ``_valid_deep_gemm`` holds, it takes DeepGEMM, and the
    weights are applied in fp32 at the unpermute-reduce over unweighted bf16
    partials (``MUL_W=False`` here).  Below that it falls back to its Triton
    grouped GEMM with ``mul_routed_weight=True, top_k=1``, i.e. the weight
    multiplies the fp32 accumulator *before* the bf16 store and the reduce is a
    plain sum (``MUL_W=True``).  Matching the wrong one costs a matched ratio
    around 0.86 and reads exactly like a kernel bug, so the switch is driven off
    the reference's own predicate rather than off a threshold constant.
    """
    pid = tl.program_id(0)
    if PDL:
        gdc_wait()
    if GROUP_M == 1:
        pid_m = pid % NUM_M
        pid_n = pid // NUM_M
    else:
        g = GROUP_M * NUM_N
        gid = pid // g
        rem = pid % g
        m0 = gid * GROUP_M
        gm = min(NUM_M - m0, GROUP_M)
        pid_m = m0 + rem % gm
        pid_n = rem // gm

    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return

    offs_token = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    token_mask = offs_token < n_valid
    if SKIP_EMPTY:
        if tl.min(offs_token) >= n_valid:
            return
    offs_token = tl.where(token_mask, offs_token, 0)
    if INT64:
        offs_token = offs_token.to(tl.int64)

    if ALIGN == BLOCK_M:
        off_e = tl.load(eid_ptr + pid_m)
    else:
        off_e = tl.load(eid_ptr + pid_m // (ALIGN // BLOCK_M))

    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + off_e * stride_be + offs_k[:, None] + offs_bn[None, :] * stride_bn
    as_ptrs = as_ptr + offs_token * stride_ask
    bs_base = bs_ptr + off_e * stride_bse + (pid_n * BLOCK_N // GROUP_N) * stride_bsn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, NUM_K, KSTEP, disallow_acc_multi_buffer=True):
        if KSTEP == 1:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            s0 = tl.load(as_ptrs + k, mask=token_mask, other=0.0) * tl.load(
                bs_base + k * stride_bsk)
            acc += tl.dot(a, b) * s0[:, None]
        elif KSTEP == 2:
            a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            a1 = tl.load(a_ptrs + BLOCK_K, mask=token_mask[:, None], other=0.0)
            b0 = tl.load(b_ptrs)
            b1 = tl.load(b_ptrs + BLOCK_K)
            s0 = tl.load(as_ptrs + k, mask=token_mask, other=0.0) * tl.load(
                bs_base + k * stride_bsk)
            s1 = tl.load(as_ptrs + k + 1, mask=token_mask, other=0.0) * tl.load(
                bs_base + (k + 1) * stride_bsk)
            t0 = tl.dot(a0, b0)
            t1 = tl.dot(a1, b1)
            acc += t0 * s0[:, None]
            acc += t1 * s1[:, None]
        else:
            a0 = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            a1 = tl.load(a_ptrs + BLOCK_K, mask=token_mask[:, None], other=0.0)
            a2 = tl.load(a_ptrs + 2 * BLOCK_K, mask=token_mask[:, None], other=0.0)
            b0 = tl.load(b_ptrs)
            b1 = tl.load(b_ptrs + BLOCK_K)
            b2 = tl.load(b_ptrs + 2 * BLOCK_K)
            s0 = tl.load(as_ptrs + k, mask=token_mask, other=0.0) * tl.load(
                bs_base + k * stride_bsk)
            s1 = tl.load(as_ptrs + k + 1, mask=token_mask, other=0.0) * tl.load(
                bs_base + (k + 1) * stride_bsk)
            s2 = tl.load(as_ptrs + k + 2, mask=token_mask, other=0.0) * tl.load(
                bs_base + (k + 2) * stride_bsk)
            t0 = tl.dot(a0, b0)
            t1 = tl.dot(a1, b1)
            t2 = tl.dot(a2, b2)
            acc += t0 * s0[:, None]
            acc += t1 * s1[:, None]
            acc += t2 * s2[:, None]
        a_ptrs += KSTEP * BLOCK_K
        b_ptrs += KSTEP * BLOCK_K

    if MUL_W:
        w = tl.load(tw_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]
    tl.store(c_ptr + offs_token[:, None] * stride_cm + offs_bn[None, :],
             acc.to(c_ptr.dtype.element_ty), mask=token_mask[:, None])
    if PDL:
        gdc_launch_dependents()


# ---------------------------------------------------------------------------
# Weighted reduction over the top_k slots of each token
# ---------------------------------------------------------------------------
@triton.jit
def _reduce_kernel(p_ptr, w_ptr, o_ptr, M, stride_pm, stride_om,
                   TOP_K: tl.constexpr, BLOCK_N: tl.constexpr,
                   NUM_N: tl.constexpr, WEIGHTED: tl.constexpr,
                   PDL: tl.constexpr):
    """``out[m] = sum_j [topk_weight[m, j] *] partial[m * TOP_K + j]`` in fp32.

    The partials sit at consecutive flat routed rows, so the unpermute is just
    the row arithmetic here -- ``[M, top_k, K]`` is never materialized.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if PDL:
        gdc_wait()
    if pid_m >= M:
        return
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    p_base = p_ptr + pid_m * TOP_K * stride_pm + offs
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        v = tl.load(p_base + j * stride_pm).to(tl.float32)
        if WEIGHTED:
            acc += v * tl.load(w_ptr + pid_m * TOP_K + j)
        else:
            # GEMM2's epilogue already applied the weight (the reference's
            # sub-128 path does the same), so this is the plain ``moe_sum``.
            acc += v
    tl.store(o_ptr + pid_m * stride_om + offs, acc.to(o_ptr.dtype.element_ty))


class _Buffers:
    """Scratch shared by every layer (layers run sequentially)."""

    __slots__ = ("h_bf16", "a_fp8", "a_scale", "h_fp8", "h_scale", "part", "out")

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def get(self, name, numel, dtype, device):
        buf = getattr(self, name)
        if buf is None or buf.numel() < numel or buf.dtype != dtype or buf.device != device:
            buf = torch.empty(numel, dtype=dtype, device=device)
            setattr(self, name, buf)
        return buf[:numel]


_BUF = _Buffers()


class FusedMoE(Qwen3MoE):
    """``Qwen3MoE`` with the expert data plane replaced (same parameters)."""

    def __init__(self, config, quant_config: dict | None = None):
        super().__init__(config, quant_config=quant_config)
        self.moe_align = MoeAlign()
        self.act_quant = PerTokenGroupQuantFp8()
        self._buf = _BUF

    # -- fused path -------------------------------------------------------
    def _ok(self) -> bool:
        return (self.use_fp8
                and self.block_shape is not None
                and list(self.block_shape) == [_GROUP, _GROUP]
                and self.w13.dtype == torch.float8_e4m3fn
                and self.w2.dtype == torch.float8_e4m3fn
                and self.hidden_size % _GROUP == 0
                and self.intermediate_per_tp % _GROUP == 0)

    def eligible(self, h: torch.Tensor) -> bool:
        return bool(_USE_FUSED and self._ok() and h.is_contiguous()
                    and h.dtype == torch.bfloat16 and h.size(0) >= _MIN_M
                    and h.dim() == 2 and h.size(1) == self.hidden_size)

    def forward_impl(self, hidden_states: torch.Tensor,
                     quant: tuple | None = None) -> torch.Tensor:
        orig_shape = hidden_states.shape
        h = hidden_states.view(-1, self.hidden_size)
        M = h.size(0)
        if not self.eligible(h):
            return super().forward_impl(hidden_states)

        router_logits = self.gate(h)
        topk_weights, topk_ids = self.topk_softmax(
            router_logits, self.top_k, renormalize=self.renormalize)

        out = self._experts(h, topk_weights, topk_ids, M, quant)
        if self.tp_size > 1 and not self._use_custom_op:
            out = self.allreduce(out)
        return out.view(orig_shape)

    def _experts(self, h, topk_weights, topk_ids, M, quant=None):
        device = h.device
        K = self.hidden_size
        N = self.intermediate_per_tp
        E = self.num_experts
        top_k = self.top_k
        rows = M * top_k
        ng = N // _GROUP
        nk = K // _GROUP
        buf = self._buf
        align, (bm1, rb1, nw1, ns1, u1), (bm2, bn2, nw2, ns2, ks2, gm2) = _plan_for(rows, E)
        # Exactly the gate ``FusedExperts.forward`` reads, so the top_k weight
        # lands where the reference put it for this shape.  Not the capturing
        # check the reference also applies: the *baseline* this is scored against
        # is never captured, so its placement is the DeepGEMM one whenever the
        # shape allows it.
        deep_ok = bool(_ref_valid_deep_gemm(h, self.w13, self.w2))

        # Activation quantization: the L1 winner's CUDA kernel (5 us cheaper than
        # the equivalent Triton kernel at M=1000 and bit-identical to it).  Only
        # reached when the fused norm+quant prologue declined the shape -- it
        # writes these two buffers itself.
        if quant is not None:
            a_fp8, a_scale = quant
        else:
            a_fp8 = buf.get("a_fp8", M * K, torch.float8_e4m3fn, device).view(M, K)
            a_scale = buf.get("a_scale", M * nk, torch.float32, device).view(M, nk)
            self.act_quant(h, a_fp8, a_scale)

        # All routing metadata in one CUDA kernel (L1 winner), device-side only.
        sorted_ids, expert_ids, ntpp = self.moe_align(topk_ids, align, E)
        EM = sorted_ids.size(0)

        h_fp8 = buf.get("h_fp8", rows * N, torch.float8_e4m3fn, device).view(rows, N)
        h_scale = buf.get("h_scale", rows * ng, torch.float32, device).view(rows, ng)
        int64_addr = max(rows * N, rows * K, E * 2 * N * K) > 2 ** 31 - 1

        bm1 = min(bm1, align)
        rb1 = min(rb1, align // bm1)
        _gemm1_kernel[(triton.cdiv(EM, bm1 * rb1) * ng,)](
            a_fp8, a_scale, self.w13, self.w13_scale, h_fp8, h_scale,
            sorted_ids, expert_ids, ntpp,
            rows,
            a_fp8.stride(0), a_scale.stride(0),
            self.w13.stride(0), self.w13.stride(1),
            self.w13_scale.stride(0), self.w13_scale.stride(1), self.w13_scale.stride(2),
            h_fp8.stride(0), h_scale.stride(0),
            NG=ng, N_HALF=N, TOP_K=top_k,
            BLOCK_M=bm1, BLOCK_N=_GROUP, BLOCK_K=_GROUP, NUM_K=nk,
            ALIGN=align, SKIP_EMPTY=bm1 < align, UNROLL=u1, RB=rb1,
            INT64=int64_addr, FP8_MAX=_FP8_MAX, PDL=_PDL,
            num_warps=nw1, num_stages=ns1, launch_pdl=_PDL,
        )

        part = buf.get("part", rows * K, h.dtype, device).view(rows, K)
        bm2 = min(bm2, align)
        num_m = triton.cdiv(EM, bm2)
        num_n = K // bn2
        _gemm2_kernel[(num_m * num_n,)](
            h_fp8, h_scale, self.w2, self.w2_scale, part, topk_weights,
            sorted_ids, expert_ids, ntpp,
            rows,
            h_fp8.stride(0), h_scale.stride(0),
            self.w2.stride(0), self.w2.stride(1),
            self.w2_scale.stride(0), self.w2_scale.stride(1), self.w2_scale.stride(2),
            part.stride(0),
            NUM_M=num_m, NUM_N=num_n, GROUP_M=gm2,
            BLOCK_M=bm2, BLOCK_N=bn2, BLOCK_K=_GROUP, NUM_K=ng,
            KSTEP=ks2, GROUP_N=_GROUP, ALIGN=align, SKIP_EMPTY=bm2 < align,
            INT64=int64_addr, MUL_W=not deep_ok, PDL=_PDL,
            num_warps=nw2, num_stages=ns2, launch_pdl=_PDL,
        )

        out = buf.get("out", M * K, h.dtype, device).view(M, K)
        _reduce_kernel[(M, K // _BN_RED)](
            part, topk_weights, out, M, part.stride(0), out.stride(0),
            TOP_K=top_k, BLOCK_N=_BN_RED, NUM_N=K // _BN_RED,
            WEIGHTED=deep_ok, PDL=_PDL,
            num_warps=8, launch_pdl=_PDL,
        )
        return out


# ---------------------------------------------------------------------------
# QK-norm + M-RoPE glue: six launches down to two
# ---------------------------------------------------------------------------
# The reference spends six launches and five allocations on the post-QKV glue
# per layer: two ``.contiguous()`` copies of the strided q/k views (free only at
# one token, 346 us at 16384), two ``rms_norm_kernel``s, a
# ``cos_sin_cache[positions]`` gather that materializes ``(3, N, 128)``, two more
# ``.contiguous()`` copies to make the cos/sin chunks dense, and
# ``_mrope_kernel``.  Every intermediate is consumed exactly once and the only
# value the sequence needs per token is 64 cos/sin pairs, so both copies and the
# gather are pure overhead: 16.4 us of a 114 us layer at one token, ~130 us at
# 1000 and ~1.1 ms at 16384.
#
# What ships is the *rope* half: the gather and both cos/sin copies go away and
# ``_mrope_kernel``'s arithmetic is reproduced verbatim, verified bit-exact at
# every scored shape.  The norm half is written, measured, faster, and disabled;
# see ``_qk_norm_one``.
#
# Two kernels and not one, for a reason worth recording.  Folding the norm into
# the rotation is ~2 us better at one token and it does not survive: the norm's
# bf16 rounding is a fact of the program only because the reference passes the
# value through bf16 memory, and in registers it goes missing.  Measured at 870
# tokens -- the norm alone differs on 21 of 7.1M elements, the rotation alone is
# bit-exact, and the naive fusion of the two moves **10% of q and k by a full
# bf16 ULP**, the same signature as doing the rotation in fp32 on purpose
# (13.6%).  Neither ``.to(bf16).to(uint16, bitcast=True)`` (the bitcast pair
# cancels) nor an explicit integer round-to-nearest-even on the fp32 bit pattern
# changed one element -- all three variants produced byte-identical output -- so
# whatever elides the rounding sits upstream of how it is written.  Keeping the
# boundary at memory is what makes the rope half provably exact.
@triton.jit
def _qk_norm_kernel(qkv_ptr, qw_ptr, kw_ptr, q_ptr, k_ptr,
                    stride_qkv, stride_q, stride_k, eps,
                    NH: tl.constexpr, NKV: tl.constexpr,
                    PNH: tl.constexpr, PNKV: tl.constexpr, HD: tl.constexpr,
                    VEC: tl.constexpr, FMA: tl.constexpr):
    """Per-head QK RMSNorm, packed QKV in, dense q/k out, one launch for both.

    Replaces two ``.contiguous()`` copies and two ``rms_norm_kernel`` launches:
    the strided per-head view is read in place (the row of a head *is*
    contiguous; only the stride between tokens is the packed QKV width) and the
    dense buffers attention wants are written directly.

    Arithmetic is ``vllm::rms_norm_kernel``'s, which is the kernel the profile
    shows for this norm (``rms_norm_kernel<BFloat16, 8, 3, true>``, the generic
    3-D instantiation): fp32 variance over the head, ``rsqrtf(var/HD + eps)``,
    and the weighted value rounded to bf16 **once** --
    ``static_cast<scalar_t>(x * s_variance * w)``.  A single rounding here and
    two would move a fifth of the elements by a ULP.  Only the fp32 reduction
    order differs (a tree over the head against cub's raking reduce over 16 lanes
    of 8): 21 of 7.1M elements at 870 tokens, and those 21 land inside the bf16
    tolerance three stages later.
    """
    tok = tl.program_id(0)
    base = qkv_ptr + tok * stride_qkv
    _qk_norm_one(base, qw_ptr, q_ptr + tok * stride_q, eps, NH, PNH, HD,
                 VEC, FMA)
    _qk_norm_one(base + NH * HD, kw_ptr, k_ptr + tok * stride_k,
                 eps, NKV, PNKV, HD, VEC, FMA)


@triton.jit
def _qk_norm_one(x_base, w_ptr, out_base, eps,
                 N: tl.constexpr, PN: tl.constexpr, HD: tl.constexpr,
                 VEC: tl.constexpr, FMA: tl.constexpr):
    """One head-block's norm, with the reference's *summation order*, not just its
    arithmetic.

    This is the part that cannot be approximated.  Measured at 1000 tokens: a
    tree reduction over the head instead of the reference's order leaves **31 of
    8.2M** q elements differing by one bf16 ULP -- and the layer's matched ratio
    falls to 0.8487, a hard fail, where the same kernel with the reference norm
    put back passes at 0.9992.  Two amplifiers stacked in series do that: the FP8
    activation quantization in front of ``o_proj``, and the router, where a
    perturbed logit reorders top-k and the token's whole 4096-wide row changes.
    31 flipped bits, 15% of the output.  So "order is cheap, rounding is not" is
    wrong for this layer, and r1's rule -- bit-exact upstream of every FP8
    boundary -- is the operative one.

    Reproducing the reference's summation *shape* was not enough, and this is as
    far as it got.  The launcher fixes that shape: ``VEC = gcd(16/sizeof(T),
    hidden)`` = 8 for bf16 at head_dim 128, and ``block = min(hidden/VEC, 1024 or
    256)`` = 16 threads, so lane ``i`` owns exactly one 8-wide vector and sums its
    squares **sequentially** inside ``vec_op``, after which ``cub::BlockReduce``
    combines the 16 lane partials as a balanced binary tree (cub's shuffle-down
    and an xor butterfly agree for lane 0 at 16 valid lanes).  That is what the
    loop below does -- ``VEC`` sequential accumulations into a
    ``(heads, hidden/VEC)`` tile, then one ``tl.sum`` over the 16 partials -- and
    it moved the differing count at 1000 tokens only from 31 to 25 of 8.2M, with
    or without ``fma`` for the accumulate (25 either way).  Whatever is left is
    somewhere in ``rsqrtf`` vs ``tl.math.rsqrt``, cub's exact raking, or how
    ``vectorize_read_with_alignment`` walks the row; the search was stopped there
    because 25 flipped bits is still a hard fail, so the kernel is off by default
    (``_GLUE_NORM``) and kept for a round that wants to finish the job.  The prize
    is real and measured: 1.35x at 16384 tokens against 1.25x, ~+5% geomean.

    The row is read twice here -- once strided for the reduction, once dense for
    the epilogue, the second out of L2 -- which is still far cheaper than the 346
    us of ``.contiguous()`` copies it replaces at 16384 tokens.
    """
    h = tl.arange(0, PN)
    d = tl.arange(0, HD)
    i = tl.arange(0, HD // VEC)
    m = h[:, None] < N
    v = tl.zeros((PN, HD // VEC), dtype=tl.float32)
    for j in tl.static_range(VEC):
        f = tl.load(x_base + h[:, None] * HD + i[None, :] * VEC + j,
                    mask=m, other=0.0).to(tl.float32)
        if FMA:
            v = tl.math.fma(f, f, v)
        else:
            v = v + f * f
    s = tl.math.rsqrt(tl.sum(v, axis=1) / HD + eps)
    x = tl.load(x_base + h[:, None] * HD + d[None, :], mask=m, other=0.0)
    w = tl.load(w_ptr + d).to(tl.float32)
    t = (x.to(tl.float32) * s[:, None] * w[None, :]).to(out_base.dtype.element_ty)
    tl.store(out_base + h[:, None] * HD + d[None, :], t, mask=m)


@triton.jit
def _qk_rope_kernel(q_ptr, k_ptr, pos_ptr, cache_ptr,
                    stride_q, stride_k, stride_pos, stride_cache,
                    NH: tl.constexpr, NKV: tl.constexpr,
                    PNH: tl.constexpr, PNKV: tl.constexpr,
                    HD: tl.constexpr, HALF: tl.constexpr,
                    ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                    IL: tl.constexpr):
    """M-RoPE in place on dense q/k, reading the table directly per token.

    The reference materializes ``cos_sin_cache[positions]`` as ``(3, N, 128)``
    and then two more dense copies of its halves before ``_mrope_kernel`` reads
    64 values per token out of them.  Here the three T/H/W rows are read straight
    out of the table with the same masks that kernel builds, so the gather and
    both copies disappear and the rotation itself is unchanged -- verified
    bit-exact against ``_mrope_kernel`` at every scored shape, which is why the
    arithmetic below is its expression verbatim, evaluated in bf16.
    """
    tok = tl.program_id(0)
    o = tl.arange(0, HALF)
    # Section -> T/H/W mapping, verbatim from ``_mrope_kernel``: the three masks
    # are disjoint and the unselected loads are 0, so summing them selects.
    if IL:
        h_mask = ((o % 3) == 1) & (o <= 3 * SH)
        w_mask = ((o % 3) == 2) & (o <= 3 * SW)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = ST
        h_end = ST + SH
        t_mask = o < ST
        h_mask = (t_end <= o) & (o < h_end)
        w_mask = (h_end <= o) & (o < HALF)
    rt = cache_ptr + tl.load(pos_ptr + tok) * stride_cache
    rh = cache_ptr + tl.load(pos_ptr + stride_pos + tok) * stride_cache
    rw = cache_ptr + tl.load(pos_ptr + 2 * stride_pos + tok) * stride_cache
    cos_row = (tl.load(rt + o, mask=t_mask, other=0)
               + tl.load(rh + o, mask=h_mask, other=0)
               + tl.load(rw + o, mask=w_mask, other=0))
    sin_row = (tl.load(rt + HALF + o, mask=t_mask, other=0)
               + tl.load(rh + HALF + o, mask=h_mask, other=0)
               + tl.load(rw + HALF + o, mask=w_mask, other=0))
    _rope_one(q_ptr + tok * stride_q, cos_row, sin_row, o, NH, PNH, HD, HALF)
    _rope_one(k_ptr + tok * stride_k, cos_row, sin_row, o, NKV, PNKV, HD, HALF)


@triton.jit
def _rope_one(base, cos_row, sin_row, o,
              N: tl.constexpr, PN: tl.constexpr, HD: tl.constexpr,
              HALF: tl.constexpr):
    h = tl.arange(0, PN)
    m = h[:, None] < N
    p = base + h[:, None] * HD + o[None, :]
    lo = tl.load(p, mask=m, other=0)
    hi = tl.load(p + HALF, mask=m, other=0)
    tl.store(p, lo * cos_row[None, :] - hi * sin_row[None, :], mask=m)
    tl.store(p + HALF, hi * cos_row[None, :] + lo * sin_row[None, :], mask=m)


class GlueAttention(LlamaAttention):
    """``LlamaAttention`` with the post-QKV glue in two launches instead of six.

    Only q and k change hands: the projection, ``Attention`` and ``o_proj`` are
    the frozen L2 winner's, all three already measured bit-identical to the
    reference here.  Any configuration this does not structurally recognise --
    a rotary module that is not the vendored M-RoPE, a partial rotary dim, 1-D
    positions, a rotary handed in at call time, no QK norm, Llama 4's weightless
    norm or temperature tuning, a cos/sin table in a dtype the reference would
    have cast, a non-2-D or non-contiguous hidden state, tracing -- falls through
    to ``super().forward``, which r1 left as the reference sequence.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glue = None      # None = unresolved, False = ineligible
        self._gbuf: dict = {}

    def _apply(self, *args, **kwargs):
        out = super()._apply(*args, **kwargs)
        self._glue = None
        self._gbuf.clear()
        return out

    def _resolve_glue(self, hidden_states):
        """Bind every per-call invariant of the glue, or disable it for good."""
        plan = False
        try:
            rope = self.rotary_emb
            cache = getattr(rope, "cos_sin_cache", None)
            hd = self.head_dim
            sec = list(getattr(rope, "mrope_section", ()) or ())
            ok = (self.q_norm is not None and self.k_norm is not None
                  and self.q_wl_norm is None and not self.nope
                  and not self.attn_temperature_tuning
                  and type(rope).__name__ in _MROPE_CLASSES
                  and getattr(type(rope).forward, "__module__", "").endswith(
                      "L1.mrope")
                  and getattr(rope, "head_dim", None) == hd
                  and getattr(rope, "rotary_dim", None) == hd
                  and isinstance(cache, torch.Tensor) and cache.dim() == 2
                  and cache.size(1) == hd and cache.stride(1) == 1
                  and len(sec) == 3 and sum(sec) == hd // 2
                  and hd % 2 == 0
                  # The reference casts the table to the activation dtype inside
                  # every forward; reading it in any other dtype changes every
                  # rotation factor, which is the one difference this layer
                  # cannot absorb.  r1 holds the buffer in bf16 for exactly this
                  # reason (``_cast_rope_cache``), so this is an identity check.
                  and cache.dtype is hidden_states.dtype
                  and hidden_states.dtype is torch.bfloat16)
            if ok:
                qw, kw = self.q_norm.weight, self.k_norm.weight
                ok = (isinstance(qw, torch.Tensor) and qw.dtype is cache.dtype
                      and qw.numel() == hd and qw.is_contiguous()
                      and isinstance(kw, torch.Tensor) and kw.dtype is cache.dtype
                      and kw.numel() == hd and kw.is_contiguous())
            if ok:
                plan = (cache, qw, kw, int(sec[0]), int(sec[1]), int(sec[2]),
                        bool(getattr(rope, "mrope_interleaved", False)))
        except Exception:
            plan = False
        self._glue = plan
        return plan

    def _glue_bufs(self, n: int, dtype, device):
        b = self._gbuf.get(n)
        if b is None:
            if len(self._gbuf) >= 8:
                self._gbuf.clear()
            b = self._gbuf[n] = (
                torch.empty(n, self._qsz, dtype=dtype, device=device),
                torch.empty(n, self.num_kv_heads * self.head_dim,
                            dtype=dtype, device=device),
            )
        return b

    def forward(self, positions, hidden_states, rotary_emb=None):
        plan = self._glue
        if (rotary_emb is None and plan is not False
                and hidden_states.dim() == 2 and hidden_states.is_contiguous()
                and positions.dim() == 2 and positions.size(0) == 3
                and positions.dtype is torch.int64
                and positions.stride(1) == 1
                and hidden_states.size(0) == positions.size(1)
                and not torch.compiler.is_compiling()):
            if plan is None:
                plan = self._resolve_glue(hidden_states)
            if plan is not False:
                cache, qw, kw, st, sh, sw, il = plan
                qkv = self.qkv_proj(hidden_states)
                if qkv.dim() == 2 and qkv.is_contiguous() and qkv.size(0):
                    # ``qkv.size(0)``: an empty batch would be a zero grid, and
                    # the reference guards its own glue the same way.
                    n = qkv.size(0)
                    hd = self.head_dim
                    nh, nkv = self.num_heads, self.num_kv_heads
                    pnh = triton.next_power_of_2(nh)
                    pnkv = triton.next_power_of_2(nkv)
                    if _GLUE_NORM:
                        q, k = self._glue_bufs(n, qkv.dtype, qkv.device)
                        _qk_norm_kernel[(n,)](
                            qkv, qw, kw, q, k,
                            qkv.stride(0), q.stride(0), k.stride(0), self._eps,
                            NH=nh, NKV=nkv, PNH=pnh, PNKV=pnkv, HD=hd,
                            VEC=_GLUE_VEC, FMA=_GLUE_FMA,
                            num_warps=_GLUE_WARPS,
                        )
                    else:
                        qs, ks, _ = qkv.split(self._split, dim=-1)
                        q = self.q_norm(qs.view(n, nh, hd)).view(n, nh * hd)
                        k = self.k_norm(ks.view(n, nkv, hd)).view(n, nkv * hd)
                    _qk_rope_kernel[(n,)](
                        q, k, positions, cache,
                        q.stride(0), k.stride(0), positions.stride(0),
                        cache.stride(0),
                        NH=nh, NKV=nkv, PNH=pnh, PNKV=pnkv,
                        HD=hd, HALF=hd // 2, ST=st, SH=sh, SW=sw, IL=il,
                        num_warps=_GLUE_WARPS,
                    )
                    v = qkv[:, self._qsz + k.size(1):]
                    return self.o_proj(self.attn(q, k, v))
        return super().forward(positions, hidden_states, rotary_emb)


class _GraphKey:
    """One captured replay of the whole layer for one (token count, residual) key.

    The layer's wall at a single token is 665 us of Python and driver calls over
    115 us of device work (iter 01), and none of that Python depends on a device
    value: the token count fixes every tile shape, every buffer size and every
    branch on the path.  So the entire forward is captured once per key and
    replayed, which is 2.1 us of host instead of 665.

    Two things make it safe rather than merely fast:

    * **The harness hands a different tensor every iteration** (its
      ``_ShiftingPool`` shifts the base address so no two calls share a
      ``data_ptr``), so the graph cannot read the caller's buffers.  The inputs
      are copied into static slots that the capture recorded, which at one token
      is 16 KB and free -- and is exactly why this shape is the right one to
      graph and 16384 tokens is not (268 MB of copy against an 11.9 ms wall).
    * **Output 1 aliases an input.**  With a residual the reference returns the
      caller's ``residual`` tensor, mutated in place by the two add-norms; with
      ``residual is None`` it returns the caller's ``hidden_states``, mutated in
      place by the *post-attention* norm (``residual`` is bound to it on the
      first line).  Either way the graph's copy of that tensor is written back
      into the caller's, so the in-place contract survives the indirection.
      Getting this wrong is invisible at the res shapes and fails every nores
      shape.

    ``pins`` is the third: a graph records *addresses*, and the scratch a
    captured kernel writes to was allocated before the capture by a holder that
    may later reallocate it for a bigger shape (the reference MoE's
    ``_SHARED_BUF.cache13`` does exactly that).  Freeing that storage would let
    the allocator hand it to someone else while a live graph still writes there.
    Holding a reference to every module-held tensor as it stood at capture time
    keeps the storage alive, so an older key's graph stays valid no matter what
    a later shape does.  (The frozen L1 winners already reason this way -- see
    ``MoeAlign._plan``'s "a plan pins the buffer views that existed when it was
    built".)
    """

    __slots__ = ("graph", "pos", "hs", "res", "out0", "out1", "res_is_hs",
                 "warm", "pins")

    def __init__(self, positions, hidden_states, residual, warm):
        self.graph = None
        self.warm = warm
        self.pins = None
        self.pos = torch.empty_like(positions)
        self.hs = torch.empty_like(hidden_states)
        self.res = None if residual is None else torch.empty_like(residual)
        self.res_is_hs = residual is None
        self.out0 = self.out1 = None

    def args(self):
        return self.pos, self.hs, self.res

    def load(self, positions, hidden_states, residual) -> None:
        if self.res is None:
            torch._foreach_copy_((self.pos, self.hs), (positions, hidden_states))
        else:
            torch._foreach_copy_((self.pos, self.hs, self.res),
                                 (positions, hidden_states, residual))

    def replay(self, positions, hidden_states, residual):
        self.load(positions, hidden_states, residual)
        self.graph.replay()
        # Output 1 is the caller's own tensor in the reference, mutated in place.
        alias = hidden_states if self.res_is_hs else residual
        alias.copy_(self.out1)
        return self.out0, alias


def _pin_module_tensors(root: nn.Module) -> list:
    """Every tensor any module on the path holds right now (see ``_GraphKey``)."""
    pins = []
    for sub in root.modules():
        d = getattr(sub, "__dict__", None)
        if not d:
            continue
        for v in list(d.values()):
            if isinstance(v, torch.Tensor):
                pins.append(v)
            elif isinstance(v, dict):
                for vv in list(v.values()):
                    if isinstance(vv, torch.Tensor):
                        pins.append(vv)
                    elif isinstance(vv, (tuple, list)):
                        pins.extend(x for x in vv if isinstance(x, torch.Tensor))
            elif isinstance(v, (tuple, list)):
                pins.extend(x for x in v if isinstance(x, torch.Tensor))
            elif hasattr(type(v), "__slots__"):
                for name in getattr(type(v), "__slots__", ()):
                    vv = getattr(v, name, None)
                    if isinstance(vv, torch.Tensor):
                        pins.append(vv)
    for s in _Buffers.__slots__:
        vv = getattr(_BUF, s, None)
        if isinstance(vv, torch.Tensor):
            pins.append(vv)
    return pins


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb: nn.Module | None = None,
                 quant_config: dict | None = None):
        super().__init__()
        self.self_attn = (GlueAttention if _GLUE else LlamaAttention)(
            config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            rotary_emb=rotary_emb,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
        )
        # This layer cannot absorb *any* upstream last-bit difference, so the two
        # places where the frozen winners are not bit-exact have to be given up.
        # The amplifier is the FP8 activation quantization in front of ``o_proj``:
        # one bf16 ULP (0.4%) on q or k moves ~3% of the attention output across
        # an FP8 rounding boundary, each by one FP8 ULP (12.5%), and a K=8192 dot
        # over that is 0.125*sqrt(0.03) = 2% -- 200x the input perturbation and
        # 2x the harness's rtol.  Measured end to end: 0.80-0.90 matched ratio at
        # every mid-M shape (whole tokens wrong, 7-200 rows of 870), i.e. a hard
        # fail, from differences invisible at L1/L2 where each op is compared on
        # its own output.
        #
        # First: the QK norm runs on a *strided* per-head view of the packed QKV
        # output, and the L1 winner reads that view with a shuffle-only reduction
        # where the reference copies to contiguous and uses a cub block reduce.
        # The two fp32 reduction orders disagree in the last bit.  Only the
        # *head-dim* norms are swapped back; the two hidden-size norms
        # (contiguous, where the winner is bit-exact) keep using the winner.
        sa = self.self_attn
        if sa.q_norm is not None:
            eps = config.rms_norm_eps
            sa.q_norm = RefRMSNorm(config.head_dim, eps=eps)
            sa.k_norm = RefRMSNorm(config.head_dim, eps=eps)
        # ... and take the attention winner's *reference sequence* rather than its
        # fused post-QKV glue, which folds the same QK norm and the RoPE into one
        # in-place pass and reads the cos/sin table in fp32 where the reference
        # casts it to bf16.  Both are wins in isolation and both change q/k in the
        # last bits.  Dropping the two rope plans is what selects that sequence
        # (``_plan1``/``_plan2`` survive ``.to(device)``, unlike ``_plan``), and it
        # keeps everything else the winner does: the fp8 projections, its
        # ``Attention`` and its rotary module, all of which are bit-exact here.
        sa._plan1 = sa._plan2 = None
        self.mlp = FusedMoE(config, quant_config=quant_config)
        if _PIN_M1:
            self._pin_small_m_paths()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._rope_dtype = None
        self._graphs = {}

    def _pin_small_m_paths(self) -> None:
        """Take the reference GEMM path for the single-token row.

        The L1 fp8 winner has a fused quantize+GEMV for ``M == 1`` and the L2
        linear winner has its own Triton GEMV plan; both compute a different
        reduction than the reference's quantize + DeepGEMM/FlashInfer, and at
        M=1 the layer's whole output is *one* row, so whether such a difference
        crosses an FP8 boundary in front of ``o_proj`` decides the whole case.
        Measured: the *same* case, seed and code gives matched=1.0000 in one
        process and 0.6853 in another, i.e. the divergence is not weight- or
        input-dependent but varies with what else is on the GPU -- the signature
        of a split-K reduction whose atomic order follows block completion.  One
        row averages nothing out, so M=1 takes the reference reduction: it costs
        two launches on a shape that is host-bound anyway.
        """
        for sub in self.modules():
            lin = getattr(sub, "linear_op", None)
            if lin is not None and hasattr(lin, "_gemv_eligible"):
                lin._gemv_eligible = _never_eligible
                lin._gemv_ok = False
            pc = getattr(sub, "_pc", None)
            w = getattr(sub, "weight", None)
            if (isinstance(pc, dict) and isinstance(w, torch.Tensor)
                    and w.dim() == 2 and w.dtype not in _FP8_DTYPES):
                # Seed the bf16 linear's per-shape plan cache with the reference
                # ``F.linear`` for the single-token row, so its own split-K GEMV
                # plan is never built for that shape.
                pc[torch.Size([1, w.shape[1]])] = torch.nn.functional.linear

    def _cast_rope_cache(self, dtype) -> None:
        """Hold the rotary cos/sin table in the activation dtype.

        The reference M-RoPE does ``cache.to(query.dtype)`` inside every forward.
        For this config the table is ``(4 * 262144, 128)`` fp32 = 537 MB, so that
        cast moves ~800 MB and measures 116 us of a 1100 us layer -- at *every*
        shape, including M=1, for a value that never changes.  Converting the
        buffer once is bit-identical (the kernel reads exactly the bf16 values the
        per-call cast would have produced -- verified exact at M=1/870/1000/16384
        on both the 1-D and M-RoPE paths) and takes the rope from 145 us to 64 us
        at M=1000.  Only fp32 -> 16-bit is folded; a genuinely fp32 activation
        leaves the buffer alone and keeps the module's own cast.
        """
        self._rope_dtype = dtype
        rope = getattr(self.self_attn, "rotary_emb", None)
        cache = getattr(rope, "cos_sin_cache", None) if rope is not None else None
        if (isinstance(cache, torch.Tensor) and cache.dtype == torch.float32
                and dtype in (torch.bfloat16, torch.float16)):
            rope.cos_sin_cache = cache.to(dtype)
            cast_cache = getattr(rope, "_cast_cache", None)
            if isinstance(cast_cache, dict):
                cast_cache.clear()
            # The attention winner memoises the table for its glue; drop the
            # stale fp32 handle so the 537 MB allocation can actually be freed.
            if getattr(self.self_attn, "_cache_t", None) is not None:
                self.self_attn._cache_t = None

    # -- whole-layer graph --------------------------------------------------
    def _graph_ok(self, positions, hidden_states, residual) -> bool:
        """Structural conditions the capture needs, all host-side constants."""
        if not (hidden_states.dim() == 2 and hidden_states.is_contiguous()
                and hidden_states.device.type == "cuda"
                and 0 < hidden_states.size(0) <= _GRAPH_MAX_M):
            return False
        if not (isinstance(positions, torch.Tensor) and positions.is_contiguous()
                and positions.device == hidden_states.device):
            return False
        if residual is not None and not (
                residual.is_contiguous() and residual.shape == hidden_states.shape
                and residual.dtype is hidden_states.dtype):
            return False
        mlp = self.mlp
        if getattr(mlp, "_use_custom_op", False) or mlp.tp_size > 1:
            return False
        return not (torch.is_grad_enabled()
                    or torch.cuda.is_current_stream_capturing()
                    or torch.compiler.is_compiling())

    def _graph_capture(self, ent, positions, hidden_states, residual) -> None:
        """Warm on a side stream, then record the forward against the static slots."""
        a = ent.args()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                ent.load(positions, hidden_states, residual)
                self._forward_eager(*a)
        torch.cuda.current_stream().wait_stream(side)
        # The warmup mutated the static slots in place (both add-norms write
        # their residual argument), so reload before recording.
        ent.load(positions, hidden_states, residual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out0, out1 = self._forward_eager(*a)
        ent.out0, ent.out1, ent.graph = out0, out1, graph
        ent.pins = _pin_module_tensors(self)

    def _graph_validate(self, ent, positions, hidden_states, residual):
        """Replay once and check it against eager, or throw the graph away.

        "Capture raised nothing" is not evidence that the replay is right.  A
        graph records *launches*; an op that reaches the driver outside the
        capture stream records nothing at all, and its replay then reads
        whatever happens to be at those addresses.  Measured here: at one token
        the replay is bit-exact, and above it (where the attention path picks up
        FA4's CuTe launcher) it is garbage -- with no error at capture time.

        So the graph is compared against an eager forward on the same inputs,
        against an eager-vs-eager noise floor so a genuinely nondeterministic
        reduction upstream cannot fail the check (this layer has one: the bf16
        router GEMM is a cuBLAS split-K).  Bit-exact eager, bit-exact required.
        Returns ``(ok, outputs)`` and leaves the caller's tensors holding the
        outputs of whichever path won.
        """
        pos_c, hs_c = positions.clone(), hidden_states.clone()
        res_c = None if residual is None else residual.clone()
        pos_d, hs_d = positions.clone(), hidden_states.clone()
        res_d = None if residual is None else residual.clone()
        got = tuple(t.clone() for t in
                    ent.replay(positions, hidden_states, residual))
        want = tuple(t.clone() for t in self._forward_eager(pos_c, hs_c, res_c))
        again = self._forward_eager(pos_d, hs_d, res_d)
        ok = True
        for g, w, n in zip(got, want, again):
            noise = (w.to(torch.float32) - n.to(torch.float32)).abs().max()
            delta = (g.to(torch.float32) - w.to(torch.float32)).abs().max()
            if not bool(delta <= 4.0 * noise):
                ok = False
                break
        if ok:
            return True, got
        alias = hidden_states if residual is None else residual
        alias.copy_(want[1])
        return False, (want[0], alias)

    def forward(self, positions, hidden_states, residual):
        if self._rope_dtype is not hidden_states.dtype:
            self._cast_rope_cache(hidden_states.dtype)
        if _GRAPH:
            key = (hidden_states.size(0), residual is None,
                   hidden_states.dtype, positions.dtype, positions.shape)
            ent = self._graphs.get(key)
            if ent is None:
                ent = (_GraphKey(positions, hidden_states, residual, _GRAPH_WARM)
                       if self._graph_ok(positions, hidden_states, residual)
                       else False)
                self._graphs[key] = ent
            if ent is not False:
                if ent.graph is not None:
                    return ent.replay(positions, hidden_states, residual)
                if ent.warm > 1:
                    ent.warm -= 1
                    return self._forward_eager(positions, hidden_states, residual)
                # The capture call must not *also* run eagerly first: the eager
                # output at one token is the reference MoE's ``MoeSum`` scratch
                # view, and the capture's own warmup re-runs the forward and
                # overwrites it.  Capture straight from the caller's (pristine)
                # inputs -- the warmup runs against the static slots, so those
                # inputs are only read -- then replay for this call's answer.
                try:
                    self._graph_capture(ent, positions, hidden_states, residual)
                    ok, out = self._graph_validate(
                        ent, positions, hidden_states, residual)
                except Exception:
                    # An unrecordable op on the path: this key stays eager.
                    ok, out = False, None
                if not ok:
                    self._graphs[key] = False
                    if out is None:
                        out = self._forward_eager(
                            positions, hidden_states, residual)
                return out
        return self._forward_eager(positions, hidden_states, residual)

    def _forward_eager(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        fused = self._norm_quant(hidden_states, residual)
        if fused is None:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        else:
            h, a_fp8, a_scale = fused
            hidden_states = self.mlp.forward_impl(h, (a_fp8, a_scale))
        return hidden_states, residual

    def _norm_quant(self, x, residual):
        """Post-attention norm + MoE activation quantization in one pass.

        Returns ``(h_bf16, a_fp8, a_scale)``, or None when the fused MoE path
        would not be taken anyway (then the reference norm runs unchanged).
        """
        mlp = self.mlp
        norm = self.post_attention_layernorm
        w = getattr(norm, "weight", None)
        if (not _FUSE_NORM or residual is None or x.dim() != 2
                or not x.is_contiguous() or not residual.is_contiguous()
                or x.dtype is not torch.bfloat16
                or not isinstance(w, torch.Tensor) or w.dtype is not x.dtype
                or not w.is_contiguous()
                or not mlp.eligible(x)):
            return None
        M, K = x.shape
        nk = K // _GROUP
        buf = mlp._buf
        h = buf.get("h_bf16", M * K, x.dtype, x.device).view(M, K)
        a_fp8 = buf.get("a_fp8", M * K, torch.float8_e4m3fn, x.device).view(M, K)
        a_scale = buf.get("a_scale", M * nk, torch.float32, x.device).view(M, nk)
        _norm_quant_kernel[(M,)](
            x, residual, w, h, a_fp8, a_scale,
            x.stride(0), residual.stride(0), h.stride(0), a_fp8.stride(0),
            a_scale.stride(0), norm.eps,
            H=K, NUM_G=nk, GROUP=_GROUP, FP8_MAX=_FP8_MAX, num_warps=8,
        )
        return h, a_fp8, a_scale
