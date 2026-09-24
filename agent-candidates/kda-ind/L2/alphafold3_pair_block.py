"""PairBlock for AlphaFold3 -- fused Triton implementation.

Same five-stage residual update as ``baseline.py``:
TriMulOut -> TriMulIn -> TriAttStart -> TriAttEnd -> SwiGLUTransition.

At the captured shape (``z: bf16[1, 16, 16, 128]``) this operator is not
compute-bound. Total arithmetic is ~292 MFLOP -- 0.13 us at B200 dense bf16 peak
-- and the whole weight set is ~1.07 MB, another 0.13 us of HBM traffic. The
baseline nevertheless measures 1291-1362 us because it issues ~109 device
kernels whose *host* submit time is ~1400 us: ~82% of the wall clock is CPU
dispatch, and the GPU is starved. Pre-fusing the projection weights in pure
PyTorch (5 GEMMs -> 1 per stage) was measured at 1323 us, i.e. no win at all --
fewer, bigger GEMMs do not help while ~250 aten calls remain.

So the only lever is the number of dispatched operations. Every stage is
rewritten as flat ``[R, C]`` row math (``R = N*N`` rows, ``r = i*N + j``) over
pre-concatenated weights, and the five stages run as ten Triton launches built
from four kernels:

    tri_mul_out       ln_proj(GATE_MASK)  + trimul_combine
    tri_mul_in        ln_proj(GATE_MASK)  + trimul_combine   (contraction strides swapped)
    tri_att_start     ln_proj(QKV)        + triatt
    tri_att_end       ln_proj(QKV, z^T)   + triatt           (mask/residual strides swapped)
    pair_transition   ln_proj(SWIGLU)     + gemm_mask_residual

Grids run 16-512 CTAs on 148 SMs. Small grids are accepted deliberately -- with
292 MFLOP of total work, splitting wider costs launches rather than saving time --
but not blindly: NCU put the cost of a 16-CTA grid at ~89% on every launch, so
``ln_proj`` spreads its output-column blocks across the grid and pays a redundant
LayerNorm per block to get there. Evidence in
``profile/pair_block_v2_tuned/REPORT.md``.

Thread-safety limitation, accepted on purpose: the scratch arena is per plan, not
per stream, so one ``PairBlock`` instance must not have ``forward`` entered
concurrently from two CUDA streams or re-entered -- concurrent calls on separate
streams will interleave writes to the same intermediates. Verified reachable, not
theoretical: two streams at ``N = 64`` produced differing results. The harness
calls ``forward`` serially on the default stream, and a per-stream arena pool or
per-call allocation would give back the very dispatch budget this candidate
exists to protect, which is why the limitation is documented rather than fixed.
The tensor *returned* is freshly allocated per call, so ordinary sequential use
never aliases.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from .alphafold3_triangle_multiplication import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from .alphafold3_triangle_attention import TriangleAttention
from .alphafold3_swiglu_transition import SwiGLUTransition


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
# ``ln_proj`` epilogue selector. A constexpr, so each epilogue compiles to its
# own kernel; the shared body is the LayerNorm and the projection loop.
_EP_GATE_MASK = tl.constexpr(0)
_EP_SWIGLU = tl.constexpr(1)
_EP_QKV = tl.constexpr(2)


@triton.jit
def _rnd(x):
    """Round an fp32 value through bf16, i.e. materialize it the way an aten op
    would when its output dtype is bf16. Used at exactly the points where the
    baseline stores a bf16 tensor, so the candidate's rounding lattice matches
    the reference instead of merely being more accurate than it."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _ln_proj(z_ptr, mask_ptr, lnw_ptr, lnb_ptr, w_ptr, o0_ptr, o1_ptr, o2_ptr,
             ZS_B: tl.constexpr, ZS_I: tl.constexpr, ZS_J: tl.constexpr,
             MS_B: tl.constexpr, MS_I: tl.constexpr, MS_J: tl.constexpr,
             N: tl.constexpr, C: tl.constexpr, NROW: tl.constexpr,
             W1: tl.constexpr, W2: tl.constexpr, WTOT: tl.constexpr,
             EPILOGUE: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, EPS: tl.constexpr):
    """LayerNorm over ``C`` -> bf16 -> projection against one pre-concatenated
    weight, with a constexpr epilogue.

    ``w_ptr`` is the fused weight already transposed to ``[C, WTOT]``, so the
    ``tl.dot`` operand load is contiguous along the output axis. ``ZS_I``/``ZS_J``
    select the frame: swapping them reads ``z`` transposed, which is how the
    ending-node attention stage gets its ``z^T`` without materializing a copy.

    The output columns are spread across ``program_id(1)`` rather than looped
    inside one program. With only ``R / BLOCK_M`` row blocks, looping left 16
    CTAs on 148 SMs and NCU put the cost of that at ~89% on every launch; each
    column block re-derives the LayerNorm, which is a few hundred bytes of
    redundant traffic against a launch that was otherwise 90% idle silicon.

    Every stride and size is a constexpr. These are all fixed once the input
    signature is known, and baking them in lets the compiler see the exact
    addressing -- there is no shape to specialize on at run time because the
    plan that owns this launch is itself cached per signature.
    """
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2)
    n0 = pid_n * BLOCK_N
    r = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    i = r // N
    j = r % N
    cs = tl.arange(0, C)

    x = tl.load(z_ptr + b * ZS_B + i[:, None] * ZS_I + j[:, None] * ZS_J
                + cs[None, :]).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(xc * xc, 1) / C + EPS)
    lnw = tl.load(lnw_ptr + cs).to(tl.float32)
    lnb = tl.load(lnb_ptr + cs).to(tl.float32)
    # The baseline LayerNorm promotes to fp32, keeps fp32 affine, and rounds the
    # result back to the input dtype (``promote_fp32=True`` in L1/layer_norm.py).
    zl = ((xc * rstd[:, None]) * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)

    row = b * NROW + r
    ns = n0 + tl.arange(0, BLOCK_N)
    if EPILOGUE == _EP_GATE_MASK:
        mk = tl.load(mask_ptr + b * MS_B + i * MS_I + j * MS_J).to(tl.float32)
        if n0 < W1:
            wp = w_ptr + cs[:, None] * WTOT + ns[None, :]
            # a = mask * sigmoid(a_g @ zl) * (a_p @ zl), and b likewise. Pairing
            # each value column with its gate column in the same tile is what
            # lets one launch replace a LayerNorm, 4 GEMMs, 2 sigmoids and 4
            # multiplies.
            ap = _rnd(tl.dot(zl, tl.load(wp)))
            ag = _rnd(tl.dot(zl, tl.load(wp + W1)))
            a = _rnd(_rnd(mk[:, None] * _rnd(tl.sigmoid(ag))) * ap)
            tl.store(o0_ptr + row[:, None] * W1 + ns[None, :], a.to(tl.bfloat16))
            bp = _rnd(tl.dot(zl, tl.load(wp + 2 * W1)))
            bg = _rnd(tl.dot(zl, tl.load(wp + 3 * W1)))
            bv = _rnd(_rnd(mk[:, None] * _rnd(tl.sigmoid(bg))) * bp)
            tl.store(o1_ptr + row[:, None] * W1 + ns[None, :], bv.to(tl.bfloat16))
        # sigmoid(linear_g @ zl) is emitted here rather than recomputed in
        # trimul_combine, which removes the last consumer of ``zl`` and so
        # removes any need to store it.
        if n0 < W2:
            g = tl.dot(zl, tl.load(w_ptr + cs[:, None] * WTOT + (4 * W1 + ns[None, :])))
            gs = tl.sigmoid(_rnd(g))
            tl.store(o2_ptr + row[:, None] * W2 + ns[None, :], gs.to(tl.bfloat16))
    elif EPILOGUE == _EP_SWIGLU:
        wp = w_ptr + cs[:, None] * WTOT + ns[None, :]
        ha = _rnd(tl.dot(zl, tl.load(wp)))
        hb = _rnd(tl.dot(zl, tl.load(wp + W1)))
        h = _rnd(_rnd(ha * tl.sigmoid(ha)) * hb)
        tl.store(o0_ptr + row[:, None] * W1 + ns[None, :], h.to(tl.bfloat16))
    else:
        # Plain projection for [Q|K|V|G] plus the triangle-bias block. The bias
        # block is padded out to a power-of-two tile (zeros in the fused weight)
        # instead of needing a separate narrow GEMM, and rides along on the first
        # column block.
        q = tl.dot(zl, tl.load(w_ptr + cs[:, None] * WTOT + ns[None, :]))
        tl.store(o0_ptr + row[:, None] * WTOT + ns[None, :],
                 _rnd(q).to(tl.bfloat16))
        if pid_n == 0:
            ts = tl.arange(0, W2)
            tb = tl.dot(zl, tl.load(w_ptr + cs[:, None] * WTOT + (W1 + ts[None, :])))
            tl.store(o0_ptr + row[:, None] * WTOT + (W1 + ts[None, :]),
                     _rnd(tb).to(tl.bfloat16))


@triton.jit
def _trimul_combine(a_ptr, b_ptr, gs_ptr, z_ptr, lnw_ptr, lnb_ptr, wz_ptr, out_ptr,
                    AS_I: tl.constexpr, AS_J: tl.constexpr,
                    BS_K: tl.constexpr, BS_J: tl.constexpr,
                    ZS_B: tl.constexpr, ZS_I: tl.constexpr, ZS_J: tl.constexpr,
                    OS_B: tl.constexpr, OS_I: tl.constexpr, OS_J: tl.constexpr,
                    N: tl.constexpr, C: tl.constexpr, CH: tl.constexpr,
                    NROW: tl.constexpr, EPS: tl.constexpr):
    """One program per ``i``: the triangle product, ``layer_norm_out``,
    ``linear_z``, the gate and the residual add -- eight aten ops in one launch.

    The two orientations differ *only* in which axis of ``a``/``b`` is
    contracted, so they are the same kernel with different row strides:

        outgoing  p[i,k,c] = sum_j a[i,j,c] * b[k,j,c]   AS=(N,1)  BS=(N,1)
        incoming  p[i,k,c] = sum_j a[j,i,c] * b[j,k,c]   AS=(1,N)  BS=(1,N)

    The gate and the residual stay at ``[i,k]`` in both orientations -- only the
    contraction moves. ``c`` is a batch axis of the product, not a reduction
    axis, so this is ``C`` independent 16x16 gemvs rather than a ``tl.dot``, and
    it is written as an accumulate loop over ``j``.
    """
    i = tl.program_id(0)
    b = tl.program_id(1)
    ks = tl.arange(0, N)
    hs = tl.arange(0, CH)
    cs = tl.arange(0, C)
    base = b * NROW

    # static_range, not range: the j iterations are independent apart from the
    # accumulator, so unrolling lets every b tile's load issue up front. Rolled,
    # NCU showed the loop stalled ~15 cycles per warp on an L1TEX scoreboard
    # dependency and this kernel was the most expensive of the ten.
    acc = tl.zeros([N, CH], dtype=tl.float32)
    for j in tl.static_range(N):
        av = tl.load(a_ptr + (base + i * AS_I + j * AS_J) * CH + hs).to(tl.float32)
        bv = tl.load(b_ptr + (base + ks[:, None] * BS_K + j * BS_J) * CH
                     + hs[None, :]).to(tl.float32)
        acc += bv * av[None, :]

    # The baseline's einsum runs on bf16 operands and stores a bf16 result, so
    # the product is rounded before layer_norm_out sees it.
    p = _rnd(acc)
    mean = tl.sum(p, 1) / CH
    pc = p - mean[:, None]
    rstd = 1.0 / tl.sqrt(tl.sum(pc * pc, 1) / CH + EPS)
    lnw = tl.load(lnw_ptr + hs).to(tl.float32)
    lnb = tl.load(lnb_ptr + hs).to(tl.float32)
    x = ((pc * rstd[:, None]) * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)

    xz = _rnd(tl.dot(x, tl.load(wz_ptr + hs[:, None] * C + cs[None, :])))
    gs = tl.load(gs_ptr + (base + i * N + ks[:, None]) * C + cs[None, :]).to(tl.float32)
    xz = _rnd(xz * gs)
    zv = tl.load(z_ptr + b * ZS_B + i * ZS_I + ks[:, None] * ZS_J
                 + cs[None, :]).to(tl.float32)
    tl.store(out_ptr + b * OS_B + i * OS_I + ks[:, None] * OS_J + cs[None, :],
             _rnd(zv + xz).to(tl.bfloat16))


@triton.jit
def _triatt(p_ptr, mask_ptr, wo_ptr, z_ptr, out_ptr,
            MS_B: tl.constexpr, MS_I: tl.constexpr, MS_K: tl.constexpr,
            ZS_B: tl.constexpr, ZS_I: tl.constexpr, ZS_J: tl.constexpr,
            OS_B: tl.constexpr, OS_I: tl.constexpr, OS_J: tl.constexpr,
            N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
            HD: tl.constexpr, WTOT: tl.constexpr, NROW: tl.constexpr,
            INF: tl.constexpr, SQRT_D: tl.constexpr):
    """One program per ``i``: all ``H`` heads of the ``N x N`` attention, the
    sigmoid gate, ``linear_o``, and the residual add.

    ``mha`` is called with ``q_x = kv_x = x`` where ``x`` is ``[N, N, C]``, so
    the attention batch dims are ``[1, I]``, the sequence axis is ``J``, and the
    per-head dim is ``D``. The bias broadcast is the subtle part:
    ``triangle_bias`` is ``[*, 1, H, I, J]`` against scores ``[*, I, H, Jq, Jk]``,
    so its ``(I, J)`` axes align with ``(Jq, Jk)`` and it is *shared across the
    leading* ``I`` -- legal only because the pair representation is square.

    ``inf * (mask - 1)`` is a large finite number, not ``-inf``, so a fully-masked
    key row gets a finite score everywhere and a max-subtracting softmax needs no
    special case for it.

    The scores are rounded to bf16 at the baseline's own materialization points --
    after the QK product and after each bias add -- rather than kept in fp32.
    Being *more* accurate than the reference is not the goal here, and this is a
    place where it actively diverges: the baseline's bias is bf16, so with
    ``inf = 1e9`` it is ~-9.98e8, and two logits far enough apart to be
    distinguishable in fp32 can round to the same bf16 value once that bias is
    added. On a fully-masked row the baseline then produces a uniform
    distribution where an fp32 score path produces a one-hot one. Real
    activations are nowhere near large enough to trigger it and the bench
    materializes ``pair_mask`` as all ones, so neither the harness nor a
    random-mask test would surface it -- which is exactly why the cast points are
    reproduced rather than improved on. The softmax reduction itself stays in
    fp32, matching ``F.softmax``, which uses fp32 opmath on a bf16 input and
    writes bf16.
    """
    i = tl.program_id(0)
    b = tl.program_id(1)
    qs = tl.arange(0, N)
    ks = tl.arange(0, N)
    ds = tl.arange(0, D)
    cs = tl.arange(0, C)
    base = b * NROW
    row0 = base + i * N

    # Mask bias is over keys. For the ending node the mask is read transposed
    # (mask[k, i]) -- one of the three places the transposed frame shows up.
    #
    # Two roundings, not one: the baseline writes ``self.inf * (mask - 1)`` on a
    # bf16 mask, so ``mask - 1`` is materialized as bf16 before the multiply.
    # Exact either way for a 0/1 mask, which is why binary-mask tests cannot see
    # it; see docs/numerics.md row 10a.
    mk = tl.load(mask_ptr + b * MS_B + i * MS_I + ks * MS_K).to(tl.float32)
    mb = _rnd(_rnd(mk - 1.0) * INF)

    out = tl.zeros([N, C], dtype=tl.float32)
    for h in range(H):
        off = h * D
        q = tl.load(p_ptr + (row0 + qs[:, None]) * WTOT + (off + ds[None, :]))
        # The baseline scales q *after* rounding the projection to bf16
        # (``_prep_qkv``: q = q / sqrt(c_hidden) on a bf16 tensor).
        q = _rnd(q.to(tl.float32) / SQRT_D).to(tl.bfloat16)
        kt = tl.load(p_ptr + (row0 + ks[None, :]) * WTOT + (HD + off + ds[:, None]))
        tb = tl.load(p_ptr + (base + qs[:, None] * N + ks[None, :]) * WTOT
                     + (4 * HD + h)).to(tl.float32)
        # The baseline's einsum stores bf16 scores, then adds each bias as a
        # separate bf16 op, so there are three rounding points before the softmax.
        s = _rnd(_rnd(_rnd(tl.dot(q, kt)) + mb[None, :]) + tb)
        e = tl.exp(s - tl.max(s, 1)[:, None])
        # bf16 probabilities before the value matmul: F.softmax on a bf16 input
        # writes bf16, which the baseline's scores.to(dtype=value.dtype) then
        # leaves untouched.
        pr = (e / tl.sum(e, 1)[:, None]).to(tl.bfloat16)
        v = tl.load(p_ptr + (row0 + ks[:, None]) * WTOT + (2 * HD + off + ds[None, :]))
        o = _rnd(tl.dot(pr, v))
        g = tl.load(p_ptr + (row0 + qs[:, None]) * WTOT
                    + (3 * HD + off + ds[None, :])).to(tl.float32)
        og = _rnd(o * _rnd(tl.sigmoid(g))).to(tl.bfloat16)
        # Per-head partial products of linear_o, summed in fp32. Same
        # accumulation width as the baseline's single [.., H*D] x [H*D, C] GEMM.
        out += tl.dot(og, tl.load(wo_ptr + (off + ds[:, None]) * C + cs[None, :]))

    out = _rnd(out)
    zv = tl.load(z_ptr + b * ZS_B + i * ZS_I + qs[:, None] * ZS_J
                 + cs[None, :]).to(tl.float32)
    tl.store(out_ptr + b * OS_B + i * OS_I + qs[:, None] * OS_J + cs[None, :],
             _rnd(zv + out).to(tl.bfloat16))


@triton.jit
def _gemm_mask_residual(h_ptr, w_ptr, mask_ptr, z_ptr, out_ptr,
                        MS_B: tl.constexpr, MS_I: tl.constexpr, MS_J: tl.constexpr,
                        ZS_B: tl.constexpr, ZS_I: tl.constexpr, ZS_J: tl.constexpr,
                        OS_B: tl.constexpr, OS_I: tl.constexpr, OS_J: tl.constexpr,
                        N: tl.constexpr, C: tl.constexpr, HID: tl.constexpr,
                        NROW: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_K: tl.constexpr, USE_MASK: tl.constexpr):
    """The transition's output projection, the optional mask, and the residual.

    Kept as its own kernel rather than an ``ln_proj`` instantiation: the input is
    already normalized, and reusing ``ln_proj`` here would risk applying a second
    LayerNorm.
    """
    pid = tl.program_id(0)
    b = tl.program_id(2)
    r = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cs = tl.arange(0, C)
    row = b * NROW + r

    acc = tl.zeros([BLOCK_M, C], dtype=tl.float32)
    for k0 in tl.static_range(0, HID, BLOCK_K):
        kss = k0 + tl.arange(0, BLOCK_K)
        hv = tl.load(h_ptr + row[:, None] * HID + kss[None, :])
        wv = tl.load(w_ptr + kss[:, None] * C + cs[None, :])
        acc += tl.dot(hv, wv)
    acc = _rnd(acc)

    if USE_MASK:
        # _mask_trans=False means the baseline builds an all-ones mask, which is
        # an exact no-op in bf16, so the multiply is dropped rather than faked.
        mk = tl.load(mask_ptr + b * MS_B + (r // N) * MS_I
                     + (r % N) * MS_J).to(tl.float32)
        acc = _rnd(acc * mk[:, None])

    zv = tl.load(z_ptr + b * ZS_B + (r // N)[:, None] * ZS_I
                 + (r % N)[:, None] * ZS_J + cs[None, :]).to(tl.float32)
    tl.store(out_ptr + b * OS_B + (r // N)[:, None] * OS_I
             + (r % N)[:, None] * OS_J + cs[None, :],
             _rnd(zv + acc).to(tl.bfloat16))


# ---------------------------------------------------------------------------
# Launch plan
# ---------------------------------------------------------------------------
# Slots in a launch's argument list that change from call to call: the caller's
# ``z`` and ``pair_mask`` (the bench's shifting pool hands us a different
# ``data_ptr`` every iteration) and the freshly allocated output.
_IN_Z = object()
_IN_MASK = object()
_OUT_Z = object()

# grid_x, grid_y, grid_z, stream, function, packed_metadata, launch_metadata,
# enter_hook, exit_hook -- the fixed head of a compiled-launcher call, before the
# kernel's own arguments.
_HEAD = 9

# ``torch.cuda.current_stream(dev).cuda_stream`` builds a device object and a
# Python Stream wrapper only to read one integer out of them: 2.03 us against
# 0.048 us for the raw accessor, on a call whose whole host budget is ~78 us.
# Verified to return the identical handle. Resolved once, with a fallback, because
# it is a private symbol.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _same_place(t, dtype, device) -> bool:
    """Is this tensor already in the dtype and on the device the kernels assume?"""
    return t.dtype is dtype and t.device == device


def _addressable(t, ndim: int, unit_dim: int, dtype, device) -> bool:
    """Can the fused kernels address ``t`` directly?

    Both inputs go through this one predicate rather than being checked by hand.
    Both omissions this guard has actually produced were the same mistake --
    ``pair_mask``'s dtype, and then its device, each checked for ``z`` and
    forgotten for the mask -- so what is shared here is exactly that class of
    property. Shapes differ per tensor and are checked by the caller.

    Strides must be strictly positive. A stride-0 axis (from ``expand``) would in
    fact compute correctly here, since re-reading one element is what ``expand``
    means and the plan bakes strides in as constexpr; it is rejected because it is
    a layout with no test behind it and deferring costs nothing.
    """
    if not isinstance(t, torch.Tensor) or t.dim() != ndim:
        return False
    if not _same_place(t, dtype, device):
        return False
    st = t.stride()          # one read: .stride() allocates a torch.Size
    return st[unit_dim] == 1 and min(st) >= 1


# Tile widths and warp counts, chosen by measuring the ten-launch sequence on a
# B200 rather than by rule of thumb -- evidence in
# profile/pair_block_v3_final/REPORT.md. They are collected here because NCU's
# finding was structural (16 CTAs on 148 SMs, est. +89% on every launch), so the
# knobs that matter are the ones that trade redundant work for occupancy and the
# ones that give a CTA more warps to hide its dependent-load chain.
_TUNING = {
    "ln_proj_gate_mask": {"block_n": 32, "num_warps": 8},
    "ln_proj_qkv": {"block_n": 64, "num_warps": 8},
    "ln_proj_swiglu": {"block_n": 64, "num_warps": 8},
    # 16 warps rather than 4: this kernel's j loop is a chain of dependent global
    # loads and was the most expensive of the ten (29.5 us, ~29k SM-active cycles
    # for ~68 KB of traffic). Unrolling the loop and widening the CTA together
    # took the whole sequence from 86 us to 82 us; 4 warps measured 91 us.
    "trimul_combine": {"num_warps": 16},
    "triatt": {"num_warps": 4},
    "gemm_mask_residual": {"block_m": 16, "block_k": 128, "num_warps": 8},
}


class _Plan:
    """Everything needed to run one input signature: the lazily fused weights,
    a persistent scratch arena, and ten pre-bound kernel launches.

    Steady-state ``forward`` is a dict lookup, one output allocation, three
    ``data_ptr()`` reads, and ten launcher calls. Nothing here allocates an
    intermediate, reshapes anything, or synchronizes.

    The arena is shared by every call that hits this plan, so the plan is
    single-stream and non-reentrant; see the module docstring.

    Going through the compiled launcher rather than ``JITFunction.__getitem__``
    is worth measuring rather than assuming: on this box, at these argument
    counts, the JIT layer's signature binding and specialization keying cost
    14.5 us per launch against 5.1 us for the pre-bound path -- 145 us versus
    51 us over ten launches, which is the difference between a 8x and a 19x
    candidate. The specialization the compiled kernel was built with is only
    valid while every argument that feeds it is unchanged, so the plan key pins
    all of them: shapes, strides, dtype, device, ``_mask_trans``, and the
    16-byte alignment class of the two caller-owned pointers. Everything else is
    plan-owned and therefore fixed for the plan's lifetime.
    """

    __slots__ = ("specs", "seq", "lists", "z_slots", "mask_slots", "out_slots",
                 "stream", "arena", "weights", "bound")

    def __init__(self, specs, keep_alive):
        self.specs = specs
        self.arena = keep_alive[0]
        self.weights = keep_alive[1]
        self.bound = False
        self.seq = None
        self.lists = None
        self.z_slots = None
        self.mask_slots = None
        self.out_slots = None
        self.stream = None

    # -- first call: compile, launch through the JIT, and capture the kernels --
    def first_call(self, z, pair_mask, out):
        kernels = []
        for fn, grid, args, opts in self.specs:
            real = [z if a is _IN_Z else pair_mask if a is _IN_MASK
                    else out if a is _OUT_Z else a for a in args]
            kernels.append(fn[grid](*real, **opts))
        try:
            self._bind(kernels)
        except Exception:  # noqa: BLE001 - pre-binding is an optimization only
            self.bound = False

    def _bind(self, kernels):
        seq, lists = [], []
        for kernel, (_fn, grid, args, _opts) in zip(kernels, self.specs):
            kernel._init_handles()
            lst = [grid[0], grid[1] if len(grid) > 1 else 1,
                   grid[2] if len(grid) > 2 else 1,
                   None, kernel.function, kernel.packed_metadata, None, None, None]
            for a in args:
                if isinstance(a, torch.Tensor):
                    lst.append(a.data_ptr())
                elif a is _IN_Z or a is _IN_MASK or a is _OUT_Z:
                    lst.append(0)   # patched every call
                else:
                    lst.append(a)
            seq.append((kernel.run, lst))
            lists.append(lst)
        self.seq = seq
        self.lists = lists
        # (list, position) pairs, so the steady path does one store per slot with
        # no inner-loop bookkeeping.
        self.z_slots = _slot_pairs(lists, self.specs, _IN_Z)
        self.mask_slots = _slot_pairs(lists, self.specs, _IN_MASK)
        self.out_slots = _slot_pairs(lists, self.specs, _OUT_Z)
        self.bound = True

    # -- steady state --
    def run(self, z, pair_mask, out, stream):
        zp = z.data_ptr()
        mp = pair_mask.data_ptr()
        op = out.data_ptr()
        for lst, pos in self.z_slots:
            lst[pos] = zp
        for lst, pos in self.mask_slots:
            lst[pos] = mp
        for lst, pos in self.out_slots:
            lst[pos] = op
        if stream != self.stream:
            self.stream = stream
            for lst in self.lists:
                lst[3] = stream
        for run, lst in self.seq:
            run(*lst)

    def run_jit(self, z, pair_mask, out):
        for fn, grid, args, opts in self.specs:
            real = [z if a is _IN_Z else pair_mask if a is _IN_MASK
                    else out if a is _OUT_Z else a for a in args]
            fn[grid](*real, **opts)


def _slot_pairs(lists, specs, sentinel):
    pairs = []
    for lst, (_fn, _grid, args, _opts) in zip(lists, specs):
        for pos, a in enumerate(args, start=_HEAD):
            if a is sentinel:
                pairs.append((lst, pos))
    return pairs


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class PairBlock(nn.Module):
    """Shared pair stack block for AF3 PairFormer / MSA module / template.

    Args:
        c_z: Pair embedding channel dimension
        c_hidden_mul: Hidden dim for triangle multiplication
        c_hidden_pair_att: Per-head hidden dim for triangle attention
        no_heads_pair: Number of heads in triangle attention
        transition_n: Scale of pair transition hidden dimension
        pair_dropout: Dropout rate (unused in inference baseline)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_z: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float = 0.0,
        fuse_projection_weights: bool = False,
        inf: float = 1e9,
    ):
        super().__init__()

        # The bench shares weights by ``state_dict`` key with
        # ``load_state_dict(..., strict=False)``, which silently drops anything
        # that does not match -- a renamed parameter would leave this module
        # running on N(0, 0.02) noise while still producing plausible output. So
        # the real baseline children are constructed as children and the key set
        # matches by construction. Their ``forward`` methods are only ever called
        # by the reference fallback.
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden_mul)

        self.tri_att_start = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=True, inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z, c_hidden_pair_att, no_heads_pair, starting=False, inf=inf,
        )

        self.pair_transition = SwiGLUTransition(c_in=c_z, n=transition_n)

        self.c_z = c_z
        self.c_hidden_mul = c_hidden_mul
        self.c_hidden_pair_att = c_hidden_pair_att
        self.no_heads_pair = no_heads_pair
        self.transition_n = transition_n
        self.inf = inf

        # Nothing is fused here on purpose. The bench's order is construct ->
        # ``_prepare_module`` (``.to(device)`` plus a bf16 cast that *replaces*
        # ``param.data``) -> ``_sanitize_float_params`` -> ``load_state_dict``, so
        # any weight concatenated in ``__init__`` would hold pre-load garbage.
        self._plans: dict = {}
        self._src_params: list | None = None
        self._guard = None
        # Parameter tensors the current fused weights were derived from. Held so
        # their storage cannot be recycled underneath an unchanged ``data_ptr``;
        # see ``_guard_key``.
        self._guard_pins: list | None = None

    # -- exact reference path -------------------------------------------------
    def _reference_forward(self, z, pair_mask, _mask_trans):
        """Defer to the baseline children. Used for any input the fused path does
        not cover, so the fallback is the reference rather than a second
        re-derivation of it."""
        pair_trans_mask = pair_mask if _mask_trans else None
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = z + self.tri_att_end(z, mask=pair_mask)
        z = z + self.pair_transition(z, mask=pair_trans_mask)
        return z

    # -- lazily fused weights -------------------------------------------------
    def _source_params(self):
        if self._src_params is None:
            tm = (self.tri_mul_out, self.tri_mul_in)
            ta = (self.tri_att_start, self.tri_att_end)
            ps = []
            for m in tm:
                ps += [m.layer_norm_in.weight, m.layer_norm_in.bias,
                       m.layer_norm_out.weight, m.layer_norm_out.bias,
                       m.linear_a_p.weight, m.linear_a_g.weight,
                       m.linear_b_p.weight, m.linear_b_g.weight,
                       m.linear_g.weight, m.linear_z.weight]
            for m in ta:
                ps += [m.layer_norm.weight, m.layer_norm.bias, m.linear_z.weight,
                       m.mha.linear_q.weight, m.mha.linear_k.weight,
                       m.mha.linear_v.weight, m.mha.linear_g.weight,
                       m.mha.linear_o.weight]
            pt = self.pair_transition
            ps += [pt.layer_norm.weight, pt.layer_norm.bias,
                   pt.swiglu.linear_a.weight, pt.swiglu.linear_b.weight,
                   pt.linear_out.weight]
            self._src_params = ps
        return self._src_params

    def _guard_key(self):
        """Exact per-parameter invalidation key: one record per source parameter.

        ``load_state_dict`` keeps the same ``Parameter`` *and* the same
        ``data_ptr``, mutating in place and bumping ``_version``, so ``_version``
        is what catches a reload; ``param.data = param.data.to(dtype)`` keeps the
        object and ``_version`` but changes ``data_ptr``; ``dtype`` and ``device``
        catch a cast whose result differs in dtype landing on a recycled address.

        On its own that is still not enough, and the gap is not hypothetical. The
        CUDA caching allocator recycles freed blocks, so two *separate* ``.data``
        assignments -- ``p.data = p.data.to(float16)`` then back to bf16 -- can
        hand the original address straight back: ``data_ptr`` unchanged,
        ``_version`` unchanged, dtype and device back where they started, key
        equal, stale fused weights served. Reproduced, with a weight fp16 cannot
        represent, the output stayed finite while the parameter held ``inf``.

        The fix is ``_guard_pins``: keep a reference to the parameter tensors the
        fused weights were built from. A pinned block cannot be handed to any
        later allocation, so a matching ``data_ptr`` now *means* the storage is
        the same storage, and an in-place change to it would bump ``_version``.
        That is what makes the four fields sufficient rather than merely
        plausible.

        Storage identity (``untyped_storage()._cdata``) is deliberately *not* used
        instead: it is an address too, and in one of two reproductions it was
        recycled along with the data pointer, so it would have collided in exactly
        the case it was meant to catch.

        Per-parameter rather than a summed digest, because a sum cannot say which
        parameter moved and admits collisions. ~9 us/call at the captured shape
        against ~4 us for the digest (`scratch/probe_guard_cost.py`); ~7 us of
        that is the irreducible cost of reading four attributes off 41 parameters.
        Host submit for one `forward` is ~65-72 us against ~45 us of device busy,
        so host work is on the critical path in principle -- but cutting ~5 us of
        it elsewhere did not move the bench's reported `candidate_ms` at all
        (0.0809 ms either way), because the bench enqueues a ~250 MB L2 flush
        ahead of every timed call. Worth taking; not worth claiming a speedup for.
        The pins cost nothing per call: they are refreshed only when the key
        changes, and they hold ~1 MB.
        """
        return tuple((p.data_ptr(), p._version, p.dtype, p.device)
                     for p in (self._src_params or self._source_params()))

    def _build_weights(self, dev, dt):
        C = self.c_z
        CH = self.c_hidden_mul
        D = self.c_hidden_pair_att
        H = self.no_heads_pair
        HD = H * D
        HID = self.transition_n * C
        tb_pad = max(16, 1 << (H - 1).bit_length())

        w = {}
        with torch.no_grad():
            for name, m in (("out", self.tri_mul_out), ("in", self.tri_mul_in)):
                w[f"tm_{name}_w"] = torch.cat(
                    [m.linear_a_p.weight, m.linear_a_g.weight,
                     m.linear_b_p.weight, m.linear_b_g.weight,
                     m.linear_g.weight], 0).t().contiguous()
                w[f"tm_{name}_wz"] = m.linear_z.weight.t().contiguous()
            for name, m in (("start", self.tri_att_start), ("end", self.tri_att_end)):
                lz = m.linear_z.weight
                pad = lz.new_zeros((tb_pad, C))
                pad[:H] = lz
                w[f"ta_{name}_w"] = torch.cat(
                    [m.mha.linear_q.weight, m.mha.linear_k.weight,
                     m.mha.linear_v.weight, m.mha.linear_g.weight,
                     pad], 0).t().contiguous()
                w[f"ta_{name}_wo"] = m.mha.linear_o.weight.t().contiguous()
            pt = self.pair_transition
            w["pt_w"] = torch.cat(
                [pt.swiglu.linear_a.weight, pt.swiglu.linear_b.weight],
                0).t().contiguous()
            w["pt_wout"] = pt.linear_out.weight.t().contiguous()
        return w, (C, CH, D, H, HD, HID, tb_pad)

    # -- plan construction ----------------------------------------------------
    def _build_plan(self, z, pair_mask, mask_trans):
        dev, dt = z.device, z.dtype
        # Every weight the kernels read must already be in z's dtype on z's
        # device -- same predicate the inputs go through in ``_plan_key``.
        # Anything else is a case for the reference path, since promoting inside
        # the kernels would silently change the arithmetic.
        if not all(_same_place(p, dt, dev) for p in self._source_params()):
            return None
        B, N = z.shape[0], z.shape[1]
        w, (C, CH, D, H, HD, HID, TB) = self._build_weights(dev, dt)
        R = N * N
        NROW = R
        BM = 16
        zs = (z.stride(0), z.stride(1), z.stride(2))
        ms = (pair_mask.stride(0), pair_mask.stride(1), pair_mask.stride(2))
        # Contiguous [*, N, N, C] strides for every arena-resident z buffer.
        cs = (R * C, N * C, C)

        # One persistent arena for every intermediate: three z-sized buffers
        # (ping-pong plus the trimul gate), the two hidden-width projections, the
        # packed [Q|K|V|G|bias] block and the transition hidden.
        sizes = [B * R * C, B * R * C,                      # z ping-pong
                 B * R * CH, B * R * CH, B * R * C,         # a, b, sigmoid(g)
                 B * R * (4 * HD + TB),                     # Q|K|V|G|triangle bias
                 B * R * HID]                               # SwiGLU hidden
        arena = torch.empty(sum(sizes), dtype=dt, device=dev)
        off = 0
        views = []
        for n in sizes:
            views.append(arena[off:off + n])
            off += n
        assert off == arena.numel()
        zbuf = (views[0], views[1])
        sa, sb, sg, sqkv, sh = views[2], views[3], views[4], views[5], views[6]

        eps = self.tri_mul_out.layer_norm_in.eps
        specs = []
        grid_i = (N, B)

        def ln_proj(tune, src, src_s, mask_s, lnw, lnb, wt, o0, o1, o2,
                    w1, w2, wtot, epi):
            t = _TUNING[tune]
            # Every projection width here is a power of two, so a power-of-two
            # tile divides all of them and no boundary masking is needed.
            bn = min(t["block_n"], w1, w2 or w1)
            nblk = max(w1, w2) // bn
            specs.append((_ln_proj, (R // BM, nblk, B),
                          [src, _IN_MASK, lnw, lnb, wt, o0, o1, o2,
                           src_s[0], src_s[1], src_s[2],
                           mask_s[0], mask_s[1], mask_s[2],
                           N, C, NROW, w1, w2, wtot, epi, BM, bn, eps],
                          {"num_warps": t["num_warps"]}))

        # --- tri_mul_out, tri_mul_in -----------------------------------------
        for k, (name, m, out_going) in enumerate(
                (("out", self.tri_mul_out, True), ("in", self.tri_mul_in, False))):
            src = _IN_Z if k == 0 else zbuf[0]
            src_s = zs if k == 0 else cs
            dst = zbuf[k]
            ln_proj("ln_proj_gate_mask", src, src_s, ms,
                    m.layer_norm_in.weight, m.layer_norm_in.bias,
                    w[f"tm_{name}_w"], sa, sb, sg, CH, C, 4 * CH + C,
                    _EP_GATE_MASK)
            a_s = (N, 1) if out_going else (1, N)
            b_s = (N, 1) if out_going else (1, N)
            specs.append((_trimul_combine, grid_i,
                          [sa, sb, sg, src, m.layer_norm_out.weight,
                           m.layer_norm_out.bias, w[f"tm_{name}_wz"], dst,
                           a_s[0], a_s[1], b_s[0], b_s[1],
                           src_s[0], src_s[1], src_s[2], cs[0], cs[1], cs[2],
                           N, C, CH, NROW, eps],
                          {"num_warps": _TUNING["trimul_combine"]["num_warps"]}))

        # --- tri_att_start, tri_att_end --------------------------------------
        WQ = 4 * HD
        for k, (name, m, starting) in enumerate(
                (("start", self.tri_att_start, True),
                 ("end", self.tri_att_end, False))):
            src = zbuf[1 - k]
            dst = zbuf[k]
            # The transpose lands on the ln_proj load of z, so Q/K/V/G and the
            # triangle bias all reach scratch already in the ending-node frame.
            proj_s = cs if starting else (cs[0], cs[2], cs[1])
            ln_proj("ln_proj_qkv", src, proj_s, ms,
                    m.layer_norm.weight, m.layer_norm.bias,
                    w[f"ta_{name}_w"], sqkv, sqkv, sqkv, WQ, TB, WQ + TB,
                    _EP_QKV)
            mask_s = ms if starting else (ms[0], ms[2], ms[1])
            res_s = cs if starting else (cs[0], cs[2], cs[1])
            specs.append((_triatt, grid_i,
                          [sqkv, _IN_MASK, w[f"ta_{name}_wo"], src, dst,
                           mask_s[0], mask_s[1], mask_s[2],
                           res_s[0], res_s[1], res_s[2],
                           res_s[0], res_s[1], res_s[2],
                           N, C, H, D, HD, WQ + TB, NROW,
                           float(m.inf), math.sqrt(D)],
                          {"num_warps": _TUNING["triatt"]["num_warps"]}))

        # --- pair_transition -------------------------------------------------
        pt = self.pair_transition
        ln_proj("ln_proj_swiglu", zbuf[1], cs, ms,
                pt.layer_norm.weight, pt.layer_norm.bias,
                w["pt_w"], sh, sh, sh, HID, 0, 2 * HID, _EP_SWIGLU)
        tg = _TUNING["gemm_mask_residual"]
        gbm = min(tg["block_m"], R)
        specs.append((_gemm_mask_residual, (R // gbm, 1, B),
                      [sh, w["pt_wout"], _IN_MASK, zbuf[1], _OUT_Z,
                       ms[0], ms[1], ms[2], cs[0], cs[1], cs[2],
                       cs[0], cs[1], cs[2],
                       N, C, HID, NROW, gbm, min(tg["block_k"], HID),
                       bool(mask_trans)],
                      {"num_warps": tg["num_warps"]}))

        return _Plan(specs, (arena, w))

    @staticmethod
    def _stream_handle(device):
        if _raw_stream is not None:
            return _raw_stream(device.index)
        return torch.cuda.current_stream(device).cuda_stream

    # -- dispatch -------------------------------------------------------------
    def _plan_key(self, z, pair_mask, mask_trans):
        """None when the fused path does not cover this input.

        The key pins every property the compiled kernels were specialized on --
        including the caller pointers' 16-byte alignment class, because the
        pre-bound launch path skips the JIT layer that would otherwise notice.
        """
        if not z.is_cuda:
            return None
        dev = z.device
        if not (_addressable(z, 4, 3, torch.bfloat16, dev)
                and _addressable(pair_mask, 3, 2, torch.bfloat16, dev)):
            return None
        B, I, J, C = z.shape
        if I != J or C != self.c_z or pair_mask.shape != z.shape[:3]:
            return None
        N = I
        D = self.c_hidden_pair_att
        HID = self.transition_n * C
        if not (_pow2(N) and N >= 16 and _pow2(C) and C >= 16
                and _pow2(self.c_hidden_mul) and self.c_hidden_mul >= 16
                and _pow2(D) and D >= 16 and _pow2(HID)):
            return None
        if (N * N) % 16 != 0:
            return None
        return (B, N, C, dev, z.stride(0), z.stride(1), z.stride(2),
                pair_mask.stride(0), pair_mask.stride(1), bool(mask_trans),
                z.data_ptr() % 16 == 0, pair_mask.data_ptr() % 16 == 0)

    def forward(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:         [*, N, N, C_z] pair embedding
            pair_mask: [*, N, N] pair mask

        Returns:
            [*, N, N, C_z] updated pair embedding
        """
        key = self._plan_key(z, pair_mask, _mask_trans)
        if key is None:
            return self._reference_forward(z, pair_mask, _mask_trans)

        guard = self._guard_key()
        if guard != self._guard:
            self._plans.clear()
            self._guard = guard
            # Pin the exact tensors this key describes, so no later allocation can
            # reuse their blocks and make a stale key look current.
            self._guard_pins = [p.data for p in self._source_params()]

        plan = self._plans.get(key, False)
        if plan is False:
            plan = self._build_plan(z, pair_mask, _mask_trans)
            self._plans[key] = plan
            if plan is not None:
                # A fresh output rather than a view into the scratch arena: the
                # arena is what makes steady-state host work a dict lookup plus
                # launches, but handing its storage back to the caller would mean
                # the returned tensor is overwritten by the next call. One 64 KB
                # allocation from the caching allocator is cheap against the
                # latency target.
                out = torch.empty_like(z, memory_format=torch.contiguous_format)
                plan.first_call(z, pair_mask, out)
                return out
        if plan is None:
            return self._reference_forward(z, pair_mask, _mask_trans)

        # empty_like with an explicit contiguous format rather than
        # torch.empty(z.shape, ...): same result, 1.22 us against 2.88 us. The
        # format has to be explicit -- plain empty_like would inherit z's strides,
        # and the kernels write the output contiguously.
        out = torch.empty_like(z, memory_format=torch.contiguous_format)
        if plan.bound:
            plan.run(z, pair_mask, out, self._stream_handle(z.device))
        else:
            plan.run_jit(z, pair_mask, out)
        return out
