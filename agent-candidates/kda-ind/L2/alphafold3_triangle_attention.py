"""Triangle attention for AlphaFold3 (L2), fused into two Triton kernels.

Implements AF3 Algorithms 14 (starting node) and 15 (ending node), matching
``baseline.py`` within the harness's bf16 tolerance. Individual runs are often
bit-identical, but that is not an invariant: the per-head output-projection
decomposition changes the fp32 summation order, so the observed max absolute
error ranges from 0 to about 1.2e-4 depending on the run and on which cuBLAS
kernel the baseline happens to pick.

The baseline dispatches 23 CUDA kernels per call for ~42 MFLOP of arithmetic, so
it is bound by launches and latency rather than by math. Everything here is
aimed at collapsing those dispatches:

* ``_pair_bias_kernel`` -- LayerNorm + ``linear_z``, producing the triangle bias.
* ``_row_attention_kernel`` -- LayerNorm + the q/k/v/g projections + the
  triangle-biased masked softmax + the sigmoid gate + the output projection,
  one block per output row of the pair representation.
* ``_row_attention_large_kernel`` -- the same for a pair axis too large to hold an
  ``[N, N]`` score tile, tiling the key axis with a two-pass softmax.

Two kernels rather than one because the triangle bias couples rows:
``z[jq, jk, h]`` is read from row ``jq`` of the pair representation, so output
row ``i`` depends on every row, while ``q``/``k``/``v``/``g`` for output row
``i`` depend on row ``i`` alone. That single cross-row dependency is the only
thing that does not fuse. A single-kernel version *is* implementable -- it was
built and measured in ``profile/measure_variants.py`` -- but it makes every block
hold a whole ``[N*N, C]`` LayerNorm tile, which costs ~10 us more GPU time than
the second launch saves on the host, so it loses end-to-end (35.8 us against
33.0 us) and only exists for small ``N``.

Rounding follows the baseline's dtype at every rounding point rather than being
more accurate than it: the target is *agreement* with a bf16 reference, so extra
internal precision is a divergence risk, not a safety margin. The LayerNorm
reduction runs in fp32 (the L1 ``LayerNorm`` promotes) and every matmul
accumulates in fp32, but each result is rounded back to the input dtype exactly
where the baseline rounds it -- including the two bias additions, which happen
in the input dtype and in the baseline's order (mask bias, then triangle bias).

Inputs outside the kernels' proven validity domain go to ``_forward_reference``,
which reproduces the baseline body through the same submodules, so the class
stays a drop-in replacement rather than a shape-specific kernel. "Proven" is
literal: the accepted tile geometries are an allowlist of tuples that
``profile/map_launch_envelope.py`` has compiled and compared against the
baseline, because the kernels' resource limits couple the tile axes and bounding
them independently admitted combinations that cannot launch at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_of3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: tuple[int, ...]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
# Both kernels address the pair representation through explicit strides. Only
# the channel axis is required to be contiguous; the (I, J) axes are not, which
# is what makes the ending-node path (where ``x`` arrives transposed) work
# without a copy. Weight tiles are read with the contraction axis carrying
# stride 1, so no host-side repacking is needed either.
#
# Padding is neutral by construction. A padded channel lane loads as zero and
# picks up a zero LayerNorm bias, so it contributes nothing to any contraction.
# A padded key column is forced to -inf *before* the row max and the exponential
# sum, so it carries exactly zero probability mass; leaving it at zero would
# corrupt the softmax normalization instead. The contraction axis of every dot
# is at least 16, which is the NVIDIA backend's minimum for 16-bit operands.


@triton.jit
def _pair_bias_kernel(
    x_ptr, ln_w_ptr, ln_b_ptr, wz_ptr, zb_ptr,
    sx_b, sx_i, sx_j,
    sln_w, sln_b,
    swz_o, swz_i,
    n_pair, N, C, H, eps,
    BT: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr, DT: tl.constexpr,
):
    """LayerNorm + ``linear_z`` -> ``zb[b, h, jq * N + jk]``, in the input dtype.

    ``zb`` holds exactly the values the baseline's ``triangle_bias`` holds. The
    flat pair index ``t = jq * N + jk`` is already the linear index of the
    (jq, jk) entry, so each head's slice of ``zb`` is a plain contiguous
    ``[N, N]`` tile and the attention kernel needs no reshape to read it.
    """
    pb = tl.program_id(0)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    tm = t < n_pair
    jq = t // N
    jk = t - jq * N

    c = tl.arange(0, BC)
    cm = c < C
    xv = tl.load(
        x_ptr + pb * sx_b + jq[:, None] * sx_i + jk[:, None] * sx_j + c[None, :],
        mask=tm[:, None] & cm[None, :], other=0.0,
    ).to(tl.float32)

    mean = tl.sum(xv, 1) / C
    xc = tl.where(cm[None, :], xv - mean[:, None], 0.0)
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    lw = tl.load(ln_w_ptr + c * sln_w, mask=cm, other=0.0).to(tl.float32)
    lb = tl.load(ln_b_ptr + c * sln_b, mask=cm, other=0.0).to(tl.float32)
    xln = (xc * rstd[:, None] * lw[None, :] + lb[None, :]).to(DT)

    h = tl.arange(0, BH)
    hm = h < H
    wz = tl.load(
        wz_ptr + h[None, :] * swz_o + c[:, None] * swz_i,
        mask=cm[:, None] & hm[None, :], other=0.0,
    )
    z = tl.dot(xln, wz).to(DT)

    tl.store(
        zb_ptr + pb * H * n_pair + h[None, :] * n_pair + t[:, None], z,
        mask=tm[:, None] & hm[None, :],
    )


@triton.jit
def _row_attention_kernel(
    x_ptr, mask_ptr, zb_ptr, out_ptr,
    ln_w_ptr, ln_b_ptr,
    wq_ptr, wk_ptr, wv_ptr, wg_ptr, wo_ptr,
    sx_b, sx_i, sx_j,
    sm_b, sm_i, sm_j,
    so_b, so_i, so_j,
    sln_w, sln_b,
    swq_o, swq_i, swk_o, swk_i, swv_o, swv_i, swg_o, swg_i, swo_o, swo_i,
    n_pair, N, C, CH, H, eps, inf, sqrt_ch,
    BN: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr, DT: tl.constexpr,
):
    """One block per output row ``i`` of the pair representation.

    Recomputes ``LayerNorm(x[b, i])`` instead of reading a materialized
    intermediate: that is one reduction over ``C`` elements against an
    ``[N*N, C]`` round trip plus an allocation plus a launch.

    The heads run in a dynamic loop rather than an unrolled one so the compiler
    cannot hoist all five weight tiles of every head into registers at once. The
    MAC count is the same as projecting ``[N, H*CH]`` once and slicing out each
    head, but that slice would be at a loop-varying column offset, which Triton
    cannot express against a register tile -- so the per-head projection is not a
    trade, just the form the language allows.
    """
    pid = tl.program_id(0)
    pb = pid // N
    i = pid % N

    j = tl.arange(0, BN)
    jm = j < N
    c = tl.arange(0, BC)
    cm = c < C
    d = tl.arange(0, BD)
    dm = d < CH

    xv = tl.load(
        x_ptr + pb * sx_b + i * sx_i + j[:, None] * sx_j + c[None, :],
        mask=jm[:, None] & cm[None, :], other=0.0,
    ).to(tl.float32)
    mean = tl.sum(xv, 1) / C
    xc = tl.where(cm[None, :], xv - mean[:, None], 0.0)
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    lw = tl.load(ln_w_ptr + c * sln_w, mask=cm, other=0.0).to(tl.float32)
    lb = tl.load(ln_b_ptr + c * sln_b, mask=cm, other=0.0).to(tl.float32)
    xln = (xc * rstd[:, None] * lw[None, :] + lb[None, :]).to(DT)

    # inf * (mask - 1), formed in the input dtype exactly as the baseline forms
    # it. Padded key columns load as 1 so they contribute a zero bias; the -inf
    # applied to the logits below is what actually neutralizes them.
    mv = tl.load(
        mask_ptr + pb * sm_b + i * sm_i + j * sm_j, mask=jm, other=1.0,
    ).to(tl.float32)
    mb = ((mv - 1.0).to(DT).to(tl.float32) * inf).to(DT)

    acc = tl.zeros([BN, BC], dtype=tl.float32)
    for h in range(H):
        hd = h * CH + d

        wq = tl.load(wq_ptr + hd[None, :] * swq_o + c[:, None] * swq_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        # The baseline scales q after rounding the projection, in the input
        # dtype; scaling the fp32 scores instead would be more accurate and
        # therefore less faithful.
        qh = (tl.dot(xln, wq).to(DT).to(tl.float32) / sqrt_ch).to(DT)

        wk = tl.load(wk_ptr + hd[None, :] * swk_o + c[:, None] * swk_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        kh = tl.dot(xln, wk).to(DT)

        s = tl.dot(qh, tl.trans(kh)).to(DT)
        s = (s.to(tl.float32) + mb[None, :].to(tl.float32)).to(DT)
        zh = tl.load(
            zb_ptr + pb * H * n_pair + h * n_pair + j[:, None] * N + j[None, :],
            mask=jm[:, None] & jm[None, :], other=0.0,
        )
        s = (s.to(tl.float32) + zh.to(tl.float32)).to(DT)

        sf = tl.where(jm[None, :], s.to(tl.float32), float("-inf"))
        sf = sf - tl.max(sf, 1)[:, None]
        e = tl.exp(sf)
        p = (e / tl.sum(e, 1)[:, None]).to(DT)

        wv = tl.load(wv_ptr + hd[None, :] * swv_o + c[:, None] * swv_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        oh = tl.dot(p, tl.dot(xln, wv).to(DT)).to(DT)

        wg = tl.load(wg_ptr + hd[None, :] * swg_o + c[:, None] * swg_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        gh = tl.sigmoid(tl.dot(xln, wg).to(DT).to(tl.float32)).to(DT)
        oh = (oh.to(tl.float32) * gh.to(tl.float32)).to(DT)

        wo = tl.load(wo_ptr + c[None, :] * swo_o + hd[:, None] * swo_i,
                     mask=dm[:, None] & cm[None, :], other=0.0)
        # One fp32 accumulator across all heads is a single reduction over the
        # baseline's full K = H * CH, just in a different summation order.
        acc += tl.dot(oh, wo)

    tl.store(
        out_ptr + pb * so_b + i * so_i + j[:, None] * so_j + c[None, :],
        acc.to(DT), mask=jm[:, None] & cm[None, :],
    )


@triton.jit
def _row_attention_large_kernel(
    x_ptr, mask_ptr, zb_ptr, out_ptr,
    ln_w_ptr, ln_b_ptr,
    wq_ptr, wk_ptr, wv_ptr, wg_ptr, wo_ptr,
    sx_b, sx_i, sx_j,
    sm_b, sm_i, sm_j,
    so_b, so_i, so_j,
    sln_w, sln_b,
    swq_o, swq_i, swk_o, swk_i, swv_o, swv_i, swg_o, swg_i, swo_o, swo_i,
    n_pair, N, C, CH, H, eps, inf, sqrt_ch, n_qtiles,
    BQ: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr, BD: tl.constexpr,
    DT: tl.constexpr,
):
    """Large pair axis: one block per (batch, output row, query tile).

    The small kernel holds a whole ``[N, N]`` score tile, which is what bounds it
    to ``N <= 128``. Here the key axis is tiled instead, so nothing of size
    ``N * N`` is ever materialized.

    **Two passes over the key tiles**, not the usual single online-softmax pass.
    The rounding schedule requires the probabilities to be rounded to the input
    dtype *after* the final normalization, exactly where the baseline's
    ``F.softmax`` rounds them -- and the normalizer is not known until every key
    tile has been seen. A one-pass weighted-value accumulator would have to either
    rescale already-rounded probabilities (changing where rounding happens) or
    keep them in fp32 (more accurate than the baseline, which is the divergence
    risk this whole schedule exists to avoid). So pass one accumulates the fp32
    running max and normalizer from logits rounded exactly as the small kernel
    rounds them, and pass two recomputes each tile and forms the final rounded
    probabilities. The key projections are computed twice; correctness wins.
    """
    pid = tl.program_id(0)
    per_batch = N * n_qtiles
    pb = pid // per_batch
    rem = pid % per_batch
    i = rem // n_qtiles
    qt = rem % n_qtiles

    jq = qt * BQ + tl.arange(0, BQ)
    qm = jq < N
    c = tl.arange(0, BC)
    cm = c < C
    d = tl.arange(0, BD)
    dm = d < CH

    lw = tl.load(ln_w_ptr + c * sln_w, mask=cm, other=0.0).to(tl.float32)
    lb = tl.load(ln_b_ptr + c * sln_b, mask=cm, other=0.0).to(tl.float32)

    xq = tl.load(
        x_ptr + pb * sx_b + i * sx_i + jq[:, None] * sx_j + c[None, :],
        mask=qm[:, None] & cm[None, :], other=0.0,
    ).to(tl.float32)
    mq = tl.sum(xq, 1) / C
    cq = tl.where(cm[None, :], xq - mq[:, None], 0.0)
    vq = tl.sum(cq * cq, 1) / C
    xln_q = (cq * (1.0 / tl.sqrt(vq + eps))[:, None] * lw[None, :]
             + lb[None, :]).to(DT)

    n_ktiles = tl.cdiv(N, BK)
    acc = tl.zeros([BQ, BC], dtype=tl.float32)
    for h in range(H):
        hd = h * CH + d
        wq = tl.load(wq_ptr + hd[None, :] * swq_o + c[:, None] * swq_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        qh = (tl.dot(xln_q, wq).to(DT).to(tl.float32) / sqrt_ch).to(DT)
        wg = tl.load(wg_ptr + hd[None, :] * swg_o + c[:, None] * swg_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        gh = tl.sigmoid(tl.dot(xln_q, wg).to(DT).to(tl.float32)).to(DT)
        wk = tl.load(wk_ptr + hd[None, :] * swk_o + c[:, None] * swk_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)

        # Pass one: fp32 running max and normalizer over all key tiles. wv is
        # loaded inside pass two rather than here: hoisting it would keep a fifth
        # weight tile live across both loops, and at BLOCK_C=128 that was enough to
        # push this kernel to 255 registers and 2.0 M spill requests.
        run_max = tl.full([BQ], float("-inf"), tl.float32)
        run_sum = tl.zeros([BQ], dtype=tl.float32)
        for kt in range(n_ktiles):
            jk = kt * BK + tl.arange(0, BK)
            km = jk < N
            xk = tl.load(
                x_ptr + pb * sx_b + i * sx_i + jk[:, None] * sx_j + c[None, :],
                mask=km[:, None] & cm[None, :], other=0.0,
            ).to(tl.float32)
            mk = tl.sum(xk, 1) / C
            ck = tl.where(cm[None, :], xk - mk[:, None], 0.0)
            vk = tl.sum(ck * ck, 1) / C
            xln_k = (ck * (1.0 / tl.sqrt(vk + eps))[:, None] * lw[None, :]
                     + lb[None, :]).to(DT)
            kh = tl.dot(xln_k, wk).to(DT)
            s = tl.dot(qh, tl.trans(kh)).to(DT)
            mv = tl.load(mask_ptr + pb * sm_b + i * sm_i + jk * sm_j,
                         mask=km, other=1.0).to(tl.float32)
            mb = ((mv - 1.0).to(DT).to(tl.float32) * inf).to(DT)
            s = (s.to(tl.float32) + mb[None, :].to(tl.float32)).to(DT)
            zh = tl.load(
                zb_ptr + pb * H * n_pair + h * n_pair + jq[:, None] * N + jk[None, :],
                mask=qm[:, None] & km[None, :], other=0.0,
            )
            s = (s.to(tl.float32) + zh.to(tl.float32)).to(DT)
            sf = tl.where(km[None, :], s.to(tl.float32), float("-inf"))
            new_max = tl.maximum(run_max, tl.max(sf, 1))
            # The first tile always has at least one valid column, so new_max is
            # finite from the start and exp(-inf - finite) is 0 rather than NaN.
            run_sum = (run_sum * tl.exp(run_max - new_max)
                       + tl.sum(tl.exp(sf - new_max[:, None]), 1))
            run_max = new_max

        # Pass two: recompute each tile and form the final rounded probabilities.
        wv = tl.load(wv_ptr + hd[None, :] * swv_o + c[:, None] * swv_i,
                     mask=cm[:, None] & dm[None, :], other=0.0)
        oh = tl.zeros([BQ, BD], dtype=tl.float32)
        for kt in range(n_ktiles):
            jk = kt * BK + tl.arange(0, BK)
            km = jk < N
            xk = tl.load(
                x_ptr + pb * sx_b + i * sx_i + jk[:, None] * sx_j + c[None, :],
                mask=km[:, None] & cm[None, :], other=0.0,
            ).to(tl.float32)
            mk = tl.sum(xk, 1) / C
            ck = tl.where(cm[None, :], xk - mk[:, None], 0.0)
            vk = tl.sum(ck * ck, 1) / C
            xln_k = (ck * (1.0 / tl.sqrt(vk + eps))[:, None] * lw[None, :]
                     + lb[None, :]).to(DT)
            kh = tl.dot(xln_k, wk).to(DT)
            s = tl.dot(qh, tl.trans(kh)).to(DT)
            mv = tl.load(mask_ptr + pb * sm_b + i * sm_i + jk * sm_j,
                         mask=km, other=1.0).to(tl.float32)
            mb = ((mv - 1.0).to(DT).to(tl.float32) * inf).to(DT)
            s = (s.to(tl.float32) + mb[None, :].to(tl.float32)).to(DT)
            zh = tl.load(
                zb_ptr + pb * H * n_pair + h * n_pair + jq[:, None] * N + jk[None, :],
                mask=qm[:, None] & km[None, :], other=0.0,
            )
            s = (s.to(tl.float32) + zh.to(tl.float32)).to(DT)
            sf = tl.where(km[None, :], s.to(tl.float32), float("-inf"))
            p = (tl.exp(sf - run_max[:, None]) / run_sum[:, None]).to(DT)
            vh = tl.dot(xln_k, wv).to(DT)
            # One fp32 accumulator across key tiles is the baseline's single
            # fp32-accumulated einsum, just in a different summation order.
            oh += tl.dot(p, vh)

        og = (oh.to(DT).to(tl.float32) * gh.to(tl.float32)).to(DT)
        wo = tl.load(wo_ptr + c[None, :] * swo_o + hd[:, None] * swo_i,
                     mask=dm[:, None] & cm[None, :], other=0.0)
        acc += tl.dot(og, wo)

    tl.store(
        out_ptr + pb * so_b + i * so_i + jq[:, None] * so_j + c[None, :],
        acc.to(DT), mask=qm[:, None] & cm[None, :],
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------
_FAST_DTYPES = (torch.bfloat16, torch.float16)
_MIN_CONTRACT = 16     # NVIDIA backend minimum contraction for 16-bit dot operands
_MAX_OFFSET = 1 << 30  # keep every kernel offset inside int32

# Launch geometries that have been compiled and matched against the baseline on
# this target. Anything else takes the reference path.
#
# An allowlist rather than independent per-axis maxima, because the resource
# limits couple the axes and bounding them separately admits combinations that
# cannot launch: the weight tiles are triple-buffered by ``num_stages=3``, so
# (16, 256, 32) fits while (64, 256, 32) exhausts shared memory, and
# (128, 512, 16) exhausts tensor memory instead.
#
# The two kernels are allowlisted SEPARATELY because their constexpr tiles are
# disjoint -- ``_pair_bias_kernel`` compiles on (BT, BC, BH) and
# ``_row_attention_kernel`` on (BN, BC, BD), sharing no tile but BC -- so they are
# separate compilations with separate resource limits and the domain is a product
# of two measured sets rather than one 4-D set. That factorization is not assumed:
# ``profile/map_launch_envelope.py`` sweeps each kernel over its own axes (BH at
# *every* supported BC, not just the captured one) and then probes full 4-D
# combinations end to end to confirm it. Evidence, including every rejected
# geometry, is committed at ``profile/analysis/launch_envelope.json`` / ``.csv``:
# 58 row tiles and 24 pair combinations accepted out of 163 probes, with
# 25 cross-product probes and head counts 1..128 all matching. Regenerate that
# script rather than editing these sets by hand -- a hand-written guess at this
# envelope was wrong in both directions.
_SUPPORTED_TILES = frozenset({
    (16, 16, 16),
    (16, 16, 32),
    (16, 16, 64),
    (16, 16, 128),
    (16, 32, 16),
    (16, 32, 32),
    (16, 32, 64),
    (16, 32, 128),
    (16, 64, 16),
    (16, 64, 32),
    (16, 64, 64),
    (16, 64, 128),
    (16, 128, 16),
    (16, 128, 32),
    (16, 128, 64),
    (16, 256, 16),
    (16, 256, 32),
    (16, 512, 16),
    (32, 16, 16),
    (32, 16, 32),
    (32, 16, 64),
    (32, 16, 128),
    (32, 32, 16),
    (32, 32, 32),
    (32, 32, 64),
    (32, 32, 128),
    (32, 64, 16),
    (32, 64, 32),
    (32, 64, 64),
    (32, 64, 128),
    (32, 128, 16),
    (32, 128, 32),
    (32, 128, 64),
    (32, 256, 16),
    (32, 256, 32),
    (32, 512, 16),
    (64, 16, 16),
    (64, 16, 32),
    (64, 16, 64),
    (64, 16, 128),
    (64, 32, 16),
    (64, 32, 32),
    (64, 32, 64),
    (64, 32, 128),
    (64, 64, 16),
    (64, 64, 32),
    (64, 64, 64),
    (64, 128, 16),
    (64, 128, 32),
    (64, 256, 16),
    (128, 16, 16),
    (128, 16, 32),
    (128, 32, 16),
    (128, 32, 32),
    (128, 64, 16),
    (128, 64, 32),
    (128, 128, 16),
    (128, 128, 32),
})
# (BLOCK_C, BLOCK_H) for the pair-bias kernel. Every combination in the swept
# range passes, so this is effectively ``BLOCK_H <= 128``, but it is recorded as
# the measured set so widening it requires new evidence.
_PAIR_TILES = frozenset({
    (16, 16),
    (16, 32),
    (16, 64),
    (16, 128),
    (32, 16),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
    (512, 16),
    (512, 32),
    (512, 64),
    (512, 128),
})
# BT is 16 only when N*N <= 16 (i.e. N <= 4) and 32 otherwise; both are exercised
# (BT=32 throughout the envelope sweep, BT=16 by the N=4 semantics cases).
_PAIR_ROW_BLOCKS = frozenset({16, 32})
_MAX_HEAD_BLOCK = 128

# The large path tiles the key axis, so it is bounded by what has been measured
# rather than by an [N, N] score tile. Query/key tiles are fixed.
_LARGE_MIN_PAIR = 129
_LARGE_MAX_PAIR = 256
_LARGE_BQ = 16
_LARGE_BK = 64
# (BLOCK_C, BLOCK_D) combinations verified for the large kernel.
_LARGE_TILES = frozenset({
    (16, 16), (16, 32), (16, 64),
    (32, 16), (32, 32), (32, 64),
    (64, 16), (64, 32), (64, 64),
    (128, 16), (128, 32), (128, 64),
    (256, 16), (256, 32),
})


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _flat_batch(t: torch.Tensor, keep: int):
    """Collapse ``t``'s leading dims to one ``(size, stride)`` pair, or ``None``.

    ``None`` means the leading dims are not a single strided run, i.e. flattening
    them would need a copy. Size-1 dims carry an arbitrary stride and are skipped
    rather than being allowed to reject a valid layout.
    """
    lead = t.shape[:-keep]
    strides = t.stride()[:len(lead)]
    size, stride = 1, 0
    for k in reversed(range(len(lead))):
        n = lead[k]
        if n == 1:
            continue
        if size == 1:
            stride = strides[k]
        elif strides[k] != stride * size:
            return None
        size *= n
    return size, stride


class TriangleAttention(nn.Module):
    """AF3 Algorithms 14/15: Triangle attention.

    Args:
        c_in: Input channel dimension
        c_hidden: Overall hidden channel dimension (not per-head)
        no_heads: Number of attention heads
        starting: If True, starting node (Alg 14); else ending node (Alg 15)
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        # Submodules stay verbatim. The harness shares weights with
        # ``load_state_dict(..., strict=False)``, which silently drops keys it
        # cannot place, so a renamed or fused parameter would run on
        # uninitialized memory and could still report "correct" by luck.
        self.layer_norm = LayerNorm(c_in)
        self.linear_z = Linear(c_in, no_heads, bias=False)

        self.mha = OF3Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        )

    # -- validity domain ---------------------------------------------------
    def _fast_path_layout(self, x: torch.Tensor, mask: torch.Tensor | None,
                          _skip_tile_check: bool = False):
        """Launch geometry plus weights, or ``None`` for the reference path.

        The conditions are exactly what the kernels are proven to handle, no
        wider. fp32 is excluded on purpose: Triton's default dot input precision
        is tf32 while torch's fp32 matmul is full fp32, and the fp32 comparison
        bound is far tighter than the 16-bit one, so an fp32 fused path would
        need ``input_precision="ieee"`` on every dot and would still carry
        reduction-order risk.

        Every condition is checked live, none is cached from construction. An
        `nn.Module` lets parameters and submodules be replaced at any time, so a
        flag set in ``__init__`` could say "no projection biases" about a module
        that has since been given one, and the kernels would silently drop that
        term. The per-call cost is kept down by fetching each ``Linear`` once and
        reading ``weight`` and ``bias`` off it, rather than walking the attribute
        chain twice.
        """
        dtype = x.dtype
        if dtype not in _FAST_DTYPES or not x.is_cuda:
            return None
        shape = x.shape
        ndim = len(shape)
        if ndim < 3 or x.stride(-1) != 1:
            return None

        # The triangle bias broadcasts its own (I, J) axes onto (jq, jk) across
        # the I axis, which needs a square pair representation. I == 1 also
        # broadcasts but is not worth a special case.
        pair = shape[-3]
        channels = shape[-1]
        if shape[-2] != pair or pair < 1 or channels != self.c_in:
            return None

        # Head geometry comes from the *attention submodule's* live attributes and
        # from the live weight shape, exactly as the baseline derives it:
        # ``_prep_qkv`` splits heads with ``view(..., self.mha.no_heads, -1)`` and
        # scales q by ``math.sqrt(self.mha.c_hidden)``. Those two are independent
        # of each other and of this module's constructor arguments, so reading
        # ``self.c_hidden`` here would silently ignore a mutation the baseline
        # honours. The kernel already takes the head dimension and the scale as
        # separate arguments, so tracking the baseline costs nothing.
        mha = self.mha
        heads = mha.no_heads
        if not isinstance(heads, int) or heads < 1:
            return None
        proj_out = mha.linear_q.weight.shape[0]
        if proj_out % heads:
            return None
        head_dim = proj_out // heads
        scale_dim = mha.c_hidden
        if not isinstance(scale_dim, (int, float)) or scale_dim <= 0:
            return None
        # The per-head dimension is the contraction axis of both the score dot
        # and the output projection, so it carries the same 16 minimum as c_in.
        if channels < _MIN_CONTRACT or head_dim < _MIN_CONTRACT:
            return None

        # The tile geometry must be one that has actually been compiled and
        # matched. Checked before anything else touches Triton, so a rejected
        # geometry never attempts a compile.
        block_n = max(_MIN_CONTRACT, _next_pow2(pair))
        block_c = _next_pow2(channels)
        block_d = max(_MIN_CONTRACT, _next_pow2(head_dim))
        block_h = max(_MIN_CONTRACT, _next_pow2(heads))
        large = pair >= _LARGE_MIN_PAIR
        if not _skip_tile_check:
            if block_h > _MAX_HEAD_BLOCK:
                return None
            if (block_c, block_h) not in _PAIR_TILES:
                return None
            if large:
                if pair > _LARGE_MAX_PAIR:
                    return None
                if (block_c, block_d) not in _LARGE_TILES:
                    return None
            elif (block_n, block_c, block_d) not in _SUPPORTED_TILES:
                return None

        ln = self.layer_norm
        if not ln.promote_fp32 or tuple(ln.normalized_shape) != (channels,):
            return None
        lz = self.linear_z
        lq, lk, lv = mha.linear_q, mha.linear_k, mha.linear_v
        lg, lo = mha.linear_g, mha.linear_o
        if lg is None:
            return None
        # Every projection in this operator is built with bias=False. A bias
        # present here would be a term the kernels do not apply.
        if (lz.bias is not None or lq.bias is not None or lk.bias is not None
                or lv.bias is not None or lg.bias is not None
                or lo.bias is not None):
            return None
        weights = (
            ln.weight, ln.bias, lz.weight,
            lq.weight, lk.weight, lv.weight, lg.weight, lo.weight,
        )
        device = x.device
        for w in weights:
            if w is None or w.dtype is not dtype or w.device != device:
                return None
        # Shapes are read live, not assumed from __init__: the kernels address
        # these tensors through strides, so a transposed or mis-sized weight
        # would be read as plausible garbage rather than raising.
        proj = (proj_out, channels)
        if (ln.weight.shape != (channels,) or ln.bias.shape != (channels,)
                or lz.weight.shape != (heads, channels)
                or lq.weight.shape != proj or lk.weight.shape != proj
                or lv.weight.shape != proj or lg.weight.shape != proj
                or lo.weight.shape != (channels, proj_out)):
            return None

        if ndim == 4:
            batch, x_stride_b = shape[0], x.stride(0)
        else:
            x_batch = _flat_batch(x, 3)
            if x_batch is None:
                return None
            batch, x_stride_b = x_batch
        n_pair = pair * pair
        if batch < 1 or batch * n_pair * max(channels, self.no_heads) > _MAX_OFFSET:
            return None
        # The kernels index with 32-bit arithmetic. The bound above covers the
        # freshly allocated output and bias scratch; ``x`` needs its own, because
        # a strided view can reach much further than its element count suggests.
        stride_i, stride_j = x.stride(-3), x.stride(-2)
        if (x_stride_b * (batch - 1) + stride_i * (pair - 1)
                + stride_j * (pair - 1) + channels > _MAX_OFFSET):
            return None

        if mask is None:
            mask_stride_b = n_pair
        else:
            if mask.dtype is not dtype or mask.device != device:
                return None
            if mask.shape != shape[:-1]:
                return None
            if ndim == 4:
                mask_stride_b = mask.stride(0)
            else:
                mask_batch = _flat_batch(mask, 2)
                if mask_batch is None:
                    return None
                mask_stride_b = mask_batch[1]
            if (mask_stride_b * (batch - 1) + mask.stride(-2) * (pair - 1)
                    + mask.stride(-1) * (pair - 1) > _MAX_OFFSET):
                return None

        # The kernels have no backward. Checking the inputs alone is not enough:
        # nn.Parameter defaults to requires_grad=True.
        if torch.is_grad_enabled():
            if x.requires_grad or (mask is not None and mask.requires_grad):
                return None
            for w in weights:
                if w.requires_grad:
                    return None

        return (batch, x_stride_b, mask_stride_b, pair, channels,
                heads, head_dim, scale_dim, large, weights)

    # -- reference path ----------------------------------------------------
    def _forward_reference(
        self, x: torch.Tensor, mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J] -> [*, 1, H, I, J]
        triangle_bias = _permute_final_dims(self.linear_z(x), (2, 0, 1))
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x

    # -- fused path --------------------------------------------------------
    def _forward_fused(self, x: torch.Tensor, mask: torch.Tensor | None,
                       layout) -> torch.Tensor:
        (batch, x_stride_b, mask_stride_b, pair, channels,
         heads, head_dim, scale_dim, large, weights) = layout
        ln_w, ln_b, wz, wq, wk, wv, wg, wo = weights
        n_pair = pair * pair
        dtype = x.dtype
        eps = self.layer_norm.eps
        x_stride_i, x_stride_j = x.stride(-3), x.stride(-2)

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        # One allocation carved into the output and the triangle-bias scratch:
        # a second torch.empty is ~2 us of the per-call budget, and this operator
        # is bound by host time, not by arithmetic.
        n_out = batch * n_pair * channels
        buf = torch.empty(n_out + batch * heads * n_pair, dtype=dtype,
                          device=x.device)
        out = buf[:n_out].view(x.shape)
        zb = buf[n_out:]

        block_c = _next_pow2(channels)
        tl_dtype = tl.bfloat16 if dtype is torch.bfloat16 else tl.float16
        # Both configurations were picked from an offline sweep (see
        # profile/tune_launch_config.py) rather than from triton.autotune, which
        # would benchmark inside the timed region and can spawn a thread.
        block_t = _MIN_CONTRACT if n_pair <= _MIN_CONTRACT else 32

        _pair_bias_kernel[(batch, triton.cdiv(n_pair, block_t))](
            x, ln_w, ln_b, wz, zb,
            x_stride_b, x_stride_i, x_stride_j,
            ln_w.stride(0), ln_b.stride(0),
            wz.stride(0), wz.stride(1),
            n_pair, pair, channels, heads, eps,
            BT=block_t, BC=block_c,
            BH=max(_MIN_CONTRACT, _next_pow2(heads)), DT=tl_dtype,
            num_warps=4, num_stages=1,
        )
        block_d = max(_MIN_CONTRACT, _next_pow2(head_dim))
        common = (
            x, mask, zb, out, ln_w, ln_b, wq, wk, wv, wg, wo,
            x_stride_b, x_stride_i, x_stride_j,
            mask_stride_b, mask.stride(-2), mask.stride(-1),
            n_pair * channels, pair * channels, channels,
            ln_w.stride(0), ln_b.stride(0),
            wq.stride(0), wq.stride(1), wk.stride(0), wk.stride(1),
            wv.stride(0), wv.stride(1), wg.stride(0), wg.stride(1),
            wo.stride(0), wo.stride(1),
            n_pair, pair, channels, head_dim, heads, eps, self.inf,
            math.sqrt(scale_dim),
        )
        if large:
            n_qtiles = triton.cdiv(pair, _LARGE_BQ)
            _row_attention_large_kernel[(batch * pair * n_qtiles,)](
                *common, n_qtiles,
                BQ=_LARGE_BQ, BK=_LARGE_BK, BC=block_c, BD=block_d, DT=tl_dtype,
                num_warps=4, num_stages=1,
            )
        else:
            _row_attention_kernel[(batch * pair,)](
                *common,
                BN=max(_MIN_CONTRACT, _next_pow2(pair)), BC=block_c,
                BD=block_d, DT=tl_dtype,
                num_warps=4, num_stages=3,
            )
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, C_in] input tensor (pair representation)

        Returns:
            [*, I, J, C_in] output tensor
        """
        # The ending node attends over the other pair axis. Transposing the
        # inputs makes their (I, J) axes non-contiguous, which the kernels
        # handle because they address x through real strides; the result is
        # transposed back and, like the baseline's, is a view.
        if self.starting:
            xt, mt = x, mask
        else:
            xt = x.transpose(-2, -3)
            mt = None if mask is None else mask.transpose(-1, -2)

        layout = self._fast_path_layout(xt, mt)
        if layout is None:
            return self._forward_reference(x, mask)

        out = self._forward_fused(xt, mt, layout)
        return out if self.starting else out.transpose(-2, -3)


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """AF3 Algorithm 15."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9):
        super().__init__(c_in=c_in, c_hidden=c_hidden, no_heads=no_heads,
                         starting=False, inf=inf)
