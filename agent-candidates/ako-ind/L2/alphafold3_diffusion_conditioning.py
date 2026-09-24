"""Diffusion conditioning for AlphaFold3.

Produces conditioned single and pair representations from trunk outputs
and diffusion time step. Implements Fourier time embedding and optional
trunk conditioning.

Reference: openfold3/core/model/layers/diffusion_conditioning.py

Optimization notes
------------------
At the shape this op is called with (N_token=16, batch=1, bf16) every GEMM and
reduction is orders of magnitude below the GPU's work threshold: the reference
forward is ~75 kernels whose cost is launch latency and host dispatch, not FLOPs.
Two independent attacks:

1. **Host cost -> one dispatch.**  The whole forward is captured once into a CUDA
   graph keyed on the input signature; each later call is a single
   ``_foreach_copy_`` of the live inputs into static buffers plus one graph
   replay.  ~75 host dispatches become 2.

2. **Device cost -> 8 fused Triton kernels.**  Everything is row-wise in the
   channel dimension, so LayerNorm + GEMM + activation + mask + residual all fuse:

   * ``_pair_head_kernel`` folds *all* of ``relpos_complex`` (a [N,N,139]
     thermometer encoding built from ~30 elementwise ops), the 267-wide concat,
     the LayerNorm over it and the bias-free 267->128 GEMM into one kernel.  The
     relpos block is never materialized: it contributes exactly ``v1+v2+se+v3``
     ones, so its LayerNorm statistics are analytic, and ``onehot @ W`` collapses
     to three lookups in prefix-summed weight tables (``cum1/cum2/cum3``) built
     at pack time.
   * ``_single_head_kernel`` fuses the 833-wide concat/LayerNorm/GEMM together
     with the entire scalar-t Fourier path (log -> cos -> LayerNorm(256) ->
     Linear(256,384)), which is a pure function of one scalar.
   * each SwiGLU transition becomes 1 kernel (pair, M=256 rows) or 2 (single,
     M=16 rows, split over the hidden dim so the weights stream from more than
     one SM), with the SiLU gate, the mask multiply and the residual add all in
     the epilogue.

   Every intermediate rounding the reference performs (LayerNorm outputs, GEMM
   outputs, the SiLU/mask/residual products, and each step of the Fourier
   recurrence) is reproduced in the low-precision dtype, so the fused path tracks
   the reference to well inside bf16 tolerance.

Anything outside the captured configuration (other shapes/dtypes, the
``use_conditioning=False`` branch, a batch without ``asym_id``, a missing
``token_mask``, a non-scalar ``t``, no Triton, a non-CUDA device) falls back to
the reference implementation verbatim.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_input_embedder import relpos_complex
from .alphafold3_swiglu_transition import SwiGLUTransition

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # pragma: no cover - Triton always present on the bench box
    _HAVE_TRITON = False


__targets__ = ["DiffusionConditioning"]


# Batch keys ``relpos_complex`` reads, in the order the fused kernel wants them.
_RELPOS_KEYS = ("residue_index", "token_index", "asym_id", "entity_id", "sym_id")

# Low-precision round codes shared by host and device.
# Only bf16 takes the fused path: the fp32->low-precision rounding barrier in
# ``_rnd`` is bf16-specific, and bf16 is the only dtype this op is captured in.
_RND = {torch.bfloat16: 1}

# A capture costs milliseconds.  A caller that alternates configurations would
# otherwise re-capture on every call, which is far slower than staying eager.
_MAX_CAPTURES = 8


class _NoCapture(Exception):
    """This call cannot be captured (not an error in the capture machinery)."""


class FourierEmbedding(nn.Module):
    """Fourier time embedding for diffusion conditioning.

    Uses random Fourier features (matching the reference's seeded initialization).

    Args:
        c: Embedding dimension (256 in the reference)
        seed: Random seed for weight initialization
    """

    def __init__(self, c: int = 256, seed: int = 42):
        super().__init__()
        self.c = c
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.register_buffer(
            "w", torch.randn(c, generator=generator),
        )
        self.register_buffer(
            "b", torch.randn(c, generator=generator),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.w + self.b
        return torch.cos(2 * math.pi * x)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAVE_TRITON:

    @triton.jit
    def _rnd(x, RND: tl.constexpr):
        """Round an fp32 value to bf16 precision, keeping it in fp32 registers.

        Deliberately done on the integer representation rather than as
        ``x.to(tl.bfloat16).to(tl.float32)``: the cast pair is transparent to
        LLVM's fast-math contraction, so a ``_rnd(a * b) + c`` chain gets fused
        into a single ``fma`` that skips the rounding -- which silently moves
        ~25% of the Fourier angles by one bf16 ULP.  Integer ops are a hard
        barrier.  (Round-to-nearest-even; NaN/Inf pass through unchanged.)
        """
        u = x.to(tl.uint32, bitcast=True)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.to(tl.float32, bitcast=True)

    @triton.jit
    def _cast(x, RND: tl.constexpr):
        return x.to(tl.bfloat16)

    @triton.jit
    def _thermo_count(pos_i, pos_j, cond, KCLIP: tl.constexpr,
                      NBINS: tl.constexpr, RND: tl.constexpr):
        """Number of ones in the reference's thermometer code for one relpos block.

        ``_binned_one_hot(final, arange(NBINS))`` sets feature ``b`` iff
        ``b < final``, so the count is ``clamp(ceil(final), 0, NBINS)``.  The
        offset arithmetic must round through the feature dtype exactly as the
        reference does -- a bare fp32 evaluation lands on the wrong side of an
        integer boundary for ~1 pair in 8.
        """
        off = _rnd(pos_i - pos_j, RND)
        u = _rnd(off + KCLIP, RND)
        cl = tl.minimum(tl.maximum(u, 0.0), 2.0 * KCLIP)
        fin = tl.where(cond, cl, 2.0 * KCLIP + 1.0)
        v = tl.minimum(tl.maximum(-tl.floor(-fin), 0.0), float(NBINS))
        return v.to(tl.int32)

    @triton.jit
    def _pair_head_kernel(
        zij_ptr, res_ptr, tok_ptr, asym_ptr, ent_ptr, sym_ptr,
        gz_ptr, wzt_ptr, cum1_ptr, cum2_ptr, cum3_ptr, ase_ptr, arel_ptr,
        out_ptr, M, N,
        CZ: tl.constexpr, BC: tl.constexpr, BM: tl.constexpr,
        NFEAT: tl.constexpr, NB1: tl.constexpr, NB3: tl.constexpr,
        KP: tl.constexpr, KC: tl.constexpr,
        EPS: tl.constexpr, RND: tl.constexpr,
    ):
        """relpos + concat + LayerNorm(267) + Linear(267 -> c_z), fused."""
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rmask = rm < M
        r = tl.where(rmask, rm, 0)
        nn2 = N * N
        b = r // nn2
        rem = r - b * nn2
        i = rem // N
        j = rem - i * N
        ii = b * N + i
        jj = b * N + j

        res_i = tl.load(res_ptr + ii).to(tl.float32)
        res_j = tl.load(res_ptr + jj).to(tl.float32)
        tok_i = tl.load(tok_ptr + ii).to(tl.float32)
        tok_j = tl.load(tok_ptr + jj).to(tl.float32)
        as_i = tl.load(asym_ptr + ii).to(tl.float32)
        as_j = tl.load(asym_ptr + jj).to(tl.float32)
        en_i = tl.load(ent_ptr + ii).to(tl.float32)
        en_j = tl.load(ent_ptr + jj).to(tl.float32)
        sy_i = tl.load(sym_ptr + ii).to(tl.float32)
        sy_j = tl.load(sym_ptr + jj).to(tl.float32)

        same_chain = as_i == as_j
        same_res = res_i == res_j
        same_ent = en_i == en_j

        v1 = _thermo_count(res_i, res_j, same_chain, KP, NB1, RND)
        v2 = _thermo_count(tok_i, tok_j, same_chain & same_res, KP, NB1, RND)
        v3 = _thermo_count(sy_i, sy_j, same_ent, KC, NB3, RND)
        se = tl.where(same_ent, 1.0, 0.0)

        # Number of ones the relpos block contributes -- both its sum and its
        # sum of squares, since every entry is 0 or 1.
        sr = v1.to(tl.float32) + v2.to(tl.float32) + v3.to(tl.float32) + se

        c = tl.arange(0, BC)
        cm = c < CZ
        z = tl.load(zij_ptr + r[:, None].to(tl.int64) * CZ + c[None, :],
                    mask=rmask[:, None] & cm[None, :], other=0.0).to(tl.float32)
        s = tl.sum(z, 1) + sr
        q = tl.sum(z * z, 1) + sr
        mean = s / NFEAT
        var = q / NFEAT - mean * mean
        rsig = 1.0 / tl.sqrt(var + EPS)

        g = tl.load(gz_ptr + c, mask=cm, other=0.0)
        y = (z - mean[:, None]) * rsig[:, None] * g[None, :]
        yb = _cast(tl.where(cm[None, :], y, 0.0), RND)

        w = tl.load(wzt_ptr + c[:, None] * CZ + c[None, :],
                    mask=cm[:, None] & cm[None, :], other=0.0)
        acc = tl.dot(yb, w)

        # onehot @ W for the thermometer blocks == prefix-summed weight rows.
        rp = tl.load(cum1_ptr + v1[:, None] * CZ + c[None, :],
                     mask=cm[None, :], other=0.0)
        rp += tl.load(cum2_ptr + v2[:, None] * CZ + c[None, :],
                      mask=cm[None, :], other=0.0)
        rp += tl.load(cum3_ptr + v3[:, None] * CZ + c[None, :],
                      mask=cm[None, :], other=0.0)
        ase = tl.load(ase_ptr + c, mask=cm, other=0.0)
        arel = tl.load(arel_ptr + c, mask=cm, other=0.0)
        rp += se[:, None] * ase[None, :] - mean[:, None] * arel[None, :]
        acc += rsig[:, None] * rp

        tl.store(out_ptr + r[:, None].to(tl.int64) * CZ + c[None, :],
                 _cast(acc, RND), mask=rmask[:, None] & cm[None, :])

    @triton.jit
    def _single_head_kernel(
        a_ptr, b_ptr, t_ptr, ga_ptr, gb_ptr, wat_ptr, wbt_ptr,
        fw_ptr, fb_ptr, gn_ptr, wnt_ptr, out_ptr, M,
        CA: tl.constexpr, CB: tl.constexpr, CS: tl.constexpr, CF: tl.constexpr,
        BKA: tl.constexpr, BKB: tl.constexpr, BKF: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
        SIGMA: tl.constexpr, EPS: tl.constexpr, RND: tl.constexpr,
    ):
        """concat + LayerNorm(c_s+c_s_input) + Linear -> c_s, plus the whole
        scalar-t Fourier embedding branch, fused."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rmask = rm < M
        r = tl.where(rmask, rm, 0)

        ka = tl.arange(0, BKA)
        kam = ka < CA
        kb = tl.arange(0, BKB)
        kbm = kb < CB
        xa = tl.load(a_ptr + r[:, None].to(tl.int64) * CA + ka[None, :],
                     mask=rmask[:, None] & kam[None, :], other=0.0).to(tl.float32)
        xb = tl.load(b_ptr + r[:, None].to(tl.int64) * CB + kb[None, :],
                     mask=rmask[:, None] & kbm[None, :], other=0.0).to(tl.float32)
        s = tl.sum(xa, 1) + tl.sum(xb, 1)
        q = tl.sum(xa * xa, 1) + tl.sum(xb * xb, 1)
        mean = s / (CA + CB)
        var = q / (CA + CB) - mean * mean
        rsig = 1.0 / tl.sqrt(var + EPS)
        ga = tl.load(ga_ptr + ka, mask=kam, other=0.0)
        gb = tl.load(gb_ptr + kb, mask=kbm, other=0.0)
        ya = _cast(tl.where(kam[None, :],
                            (xa - mean[:, None]) * rsig[:, None] * ga[None, :], 0.0), RND)
        yb = _cast(tl.where(kbm[None, :],
                            (xb - mean[:, None]) * rsig[:, None] * gb[None, :], 0.0), RND)

        n_ = pid_n * BN + tl.arange(0, BN)
        nm = n_ < CS
        wa = tl.load(wat_ptr + ka[:, None] * CS + n_[None, :],
                     mask=kam[:, None] & nm[None, :], other=0.0)
        wb = tl.load(wbt_ptr + kb[:, None] * CS + n_[None, :],
                     mask=kbm[:, None] & nm[None, :], other=0.0)
        o = _rnd(tl.dot(ya, wa) + tl.dot(yb, wb), RND)

        # Fourier branch: pure function of the scalar t, recomputed per program
        # (a few hundred flops) rather than launched as its own kernel chain.
        tv = tl.load(t_ptr).to(tl.float32)
        nv = _rnd(0.25 * _rnd(tl.log(_rnd(tv / SIGMA, RND)), RND), RND)
        kf = tl.arange(0, BKF)
        kfm = kf < CF
        fw = tl.load(fw_ptr + kf, mask=kfm, other=0.0).to(tl.float32)
        fbv = tl.load(fb_ptr + kf, mask=kfm, other=0.0).to(tl.float32)
        ang = _rnd(6.283185307179586 * _rnd(_rnd(nv * fw, RND) + fbv, RND), RND)
        ne = tl.where(kfm, _rnd(tl.cos(ang), RND), 0.0)
        mf = tl.sum(ne) / CF
        vf = tl.sum(ne * ne) / CF - mf * mf
        rf = 1.0 / tl.sqrt(vf + EPS)
        gn = tl.load(gn_ptr + kf, mask=kfm, other=0.0)
        yn = tl.where(kfm, _rnd((ne - mf) * rf * gn, RND), 0.0)
        wn = tl.load(wnt_ptr + kf[:, None] * CS + n_[None, :],
                     mask=kfm[:, None] & nm[None, :], other=0.0)
        fbias = _rnd(tl.sum(yn[:, None] * wn.to(tl.float32), 0), RND)

        res = _rnd(o + fbias[None, :], RND)
        tl.store(out_ptr + r[:, None].to(tl.int64) * CS + n_[None, :],
                 _cast(res, RND), mask=rmask[:, None] & nm[None, :])

    @triton.jit
    def _trans_fused_kernel(
        x_ptr, g_ptr, bb_ptr, wat_ptr, wbt_ptr, wot_ptr, mask_ptr, out_ptr,
        M, N,
        C: tl.constexpr, H: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr,
        BM: tl.constexpr, EPS: tl.constexpr, RND: tl.constexpr,
        MASK_MODE: tl.constexpr,
    ):
        """A whole SwiGLUTransition + mask + residual in one kernel (row-tiled)."""
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        rmask = rm < M
        r = tl.where(rmask, rm, 0)
        c = tl.arange(0, BC)
        cm = c < C
        x = tl.load(x_ptr + r[:, None].to(tl.int64) * C + c[None, :],
                    mask=rmask[:, None] & cm[None, :], other=0.0).to(tl.float32)
        mean = tl.sum(x, 1) / C
        var = tl.sum(x * x, 1) / C - mean * mean
        rsig = 1.0 / tl.sqrt(var + EPS)
        g = tl.load(g_ptr + c, mask=cm, other=0.0)
        bb = tl.load(bb_ptr + c, mask=cm, other=0.0)
        y = (x - mean[:, None]) * rsig[:, None] * g[None, :] + bb[None, :]
        yb = _cast(tl.where(cm[None, :], y, 0.0), RND)

        acc = tl.zeros((BM, BC), dtype=tl.float32)
        for h0 in range(0, H, BH):
            hh = h0 + tl.arange(0, BH)
            hm = hh < H
            wa = tl.load(wat_ptr + c[:, None] * H + hh[None, :],
                         mask=cm[:, None] & hm[None, :], other=0.0)
            wb = tl.load(wbt_ptr + c[:, None] * H + hh[None, :],
                         mask=cm[:, None] & hm[None, :], other=0.0)
            av = _rnd(tl.dot(yb, wa), RND)
            bv = _rnd(tl.dot(yb, wb), RND)
            hv = _rnd(_rnd(av * tl.sigmoid(av), RND) * bv, RND)
            hb = _cast(tl.where(hm[None, :], hv, 0.0), RND)
            wo = tl.load(wot_ptr + hh[:, None] * C + c[None, :],
                         mask=hm[:, None] & cm[None, :], other=0.0)
            acc += tl.dot(hb, wo)

        o = _rnd(acc, RND)
        if MASK_MODE == 1:
            mv = tl.load(mask_ptr + r).to(tl.float32)
            o = _rnd(o * mv[:, None], RND)
        elif MASK_MODE == 2:
            nn2 = N * N
            b = r // nn2
            rem = r - b * nn2
            i = rem // N
            j = rem - i * N
            mi = tl.load(mask_ptr + b * N + i).to(tl.float32)
            mj = tl.load(mask_ptr + b * N + j).to(tl.float32)
            o = _rnd(o * _rnd(mi * mj, RND)[:, None], RND)
        res = _rnd(x + o, RND)
        tl.store(out_ptr + r[:, None].to(tl.int64) * C + c[None, :],
                 _cast(res, RND), mask=rmask[:, None] & cm[None, :])

    @triton.jit
    def _trans_h_kernel(
        x_ptr, g_ptr, bb_ptr, wat_ptr, wbt_ptr, hout_ptr, M,
        C: tl.constexpr, H: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr,
        BM: tl.constexpr, EPS: tl.constexpr, RND: tl.constexpr,
    ):
        """LayerNorm + SwiGLU half of a transition, tiled over the hidden dim so
        the (large) projection weights stream from many SMs even at M=16."""
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rmask = rm < M
        r = tl.where(rmask, rm, 0)
        c = tl.arange(0, BC)
        cm = c < C
        x = tl.load(x_ptr + r[:, None].to(tl.int64) * C + c[None, :],
                    mask=rmask[:, None] & cm[None, :], other=0.0).to(tl.float32)
        mean = tl.sum(x, 1) / C
        var = tl.sum(x * x, 1) / C - mean * mean
        rsig = 1.0 / tl.sqrt(var + EPS)
        g = tl.load(g_ptr + c, mask=cm, other=0.0)
        bb = tl.load(bb_ptr + c, mask=cm, other=0.0)
        y = (x - mean[:, None]) * rsig[:, None] * g[None, :] + bb[None, :]
        yb = _cast(tl.where(cm[None, :], y, 0.0), RND)

        hh = pid_h * BH + tl.arange(0, BH)
        hm = hh < H
        wa = tl.load(wat_ptr + c[:, None] * H + hh[None, :],
                     mask=cm[:, None] & hm[None, :], other=0.0)
        wb = tl.load(wbt_ptr + c[:, None] * H + hh[None, :],
                     mask=cm[:, None] & hm[None, :], other=0.0)
        av = _rnd(tl.dot(yb, wa), RND)
        bv = _rnd(tl.dot(yb, wb), RND)
        hv = _rnd(_rnd(av * tl.sigmoid(av), RND) * bv, RND)
        tl.store(hout_ptr + r[:, None].to(tl.int64) * H + hh[None, :],
                 _cast(hv, RND), mask=rmask[:, None] & hm[None, :])

    @triton.jit
    def _trans_o_kernel(
        h_ptr, x_ptr, wot_ptr, mask_ptr, out_ptr, M, N,
        C: tl.constexpr, H: tl.constexpr, BC: tl.constexpr, BH: tl.constexpr,
        BM: tl.constexpr, RND: tl.constexpr, MASK_MODE: tl.constexpr,
    ):
        """Output projection + mask + residual half of a transition."""
        pid_m = tl.program_id(0)
        pid_c = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rmask = rm < M
        r = tl.where(rmask, rm, 0)
        c = pid_c * BC + tl.arange(0, BC)
        cm = c < C
        acc = tl.zeros((BM, BC), dtype=tl.float32)
        for h0 in range(0, H, BH):
            hh = h0 + tl.arange(0, BH)
            hm = hh < H
            hb = tl.load(h_ptr + r[:, None].to(tl.int64) * H + hh[None, :],
                         mask=rmask[:, None] & hm[None, :], other=0.0)
            wo = tl.load(wot_ptr + hh[:, None] * C + c[None, :],
                         mask=hm[:, None] & cm[None, :], other=0.0)
            acc += tl.dot(hb, wo)
        o = _rnd(acc, RND)
        if MASK_MODE == 1:
            mv = tl.load(mask_ptr + r).to(tl.float32)
            o = _rnd(o * mv[:, None], RND)
        elif MASK_MODE == 2:
            nn2 = N * N
            b = r // nn2
            rem = r - b * nn2
            i = rem // N
            j = rem - i * N
            mi = tl.load(mask_ptr + b * N + i).to(tl.float32)
            mj = tl.load(mask_ptr + b * N + j).to(tl.float32)
            o = _rnd(o * _rnd(mi * mj, RND)[:, None], RND)
        x = tl.load(x_ptr + r[:, None].to(tl.int64) * C + c[None, :],
                    mask=rmask[:, None] & cm[None, :], other=0.0).to(tl.float32)
        res = _rnd(x + o, RND)
        tl.store(out_ptr + r[:, None].to(tl.int64) * C + c[None, :],
                 _cast(res, RND), mask=rmask[:, None] & cm[None, :])


def _npow2(n: int) -> int:
    return max(16, 1 << (int(n) - 1).bit_length())


class DiffusionConditioning(nn.Module):
    """Conditioning for diffusion module.

    Matches the reference:
    - Pair: concat([zij_trunk, relpos], dim=-1) -> LayerNorm -> Linear -> 2x SwiGLU transition
    - Single: concat([si_trunk, si_input], dim=-1) -> LayerNorm -> Linear + fourier -> 2x SwiGLU transition

    Reference: openfold3/core/model/layers/diffusion_conditioning.py

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        c_s_input: Input single representation dimension (449)
        sigma_data: Noise level scaling for Fourier embedding
        relpos_k: Maximum relative position for pair bias
        max_relative_chain: Maximum relative chain index
        c_fourier_emb: Fourier embedding dimension (256)
        seed_fourier_emb: Fourier embedding random seed
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        sigma_data: float = 16.0,
        relpos_k: int = 32,
        max_relative_chain: int = 2,
        c_fourier_emb: int = 256,
        seed_fourier_emb: int = 42,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_input = c_s_input
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data
        self.relpos_k = relpos_k
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * relpos_k + 2
        num_rel_token_bins = 2 * relpos_k + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins + num_rel_token_bins
            + num_rel_chain_bins + num_same_entity_features
        )

        self.layer_norm_z = LayerNorm(num_relpos_dims + c_z, create_offset=False)
        self.linear_z = Linear(num_relpos_dims + c_z, c_z, bias=False)

        self.transition_z = nn.ModuleList([
            SwiGLUTransition(c_in=c_z, n=2)
            for _ in range(2)
        ])

        self.layer_norm_s = LayerNorm(c_s + c_s_input, create_offset=False)
        self.linear_s = Linear(c_s + c_s_input, c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        self.layer_norm_n = LayerNorm(c_fourier_emb, create_offset=False)
        self.linear_n = Linear(c_fourier_emb, c_s, bias=False)

        self.transition_s = nn.ModuleList([
            SwiGLUTransition(c_in=c_s, n=2)
            for _ in range(2)
        ])

        # --- fused-path state.  Deliberately plain attributes: the bench casts
        # every *parameter* to the run dtype and every *buffer* to the parameter
        # dtype, so anything registered here would be silently rewritten.
        self._pk = None            # packed / prefix-summed weights
        self._cg = None            # captured CUDA graph
        self._cg_dsts = None       # static input buffers (foreach_copy_ targets)
        self._cg_out = None        # static graph outputs
        self._cg_keys = ()         # batch keys mirrored into static buffers
        self._cg_uc = None
        self._cg_asym = None
        self._cg_nomask = None
        self._cg_dt = None
        self._cg_ts = self._cg_is = self._cg_ss = self._cg_zs = self._cg_bs = None
        self._cg_n = 0             # captures so far (bounded: see forward)
        self._cg_off = False       # capture failed once -> stop retrying
        self._fused_off = False    # packing rejected the weights -> stay eager

    # -- invalidation ---------------------------------------------------------
    # A capture bakes in weight *addresses*, and the packed tables bake in weight
    # *values*, so any reload or device/dtype move has to drop both.
    def _drop_fast(self) -> None:
        self._pk = None
        self._cg = None
        self._cg_dsts = None
        self._cg_out = None
        self._cg_keys = ()

    def _apply(self, *args, **kwargs):
        self._drop_fast()
        return super()._apply(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        self._drop_fast()
        return super()._load_from_state_dict(*args, **kwargs)

    # -- reference math -------------------------------------------------------
    def _eager(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_conditioning:
            # Pair conditioning: concat trunk pair with relpos features
            if "asym_id" in batch:
                relpos_zij = relpos_complex(
                    batch=batch,
                    max_relative_idx=self.relpos_k,
                    max_relative_chain=self.max_relative_chain,
                ).to(dtype=zij_trunk.dtype)
            else:
                relpos_dim = self.linear_z.weight.shape[-1] - self.c_z
                relpos_zij = zij_trunk.new_zeros(
                    zij_trunk.shape[:-1] + (relpos_dim,),
                )

            zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
            zij = self.linear_z(self.layer_norm_z(zij))

            # Single conditioning: concat trunk single with input
            si = torch.cat([si_trunk, si_input], dim=-1)
            si = self.linear_s(self.layer_norm_s(si))
        else:
            zij = zij_trunk.new_zeros(zij_trunk.shape)
            si = si_trunk.new_zeros(si_trunk.shape[:-1] + (self.c_s,))

        # Fourier noise embedding
        n = 0.25 * torch.log(t / self.sigma_data)
        n_emb = self.fourier_emb(n.unsqueeze(-1) if n.dim() == 0 else n)
        si = si + self.linear_n(self.layer_norm_n(n_emb)).unsqueeze(-2)

        # Apply transition layers
        token_mask = batch.get("token_mask")
        if token_mask is not None:
            pair_mask = token_mask[..., :, None] * token_mask[..., None, :]
        else:
            pair_mask = None

        for layer in self.transition_z:
            zij = zij + layer(zij, mask=pair_mask)

        for layer in self.transition_s:
            si = si + layer(si, mask=token_mask)

        return si, zij

    # -- weight packing -------------------------------------------------------
    def _pack(self) -> dict:
        """Fold the LayerNorm scales into the projections and prefix-sum the
        relpos weight blocks.  Called once, lazily, after weights are loaded."""
        cz, cs, csi, cf = self.c_z, self.c_s, self.c_s_input, self.c_fourier_emb
        nb1 = 2 * self.relpos_k + 2
        nb3 = 2 * self.max_relative_chain + 2
        nfeat = cz + 2 * nb1 + 1 + nb3

        wz = self.linear_z.weight
        dt, dev = wz.dtype, wz.device
        if dt not in _RND:
            raise TypeError("fused path needs bf16 weights")

        def ln_parts(ln, width):
            g = ln.weight
            g = (torch.ones(width, device=dev, dtype=torch.float32)
                 if g is None else g.float())
            bb = ln.bias
            bb = (torch.zeros(width, device=dev, dtype=torch.float32)
                  if bb is None else bb.float())
            return g, bb, float(ln.eps)

        p: dict = {"dt": dt, "dev": dev, "rnd": _RND[dt]}

        # --- pair head ---
        gz, bz, eps_z = ln_parts(self.layer_norm_z, nfeat)
        if bool(bz.any()):
            raise ValueError("layer_norm_z offset unsupported on fused path")
        a = wz.float() * gz                                  # [cz, nfeat]
        p["gz"] = gz[:cz].contiguous()
        p["wzt"] = wz[:, :cz].t().contiguous()               # [k=cz, n=cz]
        o1, o2, ose, o3 = cz, cz + nb1, cz + 2 * nb1, cz + 2 * nb1 + 1

        def prefix(block, nbins):
            out = torch.zeros(nbins + 1, block.shape[0], device=dev,
                              dtype=torch.float32)
            out[1:] = block.t().cumsum(0)
            return out.contiguous()

        p["cum1"] = prefix(a[:, o1:o1 + nb1], nb1)
        p["cum2"] = prefix(a[:, o2:o2 + nb1], nb1)
        p["cum3"] = prefix(a[:, o3:o3 + nb3], nb3)
        p["ase"] = a[:, ose].contiguous()
        p["arel"] = a[:, cz:].sum(1).contiguous()
        p["eps_z"] = eps_z
        p["nfeat"] = nfeat
        p["nb1"] = nb1
        p["nb3"] = nb3

        # --- single head + fourier ---
        gs, bs, eps_s = ln_parts(self.layer_norm_s, cs + csi)
        if bool(bs.any()):
            raise ValueError("layer_norm_s offset unsupported on fused path")
        ws = self.linear_s.weight
        p["gsa"] = gs[:cs].contiguous()
        p["gsb"] = gs[cs:].contiguous()
        p["wsat"] = ws[:, :cs].t().contiguous()              # [k=cs,  n=cs]
        p["wsbt"] = ws[:, cs:].t().contiguous()              # [k=csi, n=cs]
        p["eps_s"] = eps_s
        gn, bn, eps_n = ln_parts(self.layer_norm_n, cf)
        if bool(bn.any()):
            raise ValueError("layer_norm_n offset unsupported on fused path")
        p["gn"] = gn.contiguous()
        p["wnt"] = self.linear_n.weight.t().contiguous()     # [k=cf, n=cs]
        p["fw"] = self.fourier_emb.w.contiguous()
        p["fb"] = self.fourier_emb.b.contiguous()
        p["eps_n"] = eps_n

        # --- transitions ---
        for tag, mods in (("z", self.transition_z), ("s", self.transition_s)):
            packed = []
            for layer in mods:
                c_in = layer.c_in
                g, bb, eps = ln_parts(layer.layer_norm, c_in)
                wa = layer.swiglu.linear_a.weight
                wb = layer.swiglu.linear_b.weight
                wo = layer.linear_out.weight
                if (layer.swiglu.linear_a.bias is not None
                        or layer.swiglu.linear_b.bias is not None
                        or layer.linear_out.bias is not None):
                    raise ValueError("transition biases unsupported on fused path")
                packed.append({
                    "g": g, "bb": bb, "eps": eps, "c": c_in, "h": wa.shape[0],
                    "wat": wa.t().contiguous(), "wbt": wb.t().contiguous(),
                    "wot": wo.t().contiguous(),
                })
            p[tag] = packed
        return p

    # -- eligibility ----------------------------------------------------------
    def _fused_ok(self, batch, t, si_input, si_trunk, zij_trunk,
                  use_conditioning) -> bool:
        if self._fused_off or not (_HAVE_TRITON and use_conditioning and t.is_cuda):
            return False
        dt = self.linear_z.weight.dtype
        if dt not in _RND:
            return False
        if "asym_id" not in batch:
            return False
        if t.numel() != 1:
            return False
        cs, csi, cz = self.c_s, self.c_s_input, self.c_z
        if si_trunk.dim() < 2 or zij_trunk.dim() < 3:
            return False
        if si_trunk.shape[-1] != cs or si_input.shape[-1] != csi:
            return False
        if zij_trunk.shape[-1] != cz:
            return False
        n = si_trunk.shape[-2]
        if (si_input.shape[:-1] != si_trunk.shape[:-1]
                or zij_trunk.shape[:-3] != si_trunk.shape[:-2]
                or zij_trunk.shape[-3] != n or zij_trunk.shape[-2] != n):
            return False
        tensors = [t, si_input, si_trunk, zij_trunk]
        for k in _RELPOS_KEYS:
            v = batch.get(k)
            if not isinstance(v, torch.Tensor) or v.shape != si_trunk.shape[:-1]:
                return False
            tensors.append(v)
        tm = batch.get("token_mask")
        if tm is not None:
            if not isinstance(tm, torch.Tensor) or tm.shape != si_trunk.shape[:-1]:
                return False
            tensors.append(tm)
        for v in tensors:
            if v.dtype != dt or not v.is_contiguous() or not v.is_cuda:
                return False
        # weight-shape sanity (a state_dict from a differently-shaped module)
        for mods, c_in in ((self.transition_z, cz), (self.transition_s, cs)):
            for layer in mods:
                if layer.c_in != c_in:
                    return False
        return True

    # -- fused implementation -------------------------------------------------
    def _fused(self, batch, t, si_input, si_trunk, zij_trunk):
        p = self._pk
        if p is None:
            p = self._pk = self._pack()
        dt = p["dt"]
        rnd = p["rnd"]
        cz, cs, csi, cf = self.c_z, self.c_s, self.c_s_input, self.c_fourier_emb
        n = si_trunk.shape[-2]
        m_s = si_trunk.numel() // cs
        m_p = zij_trunk.numel() // cz
        tm = batch.get("token_mask")
        mask_z = 0 if tm is None else 2
        mask_s = 0 if tm is None else 1
        tm_arg = tm if tm is not None else si_trunk

        # ---- pair: relpos + LN + Linear ----
        bc = _npow2(cz)
        bm_p = 32
        zij = torch.empty(zij_trunk.shape[:-1] + (cz,), device=p["dev"], dtype=dt)
        _pair_head_kernel[(triton.cdiv(m_p, bm_p),)](
            zij_trunk, batch["residue_index"], batch["token_index"],
            batch["asym_id"], batch["entity_id"], batch["sym_id"],
            p["gz"], p["wzt"], p["cum1"], p["cum2"], p["cum3"],
            p["ase"], p["arel"], zij, m_p, n,
            CZ=cz, BC=bc, BM=bm_p, NFEAT=p["nfeat"], NB1=p["nb1"], NB3=p["nb3"],
            KP=self.relpos_k, KC=self.max_relative_chain,
            EPS=p["eps_z"], RND=rnd, num_warps=8,
        )
        for pz in p["z"]:
            out = torch.empty_like(zij)
            _trans_fused_kernel[(triton.cdiv(m_p, bm_p),)](
                zij, pz["g"], pz["bb"], pz["wat"], pz["wbt"], pz["wot"],
                tm_arg, out, m_p, n,
                C=pz["c"], H=pz["h"], BC=_npow2(pz["c"]), BH=64, BM=bm_p,
                EPS=pz["eps"], RND=rnd, MASK_MODE=mask_z, num_warps=8,
            )
            zij = out

        # ---- single: concat + LN + Linear + fourier ----
        bm_s = 16
        bn_s = 64
        si = torch.empty(si_trunk.shape[:-1] + (cs,), device=p["dev"], dtype=dt)
        _single_head_kernel[(triton.cdiv(m_s, bm_s), triton.cdiv(cs, bn_s))](
            si_trunk, si_input, t, p["gsa"], p["gsb"], p["wsat"], p["wsbt"],
            p["fw"], p["fb"], p["gn"], p["wnt"], si, m_s,
            CA=cs, CB=csi, CS=cs, CF=cf,
            BKA=_npow2(cs), BKB=_npow2(csi), BKF=_npow2(cf),
            BM=bm_s, BN=bn_s, SIGMA=float(self.sigma_data), EPS=p["eps_s"],
            RND=rnd, num_warps=4,
        )
        for ps in p["s"]:
            c_in, h = ps["c"], ps["h"]
            hbuf = torch.empty((m_s, h), device=p["dev"], dtype=dt)
            _trans_h_kernel[(triton.cdiv(m_s, bm_s), triton.cdiv(h, 64))](
                si, ps["g"], ps["bb"], ps["wat"], ps["wbt"], hbuf, m_s,
                C=c_in, H=h, BC=_npow2(c_in), BH=64, BM=bm_s,
                EPS=ps["eps"], RND=rnd, num_warps=4,
            )
            out = torch.empty_like(si)
            _trans_o_kernel[(triton.cdiv(m_s, bm_s), triton.cdiv(c_in, 64))](
                hbuf, si, ps["wot"], tm_arg, out, m_s, n,
                C=c_in, H=h, BC=64, BH=128, BM=bm_s,
                RND=rnd, MASK_MODE=mask_s, num_warps=4,
            )
            si = out
        return si, zij

    def _impl(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning):
        if self._fused_ok(batch, t, si_input, si_trunk, zij_trunk,
                          use_conditioning):
            try:
                return self._fused(batch, t, si_input, si_trunk, zij_trunk)
            except (TypeError, ValueError):
                self._pk = None
                self._fused_off = True
        return self._eager(batch, t, si_input, si_trunk, zij_trunk,
                           use_conditioning)

    # -- capture --------------------------------------------------------------
    def _capture(self, batch, t, si_input, si_trunk, zij_trunk,
                 use_conditioning):
        has_asym = "asym_id" in batch
        token_mask = batch.get("token_mask")

        keys: list[str] = []
        if token_mask is not None:
            keys.append("token_mask")
        if has_asym:
            for k in _RELPOS_KEYS:
                if not isinstance(batch.get(k), torch.Tensor):
                    raise TypeError(f"batch[{k!r}] missing")
                keys.append(k)

        srcs = [t, si_input, si_trunk, zij_trunk] + [batch[k] for k in keys]
        for v in srcs:
            if not v.is_cuda:
                raise RuntimeError("fast path needs CUDA inputs")
        dsts = [torch.empty(v.shape, dtype=v.dtype, device=v.device).copy_(v)
                for v in srcs]
        sbatch = dict(batch)
        for k, d in zip(keys, dsts[4:]):
            sbatch[k] = d
        s_t, s_in, s_tr, s_z = dsts[0], dsts[1], dsts[2], dsts[3]

        # Warm up on a side stream (Triton JIT, cuBLAS workspaces, allocator)
        # then capture one forward.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._impl(sbatch, s_t, s_in, s_tr, s_z, use_conditioning)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        self._cg_n += 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._impl(sbatch, s_t, s_in, s_tr, s_z, use_conditioning)
        # Capture records without executing: replay once so ``out`` holds the
        # answer for the inputs this very call was made with.
        graph.replay()

        self._cg = graph
        self._cg_dsts = dsts
        self._cg_out = out
        self._cg_keys = tuple(keys)
        self._cg_uc = bool(use_conditioning)
        self._cg_asym = has_asym
        self._cg_nomask = token_mask is None
        # Straight-line guard data.  Kept as scalars rather than re-read off the
        # static buffers: the replay path runs 5 us of Python against a 167 us
        # harness floor, so a 10-iteration zip over ``.shape``/``.dtype`` is a
        # measurable fraction of our whole cost.
        self._cg_dt = si_trunk.dtype
        self._cg_ts = t.shape
        self._cg_is = si_input.shape
        self._cg_ss = si_trunk.shape
        self._cg_zs = zij_trunk.shape
        self._cg_bs = si_trunk.shape[:-1]      # every mirrored batch key
        self._cg_n = 0
        return out

    # -- entry point ----------------------------------------------------------
    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:     Feature dictionary (needs asym_id, entity_id etc. for relpos)
            t:         [*] noise level
            si_input:  [*, N_token, c_s_input] input embedding
            si_trunk:  [*, N_token, c_s] trunk single rep
            zij_trunk: [*, N_token, N_token, c_z] trunk pair rep
            use_conditioning: Whether to condition with trunk reps

        Returns:
            si:  [*, N_token, c_s] conditioned single rep
            zij: [*, N_token, N_token, c_z] conditioned pair rep
        """
        dsts = self._cg_dsts
        dt = self._cg_dt
        if (dsts is not None
                and si_trunk.shape == self._cg_ss
                and si_input.shape == self._cg_is
                and zij_trunk.shape == self._cg_zs
                and t.shape == self._cg_ts
                and si_trunk.dtype is dt
                and si_input.dtype is dt
                and zij_trunk.dtype is dt
                and t.dtype is dt
                and bool(use_conditioning) is self._cg_uc
                and ("asym_id" in batch) is self._cg_asym):
            bs = self._cg_bs
            srcs = [t, si_input, si_trunk, zij_trunk]
            add = srcs.append
            for k in self._cg_keys:
                v = batch.get(k)
                if v is None or v.shape != bs or v.dtype is not dt:
                    srcs = None
                    break
                add(v)
            if srcs is not None and (
                    self._cg_nomask is (batch.get("token_mask") is None)):
                torch._foreach_copy_(dsts, srcs)
                self._cg.replay()
                return self._cg_out

        if not self._cg_off and t.is_cuda:
            try:
                if self._cg_n >= _MAX_CAPTURES:
                    raise _NoCapture("capture budget exhausted")
                return self._capture(
                    batch, t, si_input, si_trunk, zij_trunk, use_conditioning)
            except Exception:
                # Either the inputs are not capturable, capture itself failed, or
                # the callers are alternating configurations faster than a capture
                # pays for itself.  Any of those: stop trying and stay eager.
                self._cg_off = True
                self._cg = None
                self._cg_dsts = None
                self._cg_out = None
                self._cg_keys = ()
        return self._impl(batch, t, si_input, si_trunk, zij_trunk,
                          use_conditioning)
