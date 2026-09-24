"""Outer product mean for AlphaFold3 (L2).

Implements AF3 Algorithm 9. Computes an outer product of MSA
representations and averages over the MSA dimension to produce
a pair representation update.

Reference: openfold3/core/model/layers/outer_product_mean.py OuterProductMean

The captured workload is tiny -- m[1, 8, 16, 64], mask[1, 8, 16] with
c_hidden=32, c_z=128 -- about 70 MFLOP, well under a microsecond of real math on
a B200, spread by the reference over ~15 CUDA launches and ~22 ATen dispatches.
Latency is set by dispatch and launch count, not arithmetic, so the whole
operator is fused into a single Triton kernel: one allocation plus one launch,
with no intermediate ever reaching memory (in particular not the
[N_res, N_res, c_hidden**2] one, which is 8x the size of the output).

Program (ib, br, iz) produces out[ib, br, :, iz*BZ : (iz+1)*BZ]:

    out[b, d, z] = (sum_{c,e} W[z, c, e] * sum_a A[b, a, c] * B[d, a, e]
                    + bias[z]) / (sum_a mask[a, b] * mask[a, d] + eps)

The MSA contraction is one ``tl.dot`` with (residue, channel) folded into the
GEMM's N, so a whole residue row of the outer product falls out of a single MMA
and is consumed straight into the ``linear_out`` contraction in registers. Each
program re-derives the LayerNorm and projections it needs rather than reading
them from a scratch tensor: ~0.5 MFLOP of redundant work against ~2 us of saved
launch, dispatch and kernel-to-kernel serialization, which is the right trade at
this size.

``_forward_eager`` is the general path -- any rank, dtype or device -- and is
also what the first call uses while the launch plan is being built.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - the eager path stays correct
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ln_rows(x, lnw, lnb, CM: tl.constexpr, LN_EPS: tl.constexpr):
        """Row-wise LayerNorm of an fp32 tile, returned in bf16.

        Two-pass fp32 reduction, which is what ATen's bf16 LayerNorm does
        internally and what the reference buys with its explicit ``.float()``
        -> ``layer_norm`` -> ``.to(bf16)`` sandwich.
        """
        mu = tl.sum(x, axis=1) / CM
        xc = x - mu[:, None]
        var = tl.sum(xc * xc, axis=1) / CM
        rstd = 1.0 / tl.sqrt(var + LN_EPS)
        return (xc * rstd[:, None] * lnw[None, :] + lnb[None, :]).to(tl.bfloat16)

    @triton.jit
    def _opm_kernel(
        m_ptr,           # [nb, S, R, C_m]
        mask_ptr,        # [nb, S, R]
        lnw_ptr, lnb_ptr,
        proj_ptr,        # [2, C_m, H] = stack(linear_1.W.T, linear_2.W.T)
        wt_ptr,          # [H * H, C_z] = linear_out.weight.T
        bias_ptr,        # [C_z]
        out_ptr,         # [nb, R, R, C_z]
        eps,
        S: tl.constexpr,     # N_seq
        R: tl.constexpr,     # N_res
        H: tl.constexpr,     # c_hidden
        CM: tl.constexpr,    # C_m
        RH: tl.constexpr,    # R * H
        CZ: tl.constexpr,    # c_z
        SR: tl.constexpr,    # N_seq * N_res
        SP: tl.constexpr,    # N_seq padded up to the MMA K minimum
        BC: tl.constexpr,    # c_hidden tile (>= the MMA M minimum)
        BZ: tl.constexpr,    # c_z tile
        LN_EPS: tl.constexpr,
    ):
        ib = tl.program_id(0)
        br = tl.program_id(1)
        iz = tl.program_id(2)

        kk = tl.arange(0, CM)
        dv = tl.arange(0, R)
        sa = tl.arange(0, SP)
        live = sa < S
        zz = iz * BZ + tl.arange(0, BZ)

        m_base = m_ptr + ib * SR * CM
        mk_base = mask_ptr + ib * SR

        # Statement order below is load-bearing, worth ~10% of candidate_ms.
        # The kernel is latency-bound (ncu: 4.7% compute, 0.3% DRAM, 12.5%
        # occupancy pinned by 104 KB of SMEM), Triton issues these independent
        # global loads roughly in source order, and only 4 warps are available to
        # hide the HBM round trips -- so whichever load goes first gets overlapped
        # with everything after it. Measured, byte-identical arithmetic:
        #   * hoisting the *small* independent tail loads (``md``, ``bias``) to
        #     here:                                              0.0195 -> 0.0175
        #   * hoisting the *large* ``xa`` tile here as well, or even ``ma``
        #     alone:                                             0.0175 -> 0.0195
        #     (they stay live across the whole b-side chain; the added pressure
        #     costs more than the overlap wins)
        #   * moving the ``w2`` load above the ``xb`` load below:
        #                                                        0.0195 -> 0.0215
        # Re-measure with `bash scripts/bench.sh` before reordering; the local
        # wall-clock timer cannot resolve these (see ITERATIONS.md).
        lnw = tl.load(lnw_ptr + kk).to(tl.float32)
        lnb = tl.load(lnb_ptr + kk).to(tl.float32)
        md = tl.load(mk_base + sa[:, None] * R + dv[None, :],
                     mask=live[:, None], other=0.0)
        bias = tl.load(bias_ptr + zz).to(tl.float32)

        # ---- b side: every (a, d) row, projected by linear_2 -> B[a, (d, e)].
        # The MSA dim has to come out zero-padded to SP, the MMA K minimum. When
        # SP is at most 2*S (the captured case: N_seq 8 -> 16) pad the [S, R*H]
        # result with a join/permute; otherwise take the general branch below.
        # The two forms measured the same once statement order was controlled
        # for -- this one is kept for the captured shape because it does half the
        # LayerNorm work.
        if SP <= 2 * S:
            rb = tl.arange(0, SR)
            xb = tl.load(m_base + rb[:, None] * CM + kk[None, :]).to(tl.float32)
            w2 = tl.load(proj_ptr + CM * H + kk[:, None] * H
                         + tl.arange(0, H)[None, :])
            bf = tl.dot(_ln_rows(xb, lnw, lnb, CM, LN_EPS), w2).to(tl.bfloat16)
            bf = tl.reshape(bf, (S, RH))
            # mask[a, d], broadcast across e.
            mb = tl.load(mk_base + tl.arange(0, S)[:, None] * R
                         + (tl.arange(0, RH) // H)[None, :])
            bf = (bf * mb).to(tl.bfloat16)
            if SP > S:
                # tl.cat is not order-stable, so pad with join + permute.
                bf = tl.reshape(
                    tl.permute(tl.join(bf, tl.zeros((S, RH), tl.bfloat16)),
                               (2, 0, 1)), (2 * S, RH))
        else:
            # General form: over-fetch to SP * R rows and let the load mask zero
            # the padding. Works for any power-of-two SP.
            rb = tl.arange(0, SP * R)
            good = rb < SR
            xb = tl.load(m_base + rb[:, None] * CM + kk[None, :],
                         mask=good[:, None], other=0.0).to(tl.float32)
            w2 = tl.load(proj_ptr + CM * H + kk[:, None] * H
                         + tl.arange(0, H)[None, :])
            # An all-zero padded row normalizes to the LayerNorm *bias*, not to
            # zero, so it has to be forced to zero explicitly.
            lnv = tl.where(good[:, None], _ln_rows(xb, lnw, lnb, CM, LN_EPS), 0.0)
            mkr = tl.load(mk_base + rb, mask=good, other=0.0)
            bf = tl.dot(lnv.to(tl.bfloat16), w2).to(tl.bfloat16)
            bf = tl.reshape((bf * mkr[:, None]).to(tl.bfloat16), (SP, RH))

        # ---- a side: only the N_seq rows at residue br.
        xa = tl.load(m_base + (sa[:, None] * R + br) * CM + kk[None, :],
                     mask=live[:, None], other=0.0).to(tl.float32)
        la = tl.where(live[:, None], _ln_rows(xa, lnw, lnb, CM, LN_EPS),
                      0.0).to(tl.bfloat16)
        ma = tl.load(mk_base + sa * R + br, mask=live, other=0.0)

        acc = tl.zeros((R, BZ), tl.float32)
        for c0 in range(0, H, BC):
            w1 = tl.load(proj_ptr + kk[:, None] * H
                         + (c0 + tl.arange(0, BC))[None, :])
            at = tl.dot(la, w1).to(tl.bfloat16)              # [SP, BC]
            at = tl.trans((at * ma[:, None]).to(tl.bfloat16))

            o = tl.dot(at, bf)                               # [BC, RH] = [c, (d, e)]
            o = tl.reshape(o.to(tl.bfloat16), (BC, R, H))
            o = tl.reshape(tl.permute(o, (1, 0, 2)), (R, BC * H))

            w = tl.load(wt_ptr + (c0 * H + tl.arange(0, BC * H))[:, None] * CZ
                        + zz[None, :])
            acc = tl.dot(o, w, acc)

        # Valid sequence pairs per residue pair. The reference keeps this in
        # bf16 (bmm -> bf16, then ``+ eps`` in bf16); with a randomized mask the
        # denominator can land within bf16 epsilon of -eps, where being *more*
        # precise than the reference blows up the relative error of the few
        # outputs that divide by it. So round at exactly the reference's points.
        nrm = tl.sum(md.to(tl.float32) * ma.to(tl.float32)[:, None], axis=0)
        den = (nrm.to(tl.bfloat16).to(tl.float32) + eps).to(tl.bfloat16)

        num = (acc + bias[None, :]).to(tl.bfloat16)
        tl.store(out_ptr + ((ib * R + br) * R + dv)[:, None] * CZ + zz[None, :],
                 (num.to(tl.float32) / den.to(tl.float32)[:, None]).to(tl.bfloat16))


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class OuterProductMean(nn.Module):
    """AF3 Algorithm 9: Outer product mean.

    Args:
        c_m: MSA embedding channel dimension
        c_z: Pair embedding channel dimension
        c_hidden: Hidden channel dimension
        eps: Epsilon for numerical stability
    """

    def __init__(self, c_m: int, c_z: int, c_hidden: int, eps: float = 1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, bias=False)
        self.linear_2 = Linear(c_m, c_hidden, bias=False)
        self.linear_out = Linear(c_hidden ** 2, c_z, bias=True)

        # Derived weights + launch plan. The weight loader fills the parameters
        # *after* __init__ and does it in place, so parameter identity does not
        # change and an identity check would not notice; both are therefore
        # built on the first forward and dropped whenever the parameters could
        # have been reloaded or moved.
        self._cache: tuple | None = None
        self._plan: tuple | None = None
        self._plan_tries = 0
        if hasattr(self, "register_load_state_dict_post_hook"):
            self.register_load_state_dict_post_hook(
                lambda mod, incompatible_keys: mod._invalidate())

    def _invalidate(self) -> None:
        self._cache = None
        self._plan = None
        self._plan_tries = 0

    def _apply(self, *args, **kwargs):  # .to() / .cuda() / .float()
        self._invalidate()
        return super()._apply(*args, **kwargs)

    # -- cached derived weights ------------------------------------------------
    def _build_cache(self) -> tuple:
        ln = self.layer_norm
        # [2, c_m, c_hidden]: one strided-batched GEMM for both projections.
        proj = torch.stack((self.linear_1.weight.transpose(0, 1),
                            self.linear_2.weight.transpose(0, 1))).contiguous()
        cache = (proj, ln.weight, ln.bias, ln.eps,
                 self.linear_out.weight, self.linear_out.bias)
        self._cache = cache
        return cache

    def _build_plan(self, m: torch.Tensor,
                    mask: torch.Tensor | None) -> tuple | None:
        """Resolve the fused launch for this exact input signature, if eligible.

        Everything shape- and weight-derived is settled here so the hot path is
        a few comparisons, one allocation and one launch: no shape arithmetic,
        no transposes, no per-call branching. Returns None when the signature is
        not eligible, leaving any existing plan alone -- ``forward`` checks the
        shape before using it, so a plan for another shape is harmless to keep.
        """
        h = self.c_hidden
        if not (_HAS_TRITON and mask is not None and m.is_cuda
                and m.dtype is torch.bfloat16 and mask.dtype is torch.bfloat16
                and m.is_contiguous() and mask.is_contiguous()
                and m.dim() >= 3 and m.shape[:-1] == mask.shape):
            return None
        n_seq, n_res, c_m = m.shape[-3], m.shape[-2], m.shape[-1]
        sp = max(16, _next_pow2(n_seq))
        rh, cz = n_res * h, self.c_z
        bc = 16
        bz = 32 if cz % 32 == 0 else cz
        # The kernel takes its vcat b-side branch iff SP <= 2*N_seq, and that
        # branch is only *valid* for SP == N_seq (no padding) or SP == 2*N_seq
        # (one join). N_seq in 9..15 would land inside the branch with neither --
        # a non-power-of-2 tl.arange(0, SR) and a join to the wrong row count --
        # so those shapes must not be declared eligible at all.
        vcat_ok = ((sp == n_seq or sp == 2 * n_seq) and _pow2(n_seq * n_res))
        if not (c_m == self.c_m and h >= bc and h % bc == 0
                and _pow2(n_res) and _pow2(c_m) and _pow2(h) and _pow2(sp * n_res)
                and (sp > 2 * n_seq or vcat_ok)
                # Tile budgets: the b-side LayerNorm tile, the projected B tile,
                # and the linear_out weight tile all live in registers / SMEM.
                and sp * n_res * c_m <= 16384 and sp * rh <= 8192
                and bc * h * bz * 2 <= 65536):
            return None

        nb = m.numel() // (c_m * n_seq * n_res)
        grid = (nb, n_res, cz // bz)
        wt = self.linear_out.weight.transpose(0, 1).contiguous()
        bias = self.linear_out.bias
        proj, ln_w, ln_b, ln_eps, _, _ = self._cache
        ln_eps, eps = float(ln_eps), self.eps

        def launch(m, mask, out, grid=grid):
            _opm_kernel[grid](m, mask, ln_w, ln_b, proj, wt, bias, out, eps,
                              n_seq, n_res, h, c_m, rh, cz, n_seq * n_res, sp,
                              bc, bz, ln_eps, num_warps=4)

        return (m.shape, mask.shape, m.shape[:-3] + (n_res, n_res, cz),
                m.dtype, m.device, launch)

    # -- forward --------------------------------------------------------------
    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            m:    [*, N_seq, N_res, C_m] MSA embedding
            mask: [*, N_seq, N_res] MSA mask

        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        plan = self._plan
        if (plan is not None and mask is not None
                and m.shape == plan[0] and mask.shape == plan[1]
                and m.dtype is torch.bfloat16):
            out = torch.empty(plan[2], dtype=plan[3], device=plan[4])
            plan[5](m, mask, out)
            return out

        return self._forward_eager(m, mask)

    def _forward_eager(self, m: torch.Tensor,
                       mask: torch.Tensor | None) -> torch.Tensor:
        """General path: any rank / dtype / device, ~9 launches.

        Still much leaner than the reference: LayerNorm in one launch (ATen's
        bf16 LayerNorm already accumulates in fp32), both projections as one
        batched GEMM against the stacked weight, one mask multiply covering both
        of them, and both einsums as plain matmuls over free views.
        """
        cache = self._cache
        if cache is None:
            cache = self._build_cache()
        # Try to plan a fused launch for this signature. Bounded so a caller
        # that alternates between an eligible and an ineligible shape cannot
        # thrash on plan construction forever.
        plan = self._plan
        if (plan is None or plan[0] != m.shape) and self._plan_tries < 8:
            self._plan_tries += 1
            new = self._build_plan(m, mask)
            if new is not None:
                self._plan = new
        proj, ln_w, ln_b, ln_eps, out_w, out_b = cache

        shape = m.shape
        n_seq, n_res, c_m = shape[-3], shape[-2], shape[-1]
        h = self.c_hidden
        rows = m.numel() // c_m
        nb = rows // (n_seq * n_res)

        ln = F.layer_norm(m, (c_m,), ln_w, ln_b, ln_eps)
        ab = torch.matmul(ln.reshape(1, rows, c_m), proj)

        if mask is None:
            mask = m.new_ones(shape[:-1])
        ab = ab * mask.reshape(1, rows, 1)

        # outer[b, d, c, e] = sum_a a[b, a, c] * b[d, a, e], with (residue,
        # channel) folded into the GEMM's M / N.
        rh = n_res * h
        a = ab[0].view(nb, n_seq, rh)
        b = ab[1].view(nb, n_seq, rh)
        outer = torch.matmul(a.transpose(1, 2), b)           # [nb, R*H, R*H]
        outer = (outer.view(nb, n_res, h, n_res, h)
                      .permute(0, 1, 3, 2, 4)
                      .reshape(nb, n_res, n_res, h * h))

        out = F.linear(outer, out_w, out_b)                  # [nb, R, R, c_z]

        mk = mask.reshape(nb, n_seq, n_res)
        norm = torch.matmul(mk.transpose(1, 2), mk) + self.eps
        out = out / norm.unsqueeze(-1)
        return out.view(shape[:-3] + (n_res, n_res, self.c_z))
