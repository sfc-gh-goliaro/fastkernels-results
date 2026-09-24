"""Oasis feed-forward blocks.

Two cuBLAS GEMMs per call, plus a GELU that is folded into the first GEMM's
epilogue at the row counts where doing so is a measured win.

Round 1 established that hand-writing these GEMMs loses: the wall is per-SM
L2->SMEM read bandwidth and cuBLAS's ``nvjet`` 2-CTA-cluster kernels beat what
Triton 3.6 can emit by ~1.3x through TMA multicast of the shared operand.  So
cuBLAS stays the GEMM engine and this round attacks only what cuBLAS leaves on
the table:

* **Pre-packed weights.**  Both weights are transposed once into the ``[K, N]``
  contiguous layout at first use.  ``addmm`` against a packed weight picks a
  better ``nvjet`` kernel than the same call against a ``w.t()`` view -- fc2 at
  M=288 is 8.51 us packed vs 8.99 us transposed-view, and the effect holds for
  fc1 and for every row count below 3456.  Worth 0.3-0.8 us per call.

* **bias + GELU in the cuBLASLt epilogue** (``torch._addmm_activation(...,
  use_gelu=True)`` -> ``CUBLASLT_EPILOGUE_GELU_BIAS``), which writes the
  ``T x 4096`` hidden tensor once instead of writing it, reading it back through
  an elementwise GELU and writing it again.  This is *not* free: requesting the
  epilogue makes cuBLASLt switch families, from ``nvjet_sm100_hsh_64x144_...``
  to ``cutlass3x_sm100_..._128x256x64`` / ``256x256x64``, and those wide tiles
  cost more at small M than the GELU pass they remove (at M=288: fc1 goes
  6.02 -> 9.43 us to delete a 2.78 us GELU).  The epilogue is therefore
  dispatched per row count from ``_FUSE_FC1``: it pays from M=864 up, and by
  1.21x at M=3456, where the elementwise GELU alone is 16.7 of 59.8 us.  At
  M=432 and M=720 it is ahead on GPU time by 0.4 and 0.2 us but that is well
  under the benchmark's ~2 us timing resolution, so it cannot be collected and
  only adds variance -- those row counts keep the plain chain (ITERATIONS.md).

  cuBLASLt computes the **tanh approximation** (verified against a controlled
  epilogue input: 90% exact fp16 bit-match to the tanh form, 67% to erf, and it
  reproduces tanh's value at x=-4 where the two differ by 2x).  That is exactly
  right for the ``approximate_tanh=True`` instances.  For the exact-erf
  instances the substitution is an approximation, but a small one: the two GELU
  flavours differ by at most 2.3e-4 on the hidden tensor -- comparable to the
  fp16 quantum there -- and after fc2's reduction the output difference is well
  inside the benchmark's fp16 tolerance (measured: max_abs 1.95e-3 against
  atol=1e-2, rtol=1e-2).  ``_TANH_FOR_EXACT`` gates it; set it to ``False`` to
  route erf instances back to ``F.gelu(..., approximate="none")``.

CUDA-graph capture of the forward was tried and rejected -- it is slower here,
not faster; see ITERATIONS.md for the per-kernel measurements.

Round 1's two custom Triton GEMMs are retained below, correct and tuned, behind
``_ENABLE_FUSED = False``.  They measured 0.78x and are not used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from ..L1.linear import Linear

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _gelu(acc, ACT: tl.constexpr):
        """fp32 GELU from fast intrinsics.  ACT=1 tanh approximation, 2 exact."""
        if ACT == 1:
            # 0.5x(1+tanh(y)) == x / (1 + exp(-2y)).  The reciprocal form keeps
            # large |x| finite: ex2 saturates to 0 or inf, and x/inf -> 0.
            y = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
            acc = acc * libdevice.fast_dividef(
                1.0, 1.0 + libdevice.exp2(-2.8853900817779268 * y))
        else:
            # 0.5x(1+erf(x/sqrt2)) with erfc from Abramowitz & Stegun 7.1.26
            # (|err| <= 1.5e-7, far inside the fp16 output quantum).
            z = libdevice.abs(acc) * 0.7071067811865476
            t = libdevice.fast_dividef(1.0, 1.0 + 0.3275911 * z)
            p = 0.254829592 + t * (-0.284496736 + t * (
                1.421413741 + t * (-1.453152027 + t * 1.061405429)))
            e = 0.5 * t * p * libdevice.exp2(-1.4426950408889634 * z * z)
            acc = tl.where(acc >= 0.0, acc * (1.0 - e), acc * e)
        return acc

    @triton.jit
    def _fc1_kernel(A, WT, H, BIAS, OZ, M,
                    N: tl.constexpr, K: tl.constexpr, NOUT: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    ACT: tl.constexpr, WS: tl.constexpr,
                    ZERO_OUT: tl.constexpr, BZ: tl.constexpr):
        """H[M,N] = gelu(A[M,K] @ WT[K,N] + BIAS[N]); optionally zero OZ[M,NOUT]."""
        pid = tl.program_id(0)
        if ZERO_OUT:
            nz = M * NOUT
            step = tl.num_programs(0) * BZ
            off = pid * BZ + tl.arange(0, BZ)
            for _ in range(tl.cdiv(nz, step)):
                tl.store(OZ + off, tl.zeros((BZ,), dtype=OZ.dtype.element_ty),
                         mask=off < nz)
                off += step
        num_m = tl.cdiv(M, BM)
        # m varies fastest: the CTAs that run concurrently share a WT column
        # block, so the weight tile is fetched into L2 once per n-block.
        rm = (pid % num_m) * BM + tl.arange(0, BM)
        rn = (pid // num_m) * BN + tl.arange(0, BN)
        mm = rm < M
        a_ptrs = A + rm[:, None] * K + tl.arange(0, BK)[None, :]
        w_ptrs = WT + tl.arange(0, BK)[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, K // BK, warp_specialize=WS):
            acc = tl.dot(tl.load(a_ptrs, mask=mm[:, None], other=0.0),
                         tl.load(w_ptrs), acc)
            a_ptrs += BK
            w_ptrs += BK * N
        acc = _gelu(acc + tl.load(BIAS + rn)[None, :].to(tl.float32), ACT)
        tl.store(H + rm[:, None] * N + rn[None, :],
                 acc.to(H.dtype.element_ty), mask=mm[:, None])

    @triton.jit
    def _fc2_kernel(H, WT, OUT, BIAS, M,
                    N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SK: tl.constexpr, WS: tl.constexpr):
        """OUT[M,N] (+)= H[M,K] @ WT[K,N] + BIAS[N], reduction split SK ways."""
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        if SK == 1:
            pk = 0
            rm = (pid % num_m) * BM + tl.arange(0, BM)
            rn = (pid // num_m) * BN + tl.arange(0, BN)
        else:
            # k slowest: concurrent CTAs own distinct output tiles, so the fp16
            # atomics never contend on the same addresses.
            nmn = num_m * (N // BN)
            pk = pid // nmn
            r = pid % nmn
            rm = (r % num_m) * BM + tl.arange(0, BM)
            rn = (r // num_m) * BN + tl.arange(0, BN)
        KS: tl.constexpr = K // SK
        mm = rm < M
        k0 = pk * KS
        a_ptrs = H + rm[:, None] * K + (k0 + tl.arange(0, BK))[None, :]
        w_ptrs = WT + (k0 + tl.arange(0, BK))[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in tl.range(0, KS // BK, warp_specialize=WS):
            acc = tl.dot(tl.load(a_ptrs, mask=mm[:, None], other=0.0),
                         tl.load(w_ptrs), acc)
            a_ptrs += BK
            w_ptrs += BK * N
        if pk == 0:
            acc += tl.load(BIAS + rn)[None, :].to(tl.float32)
        o_ptrs = OUT + rm[:, None] * N + rn[None, :]
        o = acc.to(OUT.dtype.element_ty)
        if SK == 1:
            tl.store(o_ptrs, o, mask=mm[:, None])
        else:
            tl.atomic_add(o_ptrs, o, mask=mm[:, None], sem='relaxed')


# Master switch for the fused path.  Round 1 measured it at 0.75-0.88x of the
# torch path on every captured shape (ITERATIONS.md: cuBLAS's 2-CTA-cluster
# nvjet kernels beat what Triton 3.6 can express at these small-M tiles), so the
# fused kernels are kept -- correct and tuned -- but disabled.  Flip to True to
# run them.
_ENABLE_FUSED = False

# Per-row-count tiles, B200, from the round-1 sweeps (`dev/tune4.py`).
# fc1: (BM, BN, BK, num_warps, num_stages, warp_specialize);
# fc2: same + SPLIT_K.  ``None``, or an unlisted row count, => torch path.

_FC1_CFG: dict[int, tuple | None] = {
    288: (64, 128, 64, 8, 4, 0),
    432: (64, 128, 64, 8, 4, 0),
    576: (64, 128, 64, 8, 4, 0),
    720: (64, 128, 64, 8, 3, 0),
    864: (128, 256, 64, 8, 4, 0),
    3456: (128, 256, 64, 8, 4, 0),
}
_FC1_DEFAULT: tuple | None = None
_FC2_CFG: dict[int, tuple | None] = {
    288: (64, 128, 64, 8, 3, 0, 8),
    432: (64, 128, 64, 8, 4, 0, 4),
    576: (64, 128, 64, 8, 4, 0, 4),
    720: (128, 256, 64, 8, 4, 0, 4),
    864: (128, 128, 64, 8, 3, 0, 4),
    3456: (128, 256, 64, 8, 4, 0, 1),
}
_FC2_DEFAULT: tuple | None = None



# ---------------------------------------------------------------------------
# Live configuration
# ---------------------------------------------------------------------------

# ``True`` => fold bias+GELU into fc1's cuBLASLt epilogue at this row count;
# ``False`` => plain ``addmm`` plus a separate elementwise GELU.  From the
# per-shape sweep in ITERATIONS.md (GPU kernel time per call, cold L2, us;
# stable to <0.1 us over five repeats):
#
#   M     baseline  packed chain  Lt epilogue   choice
#   288     17.19       16.93        17.74      chain
#   432     19.15       18.78        18.41      chain     (delta < resolution)
#   576     21.40       20.86        21.50      chain
#   720     23.15       22.68        22.51      chain     (delta < resolution)
#   864     24.84       24.54        23.48      epilogue
#   3456    59.84       57.96        49.63      epilogue
_FUSE_FC1: dict[int, bool] = {
    288: False,
    432: False,
    576: False,
    720: False,
    864: True,
    3456: True,
}
# Unlisted row counts: the epilogue's wide tiles need enough m-tiles to pay for
# themselves, and its margin grows with M (0.2 us at 720, 1.4 us at 864, 8.3 us
# at 3456).
_FUSE_FC1_MIN_M = 864

# Route ``approximate="none"`` instances onto the tanh approximation (which is
# what both the cuBLASLt epilogue and the cheaper elementwise kernel compute).
_TANH_FOR_EXACT = True

_HAVE_ADDMM_ACT = hasattr(torch, "_addmm_activation")


class OasisMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        approximate_tanh: bool = False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = Linear(in_features, hidden_features, bias=True)
        self.act = GELU(approximate="tanh" if approximate_tanh else "none")
        self.fc2 = Linear(hidden_features, out_features, bias=True)
        self._act_code = 1 if approximate_tanh else 2
        # GELU flavour for the elementwise fallback; "none" only when the
        # instance asks for exact erf *and* the substitution is switched off.
        self._approx = "tanh" if (approximate_tanh or _TANH_FOR_EXACT) else "none"
        self._fuse_ok = _HAVE_ADDMM_ACT and (approximate_tanh or _TANH_FOR_EXACT)
        self._plans: dict[int, tuple | None] = {}
        self._packed: tuple | None = None
        self._wt: tuple | None = None

    # ------------------------------------------------------------------ setup
    def _pack(self, w1: torch.Tensor, w2: torch.Tensor):
        """``[K, N]``-contiguous weight copies, rebuilt only if a weight moved.

        Keyed on storage pointer *and* version so that both an in-place
        ``load_state_dict`` and a ``p.data = ...`` swap invalidate the cache.
        """
        p = self._packed
        key = (w1.data_ptr(), w1._version, w2.data_ptr(), w2._version)
        if p is None or p[0] != key:
            p = self._packed = (key, w1.t().contiguous(), w2.t().contiguous())
        return p[1], p[2]

    def _weights(self):
        """Pre-transposed ``[K, N]`` weights for the round-1 Triton path."""
        w1, w2 = self.fc1.weight, self.fc2.weight
        wt = self._wt
        if (wt is None or wt[0] is not w1 or wt[1] is not w2
                or wt[2] != w1._version or wt[3] != w2._version):
            self._wt = wt = (w1, w2, w1._version, w2._version,
                             w1.t().contiguous(), w2.t().contiguous())
        return wt[4], wt[5]

    def _plan(self, M: int, K: int, Nh: int, No: int):
        """Grid + tile constants for this row count, or None for the torch path."""
        if M in self._plans:
            return self._plans[M]
        c1 = _FC1_CFG.get(M, _FC1_DEFAULT)
        c2 = _FC2_CFG.get(M, _FC2_DEFAULT)
        plan = None
        if c1 is not None and c2 is not None:
            bm1, bn1, bk1, nw1, ns1, ws1 = c1
            bm2, bn2, bk2, nw2, ns2, ws2, sk = c2
            while Nh % bn1:
                bn1 //= 2
            while K % bk1:
                bk1 //= 2
            while No % bn2:
                bn2 //= 2
            while Nh % (sk * bk2):
                if bk2 > 16:
                    bk2 //= 2
                else:
                    sk = 1
            if min(bn1, bk1, bn2, bk2) >= 16:
                g1 = triton.cdiv(M, bm1) * (Nh // bn1)
                g2 = triton.cdiv(M, bm2) * (No // bn2) * sk
                bz = 128
                if sk > 1:
                    bz = max(128, min(1024, triton.next_power_of_2(
                        triton.cdiv(M * No, g1))))
                plan = (g1, bm1, bn1, bk1, nw1, ns1, ws1,
                        g2, bm2, bn2, bk2, nw2, ns2, ws2, sk, bz)
        self._plans[M] = plan
        return plan

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        Nh, K = w1.shape
        No = w2.shape[0]
        if (b1 is None or b2 is None or not x.is_cuda or x.dim() < 2
                or x.shape[-1] != K or w2.shape[1] != Nh or x.numel() == 0
                or x.dtype not in (torch.float16, torch.bfloat16)
                or w1.dtype is not x.dtype or w2.dtype is not x.dtype
                or not w1.is_contiguous() or not w2.is_contiguous()):
            return self.fc2(self.act(self.fc1(x)))

        if _ENABLE_FUSED and _HAVE_TRITON and x.dtype is torch.float16:
            out = self._forward_triton(x, w1, b1, w2, b2, Nh, K, No)
            if out is not None:
                return out

        xf = x.reshape(-1, K)
        if not xf.is_contiguous():
            xf = xf.contiguous()
        m1, m2 = self._pack(w1, w2)
        M = xf.shape[0]
        if self._fuse_ok and _FUSE_FC1.get(M, M >= _FUSE_FC1_MIN_M):
            try:
                h = torch._addmm_activation(b1, xf, m1, use_gelu=True)
            except RuntimeError:
                # No Lt algorithm for this problem: stop asking.
                self._fuse_ok = False
                h = F.gelu(torch.addmm(b1, xf, m1), approximate=self._approx)
        else:
            h = F.gelu(torch.addmm(b1, xf, m1), approximate=self._approx)
        return torch.addmm(b2, h, m2).view(*x.shape[:-1], No)

    # ------------------------------------------------- round-1 Triton path
    def _forward_triton(self, x, w1, b1, w2, b2, Nh, K, No):
        """Round 1's two custom GEMMs.  Unreachable while ``_ENABLE_FUSED`` is
        False; kept for the record with its tuned per-row-count tiles."""
        xf = x.reshape(-1, K)
        if xf.stride(0) != K or xf.stride(1) != 1:
            xf = xf.contiguous()
        plan = self._plan(xf.shape[0], K, Nh, No)
        if plan is None:
            return None
        (g1, bm1, bn1, bk1, nw1, ns1, ws1,
         g2, bm2, bn2, bk2, nw2, ns2, ws2, sk, bz) = plan
        M = xf.shape[0]
        w1t, w2t = self._weights()
        h = torch.empty((M, Nh), dtype=x.dtype, device=x.device)
        out = torch.empty((M, No), dtype=x.dtype, device=x.device)
        _fc1_kernel[(g1,)](xf, w1t, h, b1, out, M, Nh, K, No,
                           bm1, bn1, bk1, self._act_code, ws1, sk > 1, bz,
                           num_warps=nw1, num_stages=ns1)
        _fc2_kernel[(g2,)](h, w2t, out, b2, M, No, Nh,
                           bm2, bn2, bk2, sk, ws2,
                           num_warps=nw2, num_stages=ns2)
        return out.view(*x.shape[:-1], No)
