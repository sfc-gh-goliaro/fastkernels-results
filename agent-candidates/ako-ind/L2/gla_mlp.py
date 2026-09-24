"""Fused SwiGLU MLP for GLA / RetNet decoder layers (Triton, B200 / sm_100).

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

What the shape actually is
--------------------------
hidden=2560, intermediate=6912, so the three weight matrices are 3 x 35.4 MB =
106 MB while the activations are tiny: the benchmarked shapes are
M = batch*seq in {1, 64, 116, 256} plus one very large prefill (M = 195661).
Four of five scenarios are therefore pure *weight streaming* -- the floor is one
pass over 106 MB of HBM -- and the fifth is compute-bound.

Where the eager baseline's time goes (measured on B200 under this benchmark's
timing regime, which flushes L2 before every iteration):

* **~4.1 us of GPU-timeline cost per kernel launch.**  The eager forward is 5
  launches (gate GEMM, up GEMM, silu, mul, down GEMM), i.e. ~20 us of pure
  overhead -- about half of its 46 us at M=1.
* the ``silu`` / ``mul`` pair also round-trips the M x 6912 intermediate through
  memory three times.

cuBLAS is *close to* the HBM roofline for these skinny projections and
hand-written Triton GEMMs lose to it at every M >= 3.  So the wins here are
(a) collapsing 5 launches into 2-3, (b) keeping the intermediate out of HBM,
(c) handing cuBLAS the operand layout it actually wants at each M, and (d) at
M <= 2, where the whole MLP is a GEMV that Triton *can* carry, telling L2 not to
cache the weight stream.

One thing to know before re-deriving any of this: under this benchmark's timing
regime the achievable read bandwidth depends strongly on transfer size -- 1.99
TB/s for 35.4 MB, 2.70 for 70.8 MB, 2.90 for 106 MB, 3.32 for 212 MB -- because
the harness writes 252 MB of zeros to flush L2 immediately before each timed
iteration and that write traffic is still draining underneath the kernel.  A
stage timed on its own is therefore not a floor you can add up, and the ~4 TB/s
"roofline" that a large-transfer measurement suggests is not reachable here.

Three paths, dispatched on M
----------------------------
``M <= 2`` -- ``_row_gate_up_act`` does ``silu(x @ Wg^T) * (x @ Wu^T)`` in one
launch.  One intermediate row per CTA, so each CTA streams exactly one
*contiguous* 5120 B row of Wg and of Wu; the K loop only accumulates per-lane
partials and a single reduction closes each row.  A conventional ``[BN, BK]``
``tl.dot`` tiling of the same GEMM stalls ~30% short and is not rescued by
split-K or larger BK; nor is a slab of BI rows per CTA, which ties this form
standalone but loses 2 us end-to-end.
The two weight loads are ``evict_first``: every weight byte is read exactly
once, so the only thing L2 residency can buy is a place for x -- which *is*
re-read, by all 6912 CTAs.  That single hint is worth 3.02 -> 3.64 TB/s on the
70.8 MB pass and 37.8 -> 33.8 us on the ``[1,1,2560]`` scenario.
The row layout does not generalize past M~2 because every CTA re-reads all of x
(traffic = #CTAs x M x K x 2, which is 35 MB at M=1 but 2 GB at M=60); above
that the cuBLAS path below is faster, and at M=1 it is not close (33.8 us
against 43.0).

``M <= 512`` (decode) -- one ``gate|up`` GEMM against the concatenated weight,
then ``_silu_mul``, then the down GEMM: 3 launches instead of 5, and the
intermediate is written once rather than three times.  Both GEMMs get a
**pre-transposed, contiguous** B operand ("NN"), which cuBLAS prefers at small M.

``M > 512`` (prefill) -- same fusion, but the gate|up GEMM takes a ``.t()``
view of a ``[2I, K]`` contiguous weight ("NT"), which is what cuBLAS prefers
once M is large: at M=195661 NN costs 17.0 ms against 16.6 ms for NT, and two
separate NT GEMMs cost 17.0-17.3 ms against 16.6 for the concatenated one.  Work
is chunked over M to bound both the intermediate allocation and the int32
element offsets.

Everything is bf16 in / bf16 out with fp32 accumulation.  ``_row_gate_up_act``
rounds the two projections to bf16 before the activation, matching the
baseline's numerics (its gate/up GEMMs also land in bf16).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.silu import SiLU

# M at or below this uses the row-per-CTA fused path (one launch for gate+up+act).
_ROW_MAX = 2
# M at or below this hands cuBLAS pre-transposed contiguous operands ("NN") and
# fuses gate+up into one GEMM.  Above it, `.t()` views ("NT") + two GEMMs win.
# Measured crossover: NN wins to M=512, NT from M=1024.
_NN_MAX = 512
# _silu_mul launch shape.
_ACT_BLK = 4096
_ACT_WARPS = 8


def _chunk_rows(inter: int) -> int:
    """Rows per gate/up GEMM on the large-M path.

    A ``[M, 2*inter]`` bf16 buffer overflows int32 element offsets -- and
    cuBLAS -- once ``M * 2 * inter > 2**31`` (M ~ 155k at inter=6912), which
    surfaces as CUBLAS_STATUS_EXECUTION_FAILED followed by an illegal memory
    access.  Chunking bounds the offsets (2x margin) and the allocation.  Chunks
    want to be as large as that allows, because each one re-streams the 106 MB
    of weights.
    """
    return max(1024, (2 ** 31 - 1) // (4 * inter))


@triton.jit
def _row_gate_up_act(X, WG, WU, HO, M, K: tl.constexpr, N: tl.constexpr,
                     BM: tl.constexpr, BK: tl.constexpr):
    """h[:, n] = silu(x @ Wg[n]) * (x @ Wu[n]) -- one intermediate row per CTA.

    Both weight rows are read as a single contiguous run each.  The K loop
    accumulates per-lane partials into a ``[BM, BK]`` tile so there is exactly
    one cross-lane reduction per row; reducing inside the loop instead is ~2x
    slower.
    """
    n = tl.program_id(0)
    rm = tl.arange(0, BM)
    rk = tl.arange(0, BK)
    row_ok = rm < M
    mask = row_ok[:, None]
    ag = tl.zeros([BM, BK], tl.float32)
    au = tl.zeros([BM, BK], tl.float32)
    xp = X + rm[:, None] * K + rk[None, :]
    gp = WG + n * K + rk
    up = WU + n * K + rk
    for _ in range(0, K // BK):
        xt = tl.load(xp, mask=mask, other=0.0).to(tl.float32)
        # `evict_first` on the two weight streams: each weight byte is read
        # exactly once, so caching it only evicts the one line that *is* reused
        # (x, which every CTA re-reads).  Worth 3.6 TB/s against 3.0 -- see the
        # cost-model note in ITERATIONS.md.
        ag += xt * tl.load(gp, eviction_policy="evict_first").to(tl.float32)[None, :]
        au += xt * tl.load(up, eviction_policy="evict_first").to(tl.float32)[None, :]
        xp += BK
        gp += BK
        up += BK
    # Round to the storage dtype before the activation: the baseline's gate/up
    # GEMMs land in bf16, so matching that keeps the comparison tight.
    ot = HO.dtype.element_ty
    g = tl.sum(ag, 1).to(ot).to(tl.float32)
    u = tl.sum(au, 1).to(ot).to(tl.float32)
    tl.store(HO + rm * N + n, ((g * tl.sigmoid(g)) * u).to(ot), mask=row_ok)


@triton.jit
def _silu_mul(G, U, HO, NTOT, N: tl.constexpr, SG: tl.constexpr,
              BLK: tl.constexpr):
    """h = silu(G) * U, row stride ``SG`` on both inputs, contiguous ``[.., N]`` out.

    ``SG`` lets one kernel serve both callers: a single ``[M, 2N]`` gate|up
    buffer (G = buf, U = buf[:, N:], SG = 2N) and two separate ``[M, N]``
    buffers (SG = N).
    """
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < NTOT
    src = (off // N) * SG + (off % N)
    g = tl.load(G + src, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + src, mask=m, other=0.0).to(tl.float32)
    tl.store(HO + off, ((g * tl.sigmoid(g)) * u).to(HO.dtype.element_ty), mask=m)


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
        # Derived weight layouts, each built on first use of the path that
        # needs it.  A plain dict attribute, so it stays out of state_dict.
        #   gu_nn [K, 2I] contiguous   gate|up B operand, NN  (small M)
        #   dn_nn [I, H]  contiguous   down    B operand, NN  (small M)
        #   gu_nt [2I, K] contiguous   gate|up, used as .t()  (large M)
        self._cache: dict = {}
        # Weight loading mutates the parameters in place, which `_wkey` below
        # cannot see (`.data.copy_` does not bump `_version`, and `data_ptr`
        # is unchanged), so drop the derived layouts whenever a state dict is
        # loaded.  Registered on self; fires for nested loads too.
        self._register_load_state_dict_pre_hook(self._drop_derived)
        post = getattr(self, "register_load_state_dict_post_hook", None)
        if post is not None:
            post(lambda mod, _keys: mod._cache.clear())

    def _drop_derived(self, *_args, **_kwargs) -> None:
        self._cache.clear()

    @staticmethod
    def _wkey(*ws: torch.Tensor):
        """Storage identity, so a `.to()` / parameter reassignment invalidates
        every derived layout.  In-place weight edits are caught by the
        load_state_dict hooks above, not by this key."""
        return tuple((w.data_ptr(), tuple(w.shape), w.dtype, w.device)
                     for w in ws)

    def _derived(self, name: str, build, *ws: torch.Tensor) -> torch.Tensor:
        key = self._wkey(*ws)
        hit = self._cache.get(name)
        if hit is None or hit[0] != key:
            with torch.no_grad():
                hit = (key, build())
            self._cache[name] = hit
        return hit[1]

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gw = self.gate_proj.weight
        uw = self.up_proj.weight
        dw = self.down_proj.weight
        inter, hidden = gw.shape
        # Anything this fast path does not cover -- non-CUDA, other dtypes,
        # mismatched/non-contiguous weights, autograd (the `out=` GEMMs are not
        # differentiable), empty input -- falls back to the reference formula.
        if (not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16)
                or x.dtype != gw.dtype or gw.dtype != dw.dtype
                or x.shape[-1] != hidden or uw.shape != gw.shape
                or dw.shape[1] != inter or x.numel() == 0
                or x.numel() % hidden
                or not (gw.is_contiguous() and uw.is_contiguous()
                        and dw.is_contiguous())
                or torch.is_grad_enabled()):
            return self._eager(x)

        x2 = x.reshape(-1, hidden)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        M = x2.shape[0]
        out_shape = x.shape[:-1] + (dw.shape[0],)
        h = torch.empty((M, inter), dtype=x.dtype, device=x.device)

        if M <= _NN_MAX:
            dn_nn = self._derived(
                "dn_nn", lambda: dw.detach().t().contiguous(), dw)
            if M <= _ROW_MAX:
                _row_gate_up_act[(inter,)](
                    x2, gw, uw, h, M, K=hidden, N=inter,
                    BM=triton.next_power_of_2(M), BK=512,
                    num_warps=4, num_stages=4,
                )
            else:
                gu_nn = self._derived(
                    "gu_nn",
                    lambda: torch.cat((gw.detach(), uw.detach()), 0).t().contiguous(),
                    gw, uw)
                gu = torch.empty((M, 2 * inter), dtype=x.dtype, device=x.device)
                torch.mm(x2, gu_nn, out=gu)
                ntot = M * inter
                _silu_mul[(triton.cdiv(ntot, _ACT_BLK),)](
                    gu, gu[:, inter:], h, ntot, N=inter, SG=2 * inter,
                    BLK=_ACT_BLK, num_warps=_ACT_WARPS,
                )
            return torch.mm(h, dn_nn).view(out_shape)

        # Large M: NT operands, chunked over rows.
        gu_nt = self._derived(
            "gu_nt", lambda: torch.cat((gw.detach(), uw.detach()), 0).contiguous(),
            gw, uw).t()
        step = min(M, _chunk_rows(inter))
        gu = torch.empty((step, 2 * inter), dtype=x.dtype, device=x.device)
        for lo in range(0, M, step):
            rows = min(step, M - lo)
            guc = gu[:rows]
            torch.mm(x2[lo:lo + rows], gu_nt, out=guc)
            ntot = rows * inter
            _silu_mul[(triton.cdiv(ntot, _ACT_BLK),)](
                guc, guc[:, inter:], h[lo:lo + rows], ntot, N=inter,
                SG=2 * inter, BLK=_ACT_BLK, num_warps=_ACT_WARPS,
            )
        return torch.mm(h, dw.t()).view(out_shape)
