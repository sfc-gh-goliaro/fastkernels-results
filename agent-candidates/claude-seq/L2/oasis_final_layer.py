"""Oasis final DiT projection layer, fused into two kernels.

Baseline shape
--------------
``x`` arrives as a *permuted* view: the captured stride for
``[1, B, 9, 16, 1024]`` is ``[.., 147456, 16, 1, 144]``, i.e. the DiT keeps the
1024 channels on the **outer** axis and the 9x16 spatial grid contiguous.  The
baseline then runs ``F.layer_norm`` (which needs the normalized axis contiguous,
so torch materializes a transposed copy), a broadcast ``* (1 + scale) + shift``
over that same strided layout, and finally ``F.linear``.  Every one of those
touches ``x`` with a 288-byte element stride, so each 32-byte sector carries 2
useful bytes -- a 16x read amplification on top of five separate launches.

What this does instead
----------------------
Read ``x`` in its natural layout ``X[b, k, s]`` (k = channel, s = flattened
spatial position, contiguous) and fold the whole layer into two launches.
Writing ``xh`` for the normalized activation and ``sfac = 1 + scale``,

    out[b,s,n] = sum_k W[n,k] * (xh[b,k,s] * sfac[b,k] + shift[b,k]) + bias[n]

``mu`` / ``rstd`` come from a reduction over k, which is exactly the axis the
projection sums over, so they are not available until the K-loop ends.  Pulling
them out of the sum fixes that:

    out[b,s,n] = rstd[b,s] * (A[b,n,s] - mu[b,s] * r1[b,n]) + r2[b,n]
    A[b,n,s]  = sum_k W[n,k] * (sfac[b,k] * X[b,k,s])
    r1[b,n]   = sum_k W[n,k] * sfac[b,k]
    r2[b,n]   = sum_k W[n,k] * shift[b,k] + bias[n]

so one K-loop accumulates ``A``, ``sum_k X`` and ``sum_k X^2`` together and the
normalization becomes a two-term epilogue.  Note ``sfac`` scales ``X`` (a
BK x BSP tile) rather than ``W`` (a 64 x BK tile) -- scaling ``W`` needs an fp32
temporary 4x larger and spills.

``r1`` / ``r2`` are per-(b, n), and both collapse onto the modulation input:
since ``scale[b,k] = sum_j Wms[k,j] u[b,j] + bms[k]``,

    r1[b,n] = sum_k W[n,k](1 + bms[k])  +  sum_j (W @ Wms)[n,j] u[b,j]
    r2[b,n] = sum_k W[n,k] bms'[k] + bias[n] + sum_j (W @ Wmh)[n,j] u[b,j]

The two 64x1024 products are weight-only, so they are folded once into the
modulation matrix at ``_prepare`` time.  That drops the shift half of the
2048x1024 adaLN weight from the per-call read (4 MB -> 2.25 MB) and leaves the
projection kernel with nothing to accumulate but ``A`` and the two moments.

Measurement notes (B200, the scorer's own timer: L2 flushed per iteration)
-------------------------------------------------------------------------
On these shapes the operator moves ~4 MB, which is ~1 us of HBM traffic, so what
is actually being measured is fixed cost: an empty launch is 2.05 us and a
kernel's first memory touch adds ~2 us more.  End to end the two kernels here
measure the same 21.5 us for every captured batch size, i.e. the work itself is
already below the noise -- which is why the tuning below is about block *count*
and launch count, not bandwidth.  The modulation kernel is shaped for parallelism
(576 blocks of 4 rows) rather than large tiles: at ~50 blocks its dependent loads
cannot be hidden and the identical kernel measures 3x slower (the grid is
(ceil(NR / 4), BT) = 288 x BT blocks for the captured NR = 1152).

Two launches, not one: a single cooperative kernel with a grid barrier between
the two stages was built and measured (correct on all captured shapes).  It saves
one launch but pays ~2.1 us for the barrier, and a hand-written wmma projection
loop did not match what Triton's pipelined ``tl.dot`` gets out of the same tiles,
so it came out level at the three smaller batches and ~10% behind at the two
larger ones on both a 1155 MHz and a 1965 MHz GPU.  Not kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU


@triton.jit
def _mod_kernel(C, P, Q, MOD, NR, H: tl.constexpr, BJ: tl.constexpr, BI: tl.constexpr):
    """MOD[b, j] = Q[j] + sum_i P[j, i] * silu(C[b, i]).

    Rows 0:H hold ``sfac``; rows H:H+NF hold ``r1``; rows H+NF:H+2*NF hold
    ``r2``.  Plain fp32 FMA: the whole matvec is 7M MACs, three orders of
    magnitude below what the fp32 pipes retire in the time the 2.25 MB of ``P``
    takes to arrive, so tensor cores would buy nothing here.
    """
    b = tl.program_id(1)
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    mj = j < NR
    acc = tl.zeros((BJ,), dtype=tl.float32)
    for i0 in range(0, H, BI):
        i = i0 + tl.arange(0, BI)
        cv = tl.load(C + b * H + i).to(tl.float32)
        u = cv * tl.sigmoid(cv)
        p = tl.load(P + j[:, None] * H + i[None, :], mask=mj[:, None], other=0.).to(tl.float32)
        acc += tl.sum(p * u[None, :], 1)
    tl.store(MOD + b * NR + j, acc + tl.load(Q + j, mask=mj, other=0.), mask=mj)


@triton.jit
def _proj_kernel(X, W, MOD, OUT, SP, NST, EPS, H: tl.constexpr, NF: tl.constexpr,
                 NFP: tl.constexpr, BSP: tl.constexpr, BK: tl.constexpr,
                 NSPLIT: tl.constexpr):
    """LayerNorm + adaLN modulation + projection over one (batch, spatial) tile.

    Every pointer expression keeps a *scalar* base and builds the full offset
    inline.  Hoisting ``s`` into the base (``xb = X + ... + s``, then
    ``xb + k[:, None] * SP``) makes the base a tensor of pointers, Triton stops
    proving the access contiguous, and the tile is fetched element-by-element --
    3x slower for identical arithmetic.
    """
    pid = tl.program_id(0)
    b = pid // (NST * NSPLIT)
    r = pid % (NST * NSPLIT)
    st = r // NSPLIT
    nsp = r % NSPLIT
    s = st * BSP + tl.arange(0, BSP)
    ms = s < SP
    n = tl.arange(0, NFP)
    xb = X + b * H * SP
    wb = W + nsp * NFP * H
    modb = MOD + b * (H + 2 * NF)
    acc = tl.zeros((NFP, BSP), dtype=tl.float32)
    sx = tl.zeros((BSP,), dtype=tl.float32)
    sxx = tl.zeros((BSP,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        k = k0 + tl.arange(0, BK)
        xf = tl.load(xb + k[:, None] * SP + s[None, :], mask=ms[None, :], other=0.).to(tl.float32)
        sx += tl.sum(xf, 0)
        sxx += tl.sum(xf * xf, 0)
        sf = tl.load(modb + k)
        w = tl.load(wb + n[:, None] * H + k[None, :])
        acc = tl.dot(w, (xf * sf[:, None]).to(tl.float16), acc)
    nn = nsp * NFP + n
    r1 = tl.load(modb + H + nn)
    r2 = tl.load(modb + H + NF + nn)
    mu = sx / H
    rstd = 1.0 / tl.sqrt(sxx / H - mu * mu + EPS)
    o = (acc - mu[None, :] * r1[:, None]) * rstd[None, :] + r2[:, None]
    tl.store(OUT + b * SP * NF + s[:, None] * NF + nn[None, :],
             tl.trans(o).to(OUT.dtype.element_ty), mask=ms[:, None])


# (BJ, BI, num_warps) for the modulation matvec and (BSP, BK, num_warps,
# num_stages) for the projection, swept against the scorer's timer on B200.
_MOD_CFG = (4, 1024, 1)
_PROJ_CFG = (16, 512, 4, 3)
# Output rows one projection block owns.  32 is the swept optimum for the
# captured 64 and also caps the staged W tile at 32 KB -- at 64 rows with BK=512
# the pipelined tile no longer fits in shared memory.
_NFP = 32


class OasisFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.ModuleList(
            [
                SiLU(),
                Linear(hidden_size, 2 * hidden_size, bias=True),
            ]
        )
        self._key = None
        self._P = None
        self._Q = None

    # ------------------------------------------------------------------
    # Weight-derived constants, rebuilt only when a parameter actually moves.
    # ``_prepare_module`` rebinds ``p.data`` (dtype cast) and ``load_state_dict``
    # copies in place, so identity, storage and version are all checked.
    # ------------------------------------------------------------------
    def _prepare(self, W, bias, WM, BM):
        key = tuple((id(t), t.data_ptr(), t._version) for t in (W, bias, WM, BM))
        if key == self._key:
            return self._P, self._Q
        H = W.shape[1]
        W32 = W.float()
        Wmh, Wms = WM[:H].float(), WM[H:].float()
        bmh, bms = BM[:H].float(), BM[H:].float()
        P = torch.cat([Wms, W32 @ Wms, W32 @ Wmh]).to(WM.dtype).contiguous()
        Q = torch.cat([1.0 + bms, W32 @ (1.0 + bms), W32 @ bmh + bias.float()]).contiguous()
        self._key, self._P, self._Q = key, P, Q
        return P, Q

    def _baseline(self, x, c):
        modulation = c
        for layer in self.adaLN_modulation:
            modulation = layer(modulation)
        shift, scale = modulation.chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(-2)
            scale = scale.unsqueeze(-2)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        W, bias = self.linear.weight, self.linear.bias
        mod = self.adaLN_modulation[1]
        nb = c.dim() - 1
        H = x.shape[-1]
        if (bias is None or x.dtype != torch.float16 or W.dtype != x.dtype
                or nb < 1 or x.dim() - 1 <= nb or c.shape[-1] != H
                or W.shape[1] != H or mod.weight.shape[0] != 2 * H
                or x.shape[:nb] != c.shape[:nb] or not x.is_cuda):
            return self._baseline(x, c)
        # The permuted capture layout: hidden on the outer axis, spatial grid
        # contiguous. ``movedim`` recovers the underlying [BT, H, SP] block.
        xm = x.movedim(-1, nb)
        if not xm.is_contiguous():
            return self._baseline(x, c)

        SP = xm.stride(nb)
        BT = xm.numel() // (H * SP)
        NF = W.shape[0]
        NR = H + 2 * NF
        if SP < 16 or NF % 16:
            return self._baseline(x, c)

        BJ, BI, w1 = _MOD_CFG
        BSP, BK, w2, s2 = _PROJ_CFG
        # Both K-loops step by a constexpr and index unmasked, so shrink the step
        # to a divisor of H rather than masking a hot loop for a case the captured
        # shapes never hit (H = 1024, both steps divide it exactly).
        while H % BI:
            BI //= 2
        while H % BK:
            BK //= 2
        if BI < 16 or BK < 16:
            return self._baseline(x, c)
        P, Q = self._prepare(W, bias, mod.weight, mod.bias)
        cc = c if c.is_contiguous() else c.contiguous()
        # Split the output rows so each block owns at most _NFP of them, keeping
        # every block's row count a multiple of the 16-row MMA tile.
        NSPLIT = NF // _NFP if NF % _NFP == 0 else NF // 16

        m = torch.empty((BT, NR), device=x.device, dtype=torch.float32)
        _mod_kernel[(triton.cdiv(NR, BJ), BT)](
            cc, P, Q, m, NR, H=H, BJ=BJ, BI=BI, num_warps=w1, num_stages=3)
        out = torch.empty(x.shape[:-1] + (NF,), device=x.device, dtype=x.dtype)
        NST = triton.cdiv(SP, BSP)
        _proj_kernel[(BT * NST * NSPLIT,)](
            xm, W, m, out, SP, NST, self.norm_final.eps, H=H, NF=NF,
            NFP=NF // NSPLIT, BSP=BSP, BK=BK, NSPLIT=NSPLIT,
            num_warps=w2, num_stages=s2)
        return out
