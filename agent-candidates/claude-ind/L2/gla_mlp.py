"""SwiGLU MLP for GLA / RetNet decoder layers (fused Triton implementation).

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

The captured workload is dominated by decode-shaped calls (``M`` between 1 and a
few hundred rows against ``2560 x 6912`` projections), where the eager
``gate -> up -> silu -> mul -> down`` chain costs five kernel launches and
streams the ``6912 x 2560`` intermediate through HBM three extra times.  This
implementation collapses the whole block into two kernels:

* ``_gu_kernel``  -- one GEMM over the concatenated ``[gate; up]`` weight that
  applies ``silu(gate) * up`` on the fp32 accumulators before writing ``h``.
* ``_dn_kernel``  -- the down projection.  ``N = hidden_size`` alone does not
  produce enough CTAs to saturate HBM at decode shapes, so it is split along
  ``K`` and the partials are combined with atomics (``_gu_kernel`` pre-zeroes
  the output for that path, keeping the launch count at two).

Weights are repacked once into the ``[K, N]`` layout the kernels want and cached
on the module; anything the kernels cannot handle falls back to the eager path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.linear import Linear
from ..L1.silu import SiLU

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is expected to be present
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _gu_kernel(X, W, HO, OZ, M, ZN,
                   N: tl.constexpr, K: tl.constexpr, N2: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GM: tl.constexpr, EV: tl.constexpr, EVEN_K: tl.constexpr,
                   ZERO: tl.constexpr, ZB: tl.constexpr):
        """h = silu(x @ Wg^T) * (x @ Wu^T), with W = [K, 2N] = [Wg^T | Wu^T]."""
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn: tl.constexpr = N // BN
        nig: tl.constexpr = GM * npn
        gid = pid // nig
        fm = gid * GM
        gs = min(npm - fm, GM)
        pid_m = fm + ((pid % nig) % gs)
        pid_n = (pid % nig) // gs

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        mm = offs_m < M
        xp = X + offs_m[:, None] * K + offs_k[None, :]
        gp = W + offs_k[:, None] * N2 + offs_n[None, :]
        ag = tl.zeros([BM, BN], tl.float32)
        au = tl.zeros([BM, BN], tl.float32)
        for k in range(0, K, BK):
            if EVEN_K:
                a = tl.load(xp, mask=mm[:, None], other=0.0)
                bg = tl.load(gp, eviction_policy=EV)
                bu = tl.load(gp + N, eviction_policy=EV)
            else:
                kmask = offs_k + k < K
                a = tl.load(xp, mask=mm[:, None] & kmask[None, :], other=0.0)
                bg = tl.load(gp, mask=kmask[:, None], other=0.0, eviction_policy=EV)
                bu = tl.load(gp + N, mask=kmask[:, None], other=0.0, eviction_policy=EV)
            ag = tl.dot(a, bg, ag)
            au = tl.dot(a, bu, au)
            xp += BK
            gp += BK * N2
        # Round exactly where the eager chain does (bf16 gate/up outputs, then a
        # bf16 silu result) so ``h`` matches the reference bit-for-bit and the
        # error budget is spent only on the down projection.
        ety: tl.constexpr = HO.dtype.element_ty
        gb = ag.to(ety).to(tl.float32)
        s = (gb * tl.sigmoid(gb)).to(ety).to(tl.float32)
        h = (s * au.to(ety).to(tl.float32)).to(ety)
        tl.store(HO + offs_m[:, None] * N + offs_n[None, :], h, mask=mm[:, None])

        if ZERO:
            # Pre-zero the atomic accumulation target of the down projection so
            # that path stays a single extra launch.
            ng = tl.num_programs(0)
            zo = tl.arange(0, ZB)
            z = tl.zeros([ZB], dtype=OZ.dtype.element_ty)
            for j in range(tl.cdiv(ZN, ZB * ng)):
                idx = (j * ng + pid) * ZB + zo
                tl.store(OZ + idx, z, mask=idx < ZN)

    @triton.jit
    def _dn_kernel(HI, W, O, M,
                   N: tl.constexpr, K: tl.constexpr, KC: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GM: tl.constexpr, EV: tl.constexpr, EVEN_K: tl.constexpr,
                   ATOMIC: tl.constexpr):
        """out[:, :] (+)= h[:, kc] @ Wd^T[kc, :], with W = [K, N] = Wd^T."""
        pid = tl.program_id(0)
        k0 = tl.program_id(1) * KC
        npm = tl.cdiv(M, BM)
        npn: tl.constexpr = N // BN
        nig: tl.constexpr = GM * npn
        gid = pid // nig
        fm = gid * GM
        gs = min(npm - fm, GM)
        pid_m = fm + ((pid % nig) % gs)
        pid_n = (pid % nig) // gs

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = k0 + tl.arange(0, BK)
        mm = offs_m < M
        hp = HI + offs_m[:, None] * K + offs_k[None, :]
        wp = W + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros([BM, BN], tl.float32)
        for k in range(0, KC, BK):
            if EVEN_K:
                a = tl.load(hp, mask=mm[:, None], other=0.0)
                b = tl.load(wp, eviction_policy=EV)
            else:
                kmask = offs_k + k < K
                a = tl.load(hp, mask=mm[:, None] & kmask[None, :], other=0.0)
                b = tl.load(wp, mask=kmask[:, None], other=0.0, eviction_policy=EV)
            acc = tl.dot(a, b, acc)
            hp += BK
            wp += BK * N
        op = O + offs_m[:, None] * N + offs_n[None, :]
        v = acc.to(O.dtype.element_ty)
        if ATOMIC:
            tl.atomic_add(op, v, mask=mm[:, None], sem='relaxed')
        else:
            tl.store(op, v, mask=mm[:, None])


# Per-M-bucket schedules, autotuned on the captured shapes (B200, sm_100).
# A: (BM, BN, BK, num_stages, num_warps, GROUP_M, eviction_policy)
_A_CFG = [
    (16, (16, 64, 256, 4, 4, 1, '')),
    (64, (64, 64, 64, 8, 8, 8, '')),
    (128, (128, 64, 64, 6, 8, 1, '')),
    (2048, (128, 128, 64, 4, 8, 1, '')),
    (1 << 60, (128, 128, 32, 4, 8, 16, '')),
]
# B: (BM, BN, BK, SPLIT_K, num_stages, num_warps, GROUP_M, eviction_policy)
_B_CFG = [
    (16, (16, 32, 64, 9, 6, 4, 8, '')),
    (64, (64, 128, 64, 6, 6, 4, 8, '')),
    (128, (128, 128, 64, 6, 6, 8, 8, '')),
    (2048, (128, 256, 64, 6, 4, 8, 8, '')),
    (1 << 60, (256, 256, 64, 1, 3, 8, 4, '')),
]


def _pick(table, m):
    for lim, cfg in table:
        if m <= lim:
            return cfg
    return table[-1][1]


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
        self._packed = None
        self._pack_key = None

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    def _prepare(self):
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        wd = self.down_proj.weight
        key = (wg.data_ptr(), wu.data_ptr(), wd.data_ptr(),
               wg._version, wu._version, wd._version)
        if self._pack_key != key:
            with torch.no_grad():
                wgu = torch.empty(wg.shape[1], 2 * wg.shape[0],
                                  dtype=wg.dtype, device=wg.device)
                wgu[:, :wg.shape[0]] = wg.t()
                wgu[:, wg.shape[0]:] = wu.t()
                self._packed = (wgu, wd.t().contiguous())
            self._pack_key = key
        return self._packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        wd = self.down_proj.weight
        I, K = wg.shape          # intermediate, hidden
        H = wd.shape[0]          # hidden
        M = x.numel() // K if K else 0
        if (not _HAVE_TRITON
                or x.dtype not in (torch.bfloat16, torch.float16)
                or x.dtype != wg.dtype or wu.dtype != wg.dtype or wd.dtype != wg.dtype
                or not x.is_cuda or not x.is_contiguous()
                or x.shape[-1] != K or wu.shape != wg.shape or wd.shape[1] != I
                or K != H or M == 0
                or M * max(I, H) >= 2 ** 31):   # 32-bit tile indexing in the kernels
            return self._eager(x)

        ca = _pick(_A_CFG, M)
        cb = _pick(_B_CFG, M)
        BMa, BNa, BKa, nsa, nwa, GMa, EVa = ca
        BMb, BNb, BKb, SK, nsb, nwb, GMb, EVb = cb
        if I % BNa or H % BNb:
            return self._eager(x)
        KC = I // SK
        if SK * KC != I or KC % BKb:
            SK, KC = 1, I
        atomic = SK > 1

        try:
            wgu, wdt = self._prepare()
            xf = x.reshape(M, K)
            h = torch.empty(M, I, dtype=x.dtype, device=x.device)
            out = torch.empty(M, H, dtype=x.dtype, device=x.device)
            _gu_kernel[(triton.cdiv(M, BMa) * (I // BNa),)](
                xf, wgu, h, out, M, M * H, I, K, 2 * I,
                BMa, BNa, BKa, GMa, EVa, K % BKa == 0, atomic, 1024,
                num_stages=nsa, num_warps=nwa)
            _dn_kernel[(triton.cdiv(M, BMb) * (H // BNb), SK)](
                h, wdt, out, M, H, I, KC,
                BMb, BNb, BKb, GMb, EVb, KC % BKb == 0, atomic,
                num_stages=nsb, num_warps=nwb)
        except Exception:
            # Unsupported schedule for this device (e.g. shared-memory limits):
            # recompute from scratch on the reference path rather than fail.
            return self._eager(x)
        return out.view(*x.shape[:-1], H)
