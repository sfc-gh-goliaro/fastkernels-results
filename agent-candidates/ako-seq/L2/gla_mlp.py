"""SwiGLU MLP for GLA / RetNet decoder layers.

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

The existing ``L2.swiglu_mlp.SwiGLUMlp`` uses a different parameter
naming scheme (``fc1_g`` / ``fc1_x`` / ``fc2``), so we keep this thin
FLA-named variant rather than remapping checkpoint keys at load time.

Built from L1 ops, plus a weight-streaming kernel for the tiny-M decode path.

Why the shape of this matters more than the arithmetic (all measured on a B200
inside the benchmark's own timing loop):

  * hidden_size=2560, intermediate_size=6912, bf16 -> gate+up+down are
    3 x 2560x6912x2B = 106 MB of weights.  A decode call does O(M) work per
    weight byte, so at M <= 256 rows this is a *weight-streaming* problem.
  * A kernel that only *reads* those 106 MB takes 27.7 us = 3.9 TB/s.  That is
    the ceiling for this working-set size, not the 5.4 TB/s the part sustains at
    2 GB, and it needs >= ~600 blocks: the same read gets 2.54 TB/s at 148
    blocks, 3.25 at 296, 3.50 at 592.  **Block count, not bandwidth, is the
    knob.**
  * Each extra kernel in the chain costs 4.05 us of measured window (2.05 us of
    device quantum + ~2.0 us of host launch).  The baseline's five launches --
    gate GEMM, up GEMM, SiLU, mul, down GEMM -- spend ~20 us on launches alone,
    and its SiLU and mul touch only M x 6912 elements each.

Two paths follow from that:

**Every shape** goes through three launches instead of five.  ``gate_proj`` and
``up_proj`` are answered by *one* GEMM against a concatenated [2I, H] weight (x
read once; 22.6 us of in-chain kernel time against 2 x 14.4 us for the pair,
because one N=13824 GEMM pipelines its weight loads better than two N=6912
ones), and SiLU+mul collapse into the frozen L1 ``silu_and_mul`` winner, whose
contract is already exactly ``out[t, j] = silu(in[t, j]) * in[t, d + j]`` with
``d = in.size(-1) / 2`` -- it consumes the concatenated [gate; up] layout with no
reshuffle.

**M <= 4** additionally leaves cuBLAS entirely for ``gla_mlp_fused.cu``, which
partitions each projection over its own *output* dimension so that every block
reads contiguous weight rows exactly once and the block count is free (864 and
640 blocks).  No GEMM tiling can do that: these are skinny GEMMs, so splitting M
multiplies the weight reads and shrinking the N tile multiplies the activation
re-reads instead.  That path reads all 106 MB at 3.71 TB/s in two PDL-chained
launches -- 33.7 us against 46.2 us for the three-launch cuBLAS chain at M=1.

cuBLAS keeps M >= 8.  The streaming kernel is FMA-based, and FP32 FMA is
56.8 TFLOPS at the locked clock against M x 106 MFLOP of work, so it stops being
memory-bound at M ~= 14; measured, it is already only break-even at M=4.  Larger
M needs tensor cores, and a Triton `tl.dot` sweep lost to cuBLAS at every decode
shape (see ITERATIONS.md) -- exactly what ``candidate/L1/linear.py`` warns about.

The concatenated weight is a *derived cache*, not a parameter: this class exists
to match FLA's checkpoint keys, so ``gate_proj.weight`` / ``up_proj.weight`` /
``down_proj.weight`` stay exactly where a checkpoint expects them and
``state_dict()`` is unchanged.  The cache is built lazily on first use and
invalidated whenever either source weight is replaced or written through (storage
pointer + version counter), plus explicitly on ``load_state_dict``.  Never per
call: a per-call ``torch.cat`` would copy 71 MB and cost more than the whole op.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.linear import Linear
from ..L1.silu import SiLU
from ..L1.silu_and_mul import SiluAndMul

try:  # pins the build arch and rebuilds when the .cu changes
    from fastkernels.infra.cuda_ext import lazy_op
    _C = lazy_op("gla_mlp_fused", "gla_mlp_fused.cu")
except Exception:  # pragma: no cover - no CUDA toolchain: cuBLAS path only
    _C = None

# Largest M the streaming path is used for, and the (gu_cfg, dn_cfg) geometry
# swept for each.  cfg ids index the GU_CASE / DN_CASE tables in the .cu; the
# sweep and the numbers behind these picks are in ITERATIONS.md.
_STREAM_MAX_M = 4
_STREAM_CFG = {
    1: (0, 5),   # BI=8/8 warps/UF=2  +  BN=8/8 warps/UF=9   -> 33.70 us
    2: (5, 0),   # BI=16/8/UF=2       +  BN=4/8/UF=3         -> 35.81 us
    3: (1, 13),
    4: (1, 13),  # BI=8/8/UF=5        +  BN=4/16/UF=3        -> 44.19 us
}

# M window in which folding the activation into the gate+up epilogue actually
# beats the 3-launch cuBLAS chain.  Measured (chain3 / fused, same GPU):
#
#   M     512   1024   2048   4096   8192  16384  65536  195661
#   ratio 0.71   0.91   1.18   1.23   1.22   1.21   1.12    0.94
#
# Below ~2048 Triton's slower GEMM (1215 TFLOPS vs cuBLAS's 1663) costs more
# than the traffic the fusion saves.  Above ~64K it loses again, and the
# captured prefill shape M=195661 is on the losing side -- benched at 18.25 ms
# against 15.97-17.09 ms for the 3-launch chain, so that case keeps cuBLAS.
# Both ends are gates on *measured* wins, like `_MM_CFG` in L1 `linear.py`.
_FUSE_MIN_M = 2048
_FUSE_MAX_M = 65536
_FUSE_CFG = (256, 128, 64, 8, 3, 8)  # BM, BN, BK, num_warps, num_stages, GROUP_M


# ---------------------------------------------------------------------------
# Prefill path: gate+up GEMM with the SiLU-and-mul folded into the epilogue.
#
# This kernel is *slower than cuBLAS at the GEMM itself* -- 1215 TFLOPS against
# cuBLAS's 1663 (74% of peak) at M=16384 -- and it still wins, because folding
# the activation into the epilogue removes 4 x M x I x 2B of intermediate
# traffic: the whole silu_and_mul pass (which reads 2*M*I and writes M*I) plus
# half of the gate/up store, since h is written directly instead of a [M, 2I]
# gate/up buffer.  At the captured prefill shape that is gigabytes.  Measured
# end to end at M=16384: 1.618 ms for the 3-launch cuBLAS chain, 1.314 ms for
# this plus a cuBLAS down GEMM.
#
# It is used *only* for large M.  At decode sizes the GEMM is launch- and
# bandwidth-bound rather than FLOP-bound, the traffic saved is a few MB, and
# Triton's disadvantage dominates (42.0 us vs 25.7 us at M=256) -- see
# ITERATIONS.md for the per-M sweep behind `_FUSE_MIN_M`.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_silu_mul(X, W, HO, M, K: tl.constexpr, I: tl.constexpr,
                      s_x, s_w, s_h,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GM: tl.constexpr, EVEN_M: tl.constexpr):
    """HO[M, I] = silu(X @ Wg^T) * (X @ Wu^T), where W is [2I, K] = [Wg; Wu].

    One A tile feeds both accumulators, so x is read once for gate and up and
    the two N tiles that the activation pairs up (columns j and I+j) are always
    resident in the same program -- which is what lets the epilogue run at all.
    """
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = I // BN
    # Grouped ordering so concurrent programs share W tiles in L2.  The
    # min() keeps the last, short group correct for any M.
    per_group = GM * nn
    gid = pid // per_group
    first = gid * GM
    gsize = tl.minimum(nm - first, GM)
    pm = first + ((pid % per_group) % gsize)
    pn = (pid % per_group) // gsize

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    ap = X + rm[:, None] * s_x + rk[None, :]
    bg = W + rn[None, :] * s_w + rk[:, None]
    bu = W + (I + rn)[None, :] * s_w + rk[:, None]
    ag = tl.zeros((BM, BN), dtype=tl.float32)
    au = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap) if EVEN_M else tl.load(ap, mask=mm[:, None], other=0.0)
        ag = tl.dot(a, tl.load(bg), ag)
        au = tl.dot(a, tl.load(bu), au)
        ap += BK
        bg += BK
        bu += BK
    # Reference order: bf16 GEMM output -> fp32 SiLU -> bf16 -> bf16 multiply.
    g16 = ag.to(tl.bfloat16)
    u16 = au.to(tl.bfloat16)
    gf = g16.to(tl.float32)
    h = (gf * tl.sigmoid(gf)).to(tl.bfloat16) * u16
    hp = HO + rm[:, None] * s_h + rn[None, :]
    if EVEN_M:
        tl.store(hp, h)
    else:
        tl.store(hp, h, mask=mm[:, None])


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
        self.act_and_mul = SiluAndMul()
        # Derived caches; see the module docstring.  Plain attributes, so
        # state_dict() still holds exactly the three checkpoint weights.
        self._gu_w: torch.Tensor | None = None
        self._gu_tag: tuple | None = None
        self._stream_ok: bool | None = None

    # -- fused gate+up weight ------------------------------------------------
    @staticmethod
    def _tag(w: torch.Tensor) -> tuple:
        """Identity of a weight's *contents* as cheaply as it can be taken:
        which storage it lives in, and how many in-place writes it has seen."""
        return (w.data_ptr(), w._version, tuple(w.shape), w.dtype, w.device)

    def _gate_up_weight(self) -> torch.Tensor | None:
        """[2I, H] concatenation of gate_proj.weight and up_proj.weight, or
        None if the two cannot be concatenated (then the caller falls back)."""
        wg = self.gate_proj.weight
        wu = self.up_proj.weight
        if (wg.ndim != 2 or wu.shape != wg.shape or wu.dtype is not wg.dtype
                or wu.device != wg.device):
            return None
        tag = (self._tag(wg), self._tag(wu))
        if self._gu_tag != tag:
            self._gu_w = torch.cat((wg.detach(), wu.detach()), 0).contiguous()
            self._gu_tag = tag
        return self._gu_w

    def _load_from_state_dict(self, *args, **kwargs):
        # A load replaces the weights the cache was derived from.  The tag guard
        # already catches this; dropping it here also frees the 71 MB at once.
        self._gu_w = None
        self._gu_tag = None
        return super()._load_from_state_dict(*args, **kwargs)

    # -- streaming path ------------------------------------------------------
    def _can_stream(self, x: torch.Tensor, w: torch.Tensor, wd: torch.Tensor) -> bool:
        if _C is None or not x.is_cuda or x.dtype is not torch.bfloat16:
            return False
        if wd.dtype is not torch.bfloat16 or not wd.is_contiguous():
            return False
        # down_proj.weight is [H, I].
        H, I = wd.shape  # noqa: E741
        # The kernel walks 16-byte payloads with a 32-lane stride, so both
        # reduction lengths must be a multiple of 8*32; it also partitions I and
        # H by the block tile, the largest of which is 16.
        if H % 256 or I % 256 or I % 16 or H % 16:
            return False
        if self._stream_ok is None:
            # One-time build check: a missing nvcc must not fail a forward.
            try:
                _C.gla_mlp_stream
                self._stream_ok = True
            except Exception:  # noqa: BLE001
                self._stream_ok = False
        return bool(self._stream_ok)

    # -- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._gate_up_weight()
        if w is None:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        wd = self.down_proj.weight
        H = w.shape[1]
        I = w.shape[0] // 2  # noqa: E741
        M = x.numel() // H if H else 0
        contig = x.is_contiguous() and x.shape[-1] == H

        # 1) Tiny M: stream all 106 MB once, two PDL-chained launches.
        if 0 < M <= _STREAM_MAX_M and contig and self._can_stream(x, w, wd):
            gu_cfg, dn_cfg = _STREAM_CFG[M]
            try:
                y = _C.gla_mlp_stream(x.view(M, H), w, wd, gu_cfg, dn_cfg)
                return y.view(x.shape[:-1] + (H,))
            except Exception:  # noqa: BLE001 - unsupported geometry: use cuBLAS
                self._stream_ok = False

        # 2) Prefill: fold the activation into the gate+up epilogue, 2 launches.
        BM, BN, BK, nw, ns, gm = _FUSE_CFG
        if (_FUSE_MIN_M <= M <= _FUSE_MAX_M and contig
                and x.dtype is torch.bfloat16
                and x.is_cuda and w.is_contiguous()
                and H % BK == 0 and I % BN == 0):
            h = torch.empty((M, I), dtype=x.dtype, device=x.device)
            _gate_up_silu_mul[(triton.cdiv(M, BM) * (I // BN),)](
                x.view(M, H), w, h, M, H, I, H, w.stride(0), I,
                BM=BM, BN=BN, BK=BK, GM=gm, EVEN_M=(M % BM == 0),
                num_warps=nw, num_stages=ns)
            return self.down_proj(h).view(x.shape[:-1] + (H,))

        # 3) Otherwise three launches: gate+up GEMM, silu_and_mul, down GEMM.
        gate_up = self.gate_proj.matmul(x, w, None)
        return self.down_proj(self.act_and_mul(gate_up))
