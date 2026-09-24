"""SwiGLU MLP for GLA / RetNet decoder layers.

Three-projection variant matching FLA's checkpoint format:
  ``gate_proj.weight`` / ``up_proj.weight`` / ``down_proj.weight``

Two things make this op slow in its baseline form, and both are about *kernel
count*, not about the matmuls.

Every captured decode shape is a skinny MLP: ``M`` rows of 2560 against 106 MB
of weights (3 x 2560x6912 bf16).  The arithmetic is small next to the weight
traffic, so the op is bound by streaming those 106 MB out of HBM -- and by the
fixed cost of each kernel that does part of the streaming.  Measured on this
B200 inside the scorer's own timing loop (which writes 2xL2 of zeros before
every iteration, so the weights are *never* resident), reading all 106 MB from
one kernel costs ~34 us, while a kernel that reads nothing at all still costs
~4 us, and splitting the same 106 MB across n kernels costs ~5 us more per extra
kernel.  The baseline spends five kernels -- gate GEMM, up GEMM, silu, mul, down
GEMM -- on a problem whose floor is a single weight sweep.

Second, the two elementwise passes are not free: ``silu(g)`` and ``* u`` write
and re-read ``M x 6912`` bf16 three more times on top of the GEMM outputs.

So this candidate never runs more than three kernels, and on the decode shapes
only two:

* :func:`_gu_kernel` (ours) -- ``h = silu(x @ Wg.T) * (x @ Wu.T)`` in one pass,
  replacing two GEMMs and both elementwise kernels.
* the frozen L1 ``Linear`` for ``out = h @ Wd.T``.

``h`` is at most 3.5 MB on a decode shape, so it stays in L2 between the two and
costs essentially nothing.

The interleaved weight pack
---------------------------
The obvious way to fuse the two input projections is two accumulators and two
``tl.dot``s per K-step.  That measured **1.5x slower** than a plain single-
accumulator GEMM of the same total size (53 us vs 36 us at 256 rows): two
accumulators double the TMEM footprint and halve the MMA pipeline depth on
sm_100.

So the gate and up weights are pre-packed **row-interleaved** into one
``[2*I, K]`` tensor (``row 2j = Wg[j]``, ``row 2j+1 = Wu[j]``).  A block then
issues one ordinary ``tl.dot`` over a ``2*BLOCK_N``-wide tile of it, and its
accumulator comes out as ``[BLOCK_M, BLOCK_N, 2]`` after a reshape -- so
``tl.split`` hands back the gate and up halves for free.  One accumulator, one
MMA per K-step, and the activation becomes a register-level epilogue.  The pack
is built once, lazily, and cached against the parameters'
``data_ptr``/``_version``.

Numerics.  The reference rounds to bf16 between every stage (GEMM out, silu out,
product), so the epilogue reproduces that rounding chain instead of carrying the
f32 accumulator straight through.  With ``BLOCK_K`` at 128 the K-accumulation
order matches too, and the fused path comes out *bit-identical* to the reference
on the captured decode shapes (0 max abs error under ``fastkernels bench``).

Where the fused kernel is *not* used
------------------------------------
Past ~80 rows the op stops being bandwidth bound and becomes a GEMM problem, and
there a hand-written Triton GEMM does not beat the tensor-core path the frozen
L1 ``Linear`` already reaches: swept over tile shapes, K-blocks, stages, warp
counts and Blackwell warp specialisation, the best Triton configuration measured
0.89x of it on the gate/up shape, 0.73x on the output-projection shape and
0.78-0.80x on the captured prefill shapes.  Those inputs therefore keep the L1
ops -- but still take two composition-level wins over the baseline: the gate and
up projections become a single GEMM against a concatenated weight, and the
separate ``silu`` + ``mul`` become the one fused L1 ``silu_and_mul`` kernel.
Five kernels become three, and the activation traffic drops by a third.

Two further fusions were built and measured, and both lose on every captured
shape, so they are recorded here rather than left to be re-discovered:

* A split-K output projection (f32 partials plus a last-CTA lock reduction) to
  fill more than the 80 of 148 SMs that ``N=2560`` tiles reach on its own.  It
  is correct but 1.2-1.5x slower -- the lock and the partial traffic cost more
  than the occupancy wins.
* Folding the activation into the output projection's *operand* load (so the
  gate/up GEMM output feeds it directly and the activation kernel disappears).
  Loading and splitting both halves inside the K-loop destroys the TMA/MMA
  pipeline: 3-20x slower, whether the two halves are blocked or interleaved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl

from ..L1.linear import Linear, Matmul
from ..L1.silu import SiLU
from ..L1.silu_and_mul import SiluAndMul

try:  # TMA descriptors (sm_90+); their absence just disables the fused path
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:  # pragma: no cover
    TensorDescriptor = None


@triton.jit
def _gu_kernel(ad, bd, hd, M, K: tl.constexpr, IC: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               NSMS: tl.constexpr):
    """``h[M, IC] = silu(x @ Wg.T) * (x @ Wu.T)``, persistent over output tiles.

    ``bd`` describes the row-interleaved pack (``B[2j] = Wg[j]``,
    ``B[2j+1] = Wu[j]``), so a ``2*BN``-wide tile of it yields both halves of
    the activation in a single accumulator.
    """
    BN2: tl.constexpr = 2 * BN
    pid = tl.program_id(0)
    num_n: tl.constexpr = IC // BN
    total = tl.cdiv(M, BM) * num_n
    for tile in tl.range(pid, total, NSMS, flatten=True):
        pid_m = tile // num_n
        pid_n = tile % num_n
        om = pid_m * BM
        acc = tl.zeros((BM, BN2), tl.float32)
        for k in tl.range(0, K, BK):
            acc = tl.dot(ad.load([om, k]), tl.trans(bd.load([pid_n * BN2, k])), acc)
        g, u = tl.split(tl.reshape(acc, (BM, BN, 2)))
        # Reproduce the reference's bf16 rounding between stages.
        g = g.to(tl.bfloat16).to(tl.float32)
        u = u.to(tl.bfloat16).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        hd.store([om, pid_n * BN], (s * u).to(tl.bfloat16))


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) per row count, swept
# end-to-end against the scorer's own timing loop.  Blackwell warp
# specialisation (``tl.range(warp_specialize=True)``) and 2-CTA clusters
# (``num_ctas=2``) were part of the sweep: specialisation is a wash below 80
# rows and only pays on the shapes that end up on the L1 path anyway, and
# ``num_ctas=2`` fails to compile for this tile shape.
_GU_CFG = ((16, (16, 64, 128, 4, 5)),
           (32, (32, 64, 128, 4, 5)),
           (1 << 30, (64, 64, 128, 4, 4)))

# Past this many rows the L1 tensor-core path wins; see the module docstring.
_MAX_FUSED_ROWS = 80

_NSM = None


def _num_sms() -> int:
    global _NSM
    if _NSM is None:
        dev = torch.cuda.current_device()
        _NSM = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NSM


def _pick(table, m):
    for bound, cfg in table:
        if m <= bound:
            return cfg
    return table[-1][1]


class GLAMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)
        self.act = SiLU()
        self.silu_and_mul = SiluAndMul()
        self.matmul = Matmul()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._packed = None   # (key, row-interleaved [2I, K]) -- fused path
        self._cat = None      # (key, concatenated    [2I, K]) -- L1 path

    # -- lazily built, cached weight layouts ---------------------------------
    def _wkey(self):
        wg, wu = self.gate_proj.weight, self.up_proj.weight
        return (wg.data_ptr(), wu.data_ptr(), wg._version, wu._version)

    def _interleaved(self):
        key = self._wkey()
        if self._packed is None or self._packed[0] != key:
            wg, wu = self.gate_proj.weight, self.up_proj.weight
            p = torch.empty(2 * wg.shape[0], wg.shape[1],
                            device=wg.device, dtype=wg.dtype)
            p[0::2] = wg
            p[1::2] = wu
            self._packed = (key, p)
        return self._packed[1]

    def _concat(self):
        key = self._wkey()
        if self._cat is None or self._cat[0] != key:
            self._cat = (key, torch.cat((self.gate_proj.weight,
                                         self.up_proj.weight), 0).contiguous())
        return self._cat[1]

    # -- fused gate/up/activation kernel + L1 output projection --------------
    def _fused(self, x: torch.Tensor):
        H = self.hidden_size
        IC = self.intermediate_size
        if (TensorDescriptor is None or not x.is_cuda
                or x.dtype is not torch.bfloat16 or not x.is_contiguous()
                or x.shape[-1] != H
                or self.gate_proj.weight.dtype is not torch.bfloat16):
            return None
        M = x.numel() // H
        if M == 0 or M > _MAX_FUSED_ROWS:
            return None
        bm, bn, bk, nw, ns = _pick(_GU_CFG, M)
        if IC % bn or H % bk:
            return None

        pk = self._interleaved()
        h = torch.empty((M, IC), device=x.device, dtype=x.dtype)
        ad = TensorDescriptor(x.reshape(M, H), [M, H], [H, 1], [bm, bk])
        bd = TensorDescriptor(pk, [2 * IC, H], [H, 1], [2 * bn, bk])
        hd = TensorDescriptor(h, [M, IC], [IC, 1], [bm, bn])
        nblk = min(_num_sms(), triton.cdiv(M, bm) * (IC // bn))
        _gu_kernel[(nblk,)](ad, bd, hd, M, H, IC, BM=bm, BN=bn, BK=bk,
                            NSMS=nblk, num_warps=nw, num_stages=ns)
        return self.down_proj(h).view(x.shape[:-1] + (H,))

    # -- L1 composition: one input GEMM, one fused activation, one output GEMM
    def _composed(self, x: torch.Tensor):
        return self.down_proj(self.silu_and_mul(self.matmul(x, self._concat())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = self._fused(x)
            if out is not None:
                return out
            return self._composed(x)
        except Exception:  # pragma: no cover - keep the baseline recipe working
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
