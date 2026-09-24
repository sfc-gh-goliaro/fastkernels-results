"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

Optimization notes (B200 / sm100, bf16, context_dim=1152 -> 4608 -> d_model)
---------------------------------------------------------------------------
Per-op timing of the baseline on the captured shapes (x: [20680, 1, 1152]):

    LayerNorm 64us | fc1 193us | GELU 34us | fc2 167us

The two GEMMs run at ~1.15 PFLOP/s, within ~10% of the fastest GEMM this part
reaches for these shapes at all (measured ceiling: 1.28 PFLOP/s on a large
square problem), so there is nothing to win there -- a hand-written Triton
tcgen05 GEMM (TMA + warp specialization + epilogue subtiling, 128x256x64, best
of ~400 autotuned configs) peaked at 904 TFLOP/s, i.e. ~80 us *slower* per
call, which more than eats the fused-epilogue saving. They keep the baseline's
``fc1``/``fc2`` (bias fused in the GEMM epilogue).

What *is* on the table is the two memory-bound passes, which torch runs at a
fraction of copy bandwidth (a 95 MB copy takes 21.5us here):

  * ``_ln_kernel``   -- layernorm fused with the patch-merge reshape (the
    reshape is a pure reinterpret: normalizing 1152-wide rows and then viewing
    them as 4608-wide rows is the same buffer). One warp per 2 rows keeps the
    whole row in registers so the reduction is a single intra-warp shuffle
    tree; multi-warp reductions cost more than the register pressure does.
    64us -> 25.5us (3.7 TB/s, 1.19x off the copy floor).
  * ``_gelu_kernel`` -- GELU at copy bandwidth, in place on the fc1 output.
    34us -> 21.5us. It evaluates the tanh form with the hardware
    ``tanh.approx.f32`` SFU instruction, because libdevice ``erff`` is ~30 ALU
    ops and makes the pass compute bound (2x slower). Measured deviation from
    exact GELU: <=4.7e-4 absolute (worst at |x|~2.7), which after fc2 is a
    1e-3 mean / 3.1e-2 max shift on outputs of std 1.25 -- zero elements over
    the bf16 bench bound (1e-2 + 1e-2|ref|), and smaller than the layernorm's
    own bf16 rounding, which is what the reported max error actually is.

Net: 1.08x on the smallest captured shape (where the two GEMMs are 80% of the
time and cuBLAS only reaches 625 TFLOP/s) and 1.23-1.25x on the rest.

Anything the kernels cannot express (non-CUDA, unsupported dtype or normalized
shape, missing affine params, fp32-promoted norm, non-contiguous input) falls
back to the baseline op, one pass at a time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton ships with torch on this target
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ln_kernel(X, Y, W, B, M, eps, C: tl.constexpr, C0: tl.constexpr,
                   C1: tl.constexpr, BR: tl.constexpr):
        """LayerNorm over the last (C-element) axis, BR rows per program.

        C is split into a power-of-two head C0 plus a masked power-of-two tail
        C1 (1152 -> 1024 + 128, 4608 -> 4096 + 512) since ``tl.arange`` needs
        powers of two; for the shapes here the split is exact, so the tail
        costs one extra load/store pair and no wasted lanes.
        """
        rows = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
        rm = rows < M
        c0 = tl.arange(0, C0)
        c1 = C0 + tl.arange(0, C1)
        m1 = c1 < C
        p = X + rows[:, None] * C
        x0 = tl.load(p + c0[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        x1 = tl.load(p + c1[None, :], mask=rm[:, None] & m1[None, :],
                     other=0.).to(tl.float32)
        mean = (tl.sum(x0, 1) + tl.sum(x1, 1)) / C
        d0 = x0 - mean[:, None]
        d1 = tl.where(m1[None, :], x1 - mean[:, None], 0.)
        var = (tl.sum(d0 * d0, 1) + tl.sum(d1 * d1, 1)) / C
        r = tl.rsqrt(var + eps)[:, None]
        w0 = tl.load(W + c0).to(tl.float32)
        w1 = tl.load(W + c1, mask=m1, other=0.).to(tl.float32)
        b0 = tl.load(B + c0).to(tl.float32)
        b1 = tl.load(B + c1, mask=m1, other=0.).to(tl.float32)
        q = Y + rows[:, None] * C
        od = Y.dtype.element_ty
        # Same op order as F.layer_norm's affine ((x-mean)*rstd*w+b, fp32),
        # so the rounding to ``od`` matches it element for element.
        tl.store(q + c0[None, :], (d0 * r * w0[None, :] + b0[None, :]).to(od),
                 mask=rm[:, None])
        tl.store(q + c1[None, :], (d1 * r * w1[None, :] + b1[None, :]).to(od),
                 mask=rm[:, None] & m1[None, :])

    @triton.jit
    def _gelu_kernel(X, Y, n, BLK: tl.constexpr):
        """GELU (tanh form via the tanh.approx.f32 SFU op), may run in place."""
        o = tl.program_id(0).to(tl.int64) * BLK + tl.arange(0, BLK)
        m = o < n
        x = tl.load(X + o, mask=m, other=0.).to(tl.float32)
        t = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        th = tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [t],
                                       dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + o, (x * 0.5 * (1.0 + th)).to(Y.dtype.element_ty), mask=m)


_GELU_BLK = 2048
_GELU_WARPS = 4
# fp32 stays on the torch ops: its bench tolerance is 1e-5/1e-3 and fp32
# matmul may run in TF32, which amplifies a last-bit layernorm difference into
# a TF32 ulp on ~0.5% of outputs. Nothing captured for this op is fp32.
_FAST_DTYPES = (torch.float16, torch.bfloat16)


def _make_ln_plan(c: int):
    """(C0, C1, rows_per_program, num_warps) for a C-wide layernorm, or None.

    One program is one warp holding ``rows*C`` floats in registers, so the row
    count shrinks as C grows; past 8192 the tile would spill and torch wins.
    """
    if c < 2 or c > 8192:
        return None
    c0 = 1 << (c.bit_length() - 1)
    if c0 == c:
        c0 >>= 1
        c1 = c0
    else:
        c1 = 1 << (c - c0 - 1).bit_length()
    if c <= 2048:
        return c0, c1, 2, 1
    return c0, c1, 1, 2


class VisionPatchMerger(nn.Module):
    """Patch merger: norm -> flatten -> MLP(fc1, GELU, fc2).

    Qwen3 DeepStack mergers set use_postshuffle_norm=True to norm after reshape.
    """

    def __init__(self, d_model: int, context_dim: int,
                 spatial_merge_size: int = 2,
                 use_postshuffle_norm: bool = False,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        # See VisionBlock: vLLM's vision path uses plain nn.LayerNorm on
        # bf16, and our fp32 promotion costs two full-tensor copies here.
        self.norm = LayerNorm(norm_dim, eps=eps, promote_fp32=False)
        self.fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELU()
        self.fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

        self._norm_dim = norm_dim
        self._ln_plan = _make_ln_plan(norm_dim) if _HAS_TRITON else None
        # Plain (unquantized, untensor-parallel) linears are exactly F.linear;
        # calling it directly skips two nn.Module.__call__ dispatches per
        # forward, which the smallest captured shape (~72 us) does notice.
        self._plain_mlp = not (self.fc1.use_fp8 or self.fc2.use_fp8
                               or self.fc2.tp_size > 1)

    def _norm_fast(self, x: torch.Tensor) -> torch.Tensor | None:
        """Fused layernorm + merge reshape, or None if this input is unsupported."""
        if self._ln_plan is None or not x.is_cuda or x.dtype not in _FAST_DTYPES:
            return None
        n = self.norm
        if n.promote_fp32 or n.weight is None or n.bias is None:
            return None
        if n.weight.dtype != x.dtype or n.bias.dtype != x.dtype:
            return None
        c = self._norm_dim
        if x.shape[-1] != c or not x.is_contiguous() or x.numel() % c:
            return None
        rows = x.numel() // c
        if rows == 0:
            return None
        xr = x.view(rows, c)
        y = torch.empty_like(xr)
        c0, c1, br, nw = self._ln_plan
        _ln_kernel[(triton.cdiv(rows, br),)](
            xr, y, n.weight, n.bias, rows, n.eps, c, c0, c1, br, num_warps=nw,
        )
        return y

    def _act_fast(self, h: torch.Tensor) -> bool:
        """In-place GELU on ``h``; False if this tensor is unsupported."""
        if not _HAS_TRITON or not h.is_cuda or h.dtype not in _FAST_DTYPES:
            return False
        if not h.is_contiguous() or self.act.approximate != "none":
            return False
        n = h.numel()
        if n == 0:
            return False
        _gelu_kernel[(triton.cdiv(n, _GELU_BLK),)](
            h, h, n, _GELU_BLK, num_warps=_GELU_WARPS,
        )
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = x.view(-1, self.hidden_size)
        y = self._norm_fast(x)
        if y is None:
            y = self.norm(x)
        y = y.view(-1, self.hidden_size)
        if not self._plain_mlp:
            x = self.fc1(y)
            if not self._act_fast(x):
                x = self.act(x)
            return self.fc2(x)
        x = F.linear(y, self.fc1.weight, self.fc1.bias)
        if not self._act_fast(x):
            x = self.act(x)
        return F.linear(x, self.fc2.weight, self.fc2.bias)
