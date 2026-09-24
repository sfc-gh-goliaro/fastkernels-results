"""Patch merger for Qwen vision encoders.

Merges spatial_merge_size^2 patches into one via norm + 2-layer MLP.

Unified across Qwen2-VL and Qwen3-VL:
  - use_postshuffle_norm: Qwen3 DeepStack mergers norm after spatial reshape.

The whole module is two GEMMs plus bandwidth. Everything around them is
deleted here: the norm is one vectorized pass instead of ``F.layer_norm``'s
three, the GELU is applied in place on fc1's output instead of allocating and
writing a second 4608-wide bf16 tensor, and forward calls the two matmuls
directly rather than through the TP wrappers (which are no-ops at world
size 1). What is left costs within ~6% of the two bare cuBLAS GEMMs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - fall back to the reference path
    triton = None


if triton is not None:

    @triton.jit
    def _layer_norm_row(X, W, B, Y, EPS, NC: tl.constexpr,
                        BA: tl.constexpr, BB: tl.constexpr,
                        EXACT: tl.constexpr):
        """One normalized row per program.

        The row is covered by two power-of-two blocks (1152 = 1024 + 128,
        4608 = 4096 + 512) so the loads stay unmasked and 128-bit wide. Stats
        are accumulated in fp32 off the registers already holding the row, so
        the mean-centered second pass is free -- it costs no extra traffic and
        avoids the ``E[x^2] - E[x]^2`` cancellation a fused single pass has.
        """
        base = tl.program_id(0).to(tl.int64) * NC
        oa = tl.arange(0, BA)
        ob = BA + tl.arange(0, BB)
        mb = ob < NC
        xa = tl.load(X + base + oa).to(tl.float32)
        if EXACT:
            xb = tl.load(X + base + ob).to(tl.float32)
        else:
            xb = tl.load(X + base + ob, mask=mb, other=0.0).to(tl.float32)
        mu = (tl.sum(xa, 0) + tl.sum(xb, 0)) / NC
        ca = xa - mu
        cb = xb - mu if EXACT else tl.where(mb, xb - mu, 0.0)
        var = (tl.sum(ca * ca, 0) + tl.sum(cb * cb, 0)) / NC
        rs = tl.rsqrt(var + EPS)
        wa = tl.load(W + oa).to(tl.float32)
        ba = tl.load(B + oa).to(tl.float32)
        ya = (ca * rs * wa + ba).to(Y.dtype.element_ty)
        if EXACT:
            wb = tl.load(W + ob).to(tl.float32)
            bb = tl.load(B + ob).to(tl.float32)
            tl.store(Y + base + oa, ya)
            tl.store(Y + base + ob, (cb * rs * wb + bb).to(Y.dtype.element_ty))
        else:
            wb = tl.load(W + ob, mask=mb, other=0.0).to(tl.float32)
            bb = tl.load(B + ob, mask=mb, other=0.0).to(tl.float32)
            tl.store(Y + base + oa, ya)
            tl.store(Y + base + ob, (cb * rs * wb + bb).to(Y.dtype.element_ty),
                     mask=mb)


def _block_split(n: int) -> tuple[int, int, bool, int]:
    """Cover ``n`` columns with two power-of-two blocks, the first < ``n``.

    Returns ``(BA, BB, exact, num_warps)``; ``exact`` is True when
    BA + BB == n, i.e. no load in the kernel needs a mask.
    """
    ba = 1 << (n.bit_length() - 1)
    if ba == n:
        ba >>= 1
    rest = n - ba
    bb = 1 << (rest - 1).bit_length()
    # One warp per row is fastest at 1152 columns (36 fp32 values per lane);
    # scale the warp count with the row so wider norms do not spill.
    warps = max(1, min(8, n // 1024))
    return ba, bb, ba + bb == n, warps


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

        self.norm_dim = norm_dim
        self._ln_blocks = _block_split(norm_dim) if triton is not None else None
        # Straight-line state, resolved on the first forward rather than here:
        # the params are ``torch.empty`` at construction and get filled by a
        # weight loader (or a ``load_state_dict``) afterwards, so anything
        # derived from their values in __init__ would be stale. Keyed on the
        # parameter objects themselves so a replaced or moved param rebuilds.
        self._fast: tuple | None = None

    # -- fast path -----------------------------------------------------------
    def _build_fast(self) -> tuple | None:
        """Cache the transposed weight views, or return None to use the
        reference path (TP world size > 1, fp8, no bias, tanh GELU, ...)."""
        if triton is None:
            return None
        fc1, fc2, norm = self.fc1, self.fc2, self.norm
        if fc1.use_fp8 or fc2.use_fp8 or norm.promote_fp32:
            return None
        if fc2.tp_size > 1 or fc1.output_size_per_partition != self.hidden_size:
            return None
        if fc2.input_size_per_partition != self.hidden_size:
            return None
        if self.act.approximate != "none":
            return None
        w1, b1, w2, b2 = fc1.weight, fc1.bias, fc2.weight, fc2.bias
        nw, nb = norm.weight, norm.bias
        if any(t is None for t in (w1, b1, w2, b2, nw, nb)):
            return None
        if self.norm_dim < 64:
            return None
        if not w1.is_cuda or w1.dtype not in (torch.bfloat16, torch.float16):
            return None
        if nw.dtype != w1.dtype or nb.dtype != w1.dtype:
            return None
        # ``.t()`` is a view: no copy, no per-forward transpose. Keeping the
        # weights in their loaded [out, in] layout is also what cuBLAS wants
        # (K-major B), so there is nothing to prepack.
        return (w1, b1, w2, b2, nw, nb, w1.t(), w2.t(),
                float(norm.eps)) + self._ln_blocks

    def _layer_norm(self, x, nw, nb, eps, ba, bb, exact, warps):
        y = torch.empty_like(x)
        nc = self.norm_dim
        _layer_norm_row[(x.numel() // nc,)](
            x, nw, nb, y, eps, nc, ba, bb, exact, num_warps=warps,
        )
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st = self._fast
        if st is None or (st[0] is not self.fc1.weight
                          or st[1] is not self.fc1.bias
                          or st[2] is not self.fc2.weight
                          or st[3] is not self.fc2.bias
                          or st[4] is not self.norm.weight
                          or st[5] is not self.norm.bias):
            st = self._build_fast()
            self._fast = st
        if (st is not None and x.is_contiguous() and x.numel() > 0
                and x.dtype == st[0].dtype):
            _, b1, _, b2, nw, nb, w1t, w2t, eps, ba, bb, exact, warps = st
            if self.use_postshuffle_norm:
                h = self._layer_norm(x.view(-1, self.hidden_size),
                                     nw, nb, eps, ba, bb, exact, warps)
            elif x.shape[-1] == self.norm_dim:
                h = self._layer_norm(x, nw, nb, eps, ba, bb, exact, warps)
                h = h.view(-1, self.hidden_size)
            else:
                h = None
            if h is not None:
                # fc1 with bias + GELU in one call. On torch 2.11 / B200 this
                # is NOT a cuBLASLt fused epilogue -- the output is bitwise
                # identical to F.gelu(F.linear(...)), and a cuBLASLt GELU
                # epilogue would be the tanh approximation -- it is the GEMM
                # plus an in-place exact ``gelu_``. That is still worth 232 ->
                # 200us at M=6600, because the 4608-wide intermediate stays
                # resident in the 126MB L2 instead of being written out to a
                # freshly allocated second tensor and read back.
                h = torch._addmm_activation(b1, h, w1t, use_gelu=True)
                return torch.addmm(b2, h, w2t)

        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.fc2(self.act(self.fc1(x)))
