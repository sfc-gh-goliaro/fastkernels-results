"""T5-style RMSNorm with fp32 variance computation -- single fused Triton kernel.

The reference (HuggingFace) formulation is a chain of ~6 eager ops that
materializes three full fp32 copies of the activation:

    variance = hidden_states.to(fp32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)   # bf16 x fp32 -> fp32
    if weight.dtype in (fp16, bf16): hidden_states = hidden_states.to(weight.dtype)
    return weight * hidden_states

For the captured shape (bf16[1, 512, 4096]) that is ~60 MiB of HBM traffic and
6 launches for an op whose minimum is 4 MiB in + 4 MiB out.  Here the whole
thing is one row-per-program Triton kernel: the row is loaded once into
registers, the sum of squares is accumulated in fp32, and the same registers are
scaled and stored -- one read, one write, one launch.

Numerics are kept identical to the reference, including its dtype-dependent
branch:

* ``weight`` fp16/bf16 -> the normalized value is **rounded to that dtype**
  before the weight multiply, and the output is that dtype.
* ``weight`` fp32      -> no intermediate rounding, fp32 output.

Anything the fast path does not cover exactly (non-contiguous input, mismatched
weight, mixed low-precision dtypes, absurd hidden sizes, CPU tensors) falls back
to the eager reference so precision is never silently changed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _t5_rmsnorm_row(
    X,  # *in_dtype   [n_rows, N]
    W,  # *w_dtype    [N]
    Y,  # *out_dtype  [n_rows, N]
    N,
    eps,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    ROUND_MID: tl.constexpr,
):
    """One program per row.  Row stays in registers between reduce and scale."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    xp = X + row.to(tl.int64) * N + cols
    yp = Y + row.to(tl.int64) * N + cols

    if EVEN:
        x = tl.load(xp)
        w = tl.load(W + cols)
    else:
        mask = cols < N
        x = tl.load(xp, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / N + eps)
    yf = xf * rstd
    if ROUND_MID:
        # Reference rounds the normalized value to the weight dtype *before*
        # multiplying by the weight.  Y.dtype == weight dtype on this path.
        yf = yf.to(Y.dtype.element_ty).to(tl.float32)
    out = (yf * w.to(tl.float32)).to(Y.dtype.element_ty)

    if EVEN:
        tl.store(yp, out)
    else:
        tl.store(yp, out, mask=mask)


# Largest hidden size we are willing to hold in registers for a single-pass
# row-per-program kernel.  Above this the register/SRAM footprint stops paying
# and we hand the call back to the eager reference.
_MAX_BLOCK = 16384

_LOW_PREC = (torch.float16, torch.bfloat16)


class _Plan:
    """Everything shape/dtype dependent, resolved once and reused per call."""

    __slots__ = ("shape", "in_dtype", "w_dtype", "out_dtype", "same_dtype",
                 "N", "grid", "BLOCK", "EVEN", "ROUND_MID", "num_warps")


def _num_warps_for(block: int) -> int:
    """Hold ~16 elements per thread (``block // (32 * warps) == 16``).

    At the captured BLOCK=4096 that is 8 warps: two 128-bit loads and two
    128-bit stores per thread, ~32 registers live, no spills.  Measured flat
    across 1..16 warps at this size (the kernel is HBM-bound), so the formula is
    chosen to keep the *ratio* constant for other hidden sizes rather than to
    win at 4096.
    """
    return min(32, max(1, block // 512))


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self._plan: _Plan | None = None

    # -- reference semantics, used for anything the fast path won't cover ----
    def _reference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

    def _build_plan(self, hidden_states: torch.Tensor) -> _Plan | None:
        w = self.weight
        if not (hidden_states.is_cuda and w.is_cuda and w.device == hidden_states.device):
            return None
        if hidden_states.dim() < 1 or w.dim() != 1:
            return None
        N = hidden_states.shape[-1]
        if N != w.shape[0] or N == 0 or N > _MAX_BLOCK:
            return None
        if not (hidden_states.is_contiguous() and w.is_contiguous()):
            return None

        xdt, wdt = hidden_states.dtype, w.dtype
        if xdt not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        if wdt in _LOW_PREC:
            # Reference: normalized value rounded to wdt, then wdt * wdt.  Torch
            # would reject/promote a mixed fp16-weight-with-bf16-input product,
            # so only take the fast path when the two agree.
            if xdt is not wdt:
                return None
            out_dtype = wdt
            round_mid = True
        elif wdt is torch.float32:
            # fp32 weight: no intermediate rounding, fp32 result.
            out_dtype = torch.float32
            round_mid = False
        else:
            return None

        p = _Plan()
        p.shape = tuple(hidden_states.shape)
        p.in_dtype = xdt
        p.w_dtype = wdt
        p.out_dtype = out_dtype
        p.same_dtype = out_dtype is xdt
        p.N = N
        p.grid = (hidden_states.numel() // N,)
        p.BLOCK = triton.next_power_of_2(N)
        p.EVEN = p.BLOCK == N
        p.ROUND_MID = round_mid
        p.num_warps = _num_warps_for(p.BLOCK)
        return p

    def _slow(self, hidden_states: torch.Tensor) -> torch.Tensor:
        plan = self._build_plan(hidden_states)
        if plan is None:
            return self._reference(hidden_states)
        self._plan = plan
        return self._launch(hidden_states, plan)

    def _launch(self, hidden_states: torch.Tensor, p: _Plan) -> torch.Tensor:
        out = (torch.empty_like(hidden_states) if p.same_dtype
               else torch.empty_like(hidden_states, dtype=p.out_dtype))
        _t5_rmsnorm_row[p.grid](
            hidden_states, self.weight, out, p.N, self.variance_epsilon,
            BLOCK=p.BLOCK, EVEN=p.EVEN, ROUND_MID=p.ROUND_MID,
            num_warps=p.num_warps,
        )
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        p = self._plan
        if (p is None
                or hidden_states.dtype is not p.in_dtype
                or self.weight.dtype is not p.w_dtype
                or tuple(hidden_states.shape) != p.shape
                or not hidden_states.is_contiguous()):
            return self._slow(hidden_states)
        return self._launch(hidden_states, p)
