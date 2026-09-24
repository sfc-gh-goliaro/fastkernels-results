"""Feed-forward blocks for encoder models.

The baseline runs five ATen launches per (intermediate, output) pair: addmm ->
GELU, and addmm -> residual add -> layer_norm.  Three measured facts about this
operator on a B200 (fp16, hidden=1024, intermediate=4096, M in {64, 512, 2048})
drive the rewrite:

1. A kernel launch costs a flat ~2.05 us of measured wall time even when the
   kernel is empty, and a launch that *reads what the previous launch wrote*
   costs a further ~2 us of dependency stall.  Launch count and that stall are
   the first-order terms at M=64/512, where the real GPU work is only 2-6 us.
2. cuBLAS is at or near roofline for both GEMMs here (~1.4 PFLOPS fp16 for
   1024->4096, ~1.7 PFLOPS for 4096->1024 at M=2048).  A hand-written Triton
   GEMM tops out near 600 TFLOPS on these shapes -- both the plain `tl.dot`
   loop and the TMA + `warp_specialize` persistent form -- so every GEMM stays
   on cuBLAS.
3. What is left to remove is therefore the *pointwise* traffic: at M=2048 the
   separate GELU pass moves 32 MiB and the residual-add + LayerNorm pair moves
   another 24 MiB, all to apply arithmetic that fits in a register.

Three levers, in order of payoff:

* **Fused epilogues.**  ``EncoderOutput`` collapses ``+ input_tensor`` and
  ``F.layer_norm`` into a single Triton program per row.  hidden_size=1024 is
  one 2 KiB fp16 row, so the whole mean/variance reduction is an in-register
  ``tl.sum`` -- no second pass and no cross-program reduction.  That takes the
  block from three launches to two, and the pointwise traffic from 24 MiB to
  12 MiB.
* **Programmatic dependent launch.**  Each Triton kernel here consumes a cuBLAS
  result, so it is launched with ``launch_pdl=True`` and calls ``gdc_wait()``
  immediately before its first load.  Its CTAs become resident and finish their
  address arithmetic while the GEMM's tail drains, recovering the ~2 us
  dependency stall.  ``gdc_wait()`` still waits for the producer grid to
  complete, so the read-after-write stays ordered.  Worth ~2 us on every shape
  -- the single largest win at M=64.
* **A cheap GELU.**  ``tl.erf`` lowers to libdevice ``erff``; on 8.4M elements
  that is ~4 us of pure ALU, enough to make the GELU pass compute-bound rather
  than bandwidth-bound.  ``_gelu_tanh`` below evaluates the standard tanh
  formulation through ``tl.exp`` (hardware ``ex2.approx.f32``) instead, which
  puts the pass back on the memory roofline (~8 TB/s measured).

The one deliberately-not-taken option is ``torch._addmm_activation``, which
fuses bias+GELU into the cuBLASLt epilogue and so needs only one launch.  Its
numerics are fine, but cuBLASLt selects a slower algorithm for the epilogue
variant than for plain addmm (+3.9 us at M=64, +5.9 us at M=2048), which
exactly cancels the launch and the extra pass it saves.  It measures within
noise of the two-launch form at M=2048 and clearly behind it below that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

try:
    from triton.language.extra.cuda import gdc_wait
    _HAVE_PDL = True
except ImportError:  # pragma: no cover - Triton without the PDL intrinsics
    _HAVE_PDL = False

    @triton.jit
    def gdc_wait():
        pass


# Flipped off permanently if the installed launcher rejects ``launch_pdl``, so
# the fallback costs one exception once rather than a check per call.
_PDL = _HAVE_PDL

# GELU pass: 2048 elements per program over 8 warps is 8 fp16 per thread, i.e.
# full 128-bit loads, and leaves >=4096 programs at M=2048.
_GELU_BLOCK = 2048
_GELU_WARPS = 8

# LayerNorm pass: one program per row.  Below ~1024 rows the grid cannot fill
# 148 SMs and the kernel is latency-bound, so more warps per CTA (more rows in
# flight per SM) wins; at and above it the pass is bandwidth-bound and the
# wider per-thread vector of 4 warps (16 B/thread vs 8 B) wins instead.
_LN_BANDWIDTH_BOUND_ROWS = 1024


@triton.jit
def _gelu_tanh(x):
    """erf-GELU to <1e-3 absolute, via ``ex2.approx`` rather than ``erff``.

    ``tanh(u) = 1 - 2 / (exp(2u) + 1)``, so with ``u`` the usual
    ``sqrt(2/pi) * (x + 0.044715 x^3)`` the activation costs one exp, one
    reciprocal and a multiply.  Saturates correctly at both ends: for very
    negative ``x`` the exp underflows to 0 and the result is 0; for very
    positive ``x`` it overflows to inf, ``1/inf`` is 0 and the result is ``x``.
    The tanh and erf formulations of GELU differ by at most ~5e-4 over this
    range -- an order of magnitude inside the fp16 tolerance, and below the
    fp16 rounding of the result itself.
    """
    u = 1.5957691216057308 * (x + 0.044715 * x * x * x)  # 2 * sqrt(2/pi) * (.)
    return x * (1.0 - 1.0 / (tl.exp(u) + 1.0))


@triton.jit
def _gelu_kernel(Y, X, n_elements, BLOCK: tl.constexpr, PDL: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    if PDL:
        gdc_wait()
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + offs, _gelu_tanh(x).to(Y.dtype.element_ty), mask=mask)


def _fused_gelu(x):
    y = torch.empty_like(x)
    n = x.numel()
    _launch(_gelu_kernel, (triton.cdiv(n, _GELU_BLOCK),), (y, x, n),
            dict(BLOCK=_GELU_BLOCK), num_warps=_GELU_WARPS)
    return y


# ---------------------------------------------------------------------------
# residual add + LayerNorm, fused
#
# Precision policy mirrors ``F.layer_norm`` on an fp16 input (the baseline's
# LayerNorm is built with promote_fp32=False): the residual add happens in the
# input dtype exactly as ATen's ``+`` would, mean/variance and the affine
# transform run in fp32, and the result is rounded once on store.
# ---------------------------------------------------------------------------
@triton.jit
def _add_layernorm_kernel(
    Y, X, R, Weight, Bias, eps,
    N: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr,
    MASKED: tl.constexpr, HAS_W: tl.constexpr, HAS_B: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    off = tl.program_id(0).to(tl.int64) * N + cols
    if PDL:
        gdc_wait()
    # BLOCK == N (the common case: hidden_size is a power of two) lets every
    # access run unmasked, which is why the two arms are spelled out.
    if MASKED:
        m = cols < N
        v = (tl.load(X + off, mask=m, other=0.0)
             + tl.load(R + off, mask=m, other=0.0)).to(tl.float32)
        mean = tl.sum(v, axis=0) / N
        d = tl.where(m, v - mean, 0.0)
        var = tl.sum(d * d, axis=0) / N
        out = d * (1.0 / tl.sqrt(var + eps))
        if HAS_W:
            out = out * tl.load(Weight + cols, mask=m, other=1.0).to(tl.float32)
        if HAS_B:
            out = out + tl.load(Bias + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + off, out.to(Y.dtype.element_ty), mask=m)
    else:
        v = (tl.load(X + off) + tl.load(R + off)).to(tl.float32)
        mean = tl.sum(v, axis=0) / N
        d = v - mean
        var = tl.sum(d * d, axis=0) / N
        out = d * (1.0 / tl.sqrt(var + eps))
        if HAS_W:
            out = out * tl.load(Weight + cols).to(tl.float32)
        if HAS_B:
            out = out + tl.load(Bias + cols).to(tl.float32)
        tl.store(Y + off, out.to(Y.dtype.element_ty))


def _fused_add_layernorm(x, residual, weight, bias, eps):
    """``F.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)``."""
    n = x.shape[-1]
    rows = x.numel() // n
    block = triton.next_power_of_2(n)
    y = torch.empty_like(x)
    _launch(_add_layernorm_kernel, (rows,),
            (y, x, residual, weight if weight is not None else x,
             bias if bias is not None else x, eps),
            dict(N=n, BLOCK=block, MASKED=block != n,
                 HAS_W=weight is not None, HAS_B=bias is not None),
            num_warps=4 if rows >= _LN_BANDWIDTH_BOUND_ROWS else 8)
    return y


def _launch(kernel, grid, args, constexprs, **opts):
    """Launch *kernel* with PDL, degrading once if the launcher rejects it."""
    global _PDL
    if _PDL:
        try:
            kernel[grid](*args, PDL=True, **constexprs, launch_pdl=True, **opts)
            return
        except TypeError:
            _PDL = False
    kernel[grid](*args, PDL=False, **constexprs, **opts)


def _fusable(t):
    return (t.is_cuda and t.is_contiguous()
            and t.dtype in (torch.float16, torch.bfloat16))


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = self.dense.weight
        b = self.dense.bias
        if (b is not None and _fusable(hidden_states)
                and hidden_states.shape[-1] == w.shape[1]):
            x = hidden_states.reshape(-1, w.shape[1])
            out = _fused_gelu(torch.addmm(b, x, w.t()))
            return out.view(*hidden_states.shape[:-1], w.shape[0])
        return self.intermediate_act_fn(self.dense(hidden_states))


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False: vLLM's bert.py / roberta.py use a plain
        # nn.LayerNorm here (see encoder_embeddings for the full rationale).
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        ln = self.LayerNorm
        w = self.dense.weight
        b = self.dense.bias
        if (b is not None and not ln.promote_fp32
                and _fusable(hidden_states) and _fusable(input_tensor)
                and hidden_states.shape[-1] == w.shape[1]
                and input_tensor.shape[-1] == w.shape[0]
                and (input_tensor.numel() // w.shape[0]
                     == hidden_states.numel() // w.shape[1])):
            proj = torch.addmm(b, hidden_states.reshape(-1, w.shape[1]), w.t())
            out = _fused_add_layernorm(
                proj, input_tensor.reshape(-1, w.shape[0]),
                ln.weight, ln.bias, ln.eps)
            return out.view(input_tensor.shape)
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)
