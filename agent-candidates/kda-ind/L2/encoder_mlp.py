"""Feed-forward blocks for encoder models.

Same composition as the baseline: ``EncoderIntermediate`` is
``gelu(x @ W1^T + b1)`` and ``EncoderOutput`` is ``LN(h @ W2^T + b2 + r)``, at
the bge-m3 sizes ``hidden_size=1024`` / ``intermediate_size=4096`` in fp16.

The module skeleton is kept verbatim -- submodule names ``dense``,
``intermediate_act_fn`` and ``LayerNorm``, and the forward parameter names
``hidden_states`` / ``input_tensor``. Weights arrive through a
``load_state_dict`` that runs *after* construction, so nothing here is derived
from a weight in ``__init__`` -- and nothing is cached from one at any later point
either. Every precondition is read live on each call. An earlier revision resolved
the parameters once on the first forward and revalidated by identity; that cannot
see ``p.data = ...``, which replaces a parameter's storage in place, and it went on
serving a stale ``LayerNorm.eps``, which no parameter check covers at all. The
guard has to read the parameters anyway in order to check them, so the cache was
never saving the lookup it appeared to.

What the fast paths require of the module is therefore checked, not assumed: the
captured geometry, matching dtypes and devices, and unit stride on the three
length-N vectors that the norm kernel addresses without a stride argument.

Two decisions shape the code, both measured rather than reasoned:

``EncoderOutput`` keeps the vendor GEMM and fuses only the epilogue. The norm
reduces over the full output width (N = hidden_size), so a single-kernel
GEMM+LayerNorm would need every CTA that touches a row to see all of N: with
N-complete tiles the ``BM x N`` fp32 accumulators overflow the register file
above BM=16, and below that the per-CTA re-read of the 8 MB weight becomes ~1 GB
of L2-to-SM traffic. A cross-CTA row reduction serialises the epilogue and costs
a second pass anyway. Splitting costs one extra 4 MB read of the GEMM result at
M=2048 (~0.5 us) plus one launch, far cheaper than either alternative.

Host submission cost is a first-class constraint, not a detail. The bench
harness enqueues an ``l2.zero_()`` over 2 x 126 MiB -- about 70 us of device
work -- before it records the start event, so the host runs ahead and the timed
window shows device time *only while the host keeps up*. At M=64 the whole call
is ~24 us of device work against ~35 us of host work for the baseline, so the
margin is thin: a fast path that adds host work can lose the case outright even
though its kernels are faster. Every per-call lookup that could be hoisted
therefore is, and the guard is written to reject in as few operations as
possible. See ``profile/probe_module_dispatch.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import forward_ad as _forward_ad

from ..L1.gelu import GELU
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear

# The one captured configuration, BGEM3Config from L4/bge_m3.py: hidden_size
# 1024, intermediate_size 4096. Both fast paths are specialized to exactly this
# geometry and every other one takes the native composition.
#
# Specializing rather than generalizing is deliberate. One program normalizes a
# whole row so it can reduce in registers, which bounds the row by the register
# file; the launch configuration below was measured at this width and no other;
# and neither L3 consumer of this module has a captured forward variant, so
# breadth here is hygiene rather than something the bench ever exercises. A width
# the kernel has not been measured at is not worth the risk surface, so it falls
# back instead.
_HIDDEN = 1024
_INTERMEDIATE = 4096
_INTERMEDIATE_WEIGHT_SHAPE = (_INTERMEDIATE, _HIDDEN)
_INTERMEDIATE_BIAS_SHAPE = (_INTERMEDIATE,)
_OUTPUT_WEIGHT_SHAPE = (_HIDDEN, _INTERMEDIATE)
_OUTPUT_VECTOR_SHAPE = (_HIDDEN,)

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _fwd_ad_level() -> int:
    """The active forward-mode AD level, or -1 when none is open."""
    return getattr(_forward_ad, "_current_level", -1)


def _autograd_wanted(*tensors: torch.Tensor | None) -> bool:
    """True when a caller would expect this call to be differentiable.

    Neither fast path builds an autograd graph -- the norm kernel writes its
    output through a raw Triton launch, and the fused GEMM is a single private op
    -- so a call that wants derivatives has to take the native composition.

    Both modes have to be covered. Reverse mode is the ``requires_grad`` test.
    Forward mode is not: a dual tensor has ``requires_grad`` False, so checking
    only that let ``EncoderIntermediate`` reach the fast path and raise
    ``NotImplementedError`` while ``EncoderOutput`` silently dropped the tangent,
    where both baselines return one. ``forward_ad._current_level`` is -1 whenever
    no dual level is open, which is a 0.02 us attribute read.

    Cost under the ``torch.no_grad`` the bench times inside: two cheap reads,
    since ``is_grad_enabled`` is False and short-circuits the rest.
    """
    if torch.is_grad_enabled():
        if any(t is not None and t.requires_grad for t in tensors):
            return True
    return _fwd_ad_level() >= 0

# cuBLASLt GEMM with a fused GELU epilogue. Private, so it is resolved once
# behind a capability guard rather than called by name: on a build without it the
# module falls back to ``gelu(linear(x))`` instead of raising AttributeError.
_ADDMM_ACTIVATION = getattr(torch, "_addmm_activation", None)


@triton.jit
def _bias_add_layernorm_kernel(
    t_ptr,              # GEMM result, (M, N)
    r_ptr,              # residual, (M, N)
    gemm_b_ptr,         # GEMM bias, (N,), unit stride
    w_ptr,              # norm weight, (N,), unit stride
    b_ptr,              # norm bias, (N,), unit stride
    out_ptr,            # (M, N)
    t_row_stride,
    r_row_stride,
    out_row_stride,
    M,
    eps,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out = layer_norm(t + gemm_b + r) * w + b``, one row-block per program.

    ``t`` and ``r`` are read exactly once and their sum stays live in fp32
    registers across both reductions and the affine, so the kernel moves ``2*N``
    halves in and ``N`` out per row and nothing more -- NCU measures its DRAM read
    traffic at 1.001x of compulsory. That is about never re-reading from global;
    the two ``tl.sum`` reductions themselves still stage through shared memory
    (measured: 1040 B per block, 20480 shared accesses per launch at M=2048), and
    collapsing them into one pass is the largest remaining lever on this kernel.

    The reductions and the affine are fp32 and only the store is narrowed, which is
    what ATen's fp16 layer_norm does internally.

    Adding the GEMM bias here rather than leaving it in the GEMM epilogue lets
    the GEMM run bias-free: worth 1.95 us at M=64 and free at M=512 and M=2048
    (``profile/probe_norm_config.py``).

    The three length-N vectors are addressed as ``ptr + cols``, i.e. unit stride
    is assumed for them; the caller guarantees it. The two matrices carry explicit
    row strides instead, so they may be arbitrary row-strided views.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    col_mask = cols < N

    t = tl.load(t_ptr + rows[:, None] * t_row_stride + cols[None, :],
                mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + rows[:, None] * r_row_stride + cols[None, :],
                mask=mask, other=0.0).to(tl.float32)
    gemm_b = tl.load(gemm_b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    x = t + r + gemm_b[None, :]

    mean = tl.sum(x, axis=1) / N
    # Re-mask after centering: the padding lanes of a non-power-of-two N load as
    # 0.0 and contribute nothing to the mean, but ``0 - mean`` would contribute
    # mean^2 each to the variance. At the captured N both masks fold away at
    # compile time, since N and BLOCK_N are constexpr and equal; they are kept so
    # the kernel is correct on its own terms rather than only for its caller.
    d = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / N
    z = d * tl.rsqrt(var + eps)[:, None]

    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    z = z * w[None, :] + b[None, :]

    tl.store(out_ptr + rows[:, None] * out_row_stride + cols[None, :],
             z.to(out_ptr.dtype.element_ty), mask=mask)


def _norm_launch_config(block_n: int) -> tuple[int, int]:
    """``(ROWS, num_warps)`` for :func:`_bias_add_layernorm_kernel`.

    Closed-form rather than autotuned: an ``@triton.autotune`` sweep reachable
    from ``forward`` would benchmark during the harness's correctness rounds and
    can leave a compiler worker thread alive across the timed call, which the
    bench reports as an injected background thread.

    One row per program at every measured M. ``profile/probe_norm_config.py``
    finds ROWS of 1, 2 and 4 equal within noise at M = 64 / 512 / 2048, while
    ROWS=8 costs ~2 us at M=64 and M=512 -- 8 rows per program leaves 8 CTAs for
    148 SMs. ``num_warps`` targets ~8 fp32 lanes per thread, which is 4 warps at
    the captured N=1024; a wider row needs more warps to keep the per-thread
    register footprint bounded, capped at 16 because past that the tail of the
    reduction tree costs more than the extra parallelism returns.
    """
    return 1, min(max(block_n // 256, 4), 16)


# Resolved once, at the one width the fast path admits. Nothing here depends on
# a weight, so it belongs at module scope rather than in a per-instance cache.
_NORM_BLOCK_N = triton.next_power_of_2(_HIDDEN)
_NORM_ROWS, _NORM_WARPS = _norm_launch_config(_NORM_BLOCK_N)


def _row_view(x: torch.Tensor) -> torch.Tensor | None:
    """A 2-D ``(rows, x.shape[-1])`` view of *x* with unit-stride rows, or
    ``None`` when no such view exists without copying.

    Returning ``None`` rather than calling ``.contiguous()`` keeps the copy
    decision at the dispatch site: an input the fast path cannot address is
    handed to the native composition, not silently repacked behind the caller's
    back.
    """
    if x.dim() < 2 or x.stride(-1) != 1:
        return None
    if x.dim() == 2:
        return x
    return x.view(-1, x.shape[-1]) if x.is_contiguous() else None


class EncoderIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = GELU()

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.intermediate_act_fn(self.dense(hidden_states))

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``gelu(hidden_states @ W1^T + b1)`` as one cuBLASLt call.

        Measured 10.8 / 10.9 / 18.7 us of device time against 13.9 / 13.2 / 25.0
        for ``gelu(linear(x))``, and it submits one kernel instead of two, so it
        is cheaper on the host as well.

        On this stack (B200, torch 2.11.0+cu130) the epilogue computes the
        **tanh** GELU approximation, not the exact erf GELU that the baseline's
        ``GELU(approximate="none")`` computes. That is measured, not assumed: in
        fp32 at the captured pre-activation magnitudes the epilogue sits 5.4e-6
        from the tanh curve and 4.7e-4 from the erf curve, an 87x separation, and
        in fp16 99.9% of the elements where it deviates from the erf reference
        deviate toward tanh. A max-abs figure alone cannot show this -- one fp16
        ULP is what a tanh epilogue and a reordered erf accumulation both produce
        -- so ``profile/probe_gelu_flavor.py`` discriminates instead of asserting.

        The deviation is therefore real and bounded rather than absent. On the
        validation inputs every element passes and the largest normalized error
        ``|err| / (atol + rtol*|ref|)`` is 0.065, so about 15x of the tolerance
        budget is still unused. A different cuBLAS build could route differently,
        which is one more reason this stays behind a capability guard with a
        fallback. Any hand-written GELU here should still use exact erf, where
        ``tl.erf`` matches ``F.gelu(approximate="none")`` bit-for-bit and costs
        nothing.
        """
        out = _ADDMM_ACTIVATION(self.dense.bias, _row_view(hidden_states),
                                self.dense.weight.t(), use_gelu=True)
        if hidden_states.dim() == 2:
            return out
        return out.view(*hidden_states.shape[:-1], out.shape[-1])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Every precondition is read live; see the module docstring for why there
        # is no cache here.
        weight = self.dense.weight
        bias = self.dense.bias
        act = self.intermediate_act_fn
        if (_ADDMM_ACTIVATION is not None
                and not torch.compiler.is_compiling()
                and bias is not None
                # The fused epilogue computes GELU itself, so it is only the right
                # answer while the module's activation still *is* GELU. Swapping
                # ``intermediate_act_fn`` for anything else has to reach the
                # native composition.
                and type(act) is GELU
                and not _autograd_wanted(hidden_states, weight, bias)
                and weight.dtype in _HALF_DTYPES
                and bias.dtype is weight.dtype
                and weight.shape == _INTERMEDIATE_WEIGHT_SHAPE
                and bias.shape == _INTERMEDIATE_BIAS_SHAPE
                and weight.is_cuda
                and bias.device == weight.device
                and hidden_states.dtype is weight.dtype
                and hidden_states.device == weight.device
                and hidden_states.dim() >= 2
                and hidden_states.shape[-1] == _HIDDEN
                and _row_view(hidden_states) is not None):
            return self.forward_cuda(hidden_states)
        return self.forward_native(hidden_states)


class EncoderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        # promote_fp32=False mirrors the baseline: a plain fp16 F.layer_norm with
        # fp16 affine params, which reduces in fp32 internally and stores fp16.
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps,
                                   promote_fp32=False)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """``LN(hidden_states @ W2^T + b2 + input_tensor)``, for inputs whose
        preconditions :meth:`forward` has already checked.

        The residual is added inside the norm kernel rather than folded into the
        GEMM through ``torch.addmm``'s ``beta`` path: that arrangement measured
        25.6 / 29.6 / 40.2 us harness-style against 23.5 / 25.6 / 37.9 for this
        one (``profile/probe_beta_residual.py``). The GEMM bias travels with it for
        the same reason -- see :func:`_bias_add_layernorm_kernel`.

        One consequence of that bias placement is worth stating precisely, because
        it is a real deviation and not merely a rounding difference. The baseline
        rounds ``h @ W2^T + b2`` to fp16 *before* adding the residual; this path
        keeps ``b2`` out of the GEMM and adds it in fp32 alongside the residual.
        The two therefore differ whenever the residual very nearly cancels the
        dense output, because the baseline's pre-norm row collapses to exactly zero
        while this one retains the fp16 residue of ``b2``, and LayerNorm then
        amplifies whatever is left. Measured at ``r = -dense(h)``: 19% of elements
        within tolerance, worst normalized error 110.

        That regime is ill-conditioned rather than one path being wrong -- against
        an fp64 reference the two are equally far off (mean |err| 0.0569 against
        0.05686) -- and it cannot arise from the bench's inputs, which draw the
        residual independently of the weights. On that distribution the worst
        normalized error is 0.144. The tradeoff is 1.95 us at M=64 against a
        deviation confined to anti-correlated residuals; if a future capture ever
        produced them, moving ``b2`` back into the GEMM epilogue restores exact
        agreement and is a one-line change.

        The output is allocated per call. A module-owned buffer grown on demand
        is harness-legal and has baseline precedent in ``L1/moe_sum.py``, but it
        measured identical here (21.54 / 23.60 / 37.89 us against
        21.54 / 23.63 / 37.95) and would make consecutive calls alias, a real
        hazard for a module whose output feeds the next encoder layer.
        """
        norm = self.LayerNorm
        r2 = _row_view(input_tensor)
        rows = r2.shape[0]
        out = torch.empty((rows, _HIDDEN), dtype=r2.dtype, device=r2.device)
        if rows == 0:
            return out
        t = F.linear(_row_view(hidden_states), self.dense.weight)
        _bias_add_layernorm_kernel[(triton.cdiv(rows, _NORM_ROWS),)](
            t, r2, self.dense.bias, norm.weight, norm.bias, out,
            t.stride(0), r2.stride(0), out.stride(0),
            rows, norm.eps,
            N=_HIDDEN, ROWS=_NORM_ROWS, BLOCK_N=_NORM_BLOCK_N,
            num_warps=_NORM_WARPS,
        )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        # Read live, for the reason in the module docstring, plus one specific to
        # this class: ``eps`` is a plain Python attribute that nothing owns, so a
        # cached copy goes stale silently and the kernel normalizes with the wrong
        # epsilon. ``normalized_shape`` is checked because the kernel reduces one
        # whole row and knows no other axis, so a multi-axis norm must reach the
        # native path and raise what the baseline raises.
        norm = self.LayerNorm
        gemm_w, gemm_b = self.dense.weight, self.dense.bias
        norm_w, norm_b = norm.weight, norm.bias
        if (not torch.compiler.is_compiling()
                and gemm_b is not None and norm_w is not None and norm_b is not None
                # promote_fp32=True would make the baseline cast the whole row to
                # fp32 before F.layer_norm and apply an fp32 affine; the kernel
                # mirrors the promote_fp32=False path this class constructs, so a
                # flipped flag has to fall back.
                and norm.promote_fp32 is False
                and not _autograd_wanted(hidden_states, input_tensor,
                                         gemm_w, gemm_b, norm_w, norm_b)
                and gemm_w.dtype in _HALF_DTYPES
                and gemm_b.dtype is gemm_w.dtype
                and norm_w.dtype is gemm_w.dtype
                and norm_b.dtype is gemm_w.dtype
                and gemm_w.shape == _OUTPUT_WEIGHT_SHAPE
                and gemm_b.shape == _OUTPUT_VECTOR_SHAPE
                and norm_w.shape == _OUTPUT_VECTOR_SHAPE
                and norm_b.shape == _OUTPUT_VECTOR_SHAPE
                # Layout, not just shape. The kernel addresses these three as
                # ``ptr + cols``, so a stride other than 1 reads the wrong
                # elements, and stride 0 -- an expanded vector over a one-element
                # storage -- reads outside the allocation entirely. The baseline
                # accepts both layouts and returns the right answer, so they have
                # to reach it rather than be repacked here.
                and gemm_b.stride(0) == 1
                and norm_w.stride(0) == 1
                and norm_b.stride(0) == 1
                and norm.normalized_shape == _OUTPUT_VECTOR_SHAPE
                and gemm_w.is_cuda
                and gemm_b.device == gemm_w.device
                and norm_w.device == gemm_w.device
                and norm_b.device == gemm_w.device
                and hidden_states.dtype is gemm_w.dtype
                and input_tensor.dtype is gemm_w.dtype
                and hidden_states.device == gemm_w.device
                and input_tensor.device == gemm_w.device
                and hidden_states.dim() == input_tensor.dim() >= 2
                and hidden_states.shape[:-1] == input_tensor.shape[:-1]
                and hidden_states.shape[-1] == _INTERMEDIATE
                and input_tensor.shape[-1] == _HIDDEN
                and _row_view(hidden_states) is not None
                and _row_view(input_tensor) is not None):
            out = self.forward_cuda(hidden_states, input_tensor)
            return out if input_tensor.dim() == 2 else out.view(input_tensor.shape)
        return self.forward_native(hidden_states, input_tensor)
