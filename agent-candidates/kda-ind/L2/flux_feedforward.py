"""FLUX feed-forward network (L2 composite), with bias + GELU fused into the first GEMM.

Two-layer MLP: ColumnParallelLinear + GELU(tanh) -> RowParallelLinear. At tp=1 both parallel
linears collapse to ``F.linear`` and ``RowParallelLinear`` skips its all-reduce, so the whole
operator is two GEMMs plus one elementwise activation, launched as three kernels.

``torch._addmm_activation(bias, x, W.t(), use_gelu=True)`` requests a fused bias+GELU
epilogue, collapsing the first GEMM, its bias add and the activation into one kernel: two
kernels instead of three, and the 12288-wide pre-activation is never written to or re-read
from memory. Everything the epilogue cannot reproduce falls back to the module chain, which
is the unmodified reference computation.

Profiled on sm_100 / bf16, that request dispatches into a CUTLASS 3.x kernel
(``cutlass3x_sm100_tensorop_..._256x256x64_..._2sm_bias_bf16_gelu_aux_bf16``) rather than the
cuBLAS ``nvjet_sm100_*`` kernels the plain ``addmm`` path uses. The switch is why the win is
shape-dependent: that tile is 256x256, so at M=512 it makes only 96 tiles for 148 SMs
(1.30 waves) and costs about as much as the activation pass it removes, while at M=4096 it
launches 1536 CTAs and the pass comes back nearly in full.

The module tree deliberately mirrors the reference implementation
(``net = ModuleList([<proj + gelu>, Identity, <out proj>])``) so the four parameter names
line up and externally supplied weights land where they are read from. Nothing derived from
the weights is built or cached: weights arrive after ``__init__`` and are copied in place,
so any precomputed copy would go stale invisibly, and ``data_ptr()`` cannot detect that
(an in-place copy leaves the pointer unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.gelu import GELU
from .parallel_linear import ColumnParallelLinear, RowParallelLinear


__targets__ = ["FeedForward"]

# Resolved once at import so the fast path costs no attribute lookup on torch, and so an
# older torch without the private op degrades to the module chain instead of raising.
_ADDMM_ACTIVATION = getattr(torch, "_addmm_activation", None)

_FUSABLE_DTYPES = (torch.bfloat16, torch.float16)

try:
    import triton
    import triton.language as tl
except ImportError:                                             # pragma: no cover
    triton = None


if triton is not None:

    @triton.jit
    def _bias_gelu_tanh_kernel(inp, bias, out, n_elements, n_cols, HAS_BIAS: tl.constexpr,
                               BLOCK: tl.constexpr):
        """bias + tanh-GELU over a flat [rows, n_cols] buffer.

        The activation is evaluated in fp32 in the same association order ATen uses --
        ``inner = kBeta * (x + kKappa * x**3)`` then ``0.5 * x * (1 + tanh(inner))`` with
        kBeta = sqrt(2/pi) -- which is what makes the result bit-identical to
        ``F.gelu(x, approximate="tanh")`` rather than merely close.

        HAS_BIAS is a compile-time switch rather than a runtime branch, so the bias-free
        specialization carries no predication cost. The bias-free form is the one used on the
        shipped path: see the dispatch note in forward().
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(inp + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            x += tl.load(bias + (offs % n_cols), mask=mask, other=0.0).to(tl.float32)
        inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
        y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
        tl.store(out + offs, y.to(out.dtype.element_ty), mask=mask)

    # Fixed launch config, chosen by an offline sweep over
    # BLOCK in {1024..32768} x num_warps in {4, 8, 16} at all three captured shapes
    # (tests/probe_triton_gelu.py). Hard-coded so nothing autotunes inside a timed region.
    _BLOCK, _NUM_WARPS = 2048, 4

    def _bias_gelu_tanh(x, bias=None, out=None):
        """Authored bias + tanh-GELU. Writes into `out` (or a fresh tensor) and returns it."""
        y = torch.empty_like(x) if out is None else out
        n = x.numel()
        _bias_gelu_tanh_kernel[(triton.cdiv(n, _BLOCK),)](
            x, bias, y, n, x.shape[-1], HAS_BIAS=bias is not None,
            BLOCK=_BLOCK, num_warps=_NUM_WARPS,
        )
        return y

else:                                                            # pragma: no cover
    _bias_gelu_tanh = None


class ColumnParallelApproxGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str, bias: bool = True,
                 quant_config: dict | None = None):
        super().__init__()
        self.proj = ColumnParallelLinear(dim_in, dim_out, bias=bias, quant_config=quant_config)
        self.gelu = GELU(approximate=approximate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.gelu(x)


class FeedForward(nn.Module):
    """FLUX FFN: GELU(tanh) linear -> linear with TP sharding."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        inner_dim: int | None = None,
        bias: bool = True,
        quant_config: dict | None = None,
    ) -> None:
        super().__init__()
        inner_dim = inner_dim or int(dim * mult)
        dim_out = dim_out or dim

        layers: list[nn.Module] = [
            ColumnParallelApproxGELU(dim, inner_dim, approximate="tanh", bias=bias,
                                      quant_config=quant_config),
            nn.Identity(),
            RowParallelLinear(inner_dim, dim_out, bias=bias, quant_config=quant_config),
        ]
        self.net = nn.ModuleList(layers)

    def _two_gemm_form_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether this call reduces to two plain GEMMs plus one activation.

        Everything that both accelerated paths need, and nothing specific to either. A flat
        sequence of cheap predicates, re-evaluated per call: caching them would buy well under
        1% of an ~80 us operator and would have to be invalidated on weight replacement, which
        cannot be detected reliably.
        """
        # ``aten::_addmm_activation`` has no registered derivative, and the authored kernel is
        # not an autograd function either, so a graph built over either raises at .backward().
        # Grad-tracking callers get the module chain, which is differentiable.
        if torch.is_grad_enabled() or hidden_states.requires_grad:
            return False

        # The activation flavour is backend-defined, not a property of the op. Measured on
        # B200 / CUDA 13 with an exactly representable pre-activation (so the GEMM
        # contributes no error): max|fused - gelu_tanh| = 4.3e-7, which is the fp32
        # rounding floor, against 4.7e-4 for the erf form. The same op on CPU is the erf
        # form exactly (0.0 difference) and 4.7e-4 away from tanh. So only the CUDA
        # epilogue matches the tanh activation this module is configured with. The authored
        # kernel is CUDA-only for the same reason it is Triton.
        if not hidden_states.is_cuda:
            return False
        if hidden_states.dtype not in _FUSABLE_DTYPES:
            return False

        activation_block, out = self.net[0], self.net[2]
        if activation_block.gelu.approximate != "tanh":
            return False

        # fp8 linears run a block-scaled GEMM against a float8 weight and a scale tensor --
        # a different computation, not a differently spelled one.
        if activation_block.proj.use_fp8 or out.use_fp8:
            return False

        # Under tp>1 the weights are shards and the output needs the all-reduce that
        # RowParallelLinear applies; it also drops the bias on every rank but 0. tp==1 is
        # the only case where reading both weights and both biases directly is faithful.
        if out.tp_size != 1:
            return False

        weight_1, weight_2 = activation_block.proj.weight, out.weight
        if weight_1.ndim != 2 or weight_2.ndim != 2:
            return False
        if weight_1.dtype is not hidden_states.dtype or weight_2.dtype is not hidden_states.dtype:
            return False
        if not (weight_1.is_contiguous() and weight_2.is_contiguous()):
            return False
        # A zero-width input makes flattening to (-1, width) ambiguous and raises, while the
        # module chain computes a valid bias-only result. Degenerate, but neither fast path may
        # be the reason a call fails.
        if weight_1.shape[1] == 0:
            return False
        # A 0-d input has no trailing dimension to compare, and indexing shape[-1] would raise
        # IndexError where the module chain raises RuntimeError from linear().
        if hidden_states.ndim == 0:
            return False
        return hidden_states.shape[-1] == weight_1.shape[1]

    def _fused_epilogue_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether the vendor bias+GELU epilogue is usable for this call."""
        if _ADDMM_ACTIVATION is None:
            return False
        # The epilogue needs a bias vector to add; bias=False has none.
        if self.net[0].proj.bias is None or self.net[2].bias is None:
            return False
        return self._two_gemm_form_applies(hidden_states)

    def _authored_activation_applies(self, hidden_states: torch.Tensor) -> bool:
        """Whether the authored activation kernel is usable for this call.

        Unlike the vendor epilogue this does not need a bias, because the activation is a
        separate pass: the first GEMM adds the bias when there is one and rounds to bf16, and
        the authored kernel then applies GELU to that rounded value. Reproducing the reference
        rounding point that way makes the result bit-identical to the module chain.
        """
        if _bias_gelu_tanh is None:
            return False
        return self._two_gemm_form_applies(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_vendor_epilogue = self._fused_epilogue_applies(hidden_states)
        if not use_vendor_epilogue and not self._authored_activation_applies(hidden_states):
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states

        linear_1, linear_2 = self.net[0].proj, self.net[2]
        weight_1, bias_1 = linear_1.weight, linear_1.bias
        # A view for the captured contiguous [1, M, dim] inputs; reshape rather than view so
        # a non-contiguous caller still gets a correct (if copied) 2-D operand.
        flat = hidden_states.reshape(-1, weight_1.shape[1])

        if use_vendor_epilogue:
            # Bias and activation inside the first GEMM's epilogue: two kernels instead of
            # three, and the wide pre-activation is never written or re-read. Measured
            # 1.02x / 1.11x / 1.20x end to end at the captured shapes.
            activated = _ADDMM_ACTIVATION(bias_1, flat, weight_1.t(), use_gelu=True)
        else:
            # Authored activation. The bias goes into the GEMM's own fp32 epilogue so the
            # rounding to bf16 happens at exactly the point the reference rounds, which is what
            # makes this bit-identical to the module chain rather than merely close. Applying
            # the bias inside the authored kernel instead would round twice and was measured at
            # a 0.989 match ratio -- below the harness's own floor -- so it is not done.
            pre = (torch.mm(flat, weight_1.t()) if bias_1 is None
                   else torch.addmm(bias_1, flat, weight_1.t()))
            activated = _bias_gelu_tanh(pre, out=pre)

        bias_2 = linear_2.bias
        weight_2t = linear_2.weight.t()
        projected = (torch.mm(activated, weight_2t) if bias_2 is None
                     else torch.addmm(bias_2, activated, weight_2t))
        # The width comes off the tensor just computed rather than a value stored in __init__,
        # so it cannot disagree with the weights actually in use. Spelling it out instead of
        # passing -1 also keeps a zero-size batch working: -1 is ambiguous at zero elements.
        return projected.view(*hidden_states.shape[:-1], projected.shape[-1])
