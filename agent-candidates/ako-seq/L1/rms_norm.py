"""RMSNorm with dual dispatch: CUDA custom op (eager) and pure-PyTorch (compiled).

Mirrors vLLM's ``CustomOp`` dispatch pattern:
  - ``forward_cuda``: hand-written CUDA kernels (``rmsnorm_ako.cu``) whose launch
    geometry is specialized on the hidden size instead of being derived from it.
    The vendored vLLM launcher pins one block per row with
    ``block = min(hidden/vec, 256|1024)``, which gives 16-thread (half-warp)
    blocks at hidden=128 and 64-thread blocks at hidden=512, each paying a
    shared-memory ``cub::BlockReduce`` + ``__syncthreads`` per row.  Here a row
    is owned by a fixed group of <= 32 lanes doing 128-bit vector loads and a
    shuffle-only reduction (no shared memory, no barrier), several rows share a
    block, and the grid is sized to spread over the SMs with a grid-stride row
    loop.  The plain path also addresses a *strided* input directly, so a
    non-contiguous view (e.g. the per-head slice of a fused QKV projection) is
    gathered by the norm kernel rather than by a preceding ``.contiguous()``
    copy -- on the captured ``[651,16,128]`` stride-``[2304,128,1]`` shape that
    copy costs more than the norm itself.
  - ``forward_native``: pure PyTorch implementation (f32 promotion, variance,
    rsqrt, weight multiply).  Used when torch.compile is active so Inductor
    can inline, fuse, and optimise the norm with adjacent ops -- this is the
    key mechanism that enables RMSNorm+FP8-quant fusion.

The ``forward`` method dispatches based on ``torch.compiler.is_compiling()``.

Numerics follow the vLLM reference exactly: variance is accumulated in fp32,
the residual add happens in packed 16-bit arithmetic, and the result is written
as ``(x * rsqrt(var/H + eps)) * w`` with a single rounding.  Only the fp32
reduction order differs (warp shuffles instead of a raking block reduce).

Known limitations of the CUDA kernel (forward_cuda path):
  - The fast path needs a 16-bit dtype (bf16/fp16), ``hidden_size % 8 == 0``
    and 16B-aligned pointers.  Anything else -- fp32, odd head_dims, an
    unaligned view -- falls back to the pure-PyTorch math, which is *more*
    accurate than the vendored kernel (that one is silently wrong for hidden
    sizes which aren't a multiple of 32).
  - Has no ``torch.autograd`` backward registered, so the norm silently
    drops gradient under ``torch.func.grad``.

Use :class:`L1.rms_norm_native.RMSNormNative` instead when you need
autograd / torch.func.grad support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import lazy_op

# Distinct extension name/source from the vendored ``rms_norm`` op: both modules
# can be imported into the same process (candidate vs baseline), and
# ``cpp_extension.load`` keys its build directory and module name on ``name``.
_C = lazy_op("rms_norm_ako", "rmsnorm_ako.cu")

_FAST_DTYPES = (torch.bfloat16, torch.float16)


def _fast_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the vectorized CUDA path can take this (input, weight) pair.

    Layout is deliberately *not* part of the test: the kernel walks the leading
    dimensions through a row map, so a strided view is read in place instead of
    being copied (see ``rmsnorm_ako.cu``).  Only the row itself must be
    contiguous and 16B-aligned.
    """
    return (
        x.dtype in _FAST_DTYPES
        and weight.dtype == x.dtype
        and x.is_cuda
        and x.size(-1) % 8 == 0
        and x.stride(-1) == 1
        and weight.numel() == x.size(-1)
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


# ---------------------------------------------------------------------------
# RMSNorm module
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            # Match vLLM's has_weight=False path: use the same CUDA RMSNorm
            # kernel with a non-persistent unit scale instead of falling back
            # to torch.nn.functional.rms_norm in eager/CUDA-graph decode.
            self.register_buffer(
                "_unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    # -- Pure PyTorch path (used under torch.compile so Inductor can fuse) --

    @staticmethod
    def forward_native(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        hidden_size: int,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch RMSNorm matching vLLM's forward_static."""
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        return x, residual

    # -- CUDA kernel path (used in eager mode / CUDA graph replay) --

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None:
            if residual is not None:
                # The fused path is in-place on both tensors, so -- like the
                # vLLM reference -- it materializes contiguous copies and hands
                # those back to the caller.
                x = x.contiguous()
                residual = residual.contiguous()
                if _fast_ok(x, weight):
                    _C.fused_add_rms_norm(x, residual, weight, eps)
                    return x, residual
                return RMSNorm.forward_native(x, weight, eps, x.size(-1), residual)
            # Plain path: the kernel reads whatever layout it is given, as long
            # as each row is contiguous and aligned.  Only a genuinely awkward
            # view (non-unit innermost stride, odd hidden size, misaligned base)
            # is worth a copy, and then only to see whether that makes the
            # vectorized path usable at all.
            if not _fast_ok(x, weight) and not x.is_contiguous():
                x = x.contiguous()
            if _fast_ok(x, weight):
                # ``empty`` rather than ``empty_like``: the output is always the
                # dense row-major tensor the reference returns, whatever the
                # input's layout.
                out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
                _C.rms_norm(out, x, weight, eps)
                return out
            # dtype / alignment / hidden-size outside the vectorized path
            return RMSNorm.forward_native(x, weight, eps, x.size(-1), None)
        if residual is None:
            return F.rms_norm(x, (x.size(-1),), eps=eps)
        x = x + residual
        residual = x
        return F.rms_norm(x, (x.size(-1),), eps=eps), residual

    def forward(self, x, residual=None):
        if torch.compiler.is_compiling():
            return self.forward_native(
                x, self.weight if self.elementwise_affine else None,
                self.eps, self.hidden_size, residual,
            )
        weight = self.weight if self.elementwise_affine else self._unit_weight
        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        return self.forward_cuda(
            x, weight, self.eps, residual,
        )
