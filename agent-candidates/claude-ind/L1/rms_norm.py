"""RMSNorm -- fastkernels candidate.

Same dual-dispatch contract as the baseline (``forward_native`` under
``torch.compile`` so Inductor can fuse the norm with adjacent ops, a CUDA
kernel in eager / CUDA-graph replay), but the eager path goes to the kernels in
``rms_norm_fk.cu`` instead of the vendored vLLM ones:

* one pass over the row (the row stays in registers between the sum-of-squares
  reduction and the scaled store) instead of two,
* strided inputs -- the captured q/k-norm shapes are ``[B, H, D]`` slices of a
  fused QKV buffer -- are normalized in place instead of being staged through a
  ``.contiguous()`` copy,
* output allocation, dtype dispatch and the (no-op) weight cast happen inside
  the extension, so a forward is a single call from Python.

``forward_cuda`` keeps the baseline's signature and in-place semantics:
``residual`` is overwritten with ``x + residual`` and ``x`` with the normalized
result, and both are returned.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import load_op

_C = load_op("rms_norm_fk", "rms_norm_fk.cu")
# Bind the extension entry points as module globals: a ``LOAD_GLOBAL`` instead
# of an attribute walk through the lazy-extension wrapper on every forward.
_rmsnorm = _C.rmsnorm
_fused_add_rmsnorm = _C.fused_add_rmsnorm
_is_compiling = torch.compiler.is_compiling


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6,
                 elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            w = self.weight
        else:
            # Match vLLM's has_weight=False path: same kernel with a
            # non-persistent unit scale rather than a functional fallback.
            self.register_buffer(
                "_unit_weight", torch.ones(hidden_size), persistent=False,
            )
            w = self._unit_weight
        # Alias the scale in the instance ``__dict__`` (bypassing
        # ``nn.Module.__setattr__``, which would re-register the Parameter under
        # a second name) so ``forward`` reads it with a plain dict lookup
        # instead of going through ``nn.Module.__getattr__``. It aliases the
        # same object, so ``.to(dtype)``/``load_state_dict`` -- both in-place on
        # the Parameter -- stay visible here.
        object.__setattr__(self, "_w", w)

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
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rmsnorm(x, weight, eps)
        return _fused_add_rmsnorm(x, residual, weight, eps)

    def forward(self, x, residual=None):
        if residual is None:
            if _is_compiling():
                return self.forward_native(
                    x, self._w if self.elementwise_affine else None,
                    self.eps, self.hidden_size, None,
                )
            return _rmsnorm(x, self._w, self.eps)
        if _is_compiling():
            return self.forward_native(
                x, self._w if self.elementwise_affine else None,
                self.eps, self.hidden_size, residual,
            )
        return _fused_add_rmsnorm(x, residual, self._w, self.eps)
