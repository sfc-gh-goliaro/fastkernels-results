"""RMSNorm with dual dispatch: a single-pass CUDA kernel (eager) and pure-PyTorch (compiled).

Same surface as the vendored module this replaces:
  - ``forward_native``: pure PyTorch, reused verbatim from the baseline so the
    ``torch.compile`` path, the ``elementwise_affine=False`` path and any
    autograd-shaped use stay byte-for-byte identical.
  - ``forward_cuda``: the hand-written kernels in ``rms_norm_kernels.cu`` when
    they claim the layout, and the vendored vLLM kernels for everything else.

``forward`` dispatches on ``torch.compiler.is_compiling()``, exactly as before.

Two differences from the vendored path are the point of this module:

  * The row is read from global memory once. The vendored kernel streams the row
    to accumulate the variance and then re-reads it to normalize; every hidden
    size in this workload fits in registers, so the row is held there instead.
  * There is no unconditional ``x.contiguous()``. Leading dimensions are
    analysed on the host and collapsed to at most two (size, stride) pairs, so a
    strided view -- the K slice of a fused QKV projection, say, where the
    per-head reshape yields row stride != hidden -- normalizes where it lies.
    For that layout the copy the vendored path forces is a full gather-and-write
    plus an extra launch, and the kernel does not need it.

Layouts the kernel does not claim are reported in-band by the extension's
host-side predicates (``None`` from the norm entry point, ``False`` from the
fused one) and routed to the vendored operator, which is correct for them.
Nothing here catches an exception around a launch: the fallback is a decision,
not a recovery.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastkernels.infra.cuda_ext import load_op
from fastkernels.tasks.baseline.L1.rms_norm import RMSNorm as _Vendored

# Built eagerly at import rather than on first use. The import happens outside
# the harness's per-case error handling, so a compile failure surfaces here as a
# traceback instead of a runtime failure on the first forward.
#
# The name must not be "rms_norm": torch keys both the build directory and the
# pybind module on the extension name alone, and rebuilds whenever a source is
# newer than the ``.so``. Sharing the name with the vendored extension would
# make the two sources invalidate each other's build on every import, and both
# are live in one process because the fallback uses the vendored one.
_EXT = load_op("rms_norm_single_pass", "rms_norm_kernels.cu")

# Bound at import. A single 128-element row cannot keep the GPU busy, so for the
# smallest shapes the measured cost is host-side launch work and each avoided
# attribute lookup on the way to the kernel is a real fraction of it.
_ext_rms_norm = _EXT.rms_norm
_ext_fused_add_rms_norm = _EXT.fused_add_rms_norm
_vendored_forward_cuda = _Vendored.forward_cuda


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
            # Matches the vendored has_weight=False path: the same kernel with a
            # non-persistent unit scale, not a fallback to F.rms_norm.
            self.register_buffer(
                "_unit_weight",
                torch.ones(hidden_size),
                persistent=False,
            )

    # -- Pure PyTorch path (used under torch.compile so Inductor can fuse) --

    # Reused rather than reimplemented: this path is not the one being optimised,
    # and sharing the function makes divergence from the baseline impossible.
    forward_native = staticmethod(_Vendored.forward_native)

    # -- CUDA kernel path (used in eager mode / CUDA graph replay) --

    @staticmethod
    def forward_cuda(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        eps: float,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if weight is not None:
            if residual is None:
                # Allocates the output itself, so this is the only dispatcher
                # call on the path -- no empty_like, no contiguous.
                out = _ext_rms_norm(x, weight, eps)
                if out is not None:
                    return out
            elif _ext_fused_add_rms_norm(x, residual, weight, eps):
                # Sum written back through ``residual``, normalized row through
                # ``x``, both in place, as the vendored fused kernel does.
                return x, residual

        # Everything the kernel does not claim: an unsupported dtype or device,
        # ``stride(-1) != 1``, more than two surviving leading (size, stride)
        # pairs, a hidden size or leading stride no bundle width divides, a
        # residual that is not contiguous, or no weight at all. The predicates
        # run before any launch, so nothing has been written yet.
        return _vendored_forward_cuda(x, weight, eps, residual)

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
