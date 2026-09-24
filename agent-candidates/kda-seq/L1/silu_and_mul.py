"""SiLU-and-Mul with a flat vectorized grid-stride CUDA kernel for Blackwell.

The eager path replaces the vendored vLLM ``act_and_mul_kernel`` with a single
kernel whose launch geometry is independent of the row count, whose global
accesses are 32 bytes wide on every shape the benchmark selects, and whose
activation costs one transcendental-pipe operation per element instead of the
reference's ``expf`` plus IEEE divide. The compiled path is unchanged pure
PyTorch so Inductor can still fuse it.

The kernels live in ``silu_and_mul_kernels.cu`` beside this file. They are built
under their own extension name -- the baseline module already owns
``silu_and_mul`` and its build directory in this same process.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastkernels.infra.cuda_ext import load_op

# Eager, not lazy: the benchmark times eager forward calls, so a first build
# inside ``forward`` would land inside the timed region.
_C = load_op("fk_cand_l1_silu_and_mul_v1", "silu_and_mul_kernels.cu")

# Reproduction selector for the recorded checkpoints, unset in normal use. Values are
# the ActKind enum in silu_and_mul_kernels.cu (also exported as _C.ACT_EXP_ROUND and
# friends); `FK_SILU_ACT_KIND=0 python validate.py` reproduces the reference-exact
# expf + IEEE-divide configuration through the real benchmark.
#
# silu_and_mul_tuned already treats act_kind < 0 as "use the default", so the branch is
# not there to save a comparison -- it is there so that with the variable unset the call
# goes through the original silu_and_mul entry point unchanged, and the shipped path is
# not silently rerouted through the tuning entry point.
_ACT_KIND = int(os.environ.get("FK_SILU_ACT_KIND", "-1"))
_impl = (_C.silu_and_mul if _ACT_KIND < 0
         else functools.partial(_C.silu_and_mul_tuned, act_kind=_ACT_KIND))


class SiluAndMul(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch implementation — visible to Inductor for fusion."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    @staticmethod
    def forward_cuda(x: torch.Tensor) -> torch.Tensor:
        # One crossing: the extension allocates the output, launches, and
        # returns it.
        return _impl(x)

    def forward(self, x):
        if torch.compiler.is_compiling():
            return self.forward_native(x)
        return self.forward_cuda(x)
