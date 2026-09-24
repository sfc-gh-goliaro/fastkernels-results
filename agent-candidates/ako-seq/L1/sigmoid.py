"""Sigmoid activation: 1 / (1 + exp(-x)).

Backed by a single hand-written CUDA kernel (``sigmoid_kernel.cu``): 128-bit
vectorized elementwise, native MUFU.TANH math, and a Programmatic Dependent
Launch that hides launch latency inside the producer's tail. See the .cu header
comment for why each of those matters here.

Falls back to ``torch.sigmoid`` whenever the extension is unavailable or the
input is outside the fast path (non-CUDA or a dtype other than bfloat16/float16);
non-contiguous inputs are handled inside the extension, which falls back to
``at::sigmoid`` for layouts it cannot decompose into contiguous runs.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import torch
import torch.nn as nn

_SRC = pathlib.Path(__file__).resolve().parent / "sigmoid_kernel.cu"


def _build():
    """Compile the extension for the local arch only.

    Restricting TORCH_CUDA_ARCH_LIST matters: the default list includes sm_75,
    where the bfloat16 pair intrinsics do not exist and the source will not
    compile. The source hash goes into the module name so an edited .cu can
    never be served from a stale build directory.
    """
    if not torch.cuda.is_available() or not _SRC.is_file():
        return None
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    tag = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:10]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    try:
        return load(
            name=f"fk_sigmoid_{major}{minor}_{tag}",
            sources=[str(_SRC)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


try:
    _EXT = _build()
except Exception:  # noqa: BLE001 - never let a build problem fail the op
    _EXT = None

_FAST_DTYPES = (torch.bfloat16, torch.float16)
# int32 element indexing inside the kernel.
_MAX_ELEMS = 2**31 - 1


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (_EXT is not None and x.dtype in _FAST_DTYPES and x.is_cuda
                and x.numel() <= _MAX_ELEMS):
            return _EXT.sigmoid(x)
        return torch.sigmoid(x)
