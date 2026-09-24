"""YOLOv10 Distribution Focal Loss layer -- one fused streaming kernel.

The whole DFL collapses to

    out[b, j, a] = sum_k w[k] * softmax_k( x[b, j*c1+k, a] )

i.e. the softmax-weighted expectation of the bin index (``w = arange(c1)``).
Nothing in between needs to exist, so the reference chain's view/transpose, the
materialized [b, c1, 4, a] softmax tensor and the 1x1 Conv2d GEMM are all
replaced by a single memory-bound pass over ``x`` (see ``yolo_dfl.cu``).

``self.conv`` / ``self._softmax`` are kept so the ``__init__`` contract, the
``state_dict`` keys and a correct fallback path are all preserved.
"""

from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d
from ..L1.softmax import Softmax

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_dfl.cu")


def _local_arch() -> str | None:
    """The visible device's arch, so the JIT build targets it and nothing else.

    The farm exports a six-architecture ``TORCH_CUDA_ARCH_LIST``; honoring it
    would compile this kernel six times over for GPUs that will never run it.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 -- no device: leave the env alone
        return None
    # 9.x/10.x/12.x need the architecture-specific 'a' variant for ptxas.
    return f"{major}.{minor}a" if major in (9, 10, 12) else f"{major}.{minor}"


def _load_ext():
    """JIT-build the fused kernel. Name carries a source hash so an edited
    source can never be served the previous ``.so`` out of the build cache."""
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        from torch.utils.cpp_extension import load

        arch = _local_arch()
        if arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        with open(_SRC, "rb") as fh:
            tag = hashlib.sha1(fh.read()).hexdigest()[:12]
        return load(
            name=f"ako_yolo_dfl_{tag}",
            sources=[_SRC],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "--expt-relaxed-constexpr",
            ],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 -- no nvcc / build failure
        # Loud, because a silent fallback benches at exactly 1.00x with no hint why.
        print(f"[yolov10_dfl] CUDA extension build failed ({exc!r}); "
              f"falling back to the reference chain", file=sys.stderr, flush=True)
        return None
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_EXT = _load_ext()


class YOLODFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self._softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.c1
        w = self.conv.weight
        if (
            _EXT is not None
            and x.is_cuda
            and x.dim() == 3
            and x.size(1) == 4 * c1
            and x.stride(2) == 1
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and w.dtype == x.dtype
        ):
            return _EXT.yolo_dfl(x, w, c1)
        b, _, a = x.shape
        return self.conv(self._softmax(x.view(b, 4, c1, a).transpose(2, 1))).view(b, 4, a)
