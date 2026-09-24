"""MaxPool2d specialised for the YOLOv10n SPPF block.

The only shapes this operator ever sees are fp16 NCHW [4,128,20,20] and
[1,128,20,20] with kernel_size=5, stride=1, padding=2, ceil_mode=False, i.e.
same-size 20x20 output.  That is ~400 KB in / 400 KB out, so the cost is
entirely launch latency plus a sub-microsecond kernel, not bandwidth.

Design:
  * ONE launch per forward, no intermediates, no ATen op in the fast path
    besides the output allocation.
  * The kernel is hand-written CUDA (`maxpool_cuda.cu`, compiled once at
    import via ``cpp_extension.load_inline``).  Both axes of the separable
    5x5 max stay in registers or warp lanes -- columns are packed two per
    lane as ``half2`` and spread across the warp, so the horizontal 5-tap is
    2 ``__shfl`` + 2 ``PRMT`` + 4 ``__hmax2_nan``; rows live in each lane's
    registers, so the vertical 5-tap is 4 ``__hmax2_nan``.  No shared memory
    and no barriers anywhere: the only memory traffic is one coalesced read
    and one coalesced write of each 800-byte plane.  Measured kernel duration
    0.98 us at [4,128,20,20] vs 1.57 us for the Triton form this replaced,
    whose 4 ``tl.gather`` row shifts each cost an SMEM round trip + barrier.
    0.98 us is only 0.07 us above a *pure copy* of the same tensor, so what is
    left is almost entirely unavoidable memory traffic.
  * PDL (``cudaLaunchAttributeProgrammaticStreamSerialization`` +
    ``cudaGridDependencySynchronize()`` placed after all address arithmetic).
    This is worth 2.0 us -- measured, by A/B-ing the launch attribute: every
    timed iteration lands either at 7.1 us (launch overlapped with the
    preceding kernel) or at 9.2 us (not overlapped), and without PDL it is
    9.2 us every time.
  * Padding is handled structurally: the lane layout has 16 lanes per plane of
    which lanes 1..10 hold the 20 real columns, so the lanes either side of a
    plane hold -inf and the horizontal shuffles need no boundary test at all.
    ``__hmax2_nan`` reproduces ATen's NaN-propagating max exactly
    (ATen: ``val > max || isnan(val)``) -- verified bit-exact on a NaN-seeded
    input, 34/34 NaNs in the same places.

If the CUDA extension cannot be built or loaded the Triton kernel below is
used instead, and anything that is not this exact case falls back to
F.max_pool2d.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# hand-written CUDA kernel (preferred)
# ---------------------------------------------------------------------------
_CUDA_RUN = None


def _load_cuda():
    """Build maxpool_cuda.cu once, at import, and return its entry point."""
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "maxpool_cuda.cu")
    with open(src_path) as fh:
        src = fh.read()
    from torch.utils.cpp_extension import load_inline

    # Name the extension after the source hash so a stale cached .so can never
    # shadow an edited kernel.
    name = "fk_maxpool5_" + hashlib.md5(src.encode()).hexdigest()[:12]
    prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
    # B200 (sm_100).  Pinned so the build never has to query a device, and so
    # it does not compile the whole default arch list.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    try:
        mod = load_inline(
            name=name,
            cpp_sources="void mp5_auto(int64_t, int64_t, int64_t);",
            cuda_sources=src,
            functions=["mp5_auto"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return mod.mp5_auto


try:
    _CUDA_RUN = _load_cuda()
except Exception:  # pragma: no cover - no nvcc, odd arch, read-only cache, ...
    _CUDA_RUN = None

# ---------------------------------------------------------------------------
# Triton fallback (round-1 kernel), used if the extension is unavailable
# ---------------------------------------------------------------------------
_NEG = tl.constexpr(float("-inf"))

try:  # Triton >= 3.6 exposes the PDL intrinsics; degrade gracefully if not.
    from triton.language.extra.cuda import gdc_wait as _gdc_wait

    _HAS_PDL = True
except Exception:  # pragma: no cover - older Triton
    _HAS_PDL = False

    @triton.jit
    def _gdc_wait():
        pass


@triton.jit
def _maxpool5_s1_p2_20x20(X, O, LANES: tl.constexpr):
    """5x5 / stride 1 / pad 2 max-pool over one contiguous 20x20 fp16 plane."""
    pid = tl.program_id(0)
    offs = tl.arange(0, LANES)
    keep = offs < 400
    row = offs // 20
    col = offs - row * 20
    base = pid * 400
    rowbase = base + row * 20

    h = tl.full((LANES,), _NEG, tl.float16)
    # Everything above is address arithmetic: keep it ahead of the wait so it
    # overlaps the producer kernel's tail.
    _gdc_wait()
    # horizontal 5-tap: 5 coalesced loads, all but the first hit L1
    for dw in tl.static_range(5):
        icol = col + dw - 2
        h = tl.maximum(
            h,
            tl.load(X + (rowbase + icol), mask=keep & (icol >= 0) & (icol < 20), other=_NEG),
            propagate_nan=tl.PropagateNan.ALL,
        )
    # vertical 5-tap: 4 in-tile shifts of h by +-1 / +-2 rows (+-20 / +-40 lanes).
    # 0 <= j < 400 is exactly "source row in range", since the column is unchanged.
    acc = h
    for k in tl.static_range(4):
        j = offs + (k - 2 + (k >= 2)) * 20
        ok = (j >= 0) & (j < 400)
        acc = tl.maximum(
            acc,
            tl.where(ok, tl.gather(h, tl.where(ok, j, 0), 0), _NEG.value),
            propagate_nan=tl.PropagateNan.ALL,
        )
    tl.store(O + (base + offs), acc, mask=keep)


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        # Resolve the specialisation predicate once, so forward is a couple of
        # attribute loads plus the launch.
        k = kernel_size if isinstance(kernel_size, int) else tuple(kernel_size)
        s = self.stride if isinstance(self.stride, int) else tuple(self.stride)
        p = padding if isinstance(padding, int) else tuple(padding)
        self._fast = (
            k in (5, (5, 5)) and s in (1, (1, 1)) and p in (2, (2, 2)) and not ceil_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fast and x.dtype == torch.float16 and x.is_contiguous():
            sh = x.shape
            if len(sh) == 4 and sh[2] == 20 and sh[3] == 20:
                n = sh[0] * sh[1]
                if n > 0:
                    o = torch.empty_like(x)
                    if _CUDA_RUN is not None:
                        _CUDA_RUN(x.data_ptr(), o.data_ptr(), n)
                    elif _HAS_PDL:
                        _maxpool5_s1_p2_20x20[(n,)](
                            x, o, LANES=512, num_warps=8, launch_pdl=True
                        )
                    else:
                        _maxpool5_s1_p2_20x20[(n,)](x, o, LANES=512, num_warps=8)
                    return o
        return F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, ceil_mode=self.ceil_mode
        )
