"""Fused FLUX rotary position embedding table builder.

Drop-in replacement for ``baseline.py``'s ``FluxPosEmbed``. The baseline spends
almost all of its wall time in host dispatch: a three-iteration Python loop of
``arange`` / ``pow`` / ``reciprocal`` / ``div`` / ``outer`` / ``polar`` plus two
``cat``s and two ``.to()``s -- 33 CUDA ops for ~68 us of actual device work
inside a ~278 us call, and the latency barely moves when the row count triples.

The operator itself is one elementwise ``sincos`` over the outer product of a
length-S position vector and a length-D2 constant frequency table: no reduction,
no cross-column coupling, no reuse. So it collapses to a single kernel launch
writing a single output allocation.

Three routes, in order of preference:

``cuda_ext``
    The fused kernel. CUDA device, 2-D ``ids``, one of the six supported dtypes,
    no autograd, and an output width the launch geometry can express.
``torch_vectorized``
    Same input domain, used when the extension is unavailable: a gather of the
    position columns, one multiply, then ``cos`` and ``sin``.
``torch_reference``
    A transcription of the baseline's own per-axis loop, for everything else --
    CPU / ``mps`` / ``npu`` devices, ranks other than 2, ``requires_grad``, zero
    axes, odd ``axes_dim`` entries, widths beyond the launch guards. Being
    structured like the baseline, it reproduces the baseline's exceptions as well
    as its values.

Numerics are float64 throughout and match the baseline bit for bit. The
frequency table is built with the baseline's own torch expression on the input's
own device -- reimplementing ``theta ** x`` inside the kernel risks a different
last bit, and so does building on the host and copying. Positions are narrowed
through float32 exactly as ``ids.float()`` does, which is lossy for float64 and
int64 ``ids``, so the narrowing must not be skipped.

Both outputs are views of one ``[2, S, D2]`` allocation. They are contiguous plain
tensors covering disjoint halves of it, so an ordinary elementwise write to ``cos``
does not change any value in ``sin``. What differs from the baseline's two
independent tensors is that they share a single storage object: ``data_ptr()``,
``untyped_storage()`` and ``storage_offset()`` reveal the aliasing, a storage-level
write or a resize can reach across the boundary, and freeing one does not release
the memory while the other lives. Neither the benchmark nor FLUX's own consumer
does any of that, which is why one allocation is safe here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

__all__ = ["FluxPosEmbed", "last_route", "extension_error"]

# Largest ``blockDim.x`` CUDA will launch. One thread owns one output column
# (scalar stores) or one column pair (128-bit stores), so this caps the output
# width the fused kernel can express: 1024 columns scalar, 2048 paired.
_MAX_BLOCK_LANES = 1024

_KERNEL_DTYPES = frozenset(
    {
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    }
)

ROUTE_CUDA_EXT = "cuda_ext"
ROUTE_TORCH_VECTORIZED = "torch_vectorized"
ROUTE_TORCH_REFERENCE = "torch_reference"


try:  # private, so degrade rather than fail if it moves
    from torch._C._functorch import is_functorch_wrapped_tensor as _is_wrapped
except ImportError:
    try:
        from torch._C._functorch import is_batchedtensor as _is_wrapped
    except ImportError:
        # Last resort: the batching key is visible in the dispatch key set even
        # when the helpers are not importable.
        def _is_wrapped(t: torch.Tensor) -> bool:
            return "FuncTorchBatched" in str(torch._C._dispatch_keys(t))

_unpack_dual = torch.autograd.forward_ad.unpack_dual


def _storage_unreadable(t: torch.Tensor) -> bool:
    """Can the kernel *not* read this tensor's values out of its storage?

    Several torch tensors satisfy every attribute-level guard in
    ``_fused_tables`` -- same ``type()``, right rank, right device, supported
    dtype, ``requires_grad`` False, no negative bit, ``torch.strided`` layout --
    and still cannot be handed to a raw pointer, because their values do not live
    in readable storage:

    * inside ``torch.func.vmap`` the batching wrapper has no storage of its own and
      ``data_ptr()`` raises;
    * a forward-mode dual tensor's ``data_ptr()`` succeeds but reaches only the
      primal, so reading it would return a plausible result that has quietly
      dropped the tangent;
    * a ZeroTensor (``torch._efficientzerotensor``) has virtual values and a null
      ``data_ptr()``, so the kernel would dereference address zero -- and an illegal
      access is sticky, taking down the whole CUDA context rather than raising.

    All of these work through the baseline, which is ordinary torch ops, so all of
    them belong on the reference path. Testing the precondition itself rather than
    enumerating types keeps the next such tensor from silently qualifying.

    Dispatch keys are not usable as the general test: a dual tensor's key set is
    byte-identical to a plain tensor's, so only the batching and ZeroTensor cases
    appear there, and the string form costs about 15x the pointer check.
    """
    # Wrapper checks must come first: ``data_ptr()`` is not safe to call on every
    # tensor -- a vmap batching wrapper raises on it -- so asking about the pointer
    # before excluding wrappers would turn a routing decision into an exception.
    if _is_wrapped(t) or _unpack_dual(t).tangent is not None:
        return True
    # A null base pointer with elements to read means the values are virtual, as
    # for a ZeroTensor. An empty tensor also reports a null pointer, but it never
    # reaches a launch, so the numel term keeps S == 0 on the fused path.
    return t.data_ptr() == 0 and t.numel() != 0


_last_route: str | None = None
_extension_error: str | None = None


def last_route() -> str | None:
    """Which route served the most recent ``forward``, or None before the first."""
    return _last_route


def extension_error() -> str | None:
    """Why the fused extension is unavailable, or None if it loaded."""
    return _extension_error


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------

_EXT_NAME = "flux_pos_embed_fused"

_CPP_SOURCE = r"""
#include <tuple>

std::tuple<at::Tensor, at::Tensor> flux_rope_tables(
    const at::Tensor& ids,
    const at::Tensor& inv_freq,
    const at::Tensor& col_axis,
    bool prefer_pair_stores);
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

// One block never exceeds this many threads; the lane dimension is the output
// width (or half of it), so the row dimension takes whatever is left over.
constexpr int64_t kMaxBlockLanes = 1024;
constexpr int64_t kTargetBlockThreads = 256;

// The baseline reaches the angle through ``ids.float()``, which is lossy for
// float64 and int64 positions. Reproduce that narrowing for every dtype.
template <typename scalar_t>
__device__ __forceinline__ double as_position(scalar_t v) {
  return static_cast<double>(static_cast<float>(v));
}

// One thread per column pair: two 128-bit stores per thread per output. For the
// captured 64-column configuration a warp covers exactly one row and writes
// 512 contiguous bytes of each output.
template <typename scalar_t>
__global__ void flux_rope_pair_kernel(
    const scalar_t* __restrict__ ids,
    const double* __restrict__ inv_freq,
    const int32_t* __restrict__ col_axis,
    double* __restrict__ cos_out,
    double* __restrict__ sin_out,
    const int64_t n_rows,
    const int64_t n_cols,
    const int64_t row_stride,
    const int64_t axis_stride) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }
  const int64_t col = static_cast<int64_t>(threadIdx.x) * 2;
  const scalar_t* src = ids + row * row_stride;

  const double p0 =
      as_position(src[static_cast<int64_t>(col_axis[col]) * axis_stride]);
  const double p1 =
      as_position(src[static_cast<int64_t>(col_axis[col + 1]) * axis_stride]);

  double sin0, cos0, sin1, cos1;
  sincos(p0 * inv_freq[col], &sin0, &cos0);
  sincos(p1 * inv_freq[col + 1], &sin1, &cos1);

  const int64_t off = row * n_cols + col;
  *reinterpret_cast<double2*>(cos_out + off) = make_double2(cos0, cos1);
  *reinterpret_cast<double2*>(sin_out + off) = make_double2(sin0, sin1);
}

// One thread per column: still fully coalesced, twice the store instructions.
// Handles odd widths, which the paired mapping cannot express.
template <typename scalar_t>
__global__ void flux_rope_single_kernel(
    const scalar_t* __restrict__ ids,
    const double* __restrict__ inv_freq,
    const int32_t* __restrict__ col_axis,
    double* __restrict__ cos_out,
    double* __restrict__ sin_out,
    const int64_t n_rows,
    const int64_t n_cols,
    const int64_t row_stride,
    const int64_t axis_stride) {
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }
  const int64_t col = static_cast<int64_t>(threadIdx.x);
  const double p = as_position(
      ids[row * row_stride +
          static_cast<int64_t>(col_axis[col]) * axis_stride]);

  double sin_v, cos_v;
  sincos(p * inv_freq[col], &sin_v, &cos_v);

  const int64_t off = row * n_cols + col;
  cos_out[off] = cos_v;
  sin_out[off] = sin_v;
}

}  // namespace

#define FLUX_DISPATCH_IDS(TYPE, NAME, ...)                       \
  AT_DISPATCH_SWITCH(                                            \
      TYPE, NAME,                                                \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)    \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)        \
      AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)       \
      AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)      \
      AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)         \
      AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__))

std::tuple<at::Tensor, at::Tensor> flux_rope_tables(
    const at::Tensor& ids,
    const at::Tensor& inv_freq,
    const at::Tensor& col_axis,
    bool prefer_pair_stores) {
  TORCH_CHECK(ids.dim() == 2, "ids must be 2-D, got ", ids.dim(), "-D");
  TORCH_CHECK(ids.is_cuda(), "ids must be a CUDA tensor");
  TORCH_CHECK(
      inv_freq.device() == ids.device() && col_axis.device() == ids.device(),
      "frequency tables must be on the same device as ids");
  TORCH_CHECK(inv_freq.scalar_type() == at::kDouble, "inv_freq must be float64");
  TORCH_CHECK(col_axis.scalar_type() == at::kInt, "col_axis must be int32");
  TORCH_CHECK(
      inv_freq.is_contiguous() && col_axis.is_contiguous(),
      "frequency tables must be contiguous");
  TORCH_CHECK(
      inv_freq.numel() == col_axis.numel(),
      "frequency and axis tables must be the same length");

  const at::cuda::CUDAGuard device_guard(ids.device());

  const int64_t n_rows = ids.size(0);
  const int64_t n_cols = inv_freq.numel();

  // One allocation backs both outputs; the two views are plain contiguous
  // [n_rows, n_cols] tensors.
  at::Tensor both =
      at::empty({2, n_rows, n_cols}, ids.options().dtype(at::kDouble));
  at::Tensor cos_out = both.select(0, 0);
  at::Tensor sin_out = both.select(0, 1);
  if (n_rows == 0 || n_cols == 0) {
    return std::make_tuple(cos_out, sin_out);
  }

  double* cos_ptr = cos_out.data_ptr<double>();
  double* sin_ptr = sin_out.data_ptr<double>();

  // 128-bit stores need an even width and 16-byte aligned bases. The caching
  // allocator gives 512-byte alignment and the sine half starts at an even
  // element whenever the width is even, so this holds in practice -- but it is
  // checked rather than assumed.
  const bool pair_ok =
      (n_cols % 2 == 0) && (n_cols / 2 <= kMaxBlockLanes) &&
      (reinterpret_cast<uintptr_t>(cos_ptr) % sizeof(double2) == 0) &&
      (reinterpret_cast<uintptr_t>(sin_ptr) % sizeof(double2) == 0);
  const bool single_ok = n_cols <= kMaxBlockLanes;
  const bool use_pair = prefer_pair_stores ? (pair_ok || !single_ok) : !single_ok;
  TORCH_CHECK(
      use_pair ? pair_ok : single_ok,
      "output width ", n_cols, " is outside the fused launch geometry");

  const int64_t lanes = use_pair ? n_cols / 2 : n_cols;
  const int64_t rows_per_block =
      std::max<int64_t>(1, kTargetBlockThreads / lanes);
  const dim3 block(
      static_cast<unsigned>(lanes), static_cast<unsigned>(rows_per_block));
  const dim3 grid(
      static_cast<unsigned>((n_rows + rows_per_block - 1) / rows_per_block));

  const int64_t row_stride = ids.stride(0);
  const int64_t axis_stride = ids.stride(1);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  FLUX_DISPATCH_IDS(ids.scalar_type(), "flux_rope_tables", [&] {
    const scalar_t* src = ids.data_ptr<scalar_t>();
    const double* freq_ptr = inv_freq.data_ptr<double>();
    const int32_t* axis_ptr = col_axis.data_ptr<int32_t>();
    if (use_pair) {
      flux_rope_pair_kernel<scalar_t><<<grid, block, 0, stream>>>(
          src, freq_ptr, axis_ptr, cos_ptr, sin_ptr,
          n_rows, n_cols, row_stride, axis_stride);
    } else {
      flux_rope_single_kernel<scalar_t><<<grid, block, 0, stream>>>(
          src, freq_ptr, axis_ptr, cos_ptr, sin_ptr,
          n_rows, n_cols, row_stride, axis_stride);
    }
  });
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(cos_out, sin_out);
}
"""


def _local_cuda_arch() -> str | None:
    """Compute capability of the visible GPUs, without creating a CUDA context.

    Blackwell-family features need the architecture-specific ``a`` suffix, so
    major versions 9, 10 and 12 are mapped accordingly.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return None
    caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    mapped = []
    for cap in caps:
        major = cap.split(".")[0]
        if major in ("9", "10", "12") and not cap.endswith("a"):
            mapped.append(f"{cap}a")
        else:
            mapped.append(cap)
    return " ".join(mapped) or None


def _load_extension():
    """Compile and load the fused kernel.

    The build directory is persistent and gitignored so only the first run pays
    for nvcc. ``TORCH_CUDA_ARCH_LIST`` is narrowed to the local capability just
    around the build -- the ambient value here lists six targets, which would
    multiply compile time for nothing -- and restored afterwards.
    """
    from torch.utils.cpp_extension import load_inline

    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXT_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    arch = _local_cuda_arch()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["flux_rope_tables"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if arch:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Compiled at import so that no nvcc or ninja work, and no new thread, can land
# inside a timed region. Import must never fail: without the extension the torch
# routes still produce correct results.
_EXT = None
try:
    if torch.cuda.is_available():
        _EXT = _load_extension()
    else:
        _extension_error = "no CUDA device visible"
except Exception as exc:  # noqa: BLE001 - any build failure falls back to torch
    _EXT = None
    _extension_error = repr(exc)

# 128-bit paired stores by default; ``single`` selects the one-column-per-thread
# mapping from the same binary so the two can be compared under the harness.
_PREFER_PAIR_STORES = (
    os.environ.get("FLUX_POS_EMBED_STORE_MAPPING", "pair").strip().lower() != "single"
)


# ---------------------------------------------------------------------------
# Frequency tables
# ---------------------------------------------------------------------------

# Keyed on everything that changes the table: the device it must live on, the
# frequency dtype, theta, and the axis widths actually used. Kept out of the
# module state so ``load_state_dict(..., strict=False)`` and ``.to(...)`` cannot
# reach it -- in particular so ``.to(dtype)`` cannot downcast the float64 table.
_TABLE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _inv_freq(
    dim: int,
    theta,
    freqs_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """The baseline's inverse-frequency expression, in one place.

    Both the fused path's cached table and the reference path's per-axis loop go
    through here, so they cannot drift apart -- which is what the bit-exactness
    claim depends on. The multiplications by 1.0 are the baseline's ``ntk_factor``
    and ``linear_factor``, which FLUX leaves at 1.0; they are numerically inert for
    a float ``theta`` and load-bearing on the reference path, where a ``theta`` the
    baseline rejects (a string, a 2-D tensor) has to raise from the same operation.
    """
    theta = theta * 1.0
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)) / 1.0


def _build_tables(
    theta: float,
    axes: tuple[int, ...],
    device: torch.device,
    freqs_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenated inverse frequencies and the column-to-axis map.

    The frequency expression is the baseline's, evaluated on the target device,
    so the table is bit-identical to the one the baseline builds.
    """
    inv_freqs = []
    col_axis = []
    for axis, dim in enumerate(axes):
        inv_freqs.append(_inv_freq(dim, theta, freqs_dtype, device))
        col_axis.append(
            torch.full((dim // 2,), axis, dtype=torch.int32, device=device)
        )
    return torch.cat(inv_freqs), torch.cat(col_axis)


def _tables(
    theta: float,
    axes: tuple[int, ...],
    device: torch.device,
    freqs_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index, freqs_dtype, theta, axes)
    entry = _TABLE_CACHE.get(key)
    if entry is None:
        entry = _build_tables(theta, axes, device, freqs_dtype)
        if device.type == "cuda":
            # These are written once and read forever after, possibly from a
            # stream other than the one that filled them -- the baseline gets
            # that ordering for free by rebuilding on every call. One
            # synchronization here, outside any timed region, makes the cached
            # tables safe on every stream from then on.
            torch.cuda.current_stream(device).synchronize()
        _TABLE_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Baseline-equivalent reference path
# ---------------------------------------------------------------------------


def _rotary_1d(
    dim: int,
    pos: torch.Tensor,
    theta: float,
    freqs_dtype: torch.dtype,
) -> torch.Tensor:
    """The baseline's complex-exponential frequency tensor for one axis.

    Mirrors ``get_1d_rotary_pos_embed``'s ``use_real=False`` branch, with the
    ``ntk_factor`` and ``linear_factor`` multiplications FLUX leaves at 1.0 kept
    in place so the arithmetic is identical.
    """
    assert dim % 2 == 0
    freqs = torch.outer(pos, _inv_freq(dim, theta, freqs_dtype, pos.device))
    return torch.polar(torch.ones_like(freqs), freqs)


class FluxPosEmbed(nn.Module):
    """2D rotary position embeddings for FLUX."""

    def __init__(self, theta: int, axes_dim: list[int] | tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global _last_route
        plan = self._fused_tables(ids)
        if plan is not None:
            inv_freq, col_axis = plan
            if _EXT is not None:
                _last_route = ROUTE_CUDA_EXT
                return _EXT.flux_rope_tables(
                    ids, inv_freq, col_axis, _PREFER_PAIR_STORES
                )
            _last_route = ROUTE_TORCH_VECTORIZED
            positions = ids.float().double()
            angles = torch.index_select(positions, 1, col_axis) * inv_freq
            return angles.cos(), angles.sin()
        _last_route = ROUTE_TORCH_REFERENCE
        return self._reference(ids)

    def _fused_tables(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tables for the fused domain, or None if ``ids`` belongs to torch.

        Everything the fused kernel cannot express -- or that the baseline turns
        into an exception -- returns None and is handled by the reference path,
        which raises what the baseline raises. The configuration is inspected,
        never coerced: ``axes_dim=[2.5]`` and ``theta="10000"`` are things the
        baseline rejects, so widening them into ints and floats here would turn
        an exception into a result.
        """
        # A subclass would have its ``__torch_function__`` bypassed by the
        # kernel's raw pointer read, and a negative-view's sign bit lives in the
        # tensor rather than in its storage.
        if type(ids) is not torch.Tensor:
            return None
        if ids.dim() != 2 or ids.requires_grad or ids.device.type != "cuda":
            return None
        if ids.dtype not in _KERNEL_DTYPES or ids.is_neg():
            return None
        if _storage_unreadable(ids):
            return None
        n_axes = ids.shape[1]
        if n_axes < 1 or n_axes > len(self.axes_dim):
            return None
        if type(self.theta) not in (int, float):
            return None
        axes = tuple(self.axes_dim[:n_axes])
        if any(type(d) is not int or d < 0 or d % 2 for d in axes):
            return None
        width = sum(d // 2 for d in axes)
        if width > 2 * _MAX_BLOCK_LANES:
            return None
        if width > _MAX_BLOCK_LANES and width % 2:
            return None
        try:
            theta = float(self.theta)
        except OverflowError:  # an int too large for a double; let torch raise
            return None
        return _tables(theta, axes, ids.device, torch.float64)

    def _reference(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = (
            torch.float32 if ids.device.type in ("mps", "npu") else torch.float64
        )
        for i in range(n_axes):
            freqs_cis = _rotary_1d(
                self.axes_dim[i], pos[:, i], self.theta, freqs_dtype
            )
            cos_out.append(freqs_cis.real)
            sin_out.append(freqs_cis.imag)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
