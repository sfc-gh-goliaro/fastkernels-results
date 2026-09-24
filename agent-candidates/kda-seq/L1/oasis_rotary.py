"""Rotary embedding helpers used by Oasis, with a fused frequency-table kernel.

`OasisRotaryEmbedding.forward` builds the rotary frequency table: for positions
`t` and a 1-D frequency vector `freqs` it returns, with `F = freqs.numel()`,

    out.shape == t.shape + (2F,)              out.dtype == freqs.dtype
    out[..., i, 2f] == out[..., i, 2f+1] == cast(freqs.dtype)(t[i]) * freqs[f]

No reduction and no trigonometry: `cos`/`sin` live in `oasis_apply_rotary_emb`.
The tables are tiny -- at most a kibibyte for the shapes this op is called with
-- so the cost is host work per call, not bandwidth. The table is therefore
produced by a single fused kernel behind a plain pybind entry point, giving one
allocation and one launch per call, with a pure-PyTorch path used whenever the
fused kernel does not apply or could not be built.

Two rounding details of the reference are contractual and preserved on both
paths:

* `t` is cast to `freqs.dtype` *before* the multiply. On float32 positions
  against float16 frequencies this rounds to half first; multiplying in float32
  and rounding once at the end would give different bits.
* The output dtype follows the `freqs` *argument*, not `self.freqs` and not `t`.
"""

from __future__ import annotations

import os
from math import pi
from pathlib import Path

import torch
import torch.nn as nn

# Unique to this workspace: the name keys both the ninja build lock and the
# resulting .so, so it must not collide with any baseline op's extension.
_EXTENSION_NAME = "fk_cand_oasis_rotary_table"

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cstdint>

namespace {

constexpr int kBlockSize = 256;
constexpr int64_t kMaxBlocks = 4096;

// Both lanes of a pair hold the same value, so the pair is one naturally aligned
// wide store: the output is freshly allocated (256-byte-aligned base, never a
// caller-supplied view) and pair p sits at byte offset 2 * sizeof(out_t) * p.
// Storing through this over-aligned aggregate keeps the two elements typed as
// out_t, rather than punning them through an unrelated integer.
template <typename out_t>
struct alignas(2 * sizeof(out_t)) OutPair {
  out_t lo;
  out_t hi;
};

// One thread per output pair.
template <typename out_t, typename pos_t>
__global__ void oasis_rotary_freq_table_kernel(
    const pos_t* __restrict__ positions,
    const out_t* __restrict__ freqs,
    out_t* __restrict__ out,
    const int64_t pairs,
    const int64_t n_freqs) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < pairs;
       idx += stride) {
    const int64_t i = idx / n_freqs;
    const int64_t f = idx - i * n_freqs;

    // Cast the position to the output type first, then multiply. The product of
    // two 16-bit floats needs at most 22 mantissa bits, so it is exact in
    // float32 and rounding once here matches an in-type multiply bit for bit.
    const float scaled = static_cast<float>(static_cast<out_t>(positions[i]))
                       * static_cast<float>(freqs[f]);
    const out_t value = static_cast<out_t>(scaled);

    out_t* dst = out + i * 2 * n_freqs + 2 * f;
    *reinterpret_cast<OutPair<out_t>*>(dst) = OutPair<out_t>{value, value};
  }
}

template <typename out_t, typename pos_t>
void launch_freq_table(
    const at::Tensor& positions,
    const at::Tensor& freqs,
    at::Tensor& out,
    const int64_t pairs,
    const int64_t n_freqs,
    cudaStream_t stream) {
  const int64_t wanted = (pairs + kBlockSize - 1) / kBlockSize;
  const int blocks = static_cast<int>(wanted < kMaxBlocks ? wanted : kMaxBlocks);
  oasis_rotary_freq_table_kernel<out_t, pos_t><<<blocks, kBlockSize, 0, stream>>>(
      positions.data_ptr<pos_t>(),
      freqs.data_ptr<out_t>(),
      out.data_ptr<out_t>(),
      pairs,
      n_freqs);
}

}  // namespace

// Hand-written dispatch: an AT_DISPATCH chain over two operands would cost more
// host time than the kernel itself. Anything not covered raises, and the caller
// falls back to the PyTorch path.
#define OASIS_ROTARY_POS_CASE(pos_enum, pos_t, out_t)                        \
  case at::ScalarType::pos_enum:                                             \
    launch_freq_table<out_t, pos_t>(                                         \
        positions, freqs, out, pairs, n_freqs, stream);                      \
    return out;

#define OASIS_ROTARY_DISPATCH_POS(out_t)                                     \
  switch (positions.scalar_type()) {                                         \
    OASIS_ROTARY_POS_CASE(Float, float, out_t)                               \
    OASIS_ROTARY_POS_CASE(Half, at::Half, out_t)                             \
    OASIS_ROTARY_POS_CASE(BFloat16, at::BFloat16, out_t)                     \
    OASIS_ROTARY_POS_CASE(Long, int64_t, out_t)                              \
    OASIS_ROTARY_POS_CASE(Int, int32_t, out_t)                               \
    default:                                                                 \
      break;                                                                 \
  }

// Guards use TORCH_CHECK_VALUE, which surfaces as a Python ValueError. That is
// what makes "this input is not for me" distinguishable from a real failure: the
// caller falls back on ValueError only, so a CUDA out-of-memory or any other
// RuntimeError propagates instead of being retried on the slow path.
//
at::Tensor oasis_rotary_freq_table(
    const at::Tensor& positions, const at::Tensor& freqs) {
  TORCH_CHECK_VALUE(positions.is_cuda() && freqs.is_cuda(),
                    "oasis_rotary: fused table needs CUDA tensors");
  TORCH_CHECK_VALUE(positions.get_device() == freqs.get_device(),
                    "oasis_rotary: fused table needs both inputs on one device");
  TORCH_CHECK_VALUE(freqs.dim() == 1,
                    "oasis_rotary: fused table needs a 1-D freqs");
  TORCH_CHECK_VALUE(positions.is_contiguous() && freqs.is_contiguous(),
                    "oasis_rotary: fused table needs contiguous inputs");
  // A raw pybind entry records no autograd graph, so differentiable calls go
  // back to the PyTorch path instead of silently losing their gradient.
  TORCH_CHECK_VALUE(!(at::GradMode::is_enabled() &&
                      (positions.requires_grad() || freqs.requires_grad())),
                    "oasis_rotary: fused table is inference-only");

  const int64_t n_freqs = freqs.size(0);
  at::DimVector shape(positions.sizes().begin(), positions.sizes().end());
  shape.push_back(2 * n_freqs);

  // The guard is taken before the allocation so the output lands on the input's
  // device whatever the ambient current device is, without relying on either
  // allocator's internal guarding.
  const c10::cuda::CUDAGuard guard(positions.device());

  // at::empty, not the dispatcher-free at::detail::empty_cuda: measured
  // head-to-head through the harness's own timing primitive, the two are
  // indistinguishable, and depending on an internal ATen API would put the whole
  // extension -- including this allocator's own path -- at risk of failing to
  // build if that API changes. tools/measure_allocator.py reproduces the
  // comparison.
  at::Tensor out = at::empty(shape, freqs.options());

  const int64_t pairs = positions.numel() * n_freqs;
  if (pairs == 0) {
    return out;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (freqs.scalar_type()) {
    case at::ScalarType::Half:
      OASIS_ROTARY_DISPATCH_POS(at::Half)
      break;
    case at::ScalarType::BFloat16:
      OASIS_ROTARY_DISPATCH_POS(at::BFloat16)
      break;
    case at::ScalarType::Float:
      OASIS_ROTARY_DISPATCH_POS(float)
      break;
    default:
      break;
  }
  TORCH_CHECK_VALUE(false, "oasis_rotary: fused table has no kernel for dtype "
                    "pair ", positions.scalar_type(), " x ", freqs.scalar_type());
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor oasis_rotary_freq_table(
    const at::Tensor& positions, const at::Tensor& freqs);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oasis_rotary_freq_table", &oasis_rotary_freq_table,
        "Rotary frequency table: cast positions to the freqs dtype, multiply, "
        "and duplicate each product into an adjacent pair");
}
"""


def _local_arch_list() -> str | None:
    """Local compute capability, in the form nvcc wants for this build.

    Compute capabilities 9.0 and up need the architecture-specific `a` variant.
    Returning None leaves whatever `TORCH_CUDA_ARCH_LIST` is already set to,
    which is the right thing to do when the capability cannot be read.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _build_fused_extension():
    """Compile the fused table extension into a workspace-local build directory."""
    from torch.utils.cpp_extension import load_inline

    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    # The environment ships a multi-architecture list; compiling all of it would
    # cost minutes of wall clock for a kernel that only ever runs on this GPU.
    arch = _local_arch_list()
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch is not None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
            ],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if arch is not None:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous


# Built eagerly so nothing compiles inside a timed region, and behind a guard so
# that a build failure costs speed rather than correctness. Set
# FK_OASIS_ROTARY_FUSED=0 to exercise the PyTorch path deliberately.
_FUSED_TABLE = None
FUSED_TABLE_BUILD_ERROR: str | None = None

if os.environ.get("FK_OASIS_ROTARY_FUSED", "1") == "0":
    FUSED_TABLE_BUILD_ERROR = "disabled by FK_OASIS_ROTARY_FUSED=0"
else:
    try:
        _FUSED_TABLE = _build_fused_extension().oasis_rotary_freq_table
    except Exception as exc:  # pragma: no cover - build environment dependent
        FUSED_TABLE_BUILD_ERROR = f"{type(exc).__name__}: {exc}"

#: Whether the fused kernel is live. A benchmark taken with this False measured
#: the PyTorch path, not the kernel.
FUSED_TABLE_AVAILABLE = _FUSED_TABLE is not None


def oasis_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def oasis_apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_left = t[..., :0]
    t_middle = t[..., :rot_dim]
    t_right = t[..., rot_dim:]
    t_transformed = (t_middle * freqs.cos()) + (oasis_rotate_half(t_middle) * freqs.sin())
    return torch.cat((t_left, t_transformed, t_right), dim=-1).to(dtype)


def _reference_freq_table(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Build the frequency table in pure PyTorch, bit-for-bit like the reference.

    On CUDA, expanding `freqs` along a trailing duplicate axis makes the
    interleaved layout fall out of a contiguous reshape, so a `(N, 1, 1) x (F, 2)`
    broadcast multiply produces the answer directly -- no separate gather, and one
    kernel where the reference needs two. The cast-then-multiply order and the
    `freqs`-derived output dtype are unchanged, so the result is bit-identical.

    Everything else -- CPU tensors, and a non-1-D `freqs`, which has no meaning
    for this operator -- runs the original `einsum` expression verbatim. That
    keeps two properties the rewrite cannot promise off CUDA: identical errors,
    and identical bits when both operands are NaN. Which of two NaN payloads
    survives a multiply depends on the operand order the iterator settles on, and
    the duplicate axis changes that order; measured over a matrix of dtypes,
    widths and layouts, CUDA agrees with the reference everywhere while the CPU
    kernels do not. Only CUDA is on a latency path, so the exact expression is
    free where it is needed.
    """
    if freqs.dim() != 1 or not (positions.is_cuda and freqs.is_cuda):
        table = torch.einsum("..., f -> ... f", positions.to(freqs.dtype), freqs)
        return table.repeat_interleave(2, dim=-1)
    cast = positions if positions.dtype == freqs.dtype else positions.to(freqs.dtype)
    pairs = freqs.unsqueeze(-1).expand(freqs.size(0), 2)
    return (cast[..., None, None] * pairs).reshape(*positions.shape, 2 * freqs.size(0))


class OasisRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        freqs_for: str = "lang",
        theta: float = 10000.0,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        else:
            raise ValueError(f"unsupported rotary mode: {freqs_for}")
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def _forward_freqs(self, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # The fused entry vets its own inputs in C++, where the checks are far
        # cheaper than the equivalent Python. It rejects what it cannot handle
        # with a ValueError, and pybind raises TypeError for arguments that are
        # not tensors at all; both mean "use the PyTorch path". Anything else --
        # an out-of-memory, a real CUDA fault -- propagates rather than being
        # quietly retried on a path that would only fail again.
        if _FUSED_TABLE is not None:
            try:
                return _FUSED_TABLE(positions, freqs)
            except (ValueError, TypeError):
                pass
        return _reference_freq_table(positions, freqs)

    def forward(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        seq_len: int | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        del seq_len, offset
        return self._forward_freqs(t, freqs)

    def rotate_queries_or_keys(self, t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        positions = torch.arange(seq_len, device=t.device, dtype=t.dtype)
        seq_freqs = self.forward(positions, freqs, seq_len=seq_len)
        return oasis_apply_rotary_emb(seq_freqs, t)

    def get_axial_freqs(self, *dims: int) -> torch.Tensor:
        colon = slice(None)
        all_freqs = []
        for index, dim in enumerate(dims):
            use_pixel = self.freqs_for == "pixel" and index >= len(dims) - 2
            if use_pixel:
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)
            seq_freqs = self.forward(pos, self.freqs, seq_len=dim)
            axis = [None] * len(dims)
            axis[index] = colon
            all_freqs.append(seq_freqs[(Ellipsis, *axis, colon)])
        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)
