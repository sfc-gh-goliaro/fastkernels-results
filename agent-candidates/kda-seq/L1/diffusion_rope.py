"""Rotary position embedding for diffusion models, interleaved (GPT-J) layout.

The hot layout for this operator is the interleaved one, where a rotary pair
``(x[2i], x[2i+1])`` is *adjacent* in memory and therefore a single 32-bit
``bfloat16x2``.  That makes one 128-bit access exactly four complete rotary
pairs, so the whole head row can be read and written with wide vector accesses
and the coefficients for those four pairs fetched with one 64-bit access.  The
generic reference kernel instead gathers the pair partner through a permuted
index vector, which reads ``x`` twice and defeats vectorization entirely.

A hand-written CUDA kernel takes that fast route.  Everything it declines --
the NeoX / half-split layout, dtypes outside {bfloat16, float16}, a rotary
dimension narrower than the head, non-contiguous or misaligned inputs, a CPU
tensor, or a failed extension build -- is served by ``_rotary_reference``, a
pure-PyTorch implementation of the same math that is bit-identical to the
reference kernel on the interleaved path.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

__all__ = [
    "DiffusionRoPE",
    "rotary_extension",
    "rotary_extension_error",
    "rotary_impl_counts",
    "reset_rotary_impl_counts",
    "rotary_mapping",
    "set_rotary_mapping",
]


# ---------------------------------------------------------------------------
# Pure-PyTorch reference: the fallback, and the oracle the kernel is tested against
# ---------------------------------------------------------------------------

def _rotary_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool,
) -> torch.Tensor:
    """Out-of-place rotary embedding, fp32 accumulate, cast back to ``x.dtype``.

    ``x`` is ``(batch, seqlen, nheads, head_dim)``; ``cos``/``sin`` are
    ``(seqlen_ro, rotary_dim / 2)`` and are indexed by absolute position, so row
    ``s`` of the coefficients belongs to token ``s`` of every batch element.
    When ``rotary_dim < head_dim`` the untouched tail is copied through.
    """
    if x.dim() != 4:
        raise ValueError(
            "expected x of shape (batch, seqlen, nheads, head_dim), got "
            f"{tuple(x.shape)}"
        )
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError(
            f"expected 2-D cos/sin, got {tuple(cos.shape)} / {tuple(sin.shape)}"
        )
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}"
        )

    seqlen, head_dim = x.shape[1], x.shape[3]
    seqlen_ro, half_rotary = cos.shape
    rotary_dim = 2 * half_rotary
    if rotary_dim > head_dim:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds head_dim {head_dim}")
    if seqlen_ro < seqlen:
        raise ValueError(
            f"cos/sin cover {seqlen_ro} positions, need at least {seqlen}"
        )

    c = cos[:seqlen].to(torch.float32).reshape(1, seqlen, 1, half_rotary)
    s = sin[:seqlen].to(torch.float32).reshape(1, seqlen, 1, half_rotary)
    rot = x[..., :rotary_dim].to(torch.float32)

    if interleaved:
        even, odd = rot[..., 0::2], rot[..., 1::2]
        rotated = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
        rotated = rotated.flatten(-2)
    else:
        first, second = rot[..., :half_rotary], rot[..., half_rotary:]
        rotated = torch.cat(
            (first * c - second * s, second * c + first * s), dim=-1
        )

    out = torch.empty_like(x)
    out[..., :rotary_dim] = rotated.to(x.dtype)
    if rotary_dim < head_dim:
        out[..., rotary_dim:] = x[..., rotary_dim:]
    return out


# ---------------------------------------------------------------------------
# CUDA kernel + launcher
# ---------------------------------------------------------------------------

# The (block_size, vecs_per_thread) pairs the kernel is instantiated for.  The
# two are coupled: a thread's coefficient lane is only invariant across its
# vectors when block_size is a multiple of the head's vector count, and the
# in-flight vectors are what set register pressure.  ``vecs_per_token`` is 384
# for the hot shape (24 heads x 128 dims / 8), whose exact tilings are 128x3,
# 64x6 and 32x12.
_ROTARY_CONFIG_LIST = """
#define ROTARY_CONFIG_LIST(F)                                                  \\
  F(32, 1) F(32, 4) F(32, 6) F(32, 12)                                         \\
  F(64, 1) F(64, 2) F(64, 3) F(64, 6) F(64, 12)                                \\
  F(128, 1) F(128, 2) F(128, 3) F(128, 4) F(128, 6)                            \\
  F(256, 1) F(256, 2) F(256, 3)
"""

_ROTARY_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <optional>

__CONFIG_LIST__

namespace rotary_vec128 {

// One 128-bit access carries 8 contiguous 16-bit elements.  Interleaved layout
// puts a rotary pair in adjacent slots, so those 8 elements are 4 whole pairs
// and need 4 cos + 4 sin coefficients = one 64-bit access each.
constexpr int kVecElems = 8;
constexpr int kPairsPerVec = kVecElems / 2;

// The reference implementation asserts head_dim <= 256; matching it keeps the
// vectors-per-head count (and therefore the coefficient lane) inside 32.
constexpr int64_t kMaxHeadDim = 256;
constexpr int64_t kMaxGridYZ = 65535;
constexpr int64_t kMaxVecsPerToken = 1 << 30;

template <typename T>
struct PairTraits;

template <>
struct PairTraits<__nv_bfloat16> {
  using pair_t = __nv_bfloat162;
  static __device__ __forceinline__ float2 unpack(pair_t p) {
    return __bfloat1622float2(p);
  }
  static __device__ __forceinline__ pair_t pack(float2 f) {
    return __float22bfloat162_rn(f);
  }
};

template <>
struct PairTraits<__half> {
  using pair_t = __half2;
  static __device__ __forceinline__ float2 unpack(pair_t p) {
    return __half22float2(p);
  }
  static __device__ __forceinline__ pair_t pack(float2 f) {
    return __float22half2_rn(f);
  }
};

// Access width has to be forced through a native vector type, not merely
// requested with alignas: an aligned array of 16-bit pairs is copied element by
// element, which measured 4 store instructions and 4x the minimum store sectors
// per warp.  Moving the traffic as uint4 / uint2 and reading the pairs out of
// the union is what actually produces one 128-bit and one 64-bit instruction.
template <typename T>
union ElemVec {
  uint4 raw;
  typename PairTraits<T>::pair_t pair[kPairsPerVec];
};

template <typename T>
union CoefVec {
  uint2 raw;
  typename PairTraits<T>::pair_t pair[kPairsPerVec / 2];
};

static_assert(sizeof(ElemVec<__nv_bfloat16>) == 16, "vector must be 128-bit");
static_assert(sizeof(CoefVec<__nv_bfloat16>) == 8, "coefficients must be 64-bit");
static_assert(sizeof(ElemVec<__half>) == 16, "vector must be 128-bit");
static_assert(sizeof(CoefVec<__half>) == 8, "coefficients must be 64-bit");

// One token's share of the work for one thread: fetch this thread's four cos
// and four sin coefficients once, then rotate each of its VPT vectors.
template <typename T, int BLOCK, int VPT>
__device__ __forceinline__ void rotary_token(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    const int64_t token_base,
    const int64_t coef_base,
    const int first_vec,
    const int vecs_per_token) {
  using traits = PairTraits<T>;

  CoefVec<T> cv, sv;
  cv.raw = *reinterpret_cast<const uint2*>(cos + coef_base);
  sv.raw = *reinterpret_cast<const uint2*>(sin + coef_base);
  const float2 c_lo = traits::unpack(cv.pair[0]);
  const float2 c_hi = traits::unpack(cv.pair[1]);
  const float2 s_lo = traits::unpack(sv.pair[0]);
  const float2 s_hi = traits::unpack(sv.pair[1]);
  const float cf[kPairsPerVec] = {c_lo.x, c_lo.y, c_hi.x, c_hi.y};
  const float sf[kPairsPerVec] = {s_lo.x, s_lo.y, s_hi.x, s_hi.y};

#pragma unroll
  for (int v = 0; v < VPT; ++v) {
    const int vec = first_vec + v * BLOCK;
    // Keeps every (block_size, vecs_per_thread) pair legal for every shape;
    // predicated, so a partial tile stays correct rather than merely rare.
    if (vec < vecs_per_token) {
      const int64_t off = token_base + static_cast<int64_t>(vec) * kVecElems;
      ElemVec<T> data;
      data.raw = *reinterpret_cast<const uint4*>(x + off);
#pragma unroll
      for (int j = 0; j < kPairsPerVec; ++j) {
        const float2 e = traits::unpack(data.pair[j]);
        float2 r;
        // fp32 throughout: cos/sin arrive as unbounded values, so the
        // subtraction genuinely cancels and 16-bit arithmetic would lose it.
        r.x = e.x * cf[j] - e.y * sf[j];
        r.y = e.y * cf[j] + e.x * sf[j];
        data.pair[j] = traits::pack(r);
      }
      *reinterpret_cast<uint4*>(out + off) = data.raw;
    }
  }
}

// WALK_TOKENS makes the kernel step over the token and batch axes instead of
// reading them straight off the launch index, which is what lets a seqlen or
// batch past the 65535 launch-dimension limit run at all.  It is a template
// parameter rather than a runtime branch because the 64-bit loop bookkeeping it
// needs measured ~65 extra warp-instructions in the prologue -- around 45 % of
// the whole kernel -- and no realistic shape ever iterates more than once.
template <typename T, int BLOCK, int VPT, bool WALK_TOKENS>
__global__ __launch_bounds__(BLOCK) void rotary_interleaved_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    const int64_t batch,
    const int64_t seqlen,
    const int64_t token_stride,
    const int vecs_per_token,
    const int vecs_per_head,
    const int half_dim) {
  const int first_vec = blockIdx.x * (BLOCK * VPT) + threadIdx.x;

  // A vector's coefficients depend only on where it sits inside its head.  The
  // launcher guarantees block_size is a multiple of vecs_per_head, so that
  // offset is identical for every vector this thread handles and for every
  // block -- which is what lets the coefficient fetch be hoisted out of the
  // vector loop instead of repeated per vector.
  const int coef_off = (threadIdx.x & (vecs_per_head - 1)) * kPairsPerVec;

  if (WALK_TOKENS) {
#pragma unroll 1
    for (int64_t b = blockIdx.z; b < batch; b += gridDim.z) {
#pragma unroll 1
      for (int64_t s = blockIdx.y; s < seqlen; s += gridDim.y) {
        rotary_token<T, BLOCK, VPT>(
            out, x, cos, sin, (b * seqlen + s) * token_stride,
            s * half_dim + coef_off, first_vec, vecs_per_token);
      }
    }
  } else {
    // The grid covers every token, so the launch index is the token.
    const int64_t s = blockIdx.y;
    rotary_token<T, BLOCK, VPT>(
        out, x, cos, sin,
        (static_cast<int64_t>(blockIdx.z) * seqlen + s) * token_stride,
        s * half_dim + coef_off, first_vec, vecs_per_token);
  }
}

inline bool config_available(int block, int vpt) {
#define ROTARY_AVAIL(B, V)          \
  if (block == (B) && vpt == (V)) { \
    return true;                    \
  }
  ROTARY_CONFIG_LIST(ROTARY_AVAIL)
#undef ROTARY_AVAIL
  return false;
}

// Shape-independent default: a 128-thread block unless the token is too small
// to fill one, and the largest exact tiling of the token that keeps the
// in-flight vector count bounded.
//
// Several vectors per thread earns its keep by amortizing the coefficient
// fetch, which is two 64-bit loads plus eight unpack instructions and is
// otherwise paid once per vector.  A 24-head, 128-dim token is 384 vectors, so
// this picks 128x3: one block per token, every lane doing real work, and the
// coefficients read once for three vectors instead of once each.
inline void choose_config(int vecs_per_token, int* block_out, int* vpt_out) {
  const int block =
      vecs_per_token >= 128 ? 128 : (vecs_per_token >= 64 ? 64 : 32);
  int vpt = 1;
  for (int cand : {6, 4, 3, 2}) {
    if (config_available(block, cand) && vecs_per_token % (block * cand) == 0) {
      vpt = cand;
      break;
    }
  }
  *block_out = block;
  *vpt_out = vpt;
}

template <typename T, bool WALK_TOKENS>
bool launch_rotary_walk(
    int block,
    int vpt,
    dim3 grid,
    cudaStream_t stream,
    T* out,
    const T* x,
    const T* cos,
    const T* sin,
    int64_t batch,
    int64_t seqlen,
    int64_t token_stride,
    int vecs_per_token,
    int vecs_per_head,
    int half_dim) {
#define ROTARY_LAUNCH(B, V)                                                  \
  if (block == (B) && vpt == (V)) {                                          \
    static_assert((B) % 32 == 0 && (B) <= 1024, "illegal block size");       \
    rotary_interleaved_kernel<T, (B), (V), WALK_TOKENS>                      \
        <<<grid, (B), 0, stream>>>(                                          \
            out, x, cos, sin, batch, seqlen, token_stride, vecs_per_token,   \
            vecs_per_head, half_dim);                                        \
    return true;                                                             \
  }
  ROTARY_CONFIG_LIST(ROTARY_LAUNCH)
#undef ROTARY_LAUNCH
  return false;
}

template <typename T>
bool launch_rotary(
    bool walk_tokens,
    int block,
    int vpt,
    dim3 grid,
    cudaStream_t stream,
    T* out,
    const T* x,
    const T* cos,
    const T* sin,
    int64_t batch,
    int64_t seqlen,
    int64_t token_stride,
    int vecs_per_token,
    int vecs_per_head,
    int half_dim) {
  if (walk_tokens) {
    return launch_rotary_walk<T, true>(
        block, vpt, grid, stream, out, x, cos, sin, batch, seqlen,
        token_stride, vecs_per_token, vecs_per_head, half_dim);
  }
  return launch_rotary_walk<T, false>(
      block, vpt, grid, stream, out, x, cos, sin, batch, seqlen, token_stride,
      vecs_per_token, vecs_per_head, half_dim);
}

inline bool aligned_to(const void* p, uintptr_t bytes) {
  return (reinterpret_cast<uintptr_t>(p) % bytes) == 0;
}

}  // namespace rotary_vec128

// Returns nullopt -- None on the Python side -- for any input the vectorized
// path does not accept, so the caller needs no per-call attribute checks.
std::optional<at::Tensor> rotary_interleaved(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t block_size,
    int64_t vecs_per_thread) {
  using namespace rotary_vec128;

  if (!x.is_cuda() || !cos.is_cuda() || !sin.is_cuda()) return std::nullopt;
  if (cos.device() != x.device() || sin.device() != x.device()) {
    return std::nullopt;
  }
  if (x.dim() != 4 || cos.dim() != 2 || sin.dim() != 2) return std::nullopt;
  if (!x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous()) {
    return std::nullopt;
  }
  if (cos.sizes() != sin.sizes()) return std::nullopt;

  const auto dtype = x.scalar_type();
  if (cos.scalar_type() != dtype || sin.scalar_type() != dtype) {
    return std::nullopt;
  }
  if (dtype != at::kBFloat16 && dtype != at::kHalf) return std::nullopt;
  if (x.numel() == 0) return std::nullopt;

  const int64_t batch = x.size(0);
  const int64_t seqlen = x.size(1);
  const int64_t nheads = x.size(2);
  const int64_t head_dim = x.size(3);

  if (head_dim > kMaxHeadDim) return std::nullopt;
  if (head_dim % kVecElems != 0) return std::nullopt;
  // A power-of-two head_dim makes the coefficient lane a mask rather than a
  // modulo, and makes vecs_per_head a power of two as the mapping requires.
  if ((head_dim & (head_dim - 1)) != 0) return std::nullopt;
  // The vectorized path rotates the whole head, so there is no tail to copy.
  // Compared by division rather than doubling cos.size(1), which is a caller-
  // supplied size and would overflow the signed multiply for a degenerate one.
  if (cos.size(1) != head_dim / 2) return std::nullopt;
  // cos/sin are indexed by absolute position, so they must cover every token.
  if (cos.size(0) < seqlen) return std::nullopt;

  const int64_t vecs_per_token64 = nheads * head_dim / kVecElems;
  if (vecs_per_token64 <= 0 || vecs_per_token64 > kMaxVecsPerToken) {
    return std::nullopt;
  }

  // Contiguity does not imply alignment: a contiguous tensor may sit at a
  // nonzero storage offset, and a 128-bit access to a 16-byte-misaligned
  // address faults.  So the pointers are checked, not inferred.
  if (!aligned_to(x.const_data_ptr(), 16)) return std::nullopt;
  if (!aligned_to(cos.const_data_ptr(), 8)) return std::nullopt;
  if (!aligned_to(sin.const_data_ptr(), 8)) return std::nullopt;

  const int vecs_per_token = static_cast<int>(vecs_per_token64);
  const int vecs_per_head = static_cast<int>(head_dim / kVecElems);
  const int half_dim = static_cast<int>(head_dim / 2);

  int block = 0;
  int vpt = 0;
  if (block_size != 0 || vecs_per_thread != 0) {
    // Any nonzero value is an explicit request; only (0, 0) means "choose".
    if (block_size <= 0 || vecs_per_thread <= 0) return std::nullopt;
    if (block_size > 1024 || vecs_per_thread > 1024) return std::nullopt;
    block = static_cast<int>(block_size);
    vpt = static_cast<int>(vecs_per_thread);
  } else {
    choose_config(vecs_per_token, &block, &vpt);
  }
  if (!config_available(block, vpt)) return std::nullopt;

  // The hoisted coefficient fetch is only correct when a thread's lane inside
  // its head is invariant across its vectors and across blocks.  That needs a
  // power-of-two vecs_per_head no larger than the block, dividing it exactly --
  // enforced here so a retuned mapping cannot silently compute wrong values.
  if ((vecs_per_head & (vecs_per_head - 1)) != 0) return std::nullopt;
  if (vecs_per_head > block || block % vecs_per_head != 0) return std::nullopt;

  const int64_t tile = static_cast<int64_t>(block) * vpt;
  const int64_t grid_x = (vecs_per_token64 + tile - 1) / tile;
  const int64_t grid_y = std::min<int64_t>(seqlen, kMaxGridYZ);
  const int64_t grid_z = std::min<int64_t>(batch, kMaxGridYZ);
  const bool walk_tokens = (grid_y < seqlen) || (grid_z < batch);
  const dim3 grid(static_cast<unsigned>(grid_x),
                  static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));

  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor out = at::empty_like(x);
  if (!aligned_to(out.data_ptr(), 16)) return std::nullopt;

  const int64_t token_stride = nheads * head_dim;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  bool launched = false;
  if (dtype == at::kBFloat16) {
    launched = launch_rotary<__nv_bfloat16>(
        walk_tokens, block, vpt, grid, stream,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(sin.const_data_ptr()),
        batch, seqlen, token_stride, vecs_per_token, vecs_per_head, half_dim);
  } else {
    launched = launch_rotary<__half>(
        walk_tokens, block, vpt, grid, stream,
        reinterpret_cast<__half*>(out.data_ptr()),
        reinterpret_cast<const __half*>(x.const_data_ptr()),
        reinterpret_cast<const __half*>(cos.const_data_ptr()),
        reinterpret_cast<const __half*>(sin.const_data_ptr()),
        batch, seqlen, token_stride, vecs_per_token, vecs_per_head, half_dim);
  }
  if (!launched) return std::nullopt;
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
""".replace("__CONFIG_LIST__", _ROTARY_CONFIG_LIST.strip())

_ROTARY_CPP_SOURCE = r"""
#include <optional>

namespace py = pybind11;

std::optional<at::Tensor> rotary_interleaved(
    const at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t block_size,
    int64_t vecs_per_thread);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rotary_interleaved",
      &rotary_interleaved,
      "Interleaved rotary embedding using 128-bit vector accesses; returns "
      "None for any input the vectorized path declines.",
      py::arg("x"),
      py::arg("cos"),
      py::arg("sin"),
      py::arg("block_size") = 0,
      py::arg("vecs_per_thread") = 0);
}
"""


# ---------------------------------------------------------------------------
# Extension build (import time only -- never inside forward)
# ---------------------------------------------------------------------------

_WORKSPACE = Path(__file__).resolve().parents[2]

# Used only when no device is visible at build time; this operator's capture
# target is a Blackwell B200.  The architecture-specific variant matches the
# convention the surrounding repository uses for that family.
_DEFAULT_ARCH = "10.0a"


def _notify(message: str) -> None:
    """Announce build progress and build failures on stderr.

    stderr rather than stdout because the benchmark's stall watchdog watches the
    stderr log's mtime, so a silent multi-minute compile there looks like a hang.
    """
    print(f"[diffusion_rope] {message}", file=sys.stderr, flush=True)


def _arch_to_pin() -> str | None:
    """Target architecture list for the build, or None to leave the env alone.

    Building for every architecture the ambient ``TORCH_CUDA_ARCH_LIST`` names
    would multiply compile time by six here for no benefit, so the local device's
    architecture is pinned for the duration of the build.
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        return override.strip() or None
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}a" if major in (9, 10, 12) else f"{major}.{minor}"
    except Exception:
        pass
    return _DEFAULT_ARCH


@contextlib.contextmanager
def _pinned_arch(arch: str | None):
    if arch is None:
        yield
        return
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        yield
    finally:
        # Restored so importing this module leaves no trace on the process env.
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def _build_extension():
    """Compile the kernel, returning ``(extension, error_message)``.

    Never raises: a machine without ``nvcc``, or any compile error, degrades to
    the PyTorch fallback with a loud note on stderr rather than breaking import.
    """
    if os.environ.get("FK_DIFFUSION_ROPE_DISABLE_EXT"):
        reason = "disabled by FK_DIFFUSION_ROPE_DISABLE_EXT"
        _notify(f"CUDA extension {reason}; using the PyTorch fallback")
        return None, reason

    try:
        from torch.utils.cpp_extension import load_inline

        # Name carries a digest of the sources, so an edited kernel can never
        # load a stale .so from the persistent build directory.
        digest = hashlib.sha256(
            (_ROTARY_CPP_SOURCE + _ROTARY_CUDA_SOURCE).encode()
        ).hexdigest()[:12]
        name = f"diffusion_rope_vec128_{digest}"
        build_dir = _WORKSPACE / ".torch_extensions" / name
        build_dir.mkdir(parents=True, exist_ok=True)

        arch = _arch_to_pin()
        _notify(
            f"building CUDA extension {name!r} for arch "
            f"{arch or os.environ.get('TORCH_CUDA_ARCH_LIST', 'auto')!r} "
            f"in {build_dir} ..."
        )
        started = time.perf_counter()
        with _pinned_arch(arch):
            ext = load_inline(
                name=name,
                cpp_sources=[_ROTARY_CPP_SOURCE],
                cuda_sources=[_ROTARY_CUDA_SOURCE],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    # Source correlation for a later profile; no codegen effect.
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
                build_directory=str(build_dir),
                verbose=True,
            )
        _notify(f"CUDA extension ready in {time.perf_counter() - started:.1f}s")
        return ext, None
    except Exception as exc:  # noqa: BLE001 -- import must not fail
        reason = f"{type(exc).__name__}: {exc}"
        _notify(
            "CUDA extension build FAILED; falling back to the slower PyTorch "
            f"path -- {reason}"
        )
        return None, reason


_EXT, _EXT_ERROR = _build_extension()

_IMPL_VEC128 = "rotary_vec128_cuda"
_IMPL_FALLBACK = "rotary_torch"
_IMPL_COUNTS = {_IMPL_VEC128: 0, _IMPL_FALLBACK: 0}

# 0 means "let the launcher pick from the shape".  Overridable for mapping
# experiments; the committed default is what the launcher chooses on its own.
_BLOCK_SIZE = int(os.environ.get("FK_DIFFUSION_ROPE_BLOCK", "0"))
_VECS_PER_THREAD = int(os.environ.get("FK_DIFFUSION_ROPE_VPT", "0"))


def rotary_extension():
    """The compiled extension, or None if it is unavailable."""
    return _EXT


def rotary_extension_error() -> str | None:
    """Why the extension is unavailable, or None when it built."""
    return _EXT_ERROR


def rotary_impl_counts() -> dict[str, int]:
    """How many calls each implementation has served since the last reset."""
    return dict(_IMPL_COUNTS)


def reset_rotary_impl_counts() -> None:
    for key in _IMPL_COUNTS:
        _IMPL_COUNTS[key] = 0


def rotary_mapping() -> tuple[int, int]:
    """The requested ``(block_size, vecs_per_thread)``; ``(0, 0)`` means auto."""
    return _BLOCK_SIZE, _VECS_PER_THREAD


def set_rotary_mapping(block_size: int, vecs_per_thread: int) -> None:
    """Request a thread mapping. ``(0, 0)`` restores launcher selection.

    A mapping the kernel is not instantiated for, or one that would break the
    hoisted coefficient fetch, is declined by the launcher and served by the
    PyTorch fallback rather than computed incorrectly.
    """
    global _BLOCK_SIZE, _VECS_PER_THREAD
    _BLOCK_SIZE, _VECS_PER_THREAD = int(block_size), int(vecs_per_thread)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class DiffusionRoPE(nn.Module):
    """Apply rotary embeddings given pre-computed (cos, sin) tensors.

    Parameters
    ----------
    is_neox_style : bool
        If True, use the GPT-NeoX (half-split) layout.
        If False (default for FLUX), use the interleaved (GPT-J) layout.
    """

    def __init__(self, is_neox_style: bool = False) -> None:
        super().__init__()
        self.interleaved = not is_neox_style
        # Bound once so the call path is a plain attribute read.  The half-split
        # layout has no vectorized kernel: it does not occur in this operator's
        # workload, and its pair partner sits half a head away rather than
        # adjacent, so it would not share the interleaved kernel's structure.
        self._vectorized = (
            _EXT.rotary_interleaved
            if (_EXT is not None and self.interleaved)
            else None
        )
        # Which path this instance will *prefer*, fixed at construction.  It is
        # not a record of what served the last call: an interleaved instance
        # whose input the launcher declines still prefers the kernel while the
        # call is served by the fallback.  rotary_impl_counts() is the per-call
        # signal.
        self.preferred_impl_label = (
            _IMPL_VEC128 if self._vectorized is not None else _IMPL_FALLBACK
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        vectorized = self._vectorized
        if vectorized is not None and not (
            torch.is_grad_enabled()
            and (x.requires_grad or cos.requires_grad or sin.requires_grad)
        ):
            out = vectorized(x, cos, sin, _BLOCK_SIZE, _VECS_PER_THREAD)
            if out is not None:
                _IMPL_COUNTS[_IMPL_VEC128] += 1
                return out
        _IMPL_COUNTS[_IMPL_FALLBACK] += 1
        return _rotary_reference(x, cos, sin, self.interleaved)
