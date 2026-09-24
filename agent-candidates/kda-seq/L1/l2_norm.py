"""L2 normalization along a single dimension: x / ||x||_2.

Drop-in replacement for ``F.normalize(x, p=2, dim, eps)``.

``F.normalize`` spends three CUDA kernels per call -- ``linalg_vector_norm``,
``clamp_min(eps)``, and the broadcast divide -- and the divide re-reads ``x`` a
second time. At the shapes this operator sees (fp32 ``[rows, 1024]``, at most
16 MiB) each additional eager elementwise operation costs 3.9-4.1 us under the
benchmark harness, rising only 1.04x while the bytes moved rise 32x (four runs at
full SM clock: profile/launch_cost/REPORT.md). That increment covers dispatch,
allocation, traffic and launch together -- the experiment separates none of them --
but its near-independence of problem size is why collapsing the three operations
into one row-normalize kernel, reading ``x`` once into registers, is the win here.

The fused kernel serves a deliberately narrow envelope -- CUDA fp32, last-dim
normalization, row-major packed rows, 16-byte aligned, row length a multiple of
4 and no longer than the largest tile below. Everything else (any other dtype,
device, ``dim``, layout, alignment, row length, or a grad-requiring call) is
routed inside C++ to the ATen expression that ``F.normalize`` itself evaluates,
so it is correct by construction and the Python side stays a single
unconditional call.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Device code.
# ---------------------------------------------------------------------------

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWarp = 32;

// Largest row length the templated tiles below cover (512 threads x 4 float4).
constexpr int64_t kMaxRowLen = 8192;

__device__ __forceinline__ float dot4(const float4 v) {
  return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

// ATen's clamp_min propagates NaN from *either* operand; fmaxf deliberately does
// not -- it returns the numeric operand. Using fmaxf alone therefore turns a NaN
// row into a finite one and ignores a NaN eps entirely, both of which diverge
// from F.normalize. The measured truth table this reproduces is recorded in
// profile/nan_semantics/aten_clamp_min_truth_table.txt.
__device__ __forceinline__ float clamp_min_propagating_nan(float v, float lower) {
  // The addition is only reached when one side is already NaN, so it is just a
  // way to produce a NaN of the right kind without pulling in a header.
  if (isnan(v) || isnan(lower)) return v + lower;
  return fmaxf(v, lower);
}

// Sum ``v`` across the whole block. Five shuffles reduce within each warp, the
// warp leaders publish to shared memory, and after one barrier every thread
// sums the per-warp totals -- which avoids a second barrier and leaves every
// thread with a bit-identical result.
template <int THREADS>
__device__ __forceinline__ float block_sum(float v, float* warp_totals) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  constexpr int kWarps = THREADS / kWarp;
  if (kWarps == 1) {
    return v;
  }
  if ((threadIdx.x & (kWarp - 1)) == 0) {
    warp_totals[threadIdx.x / kWarp] = v;
  }
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    total += warp_totals[w];
  }
  return total;
}

// One block per row, one pass. The row is loaded into registers as float4,
// reduced, then scaled and stored from those same registers -- so ``x`` is read
// exactly once. ``FULL_TILE`` is set by the host when the row is exactly
// THREADS*VPT float4s, which drops the bounds check entirely.
template <int THREADS, int VPT, bool FULL_TILE>
__global__ void l2norm_rows_kernel(const float* __restrict__ x,
                                   float* __restrict__ out, int vec_len,
                                   long long in_pitch, long long out_pitch,
                                   float eps) {
  __shared__ float warp_totals[THREADS / kWarp];

  const long long row = blockIdx.x;
  const float4* __restrict__ xv =
      reinterpret_cast<const float4*>(x + row * in_pitch);
  float4* __restrict__ ov = reinterpret_cast<float4*>(out + row * out_pitch);

  float4 v[VPT];
  float sumsq = 0.f;
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (FULL_TILE || i < vec_len) {
      v[k] = xv[i];
      sumsq += dot4(v[k]);
    }
  }

  sumsq = block_sum<THREADS>(sumsq, warp_totals);

  // Divide each element by the denominator, rather than multiplying by its
  // reciprocal. The reciprocal form is the obvious optimization and it is NOT
  // used, for two measured reasons:
  //
  //   * It is wrong for small denominators. With a subnormal eps (below
  //     1/FLT_MAX ~ 2.94e-39) and a zero row, 1/eps overflows to Inf and
  //     0 * Inf = NaN, where F.normalize gives 0; a tiny nonzero row likewise
  //     goes to Inf instead of a finite value. Verified for eps down to 5e-45.
  //   * It is not faster here. Paired at this tile over 15 alternating reps in
  //     each of four processes at full SM clock, the two forms differ by
  //     0.000-0.032 us at all three captured shapes (<=0.3%, inside the harness's
  //     own spread): profile/kernel_structure/divide_paired.jsonl. Dividing is
  //     also the more accurate of the two on that sample -- 2 ULP against 3 --
  //     though over the full distribution the divide form's own worst case is
  //     2/3/3 ULP by shape (profile/numerics/error_distributions.json).
  //
  // The reduction tree and sqrtf still round, so this is not bit-identical to
  // F.normalize; the measured agreement is in
  // profile/numerics/error_distributions.json. rsqrtf is avoided because it is an
  // approximate instruction and would reintroduce the same class of divergence the
  // reciprocal caused; its error was not measured here.
  const float denom = clamp_min_propagating_nan(sqrtf(sumsq), eps);

#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int i = threadIdx.x + k * THREADS;
    if (FULL_TILE || i < vec_len) {
      float4 r;
      r.x = v[k].x / denom;
      r.y = v[k].y / denom;
      r.z = v[k].z / denom;
      r.w = v[k].w / denom;
      ov[i] = r;
    }
  }
}

struct RowPlan {
  int64_t rows;
  int64_t row_len;
  int64_t in_pitch;
};

// The fused kernel is admitted only where its addressing and its 128-bit
// accesses are provably correct. Everything it rejects goes to ATen.
bool plan_rows(const at::Tensor& x, int64_t norm_dim, RowPlan* plan) {
  if (!x.is_cuda() || x.layout() != at::kStrided) return false;
  if (x.scalar_type() != at::kFloat) return false;
  if (x.numel() <= 0) return false;

  // A raw kernel writing into a fresh tensor has no gradient relationship to
  // ``x``, so a grad-requiring call must take the differentiable ATen path.
  if (at::GradMode::is_enabled() && x.requires_grad()) return false;

  // A lazily-applied negative bit leaves the *un-negated* values in storage, so
  // reading through a raw pointer would normalize the wrong signs. (The
  // conjugate bit cannot be set on a real dtype, so it needs no check here.)
  if (x.is_neg()) return false;

  // A forward-mode dual tensor carries a tangent that a raw kernel would drop
  // without a word. These have requires_grad() == false, so the check above does
  // not cover them.
  if (x._fw_grad(/*level=*/0).defined()) return false;

  const int64_t ndim = x.dim();
  if (ndim < 1 || norm_dim != ndim - 1) return false;

  const int64_t last = ndim - 1;
  const int64_t row_len = x.size(last);
  if (row_len > kMaxRowLen) return false;
  if (row_len % 4 != 0) return false;         // a row must be whole float4s
  if (x.stride(last) != 1) return false;

  // Rows are addressed as ``row * pitch``, so the leading dimensions have to be
  // row-major packed with respect to that pitch. ``stride(-2)`` alone is not
  // enough: a tensor sliced along a middle dimension keeps the outer stride of
  // its parent, and addressing it by ``stride(-2)`` would read the wrong rows.
  // A size-1 dimension carries no observable stride, so it is skipped.
  int64_t pitch = row_len;  // only one row: the pitch is never used
  int64_t expected = -1;
  for (int64_t d = last - 1; d >= 0; --d) {
    if (x.size(d) == 1) continue;
    if (expected < 0) {
      pitch = x.stride(d);
      expected = pitch;
    }
    if (x.stride(d) != expected) return false;
    expected *= x.size(d);
  }
  if (pitch < row_len) return false;          // overlapping rows
  if (pitch % 4 != 0) return false;           // keeps every row start 16B-aligned

  const int64_t rows = x.numel() / row_len;
  if (rows > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }

  // ``at::empty`` is allocator-aligned, but a strided *view* of the input can
  // carry a storage offset that leaves the data pointer only 4-byte aligned,
  // which a float4-reinterpreting kernel must never see.
  if (reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 != 0) return false;

  plan->rows = rows;
  plan->row_len = row_len;
  plan->in_pitch = pitch;
  return true;
}

template <int THREADS, int VPT>
void launch_tile(const float* xp, float* op, int vec_len, long long in_pitch,
                 long long out_pitch, unsigned grid, float eps,
                 cudaStream_t stream) {
  if (vec_len == THREADS * VPT) {
    l2norm_rows_kernel<THREADS, VPT, true><<<grid, THREADS, 0, stream>>>(
        xp, op, vec_len, in_pitch, out_pitch, eps);
  } else {
    l2norm_rows_kernel<THREADS, VPT, false><<<grid, THREADS, 0, stream>>>(
        xp, op, vec_len, in_pitch, out_pitch, eps);
  }
}

}  // namespace

torch::Tensor l2_norm(const torch::Tensor& x, int64_t dim, double eps) {
  const int64_t ndim = x.dim();
  const int64_t norm_dim = dim < 0 ? dim + ndim : dim;

  RowPlan plan;
  if (plan_rows(x, norm_dim, &plan)) {
    auto out = at::empty(x.sizes(), x.options());

    const c10::cuda::CUDAGuard device_guard(x.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int vec_len = static_cast<int>(plan.row_len / 4);
    const auto grid = static_cast<unsigned>(plan.rows);
    const auto in_pitch = static_cast<long long>(plan.in_pitch);
    const auto out_pitch = static_cast<long long>(plan.row_len);
    const float epsf = static_cast<float>(eps);
    const float* xp = x.const_data_ptr<float>();
    float* op = out.data_ptr<float>();

    // Tile per row length, each picked by measurement, not by rule
    // (profile/scratch_tile_sweep.py sweeps every row length here; the n=1024 row
    // is additionally confirmed by a 2x2 factorial against Triton in
    // profile/scratch_backend_factorial.py).
    //
    // What is measured: at 8 MiB the tiles below cost ~13.3 us where the tile one
    // step away in thread count costs ~15.3 us -- e.g. at n=1024 and 2048 rows,
    // (128,2) is 13.28 us against (256,1) at 15.31 us. The same +2.0 us step
    // appears for Triton at the same launch geometry, so it tracks the geometry
    // rather than the backend.
    //
    // Why is still open. NCU on both tiles at [2048,1024], against this exact
    // arithmetic (profile/ncu_tile_128x2_vs_256x1_divide/REPORT.md), rules out the
    // obvious answers: neither tile spills (0 local ld/st, at 28 and 24 registers),
    // and the slower tile has the *higher* achieved occupancy (74.7% vs 60.5%) and
    // more eligible warps per cycle (2.51 vs 1.81), so the simple achieved-occupancy
    // explanation is contradicted. Latency hiding more broadly is unresolved: no
    // counter in the collected set measures outstanding loads per thread. No cache
    // condition reproduces a kernel-time difference near the end-to-end gap (736 ns
    // warm against ~2000 ns), and the two harnesses are not directly comparable, so
    // the residue is unresolved. The table records measured winners; it does not
    // claim to explain them.
    //
    // At 64 rows every tile lands within 11.23-11.30 us, i.e. below what this
    // harness can resolve, so no row of the table is tuned for the small cases.
    if (vec_len <= 128) {          // row <= 512
      launch_tile<64, 2>(xp, op, vec_len, in_pitch, out_pitch, grid, epsf, stream);
    } else if (vec_len <= 256) {   // row <= 1024  <- the captured row length
      launch_tile<128, 2>(xp, op, vec_len, in_pitch, out_pitch, grid, epsf, stream);
    } else if (vec_len <= 512) {   // row <= 2048
      launch_tile<128, 4>(xp, op, vec_len, in_pitch, out_pitch, grid, epsf, stream);
    } else if (vec_len <= 1024) {  // row <= 4096
      launch_tile<128, 8>(xp, op, vec_len, in_pitch, out_pitch, grid, epsf, stream);
    } else {                       // row <= 8192
      launch_tile<512, 4>(xp, op, vec_len, in_pitch, out_pitch, grid, epsf, stream);
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
  }

  // Exactly what F.normalize(p=2) evaluates, for any dim / dtype / device /
  // layout, and differentiable.
  const int64_t dims[1] = {dim};
  auto denom = at::linalg_vector_norm(x, 2, at::IntArrayRef(dims, 1),
                                      /*keepdim=*/true);
  return x / denom.clamp_min(eps);
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor l2_norm(const torch::Tensor& x, int64_t dim, double eps);
"""

# ---------------------------------------------------------------------------
# Build. Eager, at import -- never lazily on the first call, which would land
# inside the benchmark's timing region.
# ---------------------------------------------------------------------------

_EXT_NAME = "fk_l2_norm_fused_ext"
# <workspace>/candidate/L1/l2_norm.py -> <workspace>. Sibling agent workspaces
# share $HOME, so a default ~/.cache/torch_extensions/<name> would collide on
# both the build lock and the artifacts.
_BUILD_DIR = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXT_NAME


def _local_cuda_arch() -> str | None:
    """Compute capability of the visible device(s), in TORCH_CUDA_ARCH_LIST form.

    Queried through nvidia-smi so the lookup does not initialize CUDA. Blackwell
    wants the architecture-specific ``a`` variant, matching how the baseline
    infrastructure pins its own builds.
    """
    caps: list[str] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        caps = sorted({c.strip() for c in out.splitlines() if c.strip()})
    except Exception:  # noqa: BLE001 - fall through to the torch query
        caps = []
    if not caps:
        try:
            major, minor = torch.cuda.get_device_capability()
            caps = [f"{major}.{minor}"]
        except Exception:  # noqa: BLE001 - no device; let load_inline decide
            return None
    mapped = []
    for cap in caps:
        major = cap.split(".")[0]
        mapped.append(f"{cap}a" if major in ("9", "10", "12") and not cap.endswith("a")
                      else cap)
    return " ".join(mapped) or None


def _build():
    """Compile the extension, pinning the target architecture explicitly.

    The ambient environment already sets a multi-architecture
    ``TORCH_CUDA_ARCH_LIST``, and ``cpp_extension`` only autodetects when that
    variable is *unset* -- so without this the build would emit a fat binary and
    pay one nvcc pass per architecture. The variable is process-global, so the
    previous value is restored afterwards.
    """
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    arch = _local_cuda_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXT_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["l2_norm"],
            # -lineinfo is free at runtime and is what makes profiler source
            # attribution work. No --use_fast_math: it would rewrite the one sqrtf
            # and the per-element divides, which is exactly the arithmetic whose
            # agreement with F.normalize is being relied on.
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(_BUILD_DIR),
            verbose=not (_BUILD_DIR / f"{_EXT_NAME}.so").exists(),
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


try:
    _EXT = _build()
except Exception:  # noqa: BLE001 - stay correct even if the toolchain fails
    # Degrade to the reference implementation rather than to a failing operator,
    # but say so loudly: a silent fallback would report a meaningless ~1.0x.
    print(f"[l2_norm] CUDA extension {_EXT_NAME!r} failed to build; falling back "
          f"to F.normalize. This run is NOT a performance result.", file=sys.stderr)
    traceback.print_exc()
    sys.stderr.flush()
    _EXT = None

_FUSED_L2_NORM = None if _EXT is None else _EXT.l2_norm


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _FUSED_L2_NORM is None:
            return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)
        return _FUSED_L2_NORM(x, self.dim, self.eps)
