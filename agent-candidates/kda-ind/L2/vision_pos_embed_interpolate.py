"""Bilinear interpolation of learned 2D position embeddings (Qwen3-VL).

Every output row is an independent 4-way weighted gather from a small learned
table -- no cross-row dependency and no reduction across rows -- so the whole
call collapses into a single CUDA kernel. The eager reference issues ~25 ops per
image (~200 launches for an 8-image call) to do that trivial arithmetic, which
makes it entirely launch-overhead bound; the fused path below replaces it with
one launch whose per-image metadata travels by value in the kernel parameters,
so there is no metadata tensor, no host-to-device copy, and no scratch buffer.

``reference_forward`` reproduces the eager op sequence exactly. It serves both as
the implementation for inputs the kernel does not cover and as the oracle the
fused path is checked against.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.embedding import Embedding

# Which path each call took. The captured workload must always take the fused
# path -- a silent degradation to the eager reference would still be correct, and
# would still be fast enough to look plausible, so it has to be observable.
path_counts = {"fused": 0, "reference": 0}


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>

at::Tensor pos_embed_interp(const at::Tensor& table,
                            const std::vector<std::vector<int64_t>>& grid_thw,
                            int64_t num_grid,
                            int64_t merge_size);

at::Tensor grid_coords(int64_t n, int64_t num_grid);
"""


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kImagesPerLaunch = 32;

// 8 bfloat16 = 16 B, carried as a uint4 so the access really is one 128-bit
// transaction. Declaring this as a 16-B-aligned struct of four __nv_bfloat162 is
// not enough: nvcc then loads and stores it field by field, and since consecutive
// threads are 16 B apart those 4-B accesses spread a warp over 512 B, costing 16
// L2 sectors per instruction where 4 would do. Going through uint4 and viewing
// the words as bfloat16 pairs is what keeps LDG.E.128 / STG.E.128 in the SASS.
union Vec8 {
  uint4 raw;
  __nv_bfloat162 pair[4];
};

// 16 bfloat16 = 32 B, for a single ld.global.v4.u64 / st.global.v4.u64.
union Vec16 {
  ulonglong4 raw;
  __nv_bfloat162 pair[8];
};

struct ImageMeta {
  int h;
  int w;
  int t;
  int h_blocks;       // h / merge_size
  int w_blocks;       // w / merge_size
  int halfway_h;      // linspace's integer "halfway" for h
  int halfway_w;      // ... and for w
  float step_h;       // (num_grid - 1) / (h - 1), 0 when h == 1
  float step_w;
  long long out_row;  // first output row of this image
};

struct LaunchMeta {
  ImageMeta img[kImagesPerLaunch];
  int num_grid;
  int merge_size;
  int vectors;        // hidden_size / 8
};

// Reproduce ATen's CUDA linspace element-for-element. RangeFactories.cu computes
//   step = (end - start) / (steps - 1)          on the host, in fp32
//   val  = ind < steps/2 ? start + step * ind
//                        : end - step * (steps - ind - 1)
// with start == 0 here, so the first branch is a bare multiply. The second is
// written as an explicit fma because that is what nvcc contracts the ATen
// expression into; spelling it out makes the result independent of the
// contraction setting rather than dependent on a compiler default. A single
// wrong floor() would corrupt a whole output row-block, so this has to match
// exactly, not merely closely.
//
// steps == 1 takes ATen's fill_(start) path. The host encodes that as
// step = 0, halfway = 1, which lands index 0 in the multiply branch and yields
// 0.0f -- no branch needed here.
__device__ __forceinline__ float grid_coord(int i, int n, int halfway,
                                            float step, float last) {
  return (i < halfway) ? (step * (float)i)
                       : __fmaf_rn(-step, (float)(n - 1 - i), last);
}

struct AxisMeta {
  int halfway;
  float step;
};

// Host-side half of grid_coord. Shared by the interpolation launch and the
// verification probe so the two can never drift apart.
AxisMeta axis_meta(int64_t n, int64_t num_grid) {
  AxisMeta a;
  a.halfway = (n == 1) ? 1 : (int)(n / 2);
  a.step = (n == 1) ? 0.0f
                    : ((float)(num_grid - 1) / (float)(n - 1));
  return a;
}

// Diagnostic: evaluate grid_coord for one axis length so the index arithmetic can
// be compared against torch.linspace directly, rather than inferred from output
// differences. Not on any hot path.
__global__ void grid_coord_probe(float* __restrict__ out, int n, int halfway,
                                 float step, float last) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = grid_coord(i, n, halfway, step, last);
}

// blockIdx.(x, y, z) = (w-block, h-block, image); threadIdx.(y, z) = the row
// inside the merge tile, threadIdx.x = which 16-B chunk of the row. Picking the
// tile row out of threadIdx rather than out of a flat thread id is what keeps
// integer division and modulo out of the kernel entirely.
__global__ void pos_embed_interp_kernel(Vec8* __restrict__ out,
                                        const Vec8* __restrict__ table,
                                        const LaunchMeta meta) {
  const ImageMeta im = meta.img[blockIdx.z];
  const int hb = blockIdx.y;
  const int wb = blockIdx.x;
  // Ragged batches launch a grid as large as the largest image, so the smaller
  // ones have blocks with nothing to do. The exit is block-uniform, hence safe
  // to take before the barrier below.
  if (hb >= im.h_blocks || wb >= im.w_blocks) return;

  const int m = meta.merge_size;
  const int num_grid = meta.num_grid;
  const int vectors = meta.vectors;
  const int hi = threadIdx.y;
  const int wi = threadIdx.z;
  const int tile_row = hi * m + wi;

  // The four table rows and four blend weights depend only on the tile row, not
  // on which chunk of the row a thread owns. Recomputing them per thread would
  // repeat ~40 ALU ops `vectors` times over and cost as much as the memory
  // traffic, so the m*m threads holding threadIdx.x == 0 compute them once into
  // shared memory for the whole block.
  extern __shared__ char smem_raw[];
  int* row_base = reinterpret_cast<int*>(smem_raw);
  __nv_bfloat162* blend = reinterpret_cast<__nv_bfloat162*>(
      smem_raw + (size_t)m * m * 4 * sizeof(int));

  if (threadIdx.x == 0) {
    const float last = (float)(num_grid - 1);
    const float fy = grid_coord(hb * m + hi, im.h, im.halfway_h, im.step_h, last);
    const float fx = grid_coord(wb * m + wi, im.w, im.halfway_w, im.step_w, last);
    // Coordinates are non-negative, so truncation is floor.
    const int h_floor = (int)fy;
    const int w_floor = (int)fx;
    const int h_ceil = min(h_floor + 1, num_grid - 1);
    const int w_ceil = min(w_floor + 1, num_grid - 1);
    const float dh = fy - (float)h_floor;
    const float dw = fx - (float)w_floor;
    // Same fp32 operation order as the eager reference, then one rounding to
    // bfloat16 per tile row -- mirroring its .to(dtype) on the stacked weights.
    const float w11 = dh * dw;
    const float w10 = dh - w11;
    const float w01 = dw - w11;
    const float w00 = (1.0f - dh) - w01;

    const int slot = tile_row * 4;
    row_base[slot + 0] = h_floor * num_grid + w_floor;
    row_base[slot + 1] = h_floor * num_grid + w_ceil;
    row_base[slot + 2] = h_ceil * num_grid + w_floor;
    row_base[slot + 3] = h_ceil * num_grid + w_ceil;
    // Duplicated into both halves so __hmul2 can consume it directly.
    blend[slot + 0] = __bfloat162bfloat162(__float2bfloat16(w00));
    blend[slot + 1] = __bfloat162bfloat162(__float2bfloat16(w01));
    blend[slot + 2] = __bfloat162bfloat162(__float2bfloat16(w10));
    blend[slot + 3] = __bfloat162bfloat162(__float2bfloat16(w11));
  }
  __syncthreads();

  const int slot = tile_row * 4;
  const Vec8* src0 = table + (long long)row_base[slot + 0] * vectors;
  const Vec8* src1 = table + (long long)row_base[slot + 1] * vectors;
  const Vec8* src2 = table + (long long)row_base[slot + 2] * vectors;
  const Vec8* src3 = table + (long long)row_base[slot + 3] * vectors;
  const __nv_bfloat162 c0 = blend[slot + 0];
  const __nv_bfloat162 c1 = blend[slot + 1];
  const __nv_bfloat162 c2 = blend[slot + 2];
  const __nv_bfloat162 c3 = blend[slot + 3];

  // Merge reordering: reshape(h/m, m, w/m, m, D).permute(0, 2, 1, 3, 4), i.e.
  // flat position ((hb * w_blocks + wb) * m + hi) * m + wi. All multiplies.
  const long long row = im.out_row +
      (long long)(((hb * im.w_blocks + wb) * m + hi) * m + wi);
  Vec8* dst = out + row * vectors;
  // The t copies of an image are identical, so each blend is computed once and
  // stored t times -- halving both the gather traffic and the ALU work relative
  // to one thread per output row.
  const long long frame_stride = (long long)im.h * im.w * vectors;

  // Two vectors per thread, with both sets of gathers issued before either blend.
  // The profile showed a load-to-use bubble -- the first consumer of a gathered
  // value carried nearly all the long-scoreboard samples -- and eight 128-bit loads
  // in flight per thread covers it where more warps did not: raising occupancy via
  // a register budget measured 1.4x *slower*, while this is ~1.09x faster on the
  // count-weighted captured workload at identical output bits.
  // 32 B per thread as a single 256-bit access. Two things make this worth the inline
  // PTX. First, the pairing has to be *adjacent* -- an earlier revision gave each
  // thread chunks v and v + blockDim.x, which is 32 B of work but not 32 B of
  // contiguous address, so no 256-bit access existed to emit. Second, the width has to
  // be checked in the SASS rather than assumed: this emits
  // LDG.E.ENL2.256.CONSTANT / STG.E.ENL2.256, half the load instructions of the
  // 128-bit revision for the same bytes, and measured 1.17x on the count-weighted
  // captured workload at identical output bits.
  //
  // `vectors` must be even or the last chunk of every row would go unwritten; the
  // guard requires hidden_size % 16 == 0 and the extension TORCH_CHECKs it.
  const Vec16* s0 = reinterpret_cast<const Vec16*>(src0);
  const Vec16* s1 = reinterpret_cast<const Vec16*>(src1);
  const Vec16* s2 = reinterpret_cast<const Vec16*>(src2);
  const Vec16* s3 = reinterpret_cast<const Vec16*>(src3);
  Vec16* dst16 = reinterpret_cast<Vec16*>(dst);
  const int pairs = vectors / 2;
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
    Vec16 e0, e1, e2, e3, acc;
    asm volatile("ld.global.nc.v4.u64 {%0,%1,%2,%3}, [%4];"
                 : "=l"(e0.raw.x), "=l"(e0.raw.y), "=l"(e0.raw.z), "=l"(e0.raw.w)
                 : "l"(s0 + p));
    asm volatile("ld.global.nc.v4.u64 {%0,%1,%2,%3}, [%4];"
                 : "=l"(e1.raw.x), "=l"(e1.raw.y), "=l"(e1.raw.z), "=l"(e1.raw.w)
                 : "l"(s1 + p));
    asm volatile("ld.global.nc.v4.u64 {%0,%1,%2,%3}, [%4];"
                 : "=l"(e2.raw.x), "=l"(e2.raw.y), "=l"(e2.raw.z), "=l"(e2.raw.w)
                 : "l"(s2 + p));
    asm volatile("ld.global.nc.v4.u64 {%0,%1,%2,%3}, [%4];"
                 : "=l"(e3.raw.x), "=l"(e3.raw.y), "=l"(e3.raw.z), "=l"(e3.raw.w)
                 : "l"(s3 + p));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float2 p0 = __bfloat1622float2(__hmul2(e0.pair[j], c0));
      const float2 p1 = __bfloat1622float2(__hmul2(e1.pair[j], c1));
      const float2 p2 = __bfloat1622float2(__hmul2(e2.pair[j], c2));
      const float2 p3 = __bfloat1622float2(__hmul2(e3.pair[j], c3));
      float2 sum;
      sum.x = ((p0.x + p1.x) + p2.x) + p3.x;
      sum.y = ((p0.y + p1.y) + p2.y) + p3.y;
      acc.pair[j] = __float22bfloat162_rn(sum);
    }
    for (int ti = 0; ti < im.t; ++ti) {
      asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};" :: "l"(
                       dst16 + (long long)ti * (frame_stride / 2) + p),
                   "l"(acc.raw.x), "l"(acc.raw.y), "l"(acc.raw.z), "l"(acc.raw.w)
                   : "memory");
    }
  }
}
// Opt-in poison fill, so a coverage gap in the grid or the offsets shows up as a
// surviving NaN instead of as plausible-looking garbage from at::empty.
bool poison_output() {
  static const bool on = [] {
    const char* e = std::getenv("FK_POS_EMBED_POISON_OUTPUT");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

}  // namespace

at::Tensor pos_embed_interp(const at::Tensor& table,
                            const std::vector<std::vector<int64_t>>& grid_thw,
                            int64_t num_grid,
                            int64_t merge_size) {
  TORCH_CHECK(table.is_cuda(), "position embedding table must be a CUDA tensor");
  TORCH_CHECK(table.is_contiguous(), "position embedding table must be contiguous");
  TORCH_CHECK(table.dim() == 2, "position embedding table must be 2-D, got ",
              table.dim(), "-D");
  TORCH_CHECK(table.scalar_type() == at::kBFloat16,
              "fused path is bfloat16 only, got ", table.scalar_type());
  TORCH_CHECK(num_grid > 0 && num_grid <= 46340,
              "num_grid must be in [1, 46340] so that num_grid^2 fits the int32 "
              "table row index the kernel carries, got ", num_grid);
  TORCH_CHECK(table.size(0) == num_grid * num_grid,
              "table has ", table.size(0), " rows, expected num_grid^2 = ",
              num_grid * num_grid);
  const int64_t hidden = table.size(1);
  // % 16, not % 8: the kernel moves 32 B per thread as one 256-bit access, so it
  // walks the row in pairs of 16-B chunks. An odd chunk count would leave the last
  // one unwritten, which is a silent coverage hole rather than a wrong answer.
  TORCH_CHECK(hidden > 0 && hidden % 16 == 0,
              "hidden size must be a positive multiple of 16, got ", hidden);
  // The gathers are 128-bit, so the base must be 16-B aligned. hidden % 8 == 0
  // makes every row aligned relative to the base, and the caching allocator hands
  // out 256-B-aligned bases -- but a contiguous tensor can still carry a storage
  // offset, so this is checked rather than assumed. Misalignment would otherwise
  // surface as an opaque misaligned-address fault from inside the kernel.
  // 32-byte alignment for the 256-bit accesses. A row is hidden*2 bytes and
  // hidden % 16 == 0, so every row is 32-B aligned relative to the base; the caching
  // allocator hands out 256-B-aligned bases. A contiguous tensor can still carry a
  // storage offset, so this is checked rather than assumed -- misalignment would
  // otherwise surface as an opaque fault from inside the kernel.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(table.data_ptr()) % 32 == 0,
              "position embedding table must be 32-byte aligned for 256-bit gathers");
  TORCH_CHECK(merge_size >= 1, "spatial_merge_size must be >= 1, got ", merge_size);
  TORCH_CHECK(!grid_thw.empty(), "grid_thw_list must not be empty");

  const int m = (int)merge_size;
  const int vectors = (int)(hidden / 8);
  // 3-D block (chunk, m, m). blockDim.z caps at 64 and the product at 1024, so
  // both are checked rather than assumed -- a flat (vectors, m*m) block would
  // already overflow at m = 4 with hidden = 1152.
  TORCH_CHECK(m <= 64, "spatial_merge_size ", m, " exceeds the blockDim.z limit of 64");
  // Half the threads, two vectors each: the kernel issues both vectors' gathers
  // before either blend, which is what covers the load-to-use latency the profile
  // found. The strided loop still handles any leftover when vectors is odd.
  const int chunk_threads =
      (int)std::min<int64_t>((vectors + 1) / 2, 1024 / ((int64_t)m * m));
  TORCH_CHECK(chunk_threads >= 1,
              "spatial_merge_size ", m, " leaves no threads for the hidden dimension");

  const int64_t n_images = (int64_t)grid_thw.size();
  std::vector<int64_t> row_offset(n_images + 1, 0);
  int64_t max_h_blocks = 0, max_w_blocks = 0;
  for (int64_t i = 0; i < n_images; ++i) {
    const auto& thw = grid_thw[i];
    TORCH_CHECK(thw.size() == 3, "grid_thw_list[", i, "] must have 3 entries, got ",
                thw.size());
    const int64_t t = thw[0], h = thw[1], w = thw[2];
    TORCH_CHECK(t >= 0, "t must be non-negative, got ", t, " at image ", i);
    TORCH_CHECK(h > 0 && w > 0, "h and w must be positive, got ", h, "x", w,
                " at image ", i);
    TORCH_CHECK(h % merge_size == 0 && w % merge_size == 0,
                "h and w must be divisible by spatial_merge_size ", merge_size,
                ", got ", h, "x", w, " at image ", i);
    // Every bound below is proved *before* the corresponding multiply or add:
    // signed overflow is undefined, so a check that inspects the overflowed result
    // is not a check at all. h*w is the tile index the kernel forms in int before
    // widening to 64-bit, so it has to fit int32.
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    TORCH_CHECK(t <= kIntMax, "t must fit int32, got ", t, " at image ", i);
    TORCH_CHECK(h <= kIntMax / w,
                "h*w must fit int32, got ", h, "x", w, " at image ", i);
    const int64_t frame_rows = h * w;
    TORCH_CHECK(t == 0 || frame_rows <= std::numeric_limits<int64_t>::max() / t,
                "t*h*w overflows int64 at image ", i);
    const int64_t image_rows = t * frame_rows;
    TORCH_CHECK(row_offset[i] <= std::numeric_limits<int64_t>::max() - image_rows,
                "total output rows overflow int64 at image ", i);
    row_offset[i + 1] = row_offset[i] + image_rows;
    max_h_blocks = std::max(max_h_blocks, h / merge_size);
    max_w_blocks = std::max(max_w_blocks, w / merge_size);
  }
  TORCH_CHECK(max_h_blocks <= 65535 && max_w_blocks <= 65535,
              "grid dimensions exceed the CUDA grid limit");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(table));
  auto out = at::empty({row_offset[n_images], hidden}, table.options());
  if (out.numel() == 0) return out;
  if (poison_output()) {
    out.fill_(std::numeric_limits<float>::quiet_NaN());
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const size_t smem = (size_t)m * m * (4 * sizeof(int) + 4 * sizeof(__nv_bfloat162));
  const dim3 block(chunk_threads, m, m);

  for (int64_t first = 0; first < n_images; first += kImagesPerLaunch) {
    const int count = (int)std::min<int64_t>(kImagesPerLaunch, n_images - first);
    LaunchMeta meta;
    meta.num_grid = (int)num_grid;
    meta.merge_size = m;
    meta.vectors = vectors;
    int64_t chunk_h_blocks = 0, chunk_w_blocks = 0;
    for (int j = 0; j < count; ++j) {
      const auto& thw = grid_thw[first + j];
      const int t = (int)thw[0], h = (int)thw[1], w = (int)thw[2];
      ImageMeta& im = meta.img[j];
      im.h = h;
      im.w = w;
      im.t = t;
      im.h_blocks = h / m;
      im.w_blocks = w / m;
      // fp32 division on the host, matching where and how ATen computes it.
      const AxisMeta ah = axis_meta(h, num_grid);
      const AxisMeta aw = axis_meta(w, num_grid);
      im.halfway_h = ah.halfway;
      im.halfway_w = aw.halfway;
      im.step_h = ah.step;
      im.step_w = aw.step;
      im.out_row = (long long)row_offset[first + j];
      chunk_h_blocks = std::max<int64_t>(chunk_h_blocks, im.h_blocks);
      chunk_w_blocks = std::max<int64_t>(chunk_w_blocks, im.w_blocks);
    }
    const dim3 grid((unsigned)chunk_w_blocks, (unsigned)chunk_h_blocks,
                    (unsigned)count);
    pos_embed_interp_kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<Vec8*>(out.data_ptr()),
        reinterpret_cast<const Vec8*>(table.data_ptr()),
        meta);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

at::Tensor grid_coords(int64_t n, int64_t num_grid) {
  TORCH_CHECK(n >= 1, "n must be >= 1, got ", n);
  TORCH_CHECK(num_grid > 0, "num_grid must be positive, got ", num_grid);
  auto out = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  const AxisMeta a = axis_meta(n, num_grid);
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  grid_coord_probe<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.data_ptr<float>(), (int)n, a.halfway, a.step, (float)(num_grid - 1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


_EXTENSION_NAME = "fk_vision_pos_embed_interpolate"
_UNBUILT = object()
_extension = _UNBUILT
_build_lock = threading.Lock()


def _build_directory() -> str | None:
    """Workspace-local build cache, so repeated bench runs do not recompile and
    parallel workspaces do not contend on a shared ``~/.cache`` entry."""
    override = os.environ.get("FK_POS_EMBED_BUILD_DIR")
    root = Path(override) if override else Path(__file__).resolve().parents[2] / ".torch_extensions"
    try:
        path = root / _EXTENSION_NAME
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except OSError:
        return None  # fall back to torch's own default location


def _target_arch() -> str | None:
    """Local architecture in ``TORCH_CUDA_ARCH_LIST`` form.

    Mirrors the repo's convention (``infra/cuda_ext._local_cuda_arch``): the
    architecture-specific ``a`` suffix for major 9/10/12. Read from the live CUDA
    context instead of shelling out to ``nvidia-smi``, since the table tensor has
    already forced initialization by the time this runs.
    """
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}a" if major in (9, 10, 12) else f"{major}.{minor}"


def _compile_extension():
    from torch.utils.cpp_extension import load_inline

    # The ambient TORCH_CUDA_ARCH_LIST here lists six architectures. Pin it to
    # the local one for the build and restore it afterwards, so nothing outside
    # this function observes a change to the process environment.
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    arch = _target_arch()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["pos_embed_interp", "grid_coords"],
            extra_cflags=["-O3"],
            # Mirrors infra/cuda_ext._BASE_CUDA_CFLAGS for the parts this kernel
            # needs: torch's default nvcc flags define the bf16/half conversion
            # macros, which would block the intrinsics below.
            #
            # --use_fast_math is deliberately absent, but not for the reason it is
            # usually given here: it was measured and does *not* change the grid
            # coordinates, because fast-math leaves FMA contraction on and nvcc
            # re-contracts the expression into the same fma. What does change them is
            # -fmad=false, and the explicit __fmaf_rn survives even that. Fast-math is
            # omitted on general grounds -- it also relaxes division, transcendentals
            # and denormal handling -- not because of the index arithmetic.
            # See tools/check_fault_injection.py group [2].
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            build_directory=_build_directory(),
            verbose=False,
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def load_extension():
    """Return the compiled extension, or ``None`` if it cannot be built.

    The failure is memoized as ``None``. Caching only successes -- as
    ``functools.lru_cache`` does, since it does not cache exceptions -- would
    retry a doomed compile on every single call, which reads as correct-but-slow
    rather than as broken.
    """
    global _extension
    if _extension is _UNBUILT:
        with _build_lock:
            if _extension is _UNBUILT:
                try:
                    _extension = _compile_extension()
                except Exception as exc:  # noqa: BLE001 - degrade, do not fail
                    _extension = None
                    print(
                        "[vision_pos_embed_interpolate] CUDA extension build failed, "
                        f"using the eager reference instead: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
    return _extension


class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def _kernel_applies(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
        weight: torch.Tensor,
    ) -> bool:
        """Scalar-only guard for the fused path. Everything it rejects is handled
        by ``reference_forward``, which is why the kernel itself can stay narrow.

        This must be at least as strict as the extension's own ``TORCH_CHECK``s:
        anything the C++ would reject has to be routed here instead, or an input the
        baseline accepts would raise instead of falling back.
        """
        if dtype is not torch.bfloat16 or weight.dtype is not dtype:
            return False
        # The kernel has no backward. Grad-enabled callers get the eager reference,
        # which builds the same graph the baseline does.
        if weight.requires_grad and torch.is_grad_enabled():
            return False
        if not weight.is_cuda or weight.dim() != 2 or not weight.is_contiguous():
            return False
        num_grid = self.num_grid_per_side
        if weight.shape[0] != num_grid * num_grid or weight.shape[1] != self.hidden_size:
            return False
        # % 16, not % 8: the kernel walks each row in pairs of 16-B chunks and moves
        # 32 B per thread as one 256-bit access, so an odd chunk count would leave the
        # last chunk unwritten. Mirrors the extension's own check.
        if self.hidden_size <= 0 or self.hidden_size % 16:
            return False
        # num_grid**2 must fit the int32 table row index the kernel carries.
        if num_grid < 1 or num_grid > 46340:
            return False
        # Contiguity does not imply a 32-byte-aligned base: a contiguous tensor can
        # carry a storage offset. The gathers are 256-bit, so check it here rather
        # than let the extension raise on something the baseline handles fine.
        if weight.data_ptr() % 32:
            return False
        m = self.spatial_merge_size
        if m < 1 or m > 64 or 1024 // (m * m) < 1:
            return False
        want = torch.device(device)
        if want.type != "cuda":
            return False
        # An unindexed "cuda" means the current device. Resolving it matters on a
        # multi-GPU host: the baseline would build its indices on the current device
        # and fail the cross-device lookup, so the fused path must not quietly
        # succeed on the weight's device instead.
        want_index = want.index
        if want_index is None:
            want_index = torch.cuda.current_device()
        if want_index != weight.device.index:
            return False
        if not grid_thw_list:
            return False
        for thw in grid_thw_list:
            if len(thw) != 3:
                return False
            t, h, w = thw
            # h > 0 and w > 0 are not cosmetic: they keep the launch grid from
            # having a zero-length dimension.
            if t < 0 or h <= 0 or w <= 0 or h % m or w % m:
                return False
            # Upper bounds mirror the extension's, so an out-of-range list falls
            # back instead of raising: t must fit int32, h*w must fit the int32 tile
            # index the kernel forms, and the block counts must fit the CUDA grid.
            if t > 0x7FFFFFFF or h * w > 0x7FFFFFFF:
                return False
            if h // m > 65535 or w // m > 65535:
                return False
        return True

    def reference_forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.hidden_size

        outputs = []
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.long()
            w_floor = w_idxs.long()
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)

            embeds = self._embed(indices) * weights
            combined = embeds.sum(dim=0)
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            ).permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        weight = self._embed.emb.weight
        if self._kernel_applies(grid_thw_list, dtype, device, weight):
            ext = load_extension()
            if ext is not None:
                path_counts["fused"] += 1
                return ext.pos_embed_interp(
                    weight, grid_thw_list, self.num_grid_per_side,
                    self.spatial_merge_size)
        path_counts["reference"] += 1
        return self.reference_forward(grid_thw_list, dtype, device)


def prewarm() -> bool:
    """Compile the extension ahead of a benchmark run. Returns whether it built."""
    return load_extension() is not None
