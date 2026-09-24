"""Bilinear interpolation of learned 2D position embeddings, fused into one kernel.

The baseline resamples a ``num_grid x num_grid`` grid of embedding rows onto each ``(h, w)``
grid in ``grid_thw_list``, reshuffles the tokens into ``spatial_merge_size`` tiles, repeats
the result ``t`` times and concatenates over the list.  It expresses that as ~26 torch ops
per image, and the profile in ``tools/probe_baseline.py`` says where the time goes: for 8
images / ~14k tokens it measures 2.17 ms of CUDA-event time against 0.88 ms of summed device
time across **209** launches, with 2.29 ms of CPU wall time.  More than half the latency is
Python dispatch and launch gaps -- the GPU is starved, not saturated.  The secondary cost is
a materialized 4x gather: ``self._embed(indices)`` writes ``4*h*w*D`` elements, the multiply
rewrites them, the sum reads them again, and permute/expand/cat copy the result twice more.

So the fix is granularity, not arithmetic: one kernel for the whole call, no intermediates.
A block owns one ``m x m`` merge tile of one image and writes it into all ``t`` frames, which
works because the merge reshuffle maps an ``m x m`` source tile to ``m*m`` *consecutive*
output rows and ``expand(t, -1, -1)`` repeats the whole ``h*w`` block identically.  Per-image
metadata rides in the kernel parameter block as a by-value array, so nothing is copied to the
device and the only allocation is the output.

Equivalence is a design constraint, not a hope, so the kernel reproduces the baseline's
*operation graph* rather than merely its mathematical value:

* the per-axis coordinate is ATen's own two-sided ``linspace`` form with a float32 step,
  and its upper half is emitted as one explicitly contracted ``__fmaf_rn``.  Both halves are
  then bitwise identical to ``torch.linspace(0, num_grid - 1, n, dtype=torch.float32)`` for
  every ``n`` in ``1..4096`` -- 0 mismatches, measured by ``tools/probe_linspace_fma.py``.
  The non-contracted spelling ``end - __fmul_rn(step, k)`` also gets every *floor* right but
  differs in 257943 coordinates by a last ulp, because ATen compiles its own kernel with
  contraction enabled; a float64 step is worse still (89 wrong floors, first at ``n = 330``,
  measured by ``tools/probe_linspace.py``), because ATen uses a float32 step for a float32
  ``linspace``;
* the four bilinear weights are formed in fp32 in the baseline's order and with its exact
  rounding graph (``w00`` has two separately rounded subtractions), then rounded to ``dtype``
  to match ``.to(dtype=dtype)`` on the stacked weights;
* each product is computed in fp32 and rounded back to ``dtype`` before accumulation, which
  is what a bf16 x bf16 elementwise multiply does (``opmath_t = float``, result rounded to
  bf16), and the four rounded products accumulate in fp32 with a single rounding on store,
  which is what ``sum`` over a bf16 tensor does;
* every load-bearing site uses an explicit round-to-nearest intrinsic, so no contraction the
  optimizer might or might not apply can move any of this.

That leaves only the fp32 summation *order* between this kernel and ATen's four-element
``sum``.  The four addends are ``dtype`` values held exactly in fp32, so any order agrees to
within about one fp32 ulp, and a tie at the fp32-to-bf16 boundary can move the stored value
by at most one bf16 ulp -- roughly 0.4 % relative, far inside the harness's 1e-2.  In practice
the order has not mattered: every configuration tested reaches ``torch.equal``.

The accumulator is seeded with ``+0.0`` for the same reason the rest of the ladder is spelled
out.  Seeding it with the first product instead is arithmetically identical for every finite
nonzero input and passes ``torch.equal``, the harness tolerance and every sweep -- and is still
wrong, because a row of ``-0.0`` then stores ``-0.0`` where ATen's reduction stores ``+0.0``.
``tools/test_interpolate.py`` compares bit patterns, not values, for exactly this case.

Everything the kernel is not proven to reproduce reaches ``_reference_forward``, which
transcribes the baseline expression operation for operation with one deliberate substitution in
the gather (see that method for why the substitution is what *preserves* equivalence rather than
breaking it), so this module is a strict behavioral superset: identical results, or the same
exception from the same line.  The fallback set covers the
arguments (a ``dtype`` that does not match the table -- where the baseline would *promote*
and return a different dtype -- a ``dtype`` outside bf16/fp16/fp32, a foreign device, a grid
that is not a list of three-integer lists, a non-positive extent, an ``h`` or ``w`` that is
not a multiple of ``m``, more images than the parameter block holds), the table's layout
(not CUDA, not contiguous, not 2-D, too few rows for ``num_grid**2``, a width that
disagrees with ``hidden_size``, a row size or base pointer no vector width fits), and the
execution context, which an opaque extension call would otherwise break:

* Dynamo is detected in Python, because the call never reaches the extension; the
  TorchScript tracer is detected inside it, which refuses while a trace is running so the
  traced graph records the real operator instead of a constant;
* a tensor carrying a forward-mode tangent is refused by asking the tensor, not by asking
  whether a dual level is open, so it holds however the level was entered;
* a tensor the dispatcher must see through -- batched under ``vmap``, functionalized, fake,
  meta, sparse, or merely observed while a ``TorchDispatchMode`` is active -- is refused
  before anything is allocated or any pointer is taken;
* reverse-mode autograd and autocast fall back rather than silently dropping history or
  ignoring a cast.  The reference route gathers with ``torch.embedding`` rather than through
  the frozen L1 winner's kernel precisely so that it stays differentiable and traceable -- see
  ``_reference_forward`` for why that substitution is what preserves baseline equivalence
  instead of breaking it.

Degenerate token counts are delegated rather than invented: ``t = 0`` genuinely returns
``[0, hidden]``, ``t = -1`` behaves like ``t = 1`` because ``expand(-1, ...)`` means "keep
this size", and an empty ``grid_thw_list`` raises exactly as ``torch.cat([])`` does.  The
reference path decides all three.

Capability questions are answered with an undefined tensor (``None`` in Python) so the caller
can fall back; only a genuine contract violation raises.  The build happens once at import
and yields ``None`` on failure, so ``forward`` needs no exception handler around the fast
path -- an exception escaping import would lose every benched case instead of degrading to
the baseline expression.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..L1.embedding import Embedding

# The kernel parameter block carries the per-image metadata by value, so the image count is
# bounded by that block rather than by memory.  The captured workload peaks at 8 images; a
# longer list takes the reference path.  Must match kMaxImages in the sources below.
_MAX_IMAGES = 128

_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <torch/csrc/jit/frontend/tracer.h>

// Defined in the .cu translation unit.  Takes the grid as a flat [t, h, w] * n array of
// int64 so no type shared between the two translation units has to be kept in sync, and
// returns an undefined tensor for any input the kernel does not serve.
at::Tensor pos_embed_interpolate_cuda(const at::Tensor& weight, const int64_t* grid,
                                      int64_t num_images, int64_t num_grid,
                                      int64_t merge_size, int64_t hidden_size);

// Mirrors _MAX_IMAGES in the Python module and kMaxImages in the .cu source.  The launcher
// re-checks the bound it actually implements, so a disagreement is a fallback, not a stomp.
constexpr int64_t kMaxImagesCpp = 128;

namespace {

// Reads `obj` as list[list[int]] of length-3 rows using the C API directly, so a malformed
// argument is answered with `false` -- and therefore with the baseline's own exception from
// the reference path -- instead of a pybind TypeError raised during argument conversion.
// PyList_CheckExact and PyLong_CheckExact are deliberate: a tuple, a list subclass or a bool
// would all work in the baseline expression, and delegating them is free and cannot be wrong.
bool read_grid(PyObject* obj, int64_t merge_size, int64_t* out, int64_t* num_images,
               int64_t* total_rows) {
  if (!PyList_CheckExact(obj)) {
    return false;
  }
  const Py_ssize_t n = PyList_GET_SIZE(obj);
  if (n < 1 || n > kMaxImagesCpp) {
    return false;
  }
  // Each extent is capped so that h*w cannot overflow, and the running total is capped so that
  // the sum cannot either.  The per-extent cap alone is *not* enough: 2**20 cubed is 2**60, and
  // 128 of those sum past int64, so the two division-form checks below are what actually keeps
  // the arithmetic in range.  Both caps are orders of magnitude above any real grid -- the
  // captured workload's largest single image is 2*22*40 = 1760 rows -- so this only ever
  // refuses inputs that would be refused a moment later anyway, by the grid-dimension limit or
  // by the allocation itself.
  constexpr int64_t kMaxExtent = 1LL << 20;
  constexpr int64_t kMaxRows = 1LL << 40;
  int64_t rows = 0;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PyList_GET_ITEM(obj, i);  // borrowed
    if (!PyList_CheckExact(item) || PyList_GET_SIZE(item) != 3) {
      return false;
    }
    int64_t thw[3];
    for (int j = 0; j < 3; ++j) {
      PyObject* entry = PyList_GET_ITEM(item, j);  // borrowed
      if (!PyLong_CheckExact(entry)) {
        return false;
      }
      int overflow = 0;
      // The _AndOverflow spelling reports an out-of-range value through its out-parameter
      // instead of setting a Python exception, so refusing here leaves no error to clear.
      const long long value = PyLong_AsLongLongAndOverflow(entry, &overflow);
      if (overflow != 0) {
        return false;
      }
      thw[j] = static_cast<int64_t>(value);
    }
    const int64_t t = thw[0], h = thw[1], w = thw[2];
    if (t < 1 || h < 1 || w < 1 || t > kMaxExtent || h > kMaxExtent || w > kMaxExtent) {
      return false;  // non-positive extents, and t <= 0 in particular, are the reference's
    }
    if (h % merge_size != 0 || w % merge_size != 0) {
      return false;  // the baseline's reshape raises here; let it
    }
    const int64_t frame_rows = h * w;  // <= 2**40 by the extent cap, so this cannot overflow
    if (frame_rows > kMaxRows / t || t * frame_rows > kMaxRows - rows) {
      return false;
    }
    out[3 * i + 0] = t;
    out[3 * i + 1] = h;
    out[3 * i + 2] = w;
    rows += t * frame_rows;
  }
  *num_images = static_cast<int64_t>(n);
  *total_rows = rows;
  return true;
}

}  // namespace

at::Tensor pos_embed_interpolate(const at::Tensor& weight, pybind11::object grid_thw_list,
                                 int64_t num_grid, int64_t merge_size,
                                 int64_t hidden_size) {
  // Asked first, before anything is allocated and before any pointer is taken: a batched,
  // functionalized, fake, meta or sparse table -- or any table observed while a
  // TorchDispatchMode is active -- is still a plain torch.Tensor in Python, but its data may
  // not exist at all, and taking a raw pointer to a wrapped allocation launches a kernel
  // against memory that is not there.  isTensorSubclassLike is the predicate ATen itself
  // uses, and it reports true whenever a dispatch mode is enabled.
  if (at::isTensorSubclassLike(weight) || !weight.has_storage()) {
    return at::Tensor();
  }
  // The TorchScript tracer records dispatched operators, not this call, so a traced graph
  // that reached here would hold a constant and ignore the table.  Refusing sends the caller
  // to the baseline expression, which the tracer records faithfully.
  if (torch::jit::tracer::isTracing()) {
    return at::Tensor();
  }
  // Asks the tensor rather than asking whether a dual level is open, so it holds however the
  // level was entered.  Interpolating the primal would drop the tangent silently.
  if (weight._fw_grad(/*level=*/0).defined()) {
    return at::Tensor();
  }
  if (merge_size < 1 || num_grid < 1 || hidden_size < 1) {
    return at::Tensor();
  }

  // Stack storage, so the guard chain reaches the launcher without a heap allocation.
  int64_t grid[3 * kMaxImagesCpp];
  int64_t num_images = 0;
  int64_t total_rows = 0;
  if (!read_grid(grid_thw_list.ptr(), merge_size, grid, &num_images, &total_rows)) {
    return at::Tensor();
  }
  return pos_embed_interpolate_cuda(weight, grid, num_images, num_grid, merge_size,
                                    hidden_size);
}

// Returns the coordinates the *compiled* kernel computes for one axis, as float32 on the
// given device.  The sweeps in tools/test_interpolate.py check end-to-end equivalence, which
// a wrong floor breaks loudly; this exists so that when it does break, the divergence can be
// read directly instead of inferred.  It is a diagnostic, not part of forward.
at::Tensor pos_embed_axis_coords(int64_t num_grid, int64_t n, at::Device device);
"""

_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>

namespace {

// Must agree with kMaxImagesCpp in the .cpp source and _MAX_IMAGES in the Python module.
constexpr int kMaxImages = 128;
// Capped well under the hardware's 1024 so the register budget __launch_bounds__ implies
// stays comfortable at the widest vector; the captured recipe wants only 288 anyway.
constexpr int kMaxThreads = 512;

// What a block needs to know about its image.  The small fields are int32 and only the
// output offset is int64, which is what keeps the whole array inside the parameter block:
// 24 bytes per image, 3 KB at the cap, against the 4 KB every toolkit honors (CUDA 13.0 on
// sm_100 allows 32 KB, but there is no reason to depend on that).
struct ImageMeta {
  int t;
  int h;
  int w;
  int tiles_w;  // w / merge_size, precomputed so the kernel divides once instead of twice
  long long row_offset;
};
static_assert(sizeof(ImageMeta) == 24, "ImageMeta must stay at 24 bytes per image");

struct ImageParams {
  ImageMeta img[kMaxImages];
};

// The complete kernel argument list, with slack for the pointers, the three scalars and any
// padding the ABI inserts.  Exceeding this is a compile error rather than a launch failure.
static_assert(sizeof(ImageParams) + 2 * sizeof(void*) + 4 * sizeof(int) + 64 <= 4096,
              "kernel argument block must stay inside the conservative 4 KB limit");

template <int BYTES> struct RawVec;
template <> struct RawVec<2>  { using type = unsigned short; };
template <> struct RawVec<4>  { using type = unsigned int; };
template <> struct RawVec<8>  { using type = uint2; };
template <> struct RawVec<16> { using type = uint4; };

// Round a float through `scalar_t` and back.  This is the baseline's `.to(dtype=dtype)` on
// the stacked weights, and its rounding of each bf16 x bf16 product; for fp32 both are
// identities, which is why the fp32 path needs no separate spelling.
template <typename scalar_t> struct Quantize;

template <> struct Quantize<__nv_bfloat16> {
  static __device__ __forceinline__ float round_trip(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
  }
  static __device__ __forceinline__ __nv_bfloat16 store(float x) {
    return __float2bfloat16_rn(x);
  }
  static __device__ __forceinline__ float widen(__nv_bfloat16 x) { return __bfloat162float(x); }
};

template <> struct Quantize<__half> {
  static __device__ __forceinline__ float round_trip(float x) {
    return __half2float(__float2half_rn(x));
  }
  static __device__ __forceinline__ __half store(float x) { return __float2half_rn(x); }
  static __device__ __forceinline__ float widen(__half x) { return __half2float(x); }
};

template <> struct Quantize<float> {
  static __device__ __forceinline__ float round_trip(float x) { return x; }
  static __device__ __forceinline__ float store(float x) { return x; }
  static __device__ __forceinline__ float widen(float x) { return x; }
};

// ATen's own linspace: a float32 step, and a two-sided evaluation that anchors the upper
// half at `end` so the last element is exact.  Reproduced here down to the contraction --
// `__fmaf_rn` for the upper half, matching the fma ATen's kernel is compiled into, and a
// bare `__fmul_rn` for the lower half, where `start = 0` leaves nothing to contract.  Both
// halves are then bitwise identical to torch.linspace for every n in 1..4096 at num_grid=48
// (tools/probe_linspace_fma.py).  Spelling the intrinsics explicitly is the point: it pins
// the arithmetic to the form that was measured instead of to whatever the optimizer picks.
__device__ __forceinline__ void axis_coord(int idx, int n, int num_grid,
                                           int* floor_idx, int* ceil_idx, float* frac) {
  float v;
  if (n <= 1) {
    // ATen returns `start` for a single step rather than evaluating the form, whose step
    // would divide by zero.
    v = 0.0f;
  } else {
    const float step = __fdiv_rn(static_cast<float>(num_grid - 1), static_cast<float>(n - 1));
    if (idx < (n >> 1)) {
      v = __fmul_rn(step, static_cast<float>(idx));
    } else {
      v = __fmaf_rn(-step, static_cast<float>(n - 1 - idx), static_cast<float>(num_grid - 1));
    }
  }
  // v >= 0, so the truncating cast is a floor and matches `.long()`.
  const int f = static_cast<int>(v);
  *floor_idx = f;
  *ceil_idx = min(f + 1, num_grid - 1);
  *frac = __fsub_rn(v, static_cast<float>(f));
}

// One block per merge tile per image; grid.y indexes the image so the metadata is read
// straight out of the parameter array with no search and no prefix scan.
//
// Rows are the outer loop and the t frames the inner one, which keeps live registers at
// roughly `lanes * 5` independent of m while still paying the 4x source traffic once per
// output row rather than once per frame -- that is what makes t = 2 nearly free.
template <typename scalar_t, int BYTES>
__global__ void __launch_bounds__(kMaxThreads) pos_embed_interp_kernel(
    const scalar_t* __restrict__ table,
    scalar_t* __restrict__ out,
    ImageParams params,
    int num_grid,
    int merge,
    int vecs) {
  using Raw = typename RawVec<BYTES>::type;
  constexpr int kLanes = BYTES / static_cast<int>(sizeof(scalar_t));
  static_assert(kLanes >= 1, "vector width must hold at least one element");

  const ImageMeta meta = params.img[blockIdx.y];
  const int tiles_w = meta.tiles_w;
  const int tile = static_cast<int>(blockIdx.x);
  // The grid is rectangular at the widest image, so blocks past a shorter image's tile count
  // retire here.  For the captured ragged case that is 317 of 1760 blocks, at no cost worth
  // shaping the launch around.
  if (tile >= (meta.h / merge) * tiles_w) {
    return;
  }

  const int a = tile / tiles_w;
  const int b = tile - a * tiles_w;

  const long long row_stride = static_cast<long long>(vecs);
  const long long frame_stride = static_cast<long long>(meta.h) * meta.w * row_stride;
  const Raw* __restrict__ table_v = reinterpret_cast<const Raw*>(table);
  Raw* __restrict__ out_v = reinterpret_cast<Raw*>(out);
  // The merge reshuffle sends source tile (a, b) to m*m consecutive output rows, so a
  // block's stores are one contiguous run despite the permute.
  const long long tile_base =
      meta.row_offset + static_cast<long long>(tile) * merge * merge;

  for (int hi = 0; hi < merge; ++hi) {
    int fh, ch;
    float dh;
    axis_coord(a * merge + hi, meta.h, num_grid, &fh, &ch, &dh);
    const long long row_fh = static_cast<long long>(fh) * num_grid;
    const long long row_ch = static_cast<long long>(ch) * num_grid;

    for (int wi = 0; wi < merge; ++wi) {
      int fw, cw;
      float dw;
      axis_coord(b * merge + wi, meta.w, num_grid, &fw, &cw, &dw);

      // The baseline's graph exactly: all four weights in fp32 first -- note w00's two
      // separately rounded subtractions, which left-associativity reproduces -- and only
      // then rounded to dtype, matching `.to(dtype=dtype)` on the stacked tensor.
      const float w11 = __fmul_rn(dh, dw);
      const float w10 = __fsub_rn(dh, w11);
      const float w01 = __fsub_rn(dw, w11);
      const float w00 = __fsub_rn(__fsub_rn(1.0f, dh), w01);
      const float weight_q[4] = {
          Quantize<scalar_t>::round_trip(w00),
          Quantize<scalar_t>::round_trip(w01),
          Quantize<scalar_t>::round_trip(w10),
          Quantize<scalar_t>::round_trip(w11),
      };
      // Paired with the weights in the baseline's stacking order: h is [f, f, c, c] and w is
      // [f, c, f, c], so the rows are (fh,fw), (fh,cw), (ch,fw), (ch,cw).  Both indices come
      // from clamped floors, so every row lies inside [0, num_grid**2).
      const long long src_row[4] = {row_fh + fw, row_fh + cw, row_ch + fw, row_ch + cw};

      const long long out_row = tile_base + static_cast<long long>(hi) * merge + wi;
      Raw* __restrict__ dst = out_v + out_row * row_stride;

      // A grid-stride loop over the row's vectors, so every hidden size the byte ladder admits
      // is served without a special case; for the captured recipe vecs == blockDim.x and it
      // runs exactly once, which is why hoisting anything out of this loop would save nothing.
      for (int v = static_cast<int>(threadIdx.x); v < vecs; v += static_cast<int>(blockDim.x)) {
        // Seeded with +0.0 rather than with the first product, because that is what ATen's
        // reduction does and the difference is observable: for an all -0.0 row, +0.0 + (-0.0)
        // is +0.0 while a bare -0.0 stays -0.0.  torch.equal and the harness's tolerance both
        // read the two as identical, so only a bit-pattern comparison catches it -- which is
        // exactly why it is worth spelling out instead of reasoning away.  The extra add is one
        // FADD per lane per row against ~50 float ops in the loop.
        float acc[kLanes];
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
          acc[l] = 0.0f;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const Raw raw = __ldg(table_v + src_row[k] * row_stride + v);
          const scalar_t* elem = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
          for (int l = 0; l < kLanes; ++l) {
            // fp32 product rounded back to dtype -- what a dtype x dtype elementwise
            // multiply does -- then accumulated in fp32.
            const float prod =
                Quantize<scalar_t>::round_trip(__fmul_rn(weight_q[k], Quantize<scalar_t>::widen(elem[l])));
            acc[l] = __fadd_rn(acc[l], prod);
          }
        }
        Raw packed;
        scalar_t* packed_elem = reinterpret_cast<scalar_t*>(&packed);
#pragma unroll
        for (int l = 0; l < kLanes; ++l) {
          packed_elem[l] = Quantize<scalar_t>::store(acc[l]);  // the single rounding on store
        }
        // expand(t, -1, -1) repeats the whole h*w block identically, so the computed row is
        // stored to every frame without re-reading the source.
        for (int f = 0; f < meta.t; ++f) {
          dst[f * frame_stride + v] = packed;
        }
      }
    }
  }
}

// Writes the coordinates the kernel above computes, for the diagnostic entry point.
__global__ void axis_coord_probe_kernel(float* out, int num_grid, int n) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (i >= n) {
    return;
  }
  int f, c;
  float d;
  axis_coord(i, n, num_grid, &f, &c, &d);
  out[i] = d + static_cast<float>(f);  // reassembles v without a second code path
}

bool aligned_to(const void* p, int width) {
  return (reinterpret_cast<uintptr_t>(p) % static_cast<uintptr_t>(width)) == 0;
}

// Widest access the row size and both base pointers permit, preferring one that also leaves
// a whole number of warps' worth of vectors so no lane idles.  For the captured recipe the
// row is 1152 * 2 = 2304 bytes: 16-byte vectors give 144 of them and would idle 16 of 160
// lanes, while 8-byte vectors give 288 (a multiple of 32) and move 256 contiguous bytes per
// warp per access.  Divisibility alone does not license a wide access, so the pointers are
// checked too -- allocator pointers are 512-byte aligned in practice, but checking costs
// nothing on the host and removes an assumption.
int select_vector_bytes(const void* src, const void* dst, long long row_bytes, int elem_size) {
  static const int kWidths[] = {16, 8, 4, 2};
  int widest_divisor = 0;
  for (int i = 0; i < 4; ++i) {
    const int bytes = kWidths[i];
    if (bytes < elem_size || row_bytes % bytes != 0) {
      continue;
    }
    if (!aligned_to(src, bytes) || !aligned_to(dst, bytes)) {
      continue;
    }
    if (widest_divisor == 0) {
      widest_divisor = bytes;
    }
    if ((row_bytes / bytes) % 32 == 0) {
      return bytes;
    }
  }
  // No width leaves full warps: take the widest that fits and accept the idle lanes in the
  // last one.  The grid-stride loop already bounds the work correctly.
  return widest_divisor;
}

template <typename scalar_t, int BYTES>
void launch_kernel(const at::Tensor& weight, at::Tensor& out, const ImageParams& params,
                   int num_images, int num_grid, int merge, int vecs, int max_tiles) {
  int threads = vecs < kMaxThreads ? vecs : kMaxThreads;
  threads = ((threads + 31) / 32) * 32;
  if (threads > kMaxThreads) {
    threads = kMaxThreads;
  }
  const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>(num_images));
  pos_embed_interp_kernel<scalar_t, BYTES>
      <<<grid, dim3(static_cast<unsigned>(threads)), 0, at::cuda::getCurrentCUDAStream()>>>(
          static_cast<const scalar_t*>(weight.const_data_ptr()),
          static_cast<scalar_t*>(out.mutable_data_ptr()),
          params, num_grid, merge, vecs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
bool launch_by_width(const at::Tensor& weight, at::Tensor& out, const ImageParams& params,
                     int num_images, int num_grid, int merge, int bytes, long long row_bytes,
                     int max_tiles) {
  const int vecs = static_cast<int>(row_bytes / bytes);
  switch (bytes) {
    case 16:
      launch_kernel<scalar_t, 16>(weight, out, params, num_images, num_grid, merge, vecs, max_tiles);
      return true;
    case 8:
      launch_kernel<scalar_t, 8>(weight, out, params, num_images, num_grid, merge, vecs, max_tiles);
      return true;
    case 4:
      if constexpr (sizeof(scalar_t) <= 4) {
        launch_kernel<scalar_t, 4>(weight, out, params, num_images, num_grid, merge, vecs, max_tiles);
        return true;
      }
      return false;
    case 2:
      if constexpr (sizeof(scalar_t) <= 2) {
        launch_kernel<scalar_t, 2>(weight, out, params, num_images, num_grid, merge, vecs, max_tiles);
        return true;
      }
      return false;
    default:
      return false;
  }
}

}  // namespace

at::Tensor pos_embed_interpolate_cuda(const at::Tensor& weight, const int64_t* grid,
                                      int64_t num_images, int64_t num_grid,
                                      int64_t merge_size, int64_t hidden_size) {
  // Layout questions the caller cannot see, each answered with an undefined tensor so the
  // caller falls back rather than being handed an error it would have to interpret.
  if (num_images < 1 || num_images > kMaxImages) {
    return at::Tensor();
  }
  if (!weight.is_cuda() || weight.dim() != 2 || !weight.is_contiguous()) {
    return at::Tensor();
  }
  if (weight.size(1) != hidden_size || hidden_size < 1) {
    return at::Tensor();
  }
  // num_grid comes from int(num_position_embeddings ** 0.5), so num_grid**2 can be smaller
  // than the table but must never exceed it: every row the kernel forms is < num_grid**2.
  if (num_grid > 65535 || num_grid * num_grid > weight.size(0)) {
    return at::Tensor();
  }
  const auto dtype = weight.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf && dtype != at::kFloat) {
    return at::Tensor();
  }

  // The stream and the allocation below must follow the table's device rather than whatever
  // happens to be current.
  const c10::cuda::OptionalCUDAGuard device_guard(at::device_of(weight));

  // This extension is built for exactly one architecture (see the -gencode flag), so a table on
  // a device of a different capability has no cubin here and the launch would fail with "no
  // kernel image available" -- an exception, where the contract calls for a fallback.  Asked
  // through the device properties, which are read below anyway.
  const auto* device_props = at::cuda::getCurrentDeviceProperties();
  if (device_props->major * 10 + device_props->minor != FK_BUILT_SM) {
    return at::Tensor();
  }

  const int merge = static_cast<int>(merge_size);
  ImageParams params;
  long long total_rows = 0;
  long long max_tiles = 0;
  for (int64_t i = 0; i < num_images; ++i) {
    const int64_t t = grid[3 * i + 0];
    const int64_t h = grid[3 * i + 1];
    const int64_t w = grid[3 * i + 2];
    const int64_t tiles = (h / merge) * (w / merge);
    params.img[i].t = static_cast<int>(t);
    params.img[i].h = static_cast<int>(h);
    params.img[i].w = static_cast<int>(w);
    params.img[i].tiles_w = static_cast<int>(w / merge);
    params.img[i].row_offset = static_cast<long long>(total_rows);
    total_rows += static_cast<long long>(t) * h * w;
    if (tiles > max_tiles) {
      max_tiles = tiles;
    }
  }
  if (total_rows < 1 || max_tiles < 1) {
    return at::Tensor();
  }
  // A grid the device will not accept is a fallback, never a truncated launch.
  if (max_tiles > device_props->maxGridSize[0] || num_images > device_props->maxGridSize[1]) {
    return at::Tensor();
  }
  // The output element count has to stay addressable in the 64-bit offsets the kernel forms.
  if (total_rows > (std::numeric_limits<long long>::max() / hidden_size)) {
    return at::Tensor();
  }

  at::Tensor out = at::empty({total_rows, hidden_size}, weight.options());

  const long long row_bytes = hidden_size * weight.element_size();
  const int bytes = select_vector_bytes(weight.const_data_ptr(), out.mutable_data_ptr(),
                                       row_bytes, static_cast<int>(weight.element_size()));
  if (bytes == 0) {
    return at::Tensor();  // no width fits this row size and pointer pair
  }
  if (row_bytes / bytes > 2147483647LL) {
    return at::Tensor();  // the vector count would not survive the kernel's int32 argument
  }

  bool launched = false;
  switch (dtype) {
    case at::kBFloat16:
      launched = launch_by_width<__nv_bfloat16>(weight, out, params, static_cast<int>(num_images),
                                                static_cast<int>(num_grid), merge, bytes,
                                                row_bytes, static_cast<int>(max_tiles));
      break;
    case at::kHalf:
      launched = launch_by_width<__half>(weight, out, params, static_cast<int>(num_images),
                                         static_cast<int>(num_grid), merge, bytes, row_bytes,
                                         static_cast<int>(max_tiles));
      break;
    default:
      launched = launch_by_width<float>(weight, out, params, static_cast<int>(num_images),
                                        static_cast<int>(num_grid), merge, bytes, row_bytes,
                                        static_cast<int>(max_tiles));
      break;
  }
  if (!launched) {
    return at::Tensor();
  }
  return out;
}

at::Tensor pos_embed_axis_coords(int64_t num_grid, int64_t n, at::Device device) {
  TORCH_CHECK(n >= 1 && num_grid >= 1, "pos_embed_axis_coords needs positive n and num_grid");
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  at::Tensor out = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(device));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  axis_coord_probe_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      out.mutable_data_ptr<float>(), static_cast<int>(num_grid), static_cast<int>(n));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""


def _build_extension():
    """Compile the fused kernel once, at import, or return ``None``.

    A build failure must not raise: the harness imports this module once per worker, and an
    exception here would lose every case instead of degrading to the baseline expression.
    Deciding once, here, is also what lets ``forward`` carry no exception handler around the
    fast path -- a ``try``/``except`` there would turn a genuine kernel error into a silent
    slow path.
    """
    if not torch.cuda.is_available():
        # Nothing the kernel serves is reachable without a device, and compiling the ambient
        # six-architecture list would cost minutes for nothing.
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        # Build for the one architecture actually present, as a compile flag rather than
        # through the process-wide TORCH_CUDA_ARCH_LIST: torch skips that variable entirely
        # once the cuda flags carry an `arch`, so nothing global is disturbed and a later
        # extension build in the same process is unaffected.
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = 10, 0
        arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"

        # Build beside the workspace rather than in the shared ~/.cache/torch_extensions,
        # where a name collision with a sibling agent's build causes rebuild thrashing.
        # Passed explicitly for the same reason as the arch: no environment mutation.
        name = "fk_l2_vision_pos_embed_interpolate"
        build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / name
        build_dir.mkdir(parents=True, exist_ok=True)

        ext = load_inline(
            name=name,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["pos_embed_interpolate", "pos_embed_axis_coords"],
            # -fmad=false belts the braces on the explicit rn intrinsics: the operation graph
            # this kernel reproduces is the baseline's, and a contraction the optimizer chose
            # on its own -- rather than one written down -- could move a coordinate or fuse
            # two of the four interpolation terms.
            # FK_BUILT_SM lets the host guard refuse a device this cubin was not built for,
            # instead of letting the launch fail with "no kernel image available".
            extra_cuda_cflags=["-O3", "--fmad=false", arch_flag,
                               f"-DFK_BUILT_SM={major * 10 + minor}"],
            build_directory=str(build_dir),
            verbose=False,
        )
        return ext
    except Exception:
        return None


_ext = _build_extension()
_interpolate = getattr(_ext, "pos_embed_interpolate", None)

# Resolved once here rather than walked per call: the scored measurement is bound by per-call
# host cost, so a few microseconds of attribute lookup would be visible.
_FAST_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_is_compiling = torch.compiler.is_compiling
_is_grad_enabled = torch.is_grad_enabled
_is_autocast_enabled = torch.is_autocast_enabled

# Fallback call count, for tests that need to prove which route an argument set took.  Only
# the fallback is counted, because an increment on the fast path is the same order of
# magnitude as the whole per-call host budget.  A call that leaves this unchanged took the
# fast path; calls made under Dynamo leave it unchanged too, since they return before
# reaching it.
_FALLBACK_CALLS = 0


class VisionPosEmbedInterpolate(nn.Module):
    def __init__(self, num_position_embeddings: int, hidden_size: int,
                 spatial_merge_size: int):
        super().__init__()
        # The submodule name is load-bearing: the harness shares weights with
        # load_state_dict(..., strict=False) against a baseline whose key is
        # `_embed.emb.weight`, and a mismatched key is ignored silently rather than raised --
        # which would leave this module compared against different random weights.
        self._embed = Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size

    def _reference_forward(
        self,
        grid_thw_list: list[list[int]],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """The baseline expression, operation for operation, with one gather substitution.

        Every argument the fused path does not serve arrives here, so this is what makes the
        module a strict superset: the same result, or the same exception from the same line.

        The substitution is what makes that claim true rather than breaking it.  The gather is
        spelled ``torch.embedding(self._embed.emb.weight, indices)`` where the baseline writes
        ``self._embed(indices)``.  The two are the same ATen call and agree bitwise --
        ``nn.Embedding`` with ``padding_idx=None`` *is* ``torch.embedding`` -- but the identical
        source text resolves differently in the two modules: in the baseline ``self._embed`` is
        an ``nn.Embedding``, while here it is the frozen lower-level winner, whose own custom
        kernel is opaque to the dispatcher.  Routing through it made this module differ from the
        baseline in two observable ways: under ``torch.jit.trace`` the tracer could not see the
        gather and baked in a constant, so the traced graph returned values the baseline's traced
        graph did not (measured: max abs difference 5.98 on the dominant case, and not even a
        permutation of the right values); and a table that requires grad produced no gradient,
        because that kernel carries no autograd node.  Calling the ATen op directly fixes both
        while leaving ``self._embed`` exactly where it is, which is what the shared
        ``_embed.emb.weight`` state-dict key needs.  So this is a semantic transcription, not a
        textual one, and the semantics are the ones that matter.  The fused path is unaffected --
        it never reaches here.
        """
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

            embeds = torch.embedding(self._embed.emb.weight, indices) * weights
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
        if _is_compiling():
            # Dynamo cannot trace the extension call, so tracing gets the baseline expression
            # itself -- returned before anything else, in particular before the fallback
            # counter, since mutating a module global is itself enough to stop a full graph
            # being captured.  The TorchScript tracer is handled on the other side, by the
            # extension refusing while a trace is in progress.
            return self._reference_forward(grid_thw_list, dtype, device)

        weight = self._embed.emb.weight
        # Semantic questions only, each one the extension cannot see.  `dtype is weight.dtype`
        # is required rather than merely compatible: the baseline would *promote* a mismatched
        # dtype and return a different one, and the harness compares output dtype on the first
        # round.  Everything about layout -- contiguity, rank, table height, row size, pointer
        # alignment, grid limits -- and everything about the grid list's own structure is the
        # extension's capability question, and it answers None for anything it cannot serve.
        # The grid list is passed through untouched so a malformed one produces the baseline's
        # exception rather than a conversion error.
        if (_interpolate is not None
                and dtype is weight.dtype
                and dtype in _FAST_DTYPES
                and weight.device == device
                and not (_is_grad_enabled() and weight.requires_grad)
                and not _is_autocast_enabled("cuda")):
            out = _interpolate(weight, grid_thw_list, self.num_grid_per_side,
                               self.spatial_merge_size, self.hidden_size)
            if out is not None:  # None: an argument or a layout the kernel does not serve
                return out

        global _FALLBACK_CALLS
        _FALLBACK_CALLS += 1
        return self._reference_forward(grid_thw_list, dtype, device)
