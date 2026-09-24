"""YOLOv10 Spatial Pyramid Pooling - Fast, collapsed into one extension call.

The baseline runs fourteen GPU operations -- two 1x1 convolutions, two batch norms, two
SiLUs, three max pools (each of which also materialises an int64 argmax nobody asked for),
a concatenation and the device-to-device copies around them -- and pays fourteen host
dispatches to issue them. At the captured sizes that is the entire cost: ``N=4`` needs
262 MMAC and ~2 MiB of DRAM traffic, roughly 0.25 us of B200 tensor-core math and 0.3 us of
bandwidth, against a measured 178.7 us of wall clock. Only 81.4 us of that interval is GPU
op time; the rest is the device idling while the host issues the next launch. So the
optimisation is not to make any one of those operations faster -- it is to stop launching
them. This module issues one Python-level call, which runs three kernels: the first convolution, the
pooling and concatenation, and the second convolution.

Why three and not two, since the pooling could be folded into the first convolution's epilogue and
was: a plane cannot be split across blocks, because pooling needs all of it. Fusing the pooling
therefore caps the grid at one block per (image, channel group), and measurement across the whole
channel-group axis says that cap costs far more than the extra launch saves -- variant A runs at
0.80x of the baseline at its best (32 blocks of 148 SMs) and 0.15x at its worst, against 1.98x for
the split. See `profile/variant_a/` for the sweep and `profile/sppf_v1_wmma_naive/REPORT.md` for the
profile that first showed it.

Three rewrites make that collapse possible.

*The pool cascade is one monotone accumulation.* ``self.m`` is a same-shape pool
(``stride=1``, ``padding=k//2``), and ``F.max_pool2d`` ignores padding cells -- identical
to ``-inf`` padding when ``pad <= k//2``, because every window contains its own centre
pixel, so no window is entirely padding. Max-plus dilation then composes by Minkowski sum
of the structuring elements, so ``pool5(pool5(x))`` is ``pool9(x)`` and
``pool5(pool5(pool5(x)))`` is ``pool13(x)``, bitwise. The three cascade outputs are
therefore radii 2, 4 and 6 of one outward-growing running max, snapshotted as it grows,
rather than three separate passes over the plane.

*BatchNorm folds into an fp32 epilogue on the GEMM accumulator.* In eval with
``track_running_stats``, BN is per-channel affine, so with ``inv = rsqrt(var + eps)``,
``s = gamma * inv`` and ``b = beta - gamma * mean * inv``, the whole conv -> BN -> SiLU
block is ``SiLU(acc * s + b)``. Computed in fp32 on the accumulator and rounded to fp16
once, this is *more* accurate than the baseline, which rounds after the conv, again after
BN, and again after SiLU.

*A 1x1 convolution over an NCHW plane is a GEMM whose operands need no repacking.* The
channel axis is the contraction axis and the plane's pixels are the free axis, so
``A(m=pixel, k=channel) = x[k][m]`` is column-major with ``ldm = H*W`` and
``B(k=channel, n=out) = W[n][k]`` is column-major with ``ldm = K``. An NCHW-contiguous
activation and a ``[out_channels, in_channels]`` weight are already in exactly those
layouts, so ``wmma`` fragments load straight out of them.

What is *not* valid, and is tempting: SiLU is non-monotonic (its minimum is ~= -0.2785 at
x ~= -1.2785), so ``max(SiLU(a), SiLU(b)) != SiLU(max(a, b))``. The pools cannot be hoisted
in front of the activation and ``cv1``'s activation cannot be deferred into ``cv2``'s
epilogue; both rewrites are silently wrong wherever the pre-activation values are negative.

The fast path is claimed only for the configuration the kernels implement; see
:meth:`YOLOSPPF._derive_folded` for the module side of that predicate and :func:`_input_refusal` for
the tensor side. Everything else -- another kernel size, a non-contiguous or non-fp16 or CPU
input, training mode, a channel count that is not a multiple of the 16-wide ``wmma`` tile, a
plane whose pixel count is not, a build failure -- runs the baseline composition verbatim
through ``self.cv1`` / ``self.m`` / ``torch.cat`` / ``self.cv2``, which reproduces the
reference's own behaviour rather than an approximation of it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv


#: The wmma tile extent, in every axis. There is no masked tail path, so every tile extent
#: the kernels use must be a multiple of this and must divide its axis.
_TILE = 16

#: Lanes in a warp. The pooling stage maps one lane per plane column, which is what bounds
#: the row length the fast path accepts.
_WARP_LANES = 32


def _warps(kernel: str, default: int) -> int:
    """Warps per block for one kernel.

    Read from the environment so ``profile/sweep_warps.py`` can vary it across processes
    without editing this file; the defaults are the values measurement settled on. An
    unrecognised value falls back to the default rather than producing an unlaunchable block.
    """
    try:
        value = int(os.environ.get(f"FK_SPPF_{kernel}_WARPS", default))
    except (TypeError, ValueError):
        return default
    return value if value in (1, 2, 4, 8, 16, 32) else default


def _channel_tiles(default: int = 1) -> int:
    """The ceiling on channel tiles per warp, overridable for the sweep."""
    try:
        value = int(os.environ.get("FK_SPPF_CHANNEL_TILES", default))
    except (TypeError, ValueError):
        return default
    return value if value in (1, 2, 4) else default


#: Output channel tiles a warp accumulates at once: that many independent weight loads and
#: accumulators per contraction step, all fed by one activation fragment.
#:
#: Measurement chose 1, against the expectation that more would help. Giving a warp more tiles
#: buys instruction-level parallelism and cuts activation traffic, but it also divides the task
#: count -- and for an operator this small the task count is what decides how many SMs run at all.
#: See profile/sweep_warps.json for the table.
_MAX_CHANNEL_TILES = _channel_tiles()

#: Warps per block in each of the three kernels. The pooling kernel stages one plane per warp
#: in shared memory, so its warp count is also what bounds the plane size the fast path
#: accepts; the two GEMM kernels stage one 16x16 fp32 accumulator per warp.
#: All three chosen by the sweep in profile/sweep_warps.py, on both captured shapes, keeping the
#: configuration that wins on *each* rather than on their average -- the harness reports speedup per
#: case and never averages, so a configuration faster at [4,...] and slower at [1,...] has not
#: improved the result. The surface is nearly flat: every combination measured between 1.52x and
#: 1.61x worst-case, so this is the winner rather than a peak. It was much less flat before the task
#: decomposition was changed to vary the channel group fastest, which suggests what the warp count
#: was previously compensating for was activation-fragment traffic.
_CV1_WARPS = _warps("CV1", 4)
_POOL_WARPS = _warps("POOL", 8)
_CV2_WARPS = _warps("CV2", 2)

#: The build is keyed on the warp configuration, so a sweep's variants do not overwrite each
#: other's build directory and the default configuration keeps a stable name.
_EXTENSION_NAME = f"fk_l2_yolov10_sppf_{_CV1_WARPS}_{_POOL_WARPS}_{_CV2_WARPS}"


def _cv2_tile(axis: str, default: int) -> int:
    """One of the second GEMM's block-level tile extents, counted in 16-wide wmma tiles.

    ``M`` counts pixel tiles per block and ``N`` counts channel groups, so a block computes a
    ``16*M`` by ``16*N*_MAX_CHANNEL_TILES`` patch of the output. Overridable so
    ``profile/sweep_cv2_tiles.py`` can measure the two axes independently of ``cv1``; the extension
    reduces an extent that does not divide its axis rather than refusing the shape.
    """
    try:
        value = int(os.environ.get(f"FK_SPPF_CV2_TILE_{axis}", default))
    except (TypeError, ValueError):
        return default
    return value if 1 <= value <= 64 else default


#: ``M = 1`` with ``N = _CV2_WARPS`` is the configuration that predates the parameterisation -- one
#: pixel tile per block, one channel group per warp -- kept as the default so the sweep starts from
#: the shipped behaviour rather than from a guess.
_CV2_TILE_M = _cv2_tile("M", 1)
_CV2_TILE_N = _cv2_tile("N", _CV2_WARPS)

#: Shared memory a block may use without opting past the default carve-out. The extension
#: queries the device's real opt-in limit and declines if even that is not enough.
_MAX_SHARED_BYTES = 48 * 1024

#: The exact classes the kernels reimplement. Every one is read off ``YOLOConv`` itself rather
#: than imported, because the candidate finder resolves a relative ``..L1.silu`` import inside
#: *this* package to the candidate file when one exists -- a different class object from the one
#: ``YOLOConv`` actually builds with, which would reject every eligible module.
#:
#: These are compared with ``type(x) is ...`` and never with ``isinstance``. That distinction is the
#: whole point: a subclass passes ``isinstance`` while overriding ``forward`` to compute something
#: else entirely, and the kernels would keep running their own arithmetic regardless. A SiLU subclass
#: returning ``super().forward(x) + 1`` took the fast path under the previous ``isinstance`` check and
#: was wrong by ``max_abs = 1.487``. Recording a type in the cache key makes the cache *invalidate*
#: when the type changes; it does not make the predicate *refuse* the new type, and those are
#: different things.
_SILU_TYPE = type(YOLOConv.default_act)
_conv_block_module = sys.modules[YOLOConv.__module__]
_CONV_TYPE = _conv_block_module.Conv2d
_BATCHNORM_TYPE = _conv_block_module.BatchNorm2d


def _workspace_root() -> Path:
    """Directory to anchor build products in.

    The harness imports this file in place, so ``__file__`` is the real path inside the
    operator workspace. Keeping build products here rather than in the shared
    ``~/.cache/torch_extensions`` is what stops the concurrently running sibling operator
    workspaces from contending over one build tree.
    """
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "validate.py").is_file():
            return parent
    return here


def _target_arch() -> str:
    """The single architecture to compile for, so a cold build is not multiplied by the
    six-architecture ambient list."""
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}.{minor}"
    except Exception:
        pass
    return "10.0"


def _tensor_fingerprint(value):
    """Everything about a tensor that the fold or the eligibility predicate depends on, or None.

    ``id`` and ``shape`` are here alongside ``data_ptr`` because the caching allocator recycles
    blocks: a tensor that is replaced, freed and reallocated can land on the same address, so an
    address alone does not establish that the binding is unchanged. ``requires_grad`` is here so
    that turning gradients on after the cache is warm invalidates it -- the reference composition
    would build a graph the kernels cannot.
    """
    if not isinstance(value, torch.Tensor):
        return None
    return (id(value), value.data_ptr(), value.dtype, value.device, value.shape,
            value._version, value.requires_grad)


def _pack_for_wmma(weight_2d: torch.Tensor) -> torch.Tensor:
    """Repack ``[out_channels, in_channels]`` so each 16x16 wmma fragment is contiguous.

    The natural layout puts a fragment's 16 contraction values 16 rows apart, so loading one
    touches 16 separate cache lines; profiling the unpacked version showed 22 of every 27 warp
    cycles stalled on exactly that. Grouping into ``[channel_tile][k_tile][16][16]`` makes a
    fragment 512 contiguous bytes, which is one transaction.

    Done once, on the host, when the fold is derived -- never on the call path.
    """
    out_channels, in_channels = weight_2d.shape
    return (weight_2d.view(out_channels // _TILE, _TILE, in_channels // _TILE, _TILE)
            .permute(0, 2, 1, 3)
            .reshape(-1))


def _as_pair(value):
    """Normalise an int / 1- or 2-element sequence to an ``(h, w)`` pair, or None.

    Deliberately strict about ``bool`` and about non-int members: the reference pool
    operator rejects ``ceil_mode=0`` and ``kernel_size=5.0``, so coercing either here would
    let the fast path succeed on a configuration the reference refuses.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)):
        ints = [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
        if len(ints) != len(value):
            return None
        if len(ints) == 1:
            return (ints[0], ints[0])
        if len(ints) == 2:
            return (ints[0], ints[1])
    return None


_CPP_SOURCE = """
#include <torch/extension.h>

// The generated binding translation unit does not see the .cu sources, so the entry points
// have to be declared here for it to compile.
c10::optional<at::Tensor> yolo_sppf(
    const at::Tensor& x, const at::Tensor& w1, const at::Tensor& s1, const at::Tensor& b1,
    const at::Tensor& w2, const at::Tensor& s2, const at::Tensor& b2,
    int64_t mid_channels, int64_t max_channel_tiles, int64_t cv2_tile_m,
    int64_t cv2_tile_n);
int64_t fast_path_calls();
int64_t declined_calls();
void reset_counters();
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <limits>

using namespace nvcuda;

namespace {

constexpr int kLanes = 32;
constexpr int kTile = 16;                        // the wmma m16n16k16 extent, every axis
constexpr int kAccumulatorSlot = kTile * kTile;  // floats a warp stages per output tile

// Warps per block, and output channel tiles per warp. The channel tiles are the reason these
// kernels are not latency-bound: a warp holds that many independent accumulators, issues that
// many independent weight-fragment loads per contraction step, and reuses one activation
// fragment across all of them. Occupancy alone cannot fix exposed latency here, because the
// whole operator is only ~1600 output tiles -- there is not enough of it to fill the machine
// with warps, so the latency has to be hidden inside each warp instead.
constexpr int kCv1Warps = __CV1_WARPS__;
constexpr int kPoolWarps = __POOL_WARPS__;
constexpr int kCv2Warps = __CV2_WARPS__;

// The three radii the pool cascade produces. ``self.m`` is a same-shape 5x5 pool, so one
// application is radius 2, two is radius 4 and three is radius 6 -- and because max-plus
// dilation composes by Minkowski sum, they are snapshots of one growing window rather than
// three independent passes.
constexpr int kRadiusOnce = 2;
constexpr int kRadiusTwice = 4;
constexpr int kRadiusThrice = 6;

__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + __expf(-v));
}

// One stage of a horizontal max-dilation: widens the radius already held in `v` by `step`
// columns on each side.
//
// Lane `l` owns column `min(l, W - 1)`, so lanes at or past W shadow the last column. That
// keeps them reading a valid address and, more importantly, keeps them participating in every
// shuffle so the full-warp mask stays honest. Taps outside [0, W) clamp to the edge column.
// Clamping is sound rather than merely convenient: a replicate-padded dilation is exactly
// "max over window intersect plane", and the clamp target -- column 0 or W-1 -- always lies
// between the centre and the out-of-range index, so it can only re-read a value the window
// already covers, and `max` is idempotent. Because lane `j` holds column `j` for every j < W,
// the clamped column index doubles as the source lane index.
__device__ __forceinline__ __half dilate_columns(__half v, int col, int W, int step) {
  const int right = (col + step < W) ? (col + step) : (W - 1);
  const int left = (col - step > 0) ? (col - step) : 0;
  const __half hi = __shfl_sync(0xffffffffu, v, right);
  const __half lo = __shfl_sync(0xffffffffu, v, left);
  return __hmax_nan(__hmax_nan(v, hi), lo);
}

// Grow a value from radius 0 to RADIUS along its row, by doubling.
//
// The step sequence is 1, 1, 2, 2, reaching radii 1, 2, 4, 6 -- exactly the three the cascade
// needs, in four stages and eight shuffles rather than six stages and twelve. Each stage's
// three source windows overlap with no gap: at radius 2 stepping by 2 they cover [i-4, i],
// [i-2, i+2] and [i, i+4], whose union is [i-4, i+4], so the radius reached is exactly 4 and
// not merely at least 4. In general the three windows are [i-s-r, i-s+r], [i-r, i+r] and
// [i+s-r, i+s+r], which tile [i-s-r, i+s+r] without a gap exactly when s <= 2r + 1; every
// stage here is well inside that.
template <int RADIUS>
__device__ __forceinline__ __half dilate_row_to(__half v, int col, int W) {
  static_assert(RADIUS == 2 || RADIUS == 4 || RADIUS == 6,
                "the doubling schedule covers exactly the cascade's three radii");
  v = dilate_columns(v, col, W, 1);
  v = dilate_columns(v, col, W, 1);
  if (RADIUS >= 4) v = dilate_columns(v, col, W, 2);
  if (RADIUS >= 6) v = dilate_columns(v, col, W, 2);
  return v;
}

// ---------------------------------------------------------------------------
// The intermediate's layout.
//
// The concatenated tensor between the two convolutions is private to this extension, so its
// layout is a free choice -- and the right choice is not NCHW. A 16x16 wmma fragment of an
// NCHW plane is 16 pixels from each of 16 channels, which with a channel stride of H*W lands
// in 16 separate cache lines; the profile of the NCHW version showed exactly that, as 22 of
// every 27 warp cycles stalled on a long scoreboard. Blocking the pixel axis by the tile
// extent instead -- [image][pixel_tile][channel][16] -- makes that same fragment 512
// contiguous bytes, one coalesced load.
//
// The weights get the same treatment, but at fold time on the host rather than here: packed
// as [channel_tile][k_tile][16 channels][16 k], each fragment is again 512 contiguous bytes.
// Both operands of both GEMMs are then read in one transaction per fragment.
//
// The cost is that a channel's plane is no longer contiguous -- it is P/16 chunks of 32 bytes
// -- which the pooling kernel pays when it stages a plane into shared. That is 25 segments per
// plane against 7, on 512 planes: irrelevant next to what the GEMMs save.
// ---------------------------------------------------------------------------
__device__ __forceinline__ long long blocked_offset(
    long long image, int pixel_tiles, int channels, int pixel_tile, int channel) {
  return ((image * pixel_tiles + pixel_tile) * (long long)channels + channel) * kTile;
}

// cv1 + folded BatchNorm + SiLU, into the intermediate's first channel block.
//
// A 1x1 convolution over a plane is a GEMM with the channel axis contracting and the pixels
// free. The activation is still NCHW here -- it comes from the caller, so its layout is not
// ours to choose -- which is why each warp takes NCT output channel tiles: the activation
// fragment is loaded once per contraction step and reused NCT times, and the NCT weight loads
// and mma operations are mutually independent.
template <int WARPS, int NCT>
__global__ __launch_bounds__(kLanes* WARPS) void cv1_gemm_silu(
    const __half* __restrict__ x, const __half* __restrict__ packed_weight,
    const float* __restrict__ scale, const float* __restrict__ shift,
    __half* __restrict__ z, int P, int in_channels, int pixel_tiles, int channel_groups,
    int z_channels, int tasks) {
  extern __shared__ float staging[];
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;
  const int task = static_cast<int>(blockIdx.x) * WARPS + warp;
  // Warp-uniform: blockDim.x is a multiple of the warp size, so a whole warp leaves together
  // and every surviving warp has all 32 lanes for the epilogue.
  if (task >= tasks) return;

  const int k_tiles = in_channels / kTile;
  // The channel group varies fastest, so consecutive tasks -- and therefore the warps of one
  // block -- share a pixel tile and with it the activation fragments. That operand is the one
  // that cannot be relaid out, so sharing its cache lines across the block is the whole point.
  const int channel_group = task % channel_groups;
  const int rest = task / channel_groups;
  const int pixel_tile = rest % pixel_tiles;
  const int image = rest / pixel_tiles;

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc[NCT];
#pragma unroll
  for (int j = 0; j < NCT; ++j) wmma::fill_fragment(acc[j], 0.0f);

  const __half* __restrict__ plane =
      x + static_cast<long long>(image) * in_channels * P + pixel_tile * kTile;
#pragma unroll 4
  for (int kt = 0; kt < k_tiles; ++kt) {
    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::col_major> a;
    wmma::load_matrix_sync(a, plane + static_cast<long long>(kt) * kTile * P, P);
#pragma unroll
    for (int j = 0; j < NCT; ++j) {
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major> b;
      wmma::load_matrix_sync(
          b, packed_weight
                 + (static_cast<long long>(channel_group * NCT + j) * k_tiles + kt)
                       * kAccumulatorSlot,
          kTile);
      wmma::mma_sync(acc[j], a, b, acc[j]);
    }
  }

  // The accumulator has to go through shared memory: store_matrix_sync needs an address the
  // whole warp agrees on, and the accumulator fragment's element-to-(m, n) mapping is opaque,
  // so a per-output-channel scale and bias cannot be applied in registers.
  float* __restrict__ slot = staging + warp * kAccumulatorSlot;
#pragma unroll
  for (int j = 0; j < NCT; ++j) {
    wmma::store_matrix_sync(slot, acc[j], kTile, wmma::mem_col_major);  // slot[n * 16 + m]
    __syncwarp();
    const int channel_tile = channel_group * NCT + j;
    __half* __restrict__ dst =
        z + blocked_offset(image, pixel_tiles, z_channels, pixel_tile, channel_tile * kTile);
    for (int i = lane; i < kAccumulatorSlot; i += kLanes) {
      const int channel = channel_tile * kTile + (i >> 4);
      // Lanes 0-15 write one channel's 16 pixels and lanes 16-31 the next channel's, which in
      // this layout are adjacent: 64 contiguous bytes per instruction.
      dst[i] = __float2half(silu(slot[i] * scale[channel] + shift[channel]));
    }
    __syncwarp();  // the staging slot is reused by the next channel tile
  }
}

// Widen a running vertical max from radius FROM-1 to radius TO, by adding one row above and one
// row below per step.
//
// Out-of-range rows clamp onto the edge row. Each tap here is a single row rather than an already
// aggregated value, so the clamping argument is the simple one: when `row - d` falls below 0 the
// clamp target 0 lies between `row` and `row - d`, so it is a row the target window already
// contains, and `max` is idempotent.
template <int FROM, int TO>
__device__ __forceinline__ __half grow_rows(
    __half running, const __half* __restrict__ plane, int row, int H, int W, int col) {
#pragma unroll
  for (int d = FROM; d <= TO; ++d) {
    const int above = (row - d > 0) ? (row - d) : 0;
    const int below = (row + d < H - 1) ? (row + d) : (H - 1);
    running = __hmax_nan(running, plane[above * W + col]);
    running = __hmax_nan(running, plane[below * W + col]);
  }
  return running;
}

// All three pooled outputs for one row of one plane.
//
// A (2R+1)x(2R+1) square structuring element factors into a vertical and a horizontal dilation of
// the *same* radius, and the two commute, so this does the vertical pass straight out of the staged
// plane and then grows the row in registers. Pairing a vertical radius with a different horizontal
// radius would compute a rectangle, which is not any of the three pools.
//
// The vertical pass is grown *once*: radii 2, 4 and 6 are snapshots of one monotone accumulation,
// so it costs 12 row taps rather than the 27 that three independent maxes over 5, 9 and 13 rows
// would. That is the same observation that collapses the cascade in the first place, applied one
// level down. The horizontal dilations cannot be shared the same way -- each radius needs its own,
// because the radii have to match per axis.
__device__ __forceinline__ void pool_row(
    const __half* __restrict__ plane, __half* __restrict__ once, __half* __restrict__ twice,
    __half* __restrict__ thrice, int row, int H, int W, int z_channels, int lane) {
  const int col = (lane < W) ? lane : (W - 1);

  __half running = plane[row * W + col];
  running = grow_rows<1, kRadiusOnce>(running, plane, row, H, W, col);
  const __half vertical_once = running;
  running = grow_rows<kRadiusOnce + 1, kRadiusTwice>(running, plane, row, H, W, col);
  const __half vertical_twice = running;
  running = grow_rows<kRadiusTwice + 1, kRadiusThrice>(running, plane, row, H, W, col);

  // Every lane runs all three dilations: they contain full-warp shuffles, and the store below is
  // the only predicated part.
  const __half pooled_once = dilate_row_to<kRadiusOnce>(vertical_once, col, W);
  const __half pooled_twice = dilate_row_to<kRadiusTwice>(vertical_twice, col, W);
  const __half pooled_thrice = dilate_row_to<kRadiusThrice>(running, col, W);

  if (lane < W) {
    const int pixel = row * W + col;
    const long long offset =
        static_cast<long long>(pixel >> 4) * z_channels * kTile + (pixel & (kTile - 1));
    once[offset] = pooled_once;
    twice[offset] = pooled_twice;
    thrice[offset] = pooled_thrice;
  }
}

// All three pooled channel blocks: one *block* per (image, channel) plane, its warps splitting
// the plane's rows.
//
// This is the kernel the fused design could not afford. Pooling needs a whole plane, so a plane
// cannot be split across blocks -- which, when the pooling lived inside the cv1 kernel, capped
// the grid at N * (mid channels / group) blocks: 16 of 148 SMs at N=4 and 4 at N=1, measured at
// 137 us with 66.9% of cycles having no eligible warp. On its own the same work is
// N * mid_channels independent planes, which is 512 blocks at N=4 and 128 at N=1, so every SM
// gets one. A block rather than a warp per plane because the block count is what decides how
// many SMs run at all, while the warps within it then split the rows.
template <int WARPS>
__global__ __launch_bounds__(kLanes* WARPS) void pool_concat(
    __half* __restrict__ z, int P, int H, int W, int mid_channels, int z_channels,
    int pixel_tiles) {
  extern __shared__ __half plane[];
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;

  const int channel = static_cast<int>(blockIdx.x) % mid_channels;
  const int image = static_cast<int>(blockIdx.x) / mid_channels;

  const long long base = blocked_offset(image, pixel_tiles, z_channels, 0, channel);
  const __half* __restrict__ src = z + base;
  for (int idx = static_cast<int>(threadIdx.x); idx < P; idx += kLanes * WARPS) {
    plane[idx] = src[static_cast<long long>(idx >> 4) * z_channels * kTile
                     + (idx & (kTile - 1))];
  }
  // Every warp reads rows other warps staged, so this has to be a block-wide barrier.
  __syncthreads();

  const long long block_stride = static_cast<long long>(mid_channels) * kTile;
  __half* __restrict__ once = z + base + block_stride;
  __half* __restrict__ twice = z + base + 2 * block_stride;
  __half* __restrict__ thrice = z + base + 3 * block_stride;
  // Warp-uniform bound, so every lane runs every iteration and the shuffles inside always see a
  // full warp.
  for (int row = warp; row < H; row += WARPS) {
    pool_row(plane, once, twice, thrice, row, H, W, z_channels, lane);
  }
}

// cv2 + folded BatchNorm + SiLU, from the blocked intermediate into a plain NCHW output.
//
// Both operands are blocked here, so every fragment is one contiguous 512-byte load. The
// output has to be NCHW because that is what the caller gets back.
// `m_tiles` pixel tiles by `n_tiles` channel groups per block -- the block-level M and N tile
// extents, in units of the 16-wide wmma tile, so a block computes a
// (16 * m_tiles) x (16 * NCT * n_tiles) patch of the output. Runtime arguments rather than template
// parameters, because each warp still owns one accumulator over the full contraction, so nothing
// here needs the extents at compile time and templating them would multiply the build for nothing.
// `profile/sweep_cv2_tiles.py` measures both axes independently of `cv1`.
template <int WARPS, int NCT>
__global__ __launch_bounds__(kLanes* WARPS) void cv2_gemm_silu(
    const __half* __restrict__ z, const __half* __restrict__ packed_weight,
    const float* __restrict__ scale, const float* __restrict__ shift,
    __half* __restrict__ out, int P, int z_channels, int out_channels, int pixel_tiles,
    int channel_groups, int m_tiles, int n_tiles) {
  extern __shared__ float staging[];
  const int warp = static_cast<int>(threadIdx.x) / kLanes;
  const int lane = static_cast<int>(threadIdx.x) % kLanes;

  const int channel_blocks = channel_groups / n_tiles;
  const int pixel_blocks = pixel_tiles / m_tiles;
  const int block = static_cast<int>(blockIdx.x);
  const int channel_block = block % channel_blocks;
  const int rest = block / channel_blocks;
  const int pixel_block = rest % pixel_blocks;
  const int image = rest / pixel_blocks;

  const int k_tiles = z_channels / kTile;
  const int tasks = m_tiles * n_tiles;
  float* __restrict__ slot = staging + warp * kAccumulatorSlot;
  const long long out_base = static_cast<long long>(image) * out_channels * P;

  // Warp-uniform bound. Within a block the channel group varies fastest, so the warps running
  // concurrently share an activation tile -- the same reason the grid-level decomposition is
  // channel-major.
  for (int task = warp; task < tasks; task += WARPS) {
    const int channel_group = channel_block * n_tiles + (task % n_tiles);
    const int pixel_tile = pixel_block * m_tiles + (task / n_tiles);

    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> acc[NCT];
#pragma unroll
    for (int j = 0; j < NCT; ++j) wmma::fill_fragment(acc[j], 0.0f);

    const __half* __restrict__ tile =
        z + blocked_offset(image, pixel_tiles, z_channels, pixel_tile, 0);
#pragma unroll 4
    for (int kt = 0; kt < k_tiles; ++kt) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, __half, wmma::col_major> a;
      wmma::load_matrix_sync(a, tile + static_cast<long long>(kt) * kAccumulatorSlot, kTile);
#pragma unroll
      for (int j = 0; j < NCT; ++j) {
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, __half, wmma::col_major> b;
        wmma::load_matrix_sync(
            b, packed_weight
                   + (static_cast<long long>(channel_group * NCT + j) * k_tiles + kt)
                         * kAccumulatorSlot,
            kTile);
        wmma::mma_sync(acc[j], a, b, acc[j]);
      }
    }

#pragma unroll
    for (int j = 0; j < NCT; ++j) {
      wmma::store_matrix_sync(slot, acc[j], kTile, wmma::mem_col_major);
      __syncwarp();
      const int channel_tile = channel_group * NCT + j;
      for (int i = lane; i < kAccumulatorSlot; i += kLanes) {
        const int channel = channel_tile * kTile + (i >> 4);
        out[out_base + static_cast<long long>(channel) * P + pixel_tile * kTile
            + (i & (kTile - 1))] =
            __float2half(silu(slot[i] * scale[channel] + shift[channel]));
      }
      __syncwarp();
    }
  }
}

// Call counters, so the tests can assert that an eligible shape really took the kernels and an
// ineligible one really did not. Host-side, so they cost nothing on the device.
long long g_fast_path_calls = 0;
long long g_declined_calls = 0;

// The device's opt-in shared-memory ceiling, queried once. It is a property of the device, and
// this operator is launch-bound enough that a driver call on every forward is not free.
int max_shared_memory(int device) {
  static int cached[16] = {0};
  if (device < 0 || device >= 16) {
    int value = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return value;
  }
  if (cached[device] == 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &cached[device], cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  }
  return cached[device];
}

// wmma::load_matrix_sync requires a 256-bit (32-byte) aligned address for __half operands, and
// an ldm that is a multiple of 8 elements. Every offset the kernels add to a base pointer is a
// multiple of 16 halves -- 32 bytes -- so checking the base is enough to establish the whole
// family of fragment addresses.
//
// This is not implied by contiguity, which is the trap. ``torch.empty(n + 1).narrow(0, 1, n)`` is
// contiguous, passes every shape and stride check, and starts two bytes into its allocation. The
// bench's own timing slots advance in 256-byte steps from an allocator base and so are always
// aligned, which means this hole would never have shown up in the benchmark -- only in a caller
// who handed the fast path a narrowed view.
bool is_wmma_aligned(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 32 == 0;
}

bool is_plain_fp32_vector(const at::Tensor& t, int64_t n) {
  return t.is_cuda() && t.scalar_type() == at::kFloat && t.dim() == 1 && t.numel() == n
         && t.is_contiguous() && !t.is_neg() && !t.is_conj();
}

bool is_packed_fp16(const at::Tensor& t, int64_t elements) {
  return t.is_cuda() && t.scalar_type() == at::kHalf && t.numel() == elements
         && t.is_contiguous() && !t.is_neg() && !t.is_conj();
}

}  // namespace

// ---------------------------------------------------------------------------
// The single entry point. Declines -- returns nothing, which pybind hands back as None --
// rather than raising when a precondition does not hold, so the Python side can run the
// reference composition instead of the caller getting an error for an input the reference
// handles perfectly well.
// ---------------------------------------------------------------------------
c10::optional<at::Tensor> yolo_sppf(
    const at::Tensor& x, const at::Tensor& w1, const at::Tensor& s1, const at::Tensor& b1,
    const at::Tensor& w2, const at::Tensor& s2, const at::Tensor& b2,
    int64_t mid_channels, int64_t max_channel_tiles, int64_t cv2_tile_m,
    int64_t cv2_tile_n) {
  // Cheap re-checks of what the kernels assume. Most of these the Python predicate has already
  // established; they are repeated because this is the function that actually dereferences the
  // pointers, and because two of them -- a forward-mode dual's tangent and a lazily negated or
  // conjugated view -- are visible here essentially for free and awkward to ask for from
  // Python on every call.
  //
  // is_neg() and is_conj() matter because the kernels read raw storage through a typed pointer:
  // torch._neg_view(t) is contiguous fp16 CUDA 4-D and passes every other check, but its
  // logical values are the negation of what is in memory.
  //
  // The two gradient guards are separate mechanisms. Reverse mode is caught by requires_grad
  // under an enabled GradMode. Forward mode is not: a dual from
  // torch.autograd.forward_ad.make_dual has requires_grad false, and reading its primal through
  // a pointer would silently drop the tangent.
  if (!(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 4 && x.is_contiguous()
        && x.numel() > 0 && !x.is_neg() && !x.is_conj())) {
    ++g_declined_calls;
    return c10::nullopt;
  }
  if (at::GradMode::is_enabled()
      && (x.requires_grad() || w1.requires_grad() || w2.requires_grad())) {
    ++g_declined_calls;
    return c10::nullopt;
  }
  if (x._fw_grad(/*level=*/0).defined()) {
    ++g_declined_calls;
    return c10::nullopt;
  }

  const int64_t N = x.size(0), in_channels = x.size(1), H = x.size(2), W = x.size(3);
  const int64_t pixels = H * W;
  const int64_t z_channels = 4 * mid_channels;
  if (!(mid_channels > 0 && s2.dim() == 1)) {
    ++g_declined_calls;
    return c10::nullopt;
  }
  const int64_t out_channels = s2.numel();

  const bool shapes_ok =
      is_packed_fp16(w1, mid_channels * in_channels)
      && is_packed_fp16(w2, out_channels * z_channels)
      && is_plain_fp32_vector(s1, mid_channels) && is_plain_fp32_vector(b1, mid_channels)
      && is_plain_fp32_vector(s2, out_channels) && is_plain_fp32_vector(b2, out_channels)
      && x.device() == w1.device() && x.device() == w2.device()
      && x.device() == s1.device() && x.device() == s2.device()
      && x.device() == b1.device() && x.device() == b2.device()
      && is_wmma_aligned(x) && is_wmma_aligned(w1) && is_wmma_aligned(w2);
  if (!shapes_ok) {
    ++g_declined_calls;
    return c10::nullopt;
  }

  // No masked tail path anywhere, so every tile extent has to divide its axis; the pooling maps
  // one lane per column, which bounds the row length; and a staged plane has to fit in shared.
  const int64_t mid_tiles = mid_channels / kTile;
  const int64_t out_tiles = out_channels / kTile;
  const bool tiling_ok = W <= kLanes && pixels % kTile == 0 && in_channels % kTile == 0
                         && mid_channels % kTile == 0 && out_channels % kTile == 0;
  if (!tiling_ok) {
    ++g_declined_calls;
    return c10::nullopt;
  }

  const c10::cuda::CUDAGuard guard(x.device());
  const int max_shared = max_shared_memory(x.device().index());
  const int64_t pool_shared = pixels * static_cast<int64_t>(sizeof(__half));
  const int64_t cv1_shared =
      static_cast<int64_t>(kCv1Warps) * kAccumulatorSlot * sizeof(float);
  const int64_t cv2_shared =
      static_cast<int64_t>(kCv2Warps) * kAccumulatorSlot * sizeof(float);
  if (pool_shared > max_shared) {
    ++g_declined_calls;
    return c10::nullopt;
  }

  // A warp takes as many output channel tiles as the axis divides evenly into, up to the
  // requested ceiling: that many independent weight loads and accumulators per contraction
  // step, all fed by one activation fragment.
  const auto channel_tiles_for = [max_channel_tiles](int64_t tiles) -> int64_t {
    for (int64_t candidate : {int64_t{4}, int64_t{2}}) {
      if (candidate <= max_channel_tiles && tiles % candidate == 0) return candidate;
    }
    return 1;
  };
  const int64_t cv1_group = channel_tiles_for(mid_tiles);
  const int64_t cv2_group = channel_tiles_for(out_tiles);
  const int64_t pixel_tiles = pixels / kTile;
  const int64_t cv1_tasks = N * pixel_tiles * (mid_tiles / cv1_group);
  const int64_t cv2_channel_groups = out_tiles / cv2_group;
  // The requested block-level tile extents, reduced to the largest that actually divide their axis,
  // so an extent that does not fit this shape degrades to a smaller block rather than to a refusal
  // or, worse, a grid that does not cover the output.
  int64_t cv2_m = cv2_tile_m > 0 ? cv2_tile_m : 1;
  int64_t cv2_n = cv2_tile_n > 0 ? cv2_tile_n : 1;
  if (cv2_m > pixel_tiles) cv2_m = pixel_tiles;
  if (cv2_n > cv2_channel_groups) cv2_n = cv2_channel_groups;
  while (cv2_m > 1 && pixel_tiles % cv2_m != 0) --cv2_m;
  while (cv2_n > 1 && cv2_channel_groups % cv2_n != 0) --cv2_n;
  const int64_t cv2_tasks = N * pixel_tiles * cv2_channel_groups;
  const int64_t cv2_blocks_exact = N * (pixel_tiles / cv2_m) * (cv2_channel_groups / cv2_n);
  const int64_t pool_tasks = N * mid_channels;
  const int64_t z_elements = N * pixels * z_channels;
  const int64_t int_max = std::numeric_limits<int>::max();
  if (pixels > int_max || z_channels > int_max || cv1_tasks > int_max
      || cv2_tasks > int_max || pool_tasks > int_max || pixel_tiles > int_max
      || z_elements > int_max) {
    ++g_declined_calls;
    return c10::nullopt;
  }

  // Fresh allocations with canonical contiguous strides. empty_like would carry over the
  // input's stride metadata, which for a contiguous-degenerate input -- a singleton dimension
  // whose stride is arbitrary -- differs from what the reference returns.
  at::Tensor z = at::empty({z_elements}, x.options());
  at::Tensor out = at::empty({N, out_channels, H, W}, x.options());

  const auto stream = at::cuda::getCurrentCUDAStream();
  const __half* x_ptr = reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>());
  const __half* w1_ptr = reinterpret_cast<const __half*>(w1.const_data_ptr<at::Half>());
  const __half* w2_ptr = reinterpret_cast<const __half*>(w2.const_data_ptr<at::Half>());
  __half* z_ptr = reinterpret_cast<__half*>(z.mutable_data_ptr<at::Half>());
  __half* out_ptr = reinterpret_cast<__half*>(out.mutable_data_ptr<at::Half>());

  const int cv1_blocks = static_cast<int>((cv1_tasks + kCv1Warps - 1) / kCv1Warps);
  const int cv2_blocks = static_cast<int>(cv2_blocks_exact);
  const int pool_blocks = static_cast<int>(pool_tasks);  // one per (image, channel) plane

#define LAUNCH_CV1(NCT)                                                                     \
  cv1_gemm_silu<kCv1Warps, NCT><<<cv1_blocks, kLanes * kCv1Warps,                           \
                                  static_cast<int>(cv1_shared), stream>>>(                  \
      x_ptr, w1_ptr, s1.const_data_ptr<float>(), b1.const_data_ptr<float>(), z_ptr,         \
      static_cast<int>(pixels), static_cast<int>(in_channels),                              \
      static_cast<int>(pixel_tiles), static_cast<int>(mid_tiles / cv1_group),               \
      static_cast<int>(z_channels), static_cast<int>(cv1_tasks))
  if (cv1_group == 4) { LAUNCH_CV1(4); }
  else if (cv1_group == 2) { LAUNCH_CV1(2); }
  else { LAUNCH_CV1(1); }
#undef LAUNCH_CV1
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pool_concat<kPoolWarps><<<pool_blocks, kLanes * kPoolWarps,
                            static_cast<int>(pool_shared), stream>>>(
      z_ptr, static_cast<int>(pixels), static_cast<int>(H), static_cast<int>(W),
      static_cast<int>(mid_channels), static_cast<int>(z_channels),
      static_cast<int>(pixel_tiles));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_CV2(NCT)                                                                     \
  cv2_gemm_silu<kCv2Warps, NCT><<<cv2_blocks, kLanes * kCv2Warps,                           \
                                  static_cast<int>(cv2_shared), stream>>>(                  \
      z_ptr, w2_ptr, s2.const_data_ptr<float>(), b2.const_data_ptr<float>(), out_ptr,       \
      static_cast<int>(pixels), static_cast<int>(z_channels),                               \
      static_cast<int>(out_channels), static_cast<int>(pixel_tiles),                        \
      static_cast<int>(cv2_channel_groups), static_cast<int>(cv2_m), static_cast<int>(cv2_n))
  if (cv2_group == 4) { LAUNCH_CV2(4); }
  else if (cv2_group == 2) { LAUNCH_CV2(2); }
  else { LAUNCH_CV2(1); }
#undef LAUNCH_CV2
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ++g_fast_path_calls;
  return out;
}

int64_t fast_path_calls() { return g_fast_path_calls; }
int64_t declined_calls() { return g_declined_calls; }
void reset_counters() { g_fast_path_calls = 0; g_declined_calls = 0; }
"""


def _build_extension():
    from torch.utils.cpp_extension import load_inline

    build_dir = _workspace_root() / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = _target_arch()
    try:
        return load_inline(
            name=_EXTENSION_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=(_CUDA_SOURCE
                          .replace("__CV1_WARPS__", str(_CV1_WARPS))
                          .replace("__POOL_WARPS__", str(_POOL_WARPS))
                          .replace("__CV2_WARPS__", str(_CV2_WARPS))),
            functions=["yolo_sppf", "fast_path_calls", "declined_calls",
                       "reset_counters"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


#: Set when the extension compiled and loaded; when False the module still works, it just
#: runs the reference composition for every input.
FAST_PATH_AVAILABLE = False
#: Populated with the build failure when ``FAST_PATH_AVAILABLE`` is False.
BUILD_ERROR: str | None = None

_extension = None
if os.environ.get("FK_SPPF_FORCE_BUILD_FAILURE") == "1":
    # A deliberate failure, so the fallback-coverage test can prove that a build failure
    # leaves the module importable rather than having to break the toolchain to find out.
    BUILD_ERROR = "RuntimeError: build disabled by FK_SPPF_FORCE_BUILD_FAILURE"
else:
    try:
        _extension = _build_extension()
        FAST_PATH_AVAILABLE = True
    except Exception as exc:  # a build failure must not make this module unimportable
        BUILD_ERROR = f"{type(exc).__name__}: {exc}"
if not FAST_PATH_AVAILABLE:
    print(
        f"[{_EXTENSION_NAME}] CUDA extension unavailable, running the reference "
        f"composition for every input: {BUILD_ERROR}",
        file=sys.stderr,
        flush=True,
    )


#: The reason recorded by the most recent *refusal* of the fast path. Written only when the fast
#: path is refused -- never cleared on success, so it can be stale after a call that did take the
#: kernels. That keeps the hot path free of bookkeeping. Diagnostic only: the module's behaviour
#: never depends on it.
LAST_REFUSAL: str | None = None


def _refuse(reason: str) -> None:
    global LAST_REFUSAL
    LAST_REFUSAL = reason


class _Folded:
    """The kernels' launch arguments, derived once from a module's parameters.

    ``scale``/``shift`` are fp32 even though ``bn.weight``/``bn.bias`` are fp16 -- the
    harness casts parameters to the run dtype but leaves buffers alone, so
    ``running_mean``/``running_var`` arrive as fp32 anyway, and promoting the rest is what
    turns the baseline's three roundings into one.
    """

    __slots__ = ("w1", "s1", "b1", "w2", "s2", "b2", "in_channels", "mid_channels",
                 "out_channels", "params_need_grad")

    def __init__(self, first, second, in_channels, mid_channels, out_channels):
        self.w1, self.s1, self.b1, first_needs_grad = first
        self.w2, self.s2, self.b2, second_needs_grad = second
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        # Whether anything this fold was derived from tracks gradients. Live rather than stale:
        # the signature that guards this object includes every source's ``requires_grad``, so a
        # change to any of them invalidates the cache and this value is recomputed. Deliberately
        # narrower than ``any(self.parameters())`` -- these are exactly the tensors both the
        # kernels and the reference composition read, so a parameter elsewhere in a subclass's
        # tree cannot affect whether *this* output needs a graph.
        self.params_need_grad = first_needs_grad or second_needs_grad


def _fold_block(block, in_channels: int, out_channels: int, what: str):
    """Fold one ``YOLOConv`` into ``(weight_2d, scale, shift, sources)`` such that the block
    computes ``SiLU(weight_2d @ plane * scale + shift)``, or return None if it is not that.

    ``sources`` are the tensors the constants were derived from, which the caller watches
    for change.
    """
    if type(block) is not YOLOConv:
        _refuse(f"{what} is {type(block).__name__}, not exactly YOLOConv -- a subclass may "
                f"override forward")
        return None
    conv = getattr(block, "conv", None)
    act = getattr(block, "act", None)
    if type(conv) is not _CONV_TYPE:
        _refuse(f"{what}.conv is {type(conv).__name__}, not exactly "
                f"{_CONV_TYPE.__name__}")
        return None
    if type(act) is not _SILU_TYPE:
        # Exact, not isinstance: a subclass would satisfy isinstance while its forward computed
        # something the epilogue's silu() does not.
        _refuse(f"{what}.act is {type(act).__name__}, not exactly {_SILU_TYPE.__name__}")
        return None

    w = getattr(conv, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 4:
        _refuse(f"{what}.conv has no 4-D weight")
        return None
    if tuple(w.shape) != (out_channels, in_channels, 1, 1):
        _refuse(f"{what}.conv weight {tuple(w.shape)} is not "
                f"{(out_channels, in_channels, 1, 1)}")
        return None
    if (_as_pair(conv.stride) != (1, 1) or _as_pair(conv.padding) != (0, 0)
            or _as_pair(conv.dilation) != (1, 1) or conv.groups != 1):
        _refuse(f"{what}.conv is not a dense unit-stride unpadded convolution")
        return None
    if not (w.is_cuda and w.dtype == torch.float16 and w.is_contiguous()):
        _refuse(f"{what}.conv weight is not a contiguous fp16 CUDA tensor")
        return None
    if out_channels % _TILE != 0 or in_channels % _TILE != 0:
        _refuse(f"{what} channel counts {in_channels}->{out_channels} are not both "
                f"multiples of {_TILE}")
        return None

    conv_bias = getattr(conv, "bias", None)
    weight_2d = _pack_for_wmma(w.view(out_channels, in_channels))

    if getattr(block, "_is_fused", False):
        # fuse() has already multiplied BN into conv.weight and moved its shift into
        # conv.bias, then deleted bn. The epilogue degenerates to a unit scale plus that
        # bias, so a fused module keeps working rather than applying BN a second time.
        scale = torch.ones(out_channels, dtype=torch.float32, device=w.device)
        if isinstance(conv_bias, torch.Tensor):
            if conv_bias.numel() != out_channels:
                _refuse(f"{what}.conv bias does not match {out_channels} channels")
                return None
            return (weight_2d, scale, conv_bias.float().contiguous(),
                    w.requires_grad or conv_bias.requires_grad)
        shift = torch.zeros(out_channels, dtype=torch.float32, device=w.device)
        return weight_2d, scale, shift, w.requires_grad

    bn = getattr(block, "bn", None)
    if bn is None:
        _refuse(f"{what} has neither a bn nor a fused flag")
        return None
    if type(bn) is not _BATCHNORM_TYPE:
        _refuse(f"{what}.bn is {type(bn).__name__}, not exactly {_BATCHNORM_TYPE.__name__}")
        return None
    if bn.training or not getattr(bn, "track_running_stats", False):
        # In training mode, or without running stats, BN normalises by *batch* statistics,
        # which are not a per-channel affine map of the accumulator and so do not fold.
        _refuse(f"{what}.bn is not in eval mode with running statistics")
        return None
    gamma, beta = getattr(bn, "weight", None), getattr(bn, "bias", None)
    mean, var = getattr(bn, "running_mean", None), getattr(bn, "running_var", None)
    if not all(isinstance(t, torch.Tensor) for t in (gamma, beta, mean, var)):
        _refuse(f"{what}.bn is not affine with running statistics")
        return None
    if any(t.numel() != out_channels for t in (gamma, beta, mean, var)):
        _refuse(f"{what}.bn statistics do not match {out_channels} channels")
        return None

    inv = torch.rsqrt(var.float() + float(bn.eps))
    g = gamma.float()
    scale = g * inv
    shift = beta.float() - g * mean.float() * inv
    if isinstance(conv_bias, torch.Tensor):
        # YOLOConv builds its convolution with bias=False, so this is unreachable there; it
        # keeps the fold correct for a block that does carry one.
        if conv_bias.numel() != out_channels:
            _refuse(f"{what}.conv bias does not match {out_channels} channels")
            return None
        shift = shift + scale * conv_bias.float()
        tracked = (w, conv_bias, gamma, beta, mean, var)
    else:
        tracked = (w, gamma, beta, mean, var)
    return (weight_2d, scale.contiguous(), shift.contiguous(),
            any(t.requires_grad for t in tracked))


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # Folded constants are derived on first forward, never here: the harness re-dtypes
        # every parameter to fp16, re-randomises any it judges uninitialised, and copies the
        # baseline's weights in, all *after* construction. Constants built here would
        # describe weights that no longer exist. They are held as plain attributes rather
        # than parameters or buffers so that the harness's own preparation passes -- which
        # downcast parameters and can overwrite ones whose magnitude looks uninitialised --
        # never touch them, and so they stay out of ``state_dict``.
        self._folded: _Folded | None = None
        self._folded_signature: tuple | None = None

    # -- the reference composition, which is also the fallback ---------------------------
    def _reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1 = self.m(y)
        y2 = self.m(y1)
        return self.cv2(torch.cat((y, y1, y2, self.m(y2)), 1))

    # -- folded constants, and the key that keeps them honest ----------------------------
    def _live_signature(self) -> tuple:
        """Everything the fold and the eligibility decision depend on, read from the module *as
        it is now*.

        This is deliberately not a fingerprint of the tensors the fold was derived from. That was
        the earlier design and it was wrong in a way that only shows up after the cache is warm:
        rebinding ``cv1.conv.weight`` to a different parameter leaves the *old* tensor untouched,
        so a key built from the old objects still matches and the kernels keep running weights the
        module no longer has. Reviewing reproduced exactly that, and two siblings of it --
        replacing ``cv1.act`` with ``Identity`` and enabling gradients after the first call both
        stayed on the fast path and returned answers that no longer matched the reference.

        So the walk starts from ``self`` every time and reads the bindings it finds. Three kinds of
        thing go in, and each is here because leaving it out is a wrong answer rather than a slow
        one:

        * **Which objects are bound**, by type and by ``_is_fused`` flag. A different activation
          class is a different operator; a deleted ``bn`` is a different fold.
        * **Every semantics-bearing scalar**: the pool's kernel/stride/padding/ceil mode, each
          convolution's stride/padding/dilation/groups, and each BatchNorm's training flag,
          ``track_running_stats``, ``affine`` and ``eps``. None of these touches a tensor, so no
          tensor-level key can see them change.
        * **A fingerprint of every currently bound source tensor**, which is what catches the
          harness's own three mutations (``_prepare_module``'s recast changes ``data_ptr`` and
          dtype while leaving the version counter at 0; ``_sanitize_float_params``' in-place
          ``normal_`` and ``load_state_dict``'s copy both bump ``_version``) as well as
          ``fuse()``, which writes through ``.data`` and so changes none of them -- caught instead
          by ``_is_fused`` and by ``bn`` disappearing.

        Cheap by construction: no CUDA call, no allocation beyond the tuple, and no walk of
        ``self.parameters()`` -- the tensors visited here *are* every parameter the computation
        reads, so ``requires_grad`` comes along for free.
        """
        pool = self.m
        parts = [self.training, type(pool),
                 getattr(pool, "kernel_size", None), getattr(pool, "stride", None),
                 getattr(pool, "padding", None), getattr(pool, "ceil_mode", None)]

        for block in (self.cv1, self.cv2):
            conv = getattr(block, "conv", None)
            bn = getattr(block, "bn", None)
            parts.append((type(block), type(conv), type(getattr(block, "act", None)),
                          type(bn), getattr(block, "_is_fused", False)))
            if conv is not None:
                parts.append((getattr(conv, "stride", None), getattr(conv, "padding", None),
                              getattr(conv, "dilation", None), getattr(conv, "groups", None)))
                for name in ("weight", "bias"):
                    parts.append(_tensor_fingerprint(getattr(conv, name, None)))
            if bn is not None:
                parts.append((bn.training, getattr(bn, "track_running_stats", None),
                              getattr(bn, "affine", None), getattr(bn, "eps", None)))
                for name in ("weight", "bias", "running_mean", "running_var"):
                    parts.append(_tensor_fingerprint(getattr(bn, name, None)))
        return tuple(parts)

    def _folded_constants(self) -> _Folded | None:
        """The cached fold, re-derived whenever the live module stops matching what it was derived
        from. Returns None when this module is not the configuration the kernels implement.

        The negative result is cached too, keyed on the same signature: what
        :meth:`_derive_folded` refuses is a function of structure and metadata, never of tensor
        *values*, so a signature that produced None will produce None again. Without that, an
        ineligible module would re-walk and re-derive on every call to the fallback.
        """
        signature = self._live_signature()
        if signature is not None and self._folded_signature is not None:
            try:
                if signature == self._folded_signature:
                    return self._folded
            except Exception:  # noqa: BLE001 - an exotic config value that will not compare
                pass           # is a reason to re-derive, not to raise
        folded = self._derive_folded()
        self._folded = folded
        self._folded_signature = signature
        return folded

    def _derive_folded(self) -> _Folded | None:
        """Check that this module is the configuration the kernels implement, and if so fold
        its BatchNorms. Every check that can fail runs before any tensor arithmetic, so a
        module that will never be eligible costs attribute reads and no GPU work."""
        if not FAST_PATH_AVAILABLE:
            _refuse(f"extension unavailable: {BUILD_ERROR}")
            return None
        if self.training:
            _refuse("module is in training mode")
            return None

        pool = self.m
        if type(pool) is not MaxPool2d:
            _refuse(f"self.m is {type(pool).__name__}, not exactly MaxPool2d -- a subclass may "
                    f"override forward")
            return None
        if (_as_pair(getattr(pool, "kernel_size", None)) != (5, 5)
                or _as_pair(getattr(pool, "stride", None)) != (1, 1)
                or _as_pair(getattr(pool, "padding", None)) != (2, 2)
                or getattr(pool, "ceil_mode", True) is not False):
            # The radii 2/4/6 are the cascade of *this* pool. Any other kernel size, stride
            # or padding is a different operator, and ceil_mode must be literally False
            # because the reference rejects ``ceil_mode=0`` with a TypeError.
            _refuse("pool is not the same-shape 5x5 unit-stride pool the radii assume")
            return None

        w1 = getattr(getattr(self.cv1, "conv", None), "weight", None)
        w2 = getattr(getattr(self.cv2, "conv", None), "weight", None)
        if not (isinstance(w1, torch.Tensor) and isinstance(w2, torch.Tensor)
                and w1.dim() == 4 and w2.dim() == 4):
            _refuse("cv1 and cv2 do not both carry a 4-D convolution weight")
            return None
        in_channels, mid_channels, out_channels = w1.shape[1], w1.shape[0], w2.shape[0]
        if w2.shape[1] != 4 * mid_channels:
            _refuse(f"cv2 takes {w2.shape[1]} input channels, not 4 * {mid_channels}")
            return None

        if mid_channels % _TILE != 0:
            _refuse(f"{mid_channels} mid channels are not a multiple of {_TILE}")
            return None
        if out_channels % _TILE != 0:
            _refuse(f"{out_channels} output channels are not a multiple of {_TILE}")
            return None

        first = _fold_block(self.cv1, in_channels, mid_channels, "cv1")
        if first is None:
            return None
        second = _fold_block(self.cv2, 4 * mid_channels, out_channels, "cv2")
        if second is None:
            return None
        if first[0].device != second[0].device:
            _refuse("cv1 and cv2 weights live on different devices")
            return None

        return _Folded(first, second, in_channels, mid_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        folded = self._folded_constants()
        if folded is not None:
            refusal = _input_refusal(x, folded)
            if refusal is None:
                out = _extension.yolo_sppf(
                    x, folded.w1, folded.s1, folded.b1, folded.w2, folded.s2, folded.b2,
                    folded.mid_channels, _MAX_CHANNEL_TILES, _CV2_TILE_M, _CV2_TILE_N)
                if out is not None:
                    return out
                # The extension checks the properties only it can see cheaply -- a
                # forward-mode dual's tangent, a lazily negated or conjugated view, and whether
                # the pointer is aligned enough for a wmma fragment load -- and hands back
                # nothing rather than reading the wrong bytes.
                _refuse("the extension refused this tensor")
            else:
                _refuse(refusal)
        return self._reference_forward(x)


def _input_refusal(x: torch.Tensor, folded: _Folded) -> str | None:
    """Why ``x`` cannot go through the kernels, or None if it can.

    The kernels implement a contiguous fp16 CUDA NCHW activation whose channel count matches
    ``cv1``'s input, whose row length fits one warp (the pooling maps one lane per column),
    and whose pixel count is a multiple of both wmma tile extents in use, because there is no
    masked tail path. Reverse-mode gradient tracking is refused here; forward-mode duals, lazily
    negated or conjugated views, and pointers too poorly aligned for a wmma fragment load are
    refused by the extension, which can see all three for free where Python cannot.
    """
    if not x.is_cuda:
        return "input is not on CUDA"
    if x.dtype != torch.float16:
        return f"input dtype {x.dtype} is not float16"
    if x.dim() != 4:
        return f"input is {x.dim()}-D, not 4-D"
    if not x.is_contiguous():
        return "input is not contiguous"
    if x.numel() == 0:
        return "input is empty"
    if x.size(1) != folded.in_channels:
        return f"input has {x.size(1)} channels, not {folded.in_channels}"
    height, width = x.size(2), x.size(3)
    if width > _WARP_LANES:
        return f"row length {width} exceeds the {_WARP_LANES} lanes of one warp"
    pixels = height * width
    if pixels % _TILE != 0:
        return f"a plane of {pixels} pixels is not a multiple of {_TILE}"
    if pixels * 2 > _MAX_SHARED_BYTES:
        return (f"a plane of {pixels} pixels exceeds the shared memory a block may hold")
    if torch.is_grad_enabled() and (x.requires_grad or folded.params_need_grad):
        return "gradients are being tracked"
    return None
