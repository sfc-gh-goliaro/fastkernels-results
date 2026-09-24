"""Oasis spatial axial attention, fused down to five launches per forward.

Same ``__init__``/``forward`` contract as the baseline, and the same parameters under
the same names, so the harness's ``load_state_dict(baseline.state_dict())`` binds
every weight and the ``rotary_emb`` child still receives its dtype cast.

The baseline is **not** shaped by its data. At the captured shapes
(``x: fp16 [1, T, 9, 16, 1024]``, ``T in {2..6}``, ``heads=16``, ``dim_head=64``,
``N = 9*16 = 144``) it spends 418.9 us of host time issuing one forward against 211 us
of GPU time over 31 launches, and only 19 us of that GPU time is the work the operator
is defined by -- the qkv GEMM (6.6 us), cuDNN flash attention (7.5) and the output
projection (4.8). The other ~190 us is the rotary written as ~20 elementwise passes
over fp32 intermediates, the axial frequency table rebuilt from scratch on *every*
call, and layout repacks. Latency is flat in ``T``: tripling the work moves it under
10%. So the lever is launch count and Python, not arithmetic -- and on SM100, where PDL
is on by default and back-to-back launches already overlap on the device, host-side
issue cost is precisely what only fusion removes (KernelWiki ``hw-pdl-gdc``).

What is left is five launches:

    1  to_qkv          self.to_qkv(x.view(M, dim))              -> qkv [M, 3*Dm]
    2  fused kernel    split + rotary(q,k) + copy(v) + permute  -> [3, BT, heads, N, Dh]
    3  attention       SDPA over the three contiguous slices    -> [BT, heads, N, Dh]
    4  repack          (BT, N, heads, Dh) -> [M, Dm]
    5  to_out          self.to_out(...).view(bsz, time, height, width, -1)

Numerics that are contractual:

* The rotary is the reference's ``t*cos(f) + rotate_half(t)*sin(f)`` rearranged per
  pair index ``p``, which is exactly ``o[2p] = t[2p]*cos[2p] - t[2p+1]*sin[2p]`` and
  ``o[2p+1] = t[2p+1]*cos[2p+1] + t[2p]*sin[2p+1]``. The kernel evaluates it in fp32
  and rounds once where the reference rounds three times in fp16, which deviates by
  one to two fp16 ulps (measured ``max_abs = 3.906e-03`` on random fp16 ``q`` at
  ``T=6``, against a ``1e-2`` atol before the rtol term contributes). The deviation is
  in the *reference's* favour to remove, not ours: the baseline is the oracle, so the
  end-to-end bench is what accepts this, not the pre-softmax number.
* ``cos``/``sin`` are taken on the frequency table **in the table's own dtype**, so
  those values are bit-identical to the reference's, fp16 argument reduction included.
  Only then is the pair packed and widened to fp32. Storing the packed table in fp32
  rather than fp16 was measured to give an identical end-to-end error, and buys one
  in-kernel code path instead of two.
* ``v`` is materialized contiguous rather than handed to SDPA as the head-strided view
  the reference passes. Measured back-to-back with no L2 flush, cuDNN costs 2.4 us more
  with the strided ``v`` (12.5 -> 15.2 us of issue cost at ``T=2``, the same kernel name
  in both, so it is plan selection and not a hidden copy) -- but end-to-end inside this
  window that difference does not reproduce: substituting the strided view for the
  kernel's own ``v`` slice measured inside 0.4 us of it at every captured shape. So the
  contiguous ``v`` ships because it costs nothing, keeps the kernel's output one
  allocation, and is what a custom attention kernel would want later -- not on the
  2.4 us, which is not visible from here.

The table is built **lazily on first use, never in __init__**: ``_prepare_module``
rewrites ``freqs`` with ``p.data = p.data.to(fp16)`` and ``load_state_dict`` then
copies the baseline's values in, both *after* construction, so anything precomputed in
``__init__`` from a parameter is stale. ``candidate/L1/linear.py``'s ``Linear``
docstring calls out the same trap. The single-entry cache key therefore carries both
``freqs.data_ptr()`` (which the dtype cast moves, leaving ``_version`` at 0) and
``freqs._version`` (which ``copy_`` bumps, leaving ``data_ptr`` unchanged) -- verified
on this machine that each leg catches exactly one of the two and neither catches both.

The key is metadata, so state its limit rather than overclaim it: an in-place write
through ``freqs.data`` (``freqs.data.add_(1)``) changes neither ``data_ptr`` nor the
Parameter's ``_version`` -- ``.data`` hands out an alias with its own version counter --
and would therefore be served a stale table. Detecting that needs the table's *contents*,
which is a device read on a path whose whole budget is 2 us, so it is not affordable and
is not done. It is also not reachable through this harness: every mutation the bench
performs (the dtype cast, ``_sanitize_float_params``' ``normal_()``, and
``load_state_dict``'s ``copy_``) happens before the first forward, and the table is not
built until then. Anything that mutates a parameter through ``.data`` after benching has
begun is outside what this cache claims to survive.

Everything the kernel is not written for runs the baseline's ``forward``, reproduced
below, so an unclaimed configuration is wrong in no new way. A build failure degrades
to that path with the reason kept in :data:`FUSED_BUILD_ERROR` rather than swallowed
(the convention ``candidate/L1/dense_attention.py`` uses for ``_TINY_EXT_ERROR``), and
:data:`_FASTPATH_HITS` counts fast-path entries on the host, because a predicate that
is quietly false returns a *correct* answer at ~1.00x and is otherwise
indistinguishable from a real regression. ``OASIS_SPATIAL_DISABLE_FAST_PATH=1``, read
once at import, forces the fallback for a one-step A/B.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..L1.dense_attention import DenseAttention
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding, oasis_apply_rotary_emb

# Keys the ninja build lock and the resulting .so, so it must not collide with any
# other operator's extension in this workspace.
_EXTENSION_NAME = "fk_cand_l2_oasis_spatial_axial"

# 16-byte alignment is what the packed table's float4 load needs; the qkv read and the
# output write are 4-byte (one rotary pair per lane) and need only that, but the
# freshly allocated buffers are 256-byte aligned anyway.
_ALIGN_BYTES = 16

# Vectorized access assumes a whole number of 8-half groups per head row and per qkv
# row, which is what keeps a warp's addresses inside one 128-byte segment on both
# sides. Also the granularity at which no chunk can straddle the rotate/copy boundary
# for a *wider* vectorization than the one shipped here -- kept at 8 rather than
# relaxed to 2 so the claimed family does not silently depend on that choice.
_VECTOR_HALVES = 8

# fp16 is what the capture runs and what the standalone kernel test covers
# end-to-end; bf16 is claimed because the same test covers it, and the kernel is one
# template over both.
_CLAIMED_DTYPES = (torch.float16, torch.bfloat16)

# The kernel's grid is (ceil(N/rows), heads, bsz*time) and its block is
# (dim_head/2, rows, 3). Every one of those has a hardware cap. The C++ entry re-reads the
# real device limits; these are the cheap host-side screens, so an ineligible call never
# even reaches the qkv projection, let alone the launch. Conservative values that hold on
# every current architecture -- being conservative here only costs a fallback.
_MAX_GRID_YZ = 65535
_MAX_BLOCK_DIM_X = 1024
_MAX_BLOCK_DIM_Y = 1024
_MAX_BLOCK_DIM_Z = 64
_MAX_BLOCK_THREADS = 1024

#: q, k and v each get a `blockDim.z` plane. Fixed, but screened like the other two axes
#: rather than assumed, so all three block dimensions are checked on both sides.
_BLOCK_PLANES = 3

# Threads per q/k/v plane, which sets the row tiling: `rows = this // (dim_head/2)`, so 160
# gives 5 rows at the captured dim_head of 64 and a (32, 5, 3) block. Chosen by sweeping every
# legal tiling (rows 1..10, all of which clear one occupancy wave and are numerically
# identical) against one binary with the row count forced per call -- rows=5 measured fastest,
# though only by 0.2 us of geomean over rows=3 and rows=4, so this is a weak preference among
# ties rather than a tuned optimum. profile/sweep_row_tiling.py reproduces the sweep.
#
# The device-side `plane_geometry` uses the same formula and the two must agree, or the host
# would screen a block shape the kernel does not launch; asserted at import below.
_TARGET_PLANE_THREADS = 160


def _launch_config(dim_head: int) -> tuple[int, int, int]:
    """``(block.x, block.y, threads per block)`` for a head width, as the kernel computes it."""
    pairs_per_row = dim_head // 2
    rows = max(1, _TARGET_PLANE_THREADS // pairs_per_row)
    return pairs_per_row, rows, _BLOCK_PLANES * pairs_per_row * rows


def _launch_eligible(bsz: int, frames: int, height: int, width: int,
                     heads: int, dim_head: int) -> bool:
    """Whether these extents can be launched and addressed by the kernel.

    Split out as pure integer arithmetic on purpose: it is the part of the predicate that
    cannot otherwise be tested. The ``2**31`` offset bound would need a 4 GB allocation to
    reach through a real tensor, and the block-dimension limits would need a head width no
    model uses, so both are exercised directly against this function instead. The C++ entry
    enforces the same bounds against the device's own reported limits.
    """
    if bsz <= 0 or frames <= 0 or height <= 0 or width <= 0:
        return False
    if heads <= 0 or dim_head <= 0:
        return False
    # Vectorized access on both sides needs whole 8-half groups per head row and per qkv row.
    if dim_head % _VECTOR_HALVES:
        return False
    dim_model = heads * dim_head
    # Implied by the line above (dim_head % 8 == 0 gives dim_model % 8 == 0), and kept
    # anyway: it is the qkv *row* stride the vectorized read actually depends on, so a
    # future change to how the row is laid out should have to delete this deliberately.
    if (3 * dim_model) % _VECTOR_HALVES:
        return False
    # 32-bit element offsets inside the kernel: 3*M*Dm must fit.
    if 3 * bsz * frames * height * width * dim_model >= 2 ** 31:
        return False
    # grid.y is heads, grid.z is bsz*time.
    if heads > _MAX_GRID_YZ or bsz * frames > _MAX_GRID_YZ:
        return False
    block_x, block_y, threads = _launch_config(dim_head)
    if block_x > _MAX_BLOCK_DIM_X or block_y > _MAX_BLOCK_DIM_Y:
        return False
    if _BLOCK_PLANES > _MAX_BLOCK_DIM_Z:
        return False
    if threads > _MAX_BLOCK_THREADS:
        return False
    return True


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cstdint>

namespace {

// One rotary pair per lane, with q/k/v on the block's z axis, so one block resolves all
// three and each thread does exactly one load and one store. The invariants this geometry
// holds, all of them checked or measured rather than assumed:
//
//   * `blockDim.x == dim_head/2` makes every warp exactly one `(row, which)` plane, so
//     `which` is warp-uniform, the rotate-vs-copy branch never diverges, and each warp
//     covers one contiguous 128-byte head row on the read side and one on the write side.
//     Measured: 4.00 sectors per store request and 32 of 32 bytes used per sector.
//   * The rotate/copy boundary is at pair granularity, so no vector access can straddle it.
//   * No integer division or modulo per element -- every index comes from a thread or block
//     coordinate directly.
//   * The grid stays above one CTA wave on all 148 SMs at every captured shape, where a CTA
//     wave is `grid CTAs / (SMs * resident CTAs per SM)`. Measured: 1.568 at the smallest
//     captured shape, 4.703 at the largest.
//
// The last one is why `which` is a thread axis rather than a serial loop over three
// iterations: the serial form does the same work with a third of the threads and cannot
// reach one wave at the smallest captured shape whatever its block shape, since the thread
// count is fixed by the work. The row tiling and the two rejected `which` mappings were
// chosen by measurement; profile/phase1-measurements.md records the sweep and what each
// alternative cost.
constexpr int kTargetPlaneThreads = 160;

// Threads for one q/k/v plane, and the row tiling that follows from it. Host and device
// must agree on this arithmetic, because the host screens the block dimensions before
// admitting a call; `_launch_config` in this file is the same formula.
//
// `rows_override` exists so the row tiling can be swept from a profiling script against this
// one binary: every tiling then runs identical machine code, and the comparison isolates the
// launch geometry instead of also comparing two compilations. Zero means "use the default",
// which is what the operator always passes.
inline void plane_geometry(const int dim_head, const int rows_override,
                           int* pairs_per_row, int* rows) {
  const int pairs = dim_head / 2;
  *pairs_per_row = pairs;
  *rows = rows_override > 0
              ? rows_override
              : (kTargetPlaneThreads / pairs > 0 ? kTargetPlaneThreads / pairs : 1);
}

// Both halves of a pair are one naturally aligned 4-byte access: the qkv row base is
// 16-byte aligned and pair p sits at byte offset 2 * sizeof(scalar_t) * p.
template <typename scalar_t>
struct alignas(2 * sizeof(scalar_t)) HalfPair {
  scalar_t lo;
  scalar_t hi;
};

// qkv    [M, 3*Dm]                 contiguous, M = BT * N, Dm = heads * Dh
// table  [N, rot_dim, 2]           contiguous fp32, packed (cos, sin) per column
// out    [3, BT, heads, N, Dh]     contiguous, qkv's dtype
//
// With `bt` the frame, `s = i * width + j` the position in the token grid (exactly the
// reference's height/width flattening), `h` the head and `p` the pair inside the head
// vector:
//
//   in  = (bt*N + s)*3*Dm + which*Dm + h*Dh + 2p
//   out = ((which*BT + bt)*heads + h)*N*Dh + s*Dh + 2p
//   tab = (s*rot_dim + 2p)*2                       -> cos[2p], sin[2p], cos[2p+1], sin[2p+1]
//
// No integer division or modulo anywhere: p is threadIdx.x, s is
// blockIdx.x*blockDim.y + threadIdx.y, h is blockIdx.y, bt is blockIdx.z, and `which` is
// threadIdx.z -- warp-uniform, so the rotate-vs-copy branch is resolved per warp and the v
// plane never reads the table.
template <typename scalar_t>
__global__ void oasis_qkv_rotary_kernel(
    const scalar_t* __restrict__ qkv,
    const float* __restrict__ table,
    scalar_t* __restrict__ out,
    const int n_positions,
    const int heads,
    const int dim_head,
    const int rot_pairs,
    const int frames) {
  const int p = threadIdx.x;
  const int s = blockIdx.x * blockDim.y + threadIdx.y;
  if (s >= n_positions) {
    return;
  }
  const int which = threadIdx.z;
  const int h = blockIdx.y;
  const int bt = blockIdx.z;

  const int dim_model = heads * dim_head;

  // 32-bit throughout: the host predicate bounds 3*M*Dm below 2**31 and both offsets are
  // inside that bound, so no element offset can wrap. `plane` is M*Dm, a third of it.
  const int in_offset = (bt * n_positions + s) * 3 * dim_model
                      + which * dim_model + h * dim_head + 2 * p;
  const int plane = frames * heads * n_positions * dim_head;
  const int out_offset = which * plane
                       + ((bt * heads + h) * n_positions + s) * dim_head + 2 * p;

  using Pair = HalfPair<scalar_t>;
  const Pair in = *reinterpret_cast<const Pair*>(qkv + in_offset);
  Pair result;

  // Columns 2p and 2p+1 are rotated only in the q and k planes, and only inside rot_dim;
  // beyond it the reference passes the head vector through untouched (its `t_right`
  // slice), and the v plane is a pure copy that never reads the table at all.
  if (which < 2 && p < rot_pairs) {
    // (s*rot_dim + 2p)*2 == 4*(s*rot_pairs + p) floats: one float4, 16-byte aligned for a
    // 16-byte aligned base, and contiguous across the warp.
    const float4 cs =
        *reinterpret_cast<const float4*>(table + 4 * (s * rot_pairs + p));
    // fp32 with one final round. An fp16 x fp16 product needs at most 22 significand
    // bits, so each product is exact in fp32 and separate multiplies, an FMA and
    // -fmad=false all give the same sum and the same rounded result; none of them
    // reproduces the reference's three fp16 roundings, which is where the ~1 ulp
    // deviation comes from.
    const float lo = static_cast<float>(in.lo);
    const float hi = static_cast<float>(in.hi);
    result.lo = static_cast<scalar_t>(lo * cs.x - hi * cs.y);
    result.hi = static_cast<scalar_t>(hi * cs.z + lo * cs.w);
  } else {
    result = in;
  }
  *reinterpret_cast<Pair*>(out + out_offset) = result;
}

template <typename scalar_t>
void launch_qkv_rotary(
    const at::Tensor& qkv,
    const at::Tensor& table,
    at::Tensor& out,
    const int n_positions,
    const int heads,
    const int dim_head,
    const int rot_pairs,
    const int frames,
    const int rows_override,
    cudaStream_t stream) {
  int pairs_per_row = 0;
  int rows = 0;
  plane_geometry(dim_head, rows_override, &pairs_per_row, &rows);
  const dim3 block(pairs_per_row, rows, 3);
  const dim3 grid((n_positions + rows - 1) / rows, heads, frames);
  oasis_qkv_rotary_kernel<scalar_t><<<grid, block, 0, stream>>>(
      qkv.data_ptr<scalar_t>(),
      table.data_ptr<float>(),
      out.data_ptr<scalar_t>(),
      n_positions,
      heads,
      dim_head,
      rot_pairs,
      frames);
  // A hard check, not a TORCH_CHECK_VALUE: past the eligibility guards above, a failed
  // launch is a real fault and must propagate rather than be retried on the slow path.
  // Without it an over-large grid returns silently and leaves the output uninitialized
  // until some later, unrelated CUDA call reports the error.
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The entry point re-validates its own contract rather than trusting the host
// predicate, so editing the Python side later cannot reach the kernel with operands it
// was not written for. Rejections that mean "this input is not for me" use
// TORCH_CHECK_VALUE, which surfaces as a Python ValueError so the caller can fall back;
// a real CUDA failure keeps propagating. That is the convention
// candidate/L1/oasis_rotary.py already establishes.
at::Tensor oasis_qkv_rotary(
    const at::Tensor& qkv,
    const at::Tensor& table,
    const int64_t heads,
    const int64_t frames,
    const int64_t rows_override) {
  TORCH_CHECK_VALUE(qkv.is_cuda() && table.is_cuda(),
                    "oasis_spatial_axial: fused kernel needs CUDA tensors");
  TORCH_CHECK_VALUE(qkv.device() == table.device(),
                    "oasis_spatial_axial: qkv and table must share one device");
  // Taken here, before anything device-dependent: every property query, the allocation
  // and the current-stream lookup below must all resolve against the *input's* device,
  // not whatever device happened to be current on entry. Querying
  // getCurrentDeviceProperties() ahead of this guard would validate the grid limits
  // against the ambient device, which can refuse a legal call on another device or admit
  // one that then fails hard at launch instead of being refused with a ValueError.
  const c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK_VALUE(qkv.dim() == 2,
                    "oasis_spatial_axial: qkv must be [M, 3*heads*dim_head], got dim ",
                    qkv.dim());
  TORCH_CHECK_VALUE(table.dim() == 3 && table.size(2) == 2,
                    "oasis_spatial_axial: table must be [N, rot_dim, 2]");
  TORCH_CHECK_VALUE(qkv.is_contiguous() && table.is_contiguous(),
                    "oasis_spatial_axial: fused kernel needs contiguous inputs");
  TORCH_CHECK_VALUE(table.scalar_type() == at::ScalarType::Float,
                    "oasis_spatial_axial: packed table must be float32, got ",
                    table.scalar_type());
  TORCH_CHECK_VALUE(qkv.scalar_type() == at::ScalarType::Half ||
                        qkv.scalar_type() == at::ScalarType::BFloat16,
                    "oasis_spatial_axial: fused kernel has no path for dtype ",
                    qkv.scalar_type());
  // A raw pybind entry records no autograd node, so a differentiable call has to go
  // back to the PyTorch path instead of silently losing its gradient.
  TORCH_CHECK_VALUE(!(at::GradMode::is_enabled() &&
                      (qkv.requires_grad() || table.requires_grad())),
                    "oasis_spatial_axial: fused kernel is inference-only");

  TORCH_CHECK_VALUE(heads > 0 && frames > 0,
                    "oasis_spatial_axial: heads and frames must be positive");
  const int64_t n_positions = table.size(0);
  const int64_t rot_dim = table.size(1);
  TORCH_CHECK_VALUE(n_positions > 0 && rot_dim > 0,
                    "oasis_spatial_axial: table must be non-empty");
  const int64_t qkv_width = qkv.size(1);
  TORCH_CHECK_VALUE(qkv_width % (3 * heads) == 0,
                    "oasis_spatial_axial: qkv width ", qkv_width,
                    " is not 3*heads*dim_head for heads=", heads);
  const int64_t dim_head = qkv_width / (3 * heads);
  TORCH_CHECK_VALUE(dim_head > 0 && dim_head % 8 == 0,
                    "oasis_spatial_axial: dim_head must be a positive multiple of 8, "
                    "got ", dim_head);
  TORCH_CHECK_VALUE(rot_dim <= dim_head && rot_dim % 8 == 0,
                    "oasis_spatial_axial: rot_dim ", rot_dim,
                    " must be a multiple of 8 and at most dim_head ", dim_head);
  // Sizes have to agree, not merely be individually plausible: a table whose N does
  // not divide the qkv rows would otherwise index past the end of one of them. Written
  // as a division rather than `qkv.size(0) == frames * n_positions` so that a wild
  // `frames` cannot overflow the guard that is supposed to catch it.
  TORCH_CHECK_VALUE(qkv.size(0) % n_positions == 0 &&
                        qkv.size(0) / n_positions == frames,
                    "oasis_spatial_axial: qkv rows ", qkv.size(0),
                    " is not frames*N for frames=", frames, ", N=", n_positions);
  // 3*M*Dm is exactly qkv.numel(), since 3*heads*dim_head == qkv.size(1) by the
  // division above -- so the bound needs no multiplication of its own.
  TORCH_CHECK_VALUE(qkv.numel() < (int64_t{1} << 31),
                    "oasis_spatial_axial: ", qkv.numel(),
                    " elements overflows the kernel's 32-bit offsets");
  // Every launch dimension, read from the device rather than hard-coded, and refused
  // rather than attempted: an over-large grid or block is an invalid launch, not a slow
  // one. grid.y is heads and grid.z is frames; the block is
  // (dim_head/2, rows, 3) with rows from the shared plane_geometry.
  const auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK_VALUE(heads <= props->maxGridSize[1],
                    "oasis_spatial_axial: heads ", heads,
                    " exceeds the grid y limit ", props->maxGridSize[1]);
  TORCH_CHECK_VALUE(frames <= props->maxGridSize[2],
                    "oasis_spatial_axial: frames ", frames,
                    " exceeds the grid z limit ", props->maxGridSize[2]);
  int pairs_per_row = 0;
  int rows = 0;
  TORCH_CHECK_VALUE(rows_override >= 0 && rows_override <= props->maxThreadsDim[1],
                    "oasis_spatial_axial: rows_override ", rows_override,
                    " is negative or beyond the block y limit ", props->maxThreadsDim[1]);
  plane_geometry(static_cast<int>(dim_head), static_cast<int>(rows_override),
                 &pairs_per_row, &rows);
  TORCH_CHECK_VALUE(pairs_per_row <= props->maxThreadsDim[0],
                    "oasis_spatial_axial: block x ", pairs_per_row,
                    " exceeds the block x limit ", props->maxThreadsDim[0]);
  TORCH_CHECK_VALUE(rows <= props->maxThreadsDim[1],
                    "oasis_spatial_axial: block y ", rows,
                    " exceeds the block y limit ", props->maxThreadsDim[1]);
  TORCH_CHECK_VALUE(3 <= props->maxThreadsDim[2],
                    "oasis_spatial_axial: block z 3 exceeds the block z limit ",
                    props->maxThreadsDim[2]);
  TORCH_CHECK_VALUE(3 * pairs_per_row * rows <= props->maxThreadsPerBlock,
                    "oasis_spatial_axial: ", 3 * pairs_per_row * rows,
                    " threads per block exceeds the limit ", props->maxThreadsPerBlock);
  TORCH_CHECK_VALUE(reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0 &&
                        reinterpret_cast<uintptr_t>(table.data_ptr()) % 16 == 0,
                    "oasis_spatial_axial: inputs must be 16-byte aligned");

  at::Tensor out = at::empty({3, frames, heads, n_positions, dim_head}, qkv.options());
  // TORCH_CHECK, not TORCH_CHECK_VALUE: this buffer is ours, so its misalignment is a
  // broken internal invariant rather than "this input is not for me", and it must not
  // be swallowed by the caller's fall-back-on-ValueError.
  TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "oasis_spatial_axial: output allocation is not 16-byte aligned");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int n_positions_i = static_cast<int>(n_positions);
  const int heads_i = static_cast<int>(heads);
  const int dim_head_i = static_cast<int>(dim_head);
  const int rot_pairs_i = static_cast<int>(rot_dim / 2);
  const int frames_i = static_cast<int>(frames);
  const int rows_i = static_cast<int>(rows_override);

  if (qkv.scalar_type() == at::ScalarType::Half) {
    launch_qkv_rotary<at::Half>(qkv, table, out, n_positions_i, heads_i, dim_head_i,
                                rot_pairs_i, frames_i, rows_i, stream);
  } else {
    launch_qkv_rotary<at::BFloat16>(qkv, table, out, n_positions_i, heads_i, dim_head_i,
                                    rot_pairs_i, frames_i, rows_i, stream);
  }
  return out;
}
"""

_CPP_SOURCE = r"""
#include <torch/extension.h>

at::Tensor oasis_qkv_rotary(
    const at::Tensor& qkv,
    const at::Tensor& table,
    const int64_t heads,
    const int64_t frames,
    const int64_t rows_override);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("oasis_qkv_rotary", &oasis_qkv_rotary,
        "Split a fused qkv row into q/k/v, apply the packed axial rotary to q and k, "
        "and write [3, frames, heads, N, dim_head] contiguous",
        py::arg("qkv"), py::arg("table"), py::arg("heads"), py::arg("frames"),
        py::arg("rows_override") = 0);
}
"""


# `_TARGET_PLANE_THREADS` above and `kTargetPlaneThreads` in the CUDA source are the same
# number written twice, and the host screens a block shape the kernel then launches -- so a
# silent divergence would mean screening the wrong shape. Checked at import, where it costs
# one substring search and cannot be got wrong later.
assert f"kTargetPlaneThreads = {_TARGET_PLANE_THREADS};" in _CUDA_SOURCE, (
    "the host and device plane-thread constants have diverged")


def _local_arch_list() -> str | None:
    """Local compute capability, in the form nvcc wants for this build.

    Compute capabilities 9.0 and up need the architecture-specific ``a`` variant.
    Returning None leaves ``TORCH_CUDA_ARCH_LIST`` alone, which is the right thing when
    the capability cannot be read.
    """
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _build_fused_extension():
    """Compile the fused kernel into a workspace-local build directory."""
    from torch.utils.cpp_extension import load_inline

    build_dir = Path(__file__).resolve().parents[2] / ".torch_extensions" / _EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)

    # The environment ships a multi-architecture list; compiling all of it would cost
    # minutes of wall clock for a kernel that only ever runs on this GPU.
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


#: Set ``OASIS_SPATIAL_DISABLE_FAST_PATH`` to anything but "" or "0" to force the
#: baseline path. Read once, at import, so it can never be part of a timed call.
FAST_PATH_DISABLED = os.environ.get("OASIS_SPATIAL_DISABLE_FAST_PATH", "") not in ("", "0")

#: The build failure, if there was one, kept rather than swallowed: a silent 1.00x has
#: to be distinguishable from a real regression.
FUSED_BUILD_ERROR: str | None = None

_FUSED = None

if FAST_PATH_DISABLED:
    FUSED_BUILD_ERROR = "disabled by OASIS_SPATIAL_DISABLE_FAST_PATH"
    # One line at import, on stderr, so a bench run taken with the fast path off
    # identifies itself in the per-operator log instead of looking like a 1.00x
    # regression. Nothing in the harness reads FUSED_BUILD_ERROR or the hit counter.
    print(f"[candidate L2/oasis_spatial_axial_attention] fast path disabled by "
          f"OASIS_SPATIAL_DISABLE_FAST_PATH; running the baseline path",
          file=sys.stderr, flush=True)
else:
    # Built at import, never inside forward: it keeps compilation out of every timed
    # region and clear of the harness's no-new-threads snapshot, and it happens while
    # the worker is still producing output rather than under the stall watchdog.
    try:
        _FUSED = _build_fused_extension().oasis_qkv_rotary
    except Exception as exc:  # pragma: no cover - build environment dependent
        FUSED_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"[candidate L2/oasis_spatial_axial_attention] fused kernel unavailable, "
              f"delegating to the baseline path: {FUSED_BUILD_ERROR}",
              file=sys.stderr, flush=True)

#: Whether the fused kernel is live. A benchmark taken with this False measured the
#: baseline path, not the kernel.
FUSED_AVAILABLE = _FUSED is not None

#: Fast-path entries, keyed by input shape. Plain ints incremented on the host: no
#: threads, no device sync, nothing the harness's integrity guards watch. This is what
#: separates "the fast path ran" from "the predicate was quietly false".
_FASTPATH_HITS: dict[tuple[int, ...], int] = {}


def fastpath_hits() -> int:
    """Total fast-path entries since import."""
    return sum(_FASTPATH_HITS.values())


class OasisSpatialAxialAttention(nn.Module):
    """Spatial (per-frame) self-attention over a height x width token grid.

    Registers exactly the baseline's parameters, under the baseline's names, and
    derives nothing from them in ``__init__``: the harness casts every high-precision
    parameter and only *then* loads the baseline's ``state_dict``, so a table or a
    transposed weight computed here would be built from values that no longer exist.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.heads = heads
        self.to_qkv = Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = Linear(dim_head * heads, dim, bias=True)
        self.rotary_emb = rotary_emb
        self.attn = DenseAttention(backend="sdpa")
        # Constructor scalars, not derived state: the predicate needs dim_head, and
        # reading it back off a weight shape would be the same number by a longer road.
        self.dim_head = dim_head
        # Single-entry packed-table cache. Plain attributes, so neither appears in
        # state_dict and the harness's weight sharing cannot touch them.
        self._packed_key: tuple | None = None
        self._packed_table: torch.Tensor | None = None

    # -- the packed (cos, sin) table ------------------------------------------------

    def _packed_rotary_table(self, height: int, width: int) -> torch.Tensor | None:
        """Return the packed ``[N, rot_dim, 2]`` fp32 table, or None to fall back.

        Built on first use and cached against a key that catches every way ``freqs``
        can change under this harness. Both the ``data_ptr`` and the ``_version`` leg
        are load-bearing and neither alone suffices: ``p.data = p.data.to(fp16)`` moves
        the storage while leaving ``_version`` at 0, and ``copy_`` bumps ``_version``
        while leaving the storage where it was. Since height and width are constant
        across the shape mix and ``T`` is not part of the key, this hits on every call
        after the first -- worth about ten launches and a good deal of Python by itself.
        """
        rotary = self.rotary_emb
        freqs = getattr(rotary, "freqs", None)
        if not isinstance(freqs, torch.Tensor):
            return None
        key = (height, width, id(rotary), getattr(rotary, "freqs_for", None),
               freqs.data_ptr(), freqs._version, freqs.dtype, freqs.device)
        if self._packed_key == key:
            return self._packed_table

        # A rejection is cached under the same key as a success, so an unclaimed rotary
        # does not rebuild the axial table on every call just to be turned down again.
        self._packed_key = key
        self._packed_table = None

        source = rotary.get_axial_freqs(height, width)
        if (not isinstance(source, torch.Tensor) or source.dim() != 3
                or source.shape[0] != height or source.shape[1] != width):
            return None
        rot_dim = source.shape[2]
        if rot_dim <= 0 or rot_dim % _VECTOR_HALVES or rot_dim > self.dim_head:
            return None
        if not source.is_cuda or source.device != freqs.device:
            return None
        if source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None

        # cos/sin in the table's *own* dtype, so those values are bit-identical to the
        # reference's -- fp16 argument reduction included -- and only then widened. An
        # fp16 packed table measured the same end-to-end error, so fp32 costs nothing
        # here and leaves the kernel one code path instead of two.
        flat = source.reshape(height * width, rot_dim)
        packed = torch.stack((flat.cos(), flat.sin()), dim=-1).float().contiguous()
        if tuple(packed.shape) != (height * width, rot_dim, 2):
            return None
        self._packed_table = packed
        return packed

    # -- dispatch --------------------------------------------------------------------

    def _fast_path_table(self, x: torch.Tensor) -> torch.Tensor | None:
        """The packed table if this call is claimed, else None.

        A flat conjunction of *sufficient* conditions, each guarding something the
        kernel relies on, and all of them host-side attribute or integer comparisons:
        no device work and no synchronization.
        """
        if _FUSED is None:
            return None
        # The kernel is a raw pybind entry that builds no graph, so anything that wants
        # a gradient has to go elsewhere. The harness runs both the correctness
        # forwards and the timed loop under no_grad, so this is false in-bench.
        if torch.is_grad_enabled():
            return None
        if x.dim() != 5 or not x.is_cuda or x.dtype not in _CLAIMED_DTYPES:
            return None
        # A hidden .contiguous() would cost a launch, and the index map assumes the
        # captured row-major layout outright.
        if not x.is_contiguous():
            return None
        bsz, frames, height, width, _ = x.shape
        heads, dim_head = self.heads, self.dim_head
        # Positive extents, whole-vector widths, the 32-bit offset bound, and every grid
        # and block dimension against its hardware cap -- all integer arithmetic, and all
        # of it before the projection runs, so an ineligible call costs no GPU work.
        if not _launch_eligible(bsz, frames, height, width, heads, dim_head):
            return None
        dim_model = heads * dim_head
        # The qkv the projection will hand the kernel has to be exactly 3*heads*dim_head
        # wide; a shape read, not a value read, so the cast-then-load order is irrelevant.
        weight = self.to_qkv.weight
        if weight.dim() != 2 or weight.shape[0] != 3 * dim_model:
            return None

        table = self._packed_rotary_table(height, width)
        if table is None:
            return None
        # The packed table's own metadata, re-checked here rather than trusted from where it
        # was built: this runs on the cache-hit path, so it is what stands between a table
        # that was replaced or reshaped since and a kernel that indexes it as
        # [N, rot_dim, 2] fp32 contiguous.
        if table.dim() != 3 or table.shape[0] != height * width or table.shape[2] != 2:
            return None
        rot_dim = table.shape[1]
        if rot_dim <= 0 or rot_dim % _VECTOR_HALVES or rot_dim > dim_head:
            return None
        if table.dtype is not torch.float32 or not table.is_contiguous():
            return None
        if table.device != x.device:
            return None
        # The float4 table load needs a 16-byte aligned base. The freshly allocated
        # table always is, but that is a property of the allocator, not a guarantee.
        if table.data_ptr() % _ALIGN_BYTES:
            return None
        return table

    def _qkv_eligible(self, qkv: torch.Tensor, table: torch.Tensor,
                      rows: int, dim_model: int) -> bool:
        """Whether the *produced* projection output is what the kernel indexes.

        The predicate that admitted the call reasoned about `x` and about the weight's shape;
        this reasons about the tensor that actually reaches the kernel. It cannot run any
        earlier -- the buffer does not exist until the projection has run -- so an ineligible
        qkv costs one GEMM and then falls back. That is the price of checking the real operand
        instead of inferring it, and it is never paid on the captured shapes.
        """
        if qkv.dim() != 2 or qkv.shape[0] != rows or qkv.shape[1] != _BLOCK_PLANES * dim_model:
            return False
        if qkv.dtype not in _CLAIMED_DTYPES or not qkv.is_contiguous():
            return False
        # One kernel, one device: the launch goes where the input is, and the table has to be
        # there too.
        if qkv.device != table.device:
            return False
        # 16-byte alignment for the vectorized row access.
        if qkv.data_ptr() % _ALIGN_BYTES:
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        table = self._fast_path_table(x)
        if table is None:
            return self._baseline_forward(x)

        bsz, frames, height, width, dim = x.shape
        batch_frames = bsz * frames
        positions = height * width
        rows = batch_frames * positions

        qkv = self.to_qkv(x.view(rows, dim))
        if not self._qkv_eligible(qkv, table, rows, self.heads * self.dim_head):
            return self._baseline_forward(x)
        try:
            # One allocation, three contiguous [batch_frames, heads, N, dim_head]
            # slices, no further Python. The C++ side allocates and returns rather than
            # taking an out parameter, which removes one Python-level op and makes a
            # size disagreement between qkv and out structurally impossible instead of
            # merely checked.
            packed = _FUSED(qkv, table, self.heads, batch_frames)
        except (ValueError, TypeError):
            # "Not for me": something the host predicate did not cover, so redo the
            # call on the path that handles everything. A CUDA fault or an OOM is a
            # RuntimeError and keeps propagating rather than being quietly retried.
            return self._baseline_forward(x)

        shape = tuple(x.shape)
        _FASTPATH_HITS[shape] = _FASTPATH_HITS.get(shape, 0) + 1

        # self.attn takes (batch, seq, heads, head_dim) and permutes straight back to
        # the (batch, heads, seq, head_dim) tensors the kernel already produced. Kept in
        # the path rather than calling SDPA directly so that a future L1 DenseAttention
        # win reaches this operator for free, which is worth having because it costs
        # nothing: measured against direct `F.scaled_dot_product_attention` on the
        # kernel's own layout, the two agree bit-for-bit and land within 0.1 us of each
        # other at every captured shape (50.3/50.3 us at T=2, 60.3/60.3 at T=6). At
        # S = 144 the module's tiny-sequence path declines on one integer comparison
        # (0 < seq <= 8) and its backend is "sdpa", so its remaining body *is*
        # `F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
        # is_causal=False, scale=None)` -- the same primitive with the same arguments.
        #
        # Pinning SDPBackend.CUDNN_ATTENTION is not worth it either: the heuristic
        # already picks the cuDNN flash kernel at these shapes, and pinning it measured
        # inside the same 0.4 us noise band, so it buys an sdpa_kernel context for
        # nothing. (An earlier, *ordered* sweep appeared to show pinning costing 18 us
        # and this wrapper costing 14; both were an artifact of the first variant timed
        # in a process reading high. profile/ab_candidate.py now interleaves.)
        out = self.attn(
            packed[0].transpose(1, 2),
            packed[1].transpose(1, 2),
            packed[2].transpose(1, 2),
            causal=False,
        )
        # (batch_frames, N, heads, dim_head) -> [rows, dim_model]. The reshape is the
        # one repack copy; an explicit torch.empty + copy_ measured identical, so the
        # reshape stays. Removing the copy altogether means fusing the transpose into an
        # attention epilogue, which is later work.
        return self.to_out(out.reshape(rows, self.heads * self.dim_head)).view(
            bsz, frames, height, width, -1)

    # -- fallback ---------------------------------------------------------------------

    def _baseline_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The baseline's forward, reproduced.

        Everything the fast path does not claim runs here, so an unclaimed
        configuration is wrong in no new way -- and a build failure costs speed rather
        than correctness.
        """
        bsz, time, height, width, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)
        v = v.reshape(bsz * time, height, width, self.heads, -1).permute(0, 3, 1, 2, 4)

        freqs = self.rotary_emb.get_axial_freqs(height, width)
        q = oasis_apply_rotary_emb(freqs, q)
        k = oasis_apply_rotary_emb(freqs, k)

        q = q.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        k = k.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        v = v.reshape(bsz * time, self.heads, height * width, -1).transpose(1, 2)
        out = self.attn(q, k, v, causal=False)
        out = out.reshape(bsz, time, height, width, self.heads, -1).reshape(
            bsz, time, height, width, -1)
        return self.to_out(out.to(q.dtype))
