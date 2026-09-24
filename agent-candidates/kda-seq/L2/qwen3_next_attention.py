"""Qwen3-Next full attention: the baseline's composition with its glue fused into CUDA.

Semantics are ``baseline.py``'s, unchanged: GQA over 16 query heads and 2 KV heads at
head_dim 256, per-head QK-RMSNorm, partial NeoX RoPE over the first 64 dimensions,
``out * sigmoid(gate)``, and a paged HND KV cache written through the engine's state
manager.  The frozen L1 winners still do the work they won -- ``TRTLLMPrefill`` and
``TRTLLMDecode`` serve attention, ``GemmaRMSNorm`` owns the norm weights,
``QKVParallelLinear`` / ``RowParallelLinear`` own the projections, ``StoreKVCacheHND``
serves the fallback.  What is different is the glue between them.

Why the glue is the problem.  Measured on B200 (``profile/phase1-baseline-measurements.md``),
four of the five benched shapes are *host* bound: at N=60 the baseline spends 277.6 us of
host time issuing 16-17 kernels while the GPU finishes in 72.0 us and then idles for
~158 us.  The same is true at N=1 (~176 us idle) and N=445 (~156 us).  Only N=16384 is
device bound, and there the Triton normalization moves ~570 MB in 410 us -- about 5x off
this device's roofline -- because its grid gives each thread about four bytes to carry.
So a single change addresses both regimes: issue fewer kernels, and make each one move a
full 16 bytes per lane.

Three fused kernels replace nine launches' worth of glue:

* ``qk_norm_rope_store`` -- QK-RMSNorm, partial RoPE and the paged K/V store in one pass.
  Replaces the vendored Triton ``fused_qk_rmsnorm_rope_gate`` *and* ``StoreKVCacheHND``
  (47.6 us of host time, ~440 us of device time at N=16384).  Two intermediates stop
  existing as a consequence: normalized K is written straight into the cache instead of
  into a tensor nothing downstream reads, and the gate is *not copied at all* -- the
  Triton baseline writes a fresh (N, 4096) gate buffer, which at N=16384 is 134 MB
  written and 134 MB read back so that a later kernel can read what was already in
  ``qkv``.
* ``gate_mul`` -- ``sigmoid`` + ``mul`` + the output ``Memcpy DtoD`` in one pass, reading
  the gate strided in place out of ``qkv`` (224 us of device time at N=16384).
* ``prefill_cu_seqlens`` -- one single-block scan in place of ``sub`` + ``zeros`` +
  ``cumsum`` + slice-assign (44.0 us of host time).

Why CUDA C++ rather than Triton, given the baseline's kernel is already Triton: on these
shapes the *launcher* is a first-order cost.  Measured in this workspace, a minimal
``load_inline`` pybind call plus one launch is 5.35 us, an aten op is 8-12 us, and one
Triton launch with 25 arguments is 26.5 us.  Triton's launcher alone is roughly a third
of the whole host budget this module is trying to reach, which settles the language for
the fast path.  Triton stays on the fallback path, where it is the baseline's own code.

The extension is built at *import*, never on the first call: compilation then sits outside
the bench's timed window and outside its no-new-threads guard, which wraps only the
candidate's timing loop.  If the build fails the module records why and every call takes
the exact baseline composition -- a silent fallback would score about 1.00x and read like
an honest result, so ``BUILD_ERROR`` is kept and the smoke test asserts on it.  This is
the convention the frozen ``candidate/L1/store_kvcache.py`` already established.

Numerics are matched deliberately rather than approximately.  The normalized value is
rounded to the storage dtype and widened back to fp32 *before* the rotation, because that
is what the vendored kernel does (``.to(INPUT_DTYPE).to(tl.float32)``), and that round
trip exists precisely to match the unfused store-then-rotate reference.  ``gate_mul``
follows whichever baseline branch it replaces: aten's ``out * torch.sigmoid(gate)``
rounds the sigmoid to the storage dtype and then rounds the product again, while the
baseline's Triton ``_gate_mul_inplace`` keeps the sigmoid in fp32 through the multiply.
The two baseline branches therefore already disagree slightly; picking one unconditionally
would silently change decode.

What is *not* claimed is bit-exactness.  The variance reduction here is a warp shuffle
butterfly where Triton's is a block tree, ``rsqrt.approx.f32`` is reached through inline
PTX rather than through Triton's ``ftz`` variant, and FMA contraction in the rotation is
left to the compiler.  Each perturbs an fp32 intermediate by about 1e-7 relative, which
is four orders of magnitude below bf16's 2^-8 quantum -- but a value sitting on a bf16
rounding midpoint can still round the other way, so the honest claim is a measured
tolerance claim against the baseline composition, not equality.  ``tests/`` measures it.

Preconditions the fast path checks, on every call, because the bench's ``_ShiftingPool``
hands every contiguous input a different ``data_ptr`` on each iteration and no
per-pointer answer stays valid:

* the extension built, the trtllm/HND backend selected, and CUDA tensors throughout;
* a 2-byte float dtype, ``head_dim`` a multiple of 8 and no wider than 256 (one warp
  covers one head row with one 16-byte access per lane, so a row is at most 32 units);
* ``rotary_dim`` even, no wider than ``head_dim``, and a multiple of 16 whose half is a
  power-of-two number of lanes -- that last clause is not decoration.  The NeoX pairing
  (i, i + rotary_dim/2) crosses lanes, and it is exchanged with ``__shfl_xor_sync`` at a
  fixed distance, which only equals "+ half" when that distance is a power of two.  A
  geometry outside it is refused rather than mis-rotated;
* an HND cache whose trailing shape is exactly ``(num_kv_heads, page_size, head_dim)``,
  contiguous, distinct from the value cache, in the input dtype;
* unit innermost strides and 16-byte-aligned bases on ``qkv``, both caches and
  ``cos_sin_cache``, plus a ``qkv`` row stride that keeps every head's 16-byte units
  aligned;
* ``positions`` and ``slot_mapping`` int32 or int64, contiguous, long enough for the
  token count;
* prefill metadata whose ``query_start_loc`` carries exactly ``num_decodes +
  num_prefills + 1`` entries, so the cumulative vectors this module builds hold the same
  values the baseline's slicing would produce.

Preconditions it deliberately does *not* check, because the baseline does not either and
checking would need a device read: that every non-negative slot is in range, and that
slots are distinct.  A positive out-of-range slot is an out-of-bounds device write here
exactly as it is in the frozen store; duplicates are a data race in both.  That is
inherited undefined behaviour, not a new contract.  ``slot_mapping`` shorter than the
token count is refused by the predicate so that the frozen store raises its own error,
which keeps the observable failure identical.

Anything the predicate rejects runs the baseline composition verbatim, and ``LAST_PATH``
records what served each stage of the last call, so "the fast path never ran" is
distinguishable from "the fast path ran and tied".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ....infra.context import get_attn_backend_config, get_context
from ....infra.tp import _tp_size
from ..L1.flash_attn_decode import FlashAttnDecode
from ..L1.flash_attn_prefill import FlashAttnPrefill
from ..L1.gemma_rms_norm import GemmaRMSNorm
from ..L1.store_kvcache import StoreKVCacheHND
from .fused_qk_norm_rope import (
    fused_qk_rmsnorm_rope_gate as _vllm_fused_qk_rmsnorm_rope_gate,
)
from .parallel_linear import QKVParallelLinear, RowParallelLinear

try:
    from ..L1.store_kvcache import StoreKVCache
except ImportError:  # pragma: no cover -- non-trtllm backends only
    # The frozen L1 winner replaces the HND store only; the NHD store it does
    # not implement stays the baseline's, and this module reaches it only on a
    # device that selects the non-trtllm backend.
    from ....tasks.baseline.L1.store_kvcache import StoreKVCache

_WORKSPACE = Path(__file__).resolve().parents[2]

#: Where the inline extension is built.  Workspace-local, so sibling operator
#: workspaces never share a directory; concurrent bench workers inside this
#: workspace are serialised by ``load_inline``'s own file lock.
EXTENSION_DIR = _WORKSPACE / ".torch_extensions"

#: ``"cuda"`` once the extension is loaded, ``"torch"`` if it could not be built.
BACKEND = "torch"
#: Why the build failed, if it did.  Kept rather than swallowed: a fallback scores
#: about 1.00x, which is exactly what a working candidate that ties would score.
BUILD_ERROR: str | None = None
#: The ``TORCH_CUDA_ARCH_LIST`` the build actually saw.
ARCH_LIST_USED: str | None = None
#: The ambient value, restored after the build.  This environment names six
#: architectures, which would otherwise be compiled six times over.
AMBIENT_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", "__unset__")

#: What served each stage of the most recent call.  Per stage rather than one
#: coarse label, because a partial fallback in the metadata or the gate would
#: otherwise hide behind a "fused" verdict on the kernel that did run.
#:   qk_norm_rope_store: "fused" | "reference"
#:   prefill_cu_seqlens: "fused" | "aten" | "none" (decode-only) | "reference"
#:   gate_mul:           "fused" | "reference"
#:   attn_output:        "direct" | "staged" | "reference"
LAST_PATH: dict[str, str] = {
    "qk_norm_rope_store": "none",
    "prefill_cu_seqlens": "none",
    "gate_mul": "none",
    "attn_output": "none",
}
#: Which precondition sent the last call to the baseline composition, or ``None``
#: when the fast path served it.
LAST_REJECTION: str | None = None

_EXT_MODULE_NAME = "qwen3_next_attention_fused_ext"

_LANE_ELEMS = 8       # a 16-byte unit holds eight 2-byte elements
_WARP = 32
#: Head rows per block in the fusion kernel.  The scored geometry has 20 rows
#: (16 q + 2 k + 2 v), so 4 divides it exactly into four pure-q blocks and one
#: k/k/v/v block, with no idle warp.  Swept over {1, 2, 4, 5, 10, 20} on the only
#: shape where this kernel's device time is not hidden behind host cost.  At
#: N=16384, 1155 MHz SM clock, it measures 351.5 / 257.5 / 251.4 / 255.3 / 273.7 /
#: 304.7 us -- reproducible to 0.1 us across three passes -- so 4 is the minimum
#: and both wings are real: 1 starves the SM of warps, 20 puts 640 threads in a
#: block and loses occupancy.  The clock is stated because it is not locked here
#: and moves absolute numbers about 1.7x between GPU leases; the *ordering* held
#: on every lease measured.  On the host-bound shapes every value lands within a
#: microsecond of every other, as expected of a kernel that is 3 us of a 66 us
#: call there.
_ROWS_PER_BLOCK = 4
#: A row is one warp holding one 16-byte unit per lane, so a row is at most 32
#: units wide.
_MAX_HEAD_DIM = _LANE_ELEMS * _WARP
#: The metadata scan is one block, so the prefill count is bounded by the largest
#: block that can hold ``num_prefills + 1`` threads.  Above it, the aten form runs.
_SCAN_MAX_PREFILLS = 1023

_INT_DTYPES = (torch.int32, torch.int64)
_HALF_DTYPES = (torch.bfloat16, torch.float16)


_CPP_SOURCE = r"""
#include <torch/extension.h>

void qk_norm_rope_store(const at::Tensor& qkv, at::Tensor q_out,
                        at::Tensor k_cache, at::Tensor v_cache,
                        const at::Tensor& q_gain, const at::Tensor& k_gain,
                        const at::Tensor& cos_sin_cache,
                        const at::Tensor& positions,
                        const at::Tensor& slot_mapping, double eps,
                        int64_t num_q_heads, int64_t num_kv_heads,
                        int64_t head_dim, int64_t rotary_dim,
                        int64_t page_size, int64_t rows_per_block);

void gate_mul(at::Tensor out, const at::Tensor& qkv, int64_t num_q_heads,
              int64_t head_dim, int64_t aten_rounding);

void prefill_cu_seqlens(const at::Tensor& seq_lens,
                        const at::Tensor& query_start_loc, at::Tensor cu_k,
                        at::Tensor cu_q, int64_t num_prefills,
                        int64_t decode_offset, int64_t write_cu_q);
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <limits>

namespace {

using Unit = uint4;                      // one 128-bit access
constexpr int kLaneElems = 8;            // ... of 2-byte elements
constexpr int kWarp = 32;
constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kGateThreads = 256;
constexpr int kMaxScanThreads = 1024;

union Pack {
  Unit u4;
  uint16_t h[kLaneElems];
};

// Storage-dtype conversion kept behind a trait so the kernel body is written once.
// Both directions are round-to-nearest-even, which is what Triton's
// ``.to(tl.bfloat16)`` / ``.to(tl.float16)`` emit, so the rounding the baseline
// applies mid-computation is reproduced rather than approximated.
struct BF16 {
  __device__ __forceinline__ static float up(uint16_t r) {
    return __bfloat162float(__ushort_as_bfloat16(r));
  }
  __device__ __forceinline__ static uint16_t down(float v) {
    return __bfloat16_as_ushort(__float2bfloat16(v));
  }
};

struct FP16 {
  __device__ __forceinline__ static float up(uint16_t r) {
    return __half2float(__ushort_as_half(r));
  }
  __device__ __forceinline__ static uint16_t down(float v) {
    return __half_as_ushort(__float2half(v));
  }
};

// Triton lowers ``tl.rsqrt`` to ``rsqrt.approx.ftz.f32``.  Reaching the same
// instruction through inline PTX keeps the reciprocal square root on the baseline's
// approximation rather than on nvcc's refined ``rsqrtf``, which would be a second,
// avoidable source of divergence.  The ``ftz`` qualifier is left off: it only
// matters when ``var + eps`` is subnormal, which needs eps == 0, and then this form
// is the more accurate of the two.
__device__ __forceinline__ float rsqrt_approx(float x) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// A streaming read that is used exactly once.  ``.nc`` is the non-coherent path
// ``__ldg`` uses -- safe because everything read here was written by an earlier
// kernel, so L1 has already been invalidated.  ``.L1::no_allocate`` additionally
// keeps single-use payload from evicting the gain and cos/sin lines, which every
// block re-reads.
__device__ __forceinline__ Unit ld_stream(const Unit* p) {
  Unit r;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p));
  return r;
}

// The counterpart to ``ld_stream`` for data that is read once *per block* but
// re-read by every block in the grid: the RMSNorm gain vectors and the row's
// cos/sin slice.  ``L1::evict_last`` keeps those lines resident while the
// single-use payload streams past under ``L1::no_allocate``, which is the cache
// policy differentiation KernelWiki ``technique-vectorized-loads`` calls for on a
// memory-bound kernel with two access patterns.  It is also the fix NCU pointed
// at: 60% of this kernel's long-scoreboard stall sat on these two reads.
// Measured 1.27x on the kernel at N=16384, bit-identical output.  It does spend
// resources -- the v4 loads take registers from 35 to 47 and resident warps from
// 69.0% to 55.9% of peak -- and wins anyway, where hoisting the same two reads
// into registers spent registers and lost 9%.  The difference is what the
// registers buy: cache residency the kernel had none of, rather than
// memory-level parallelism it already had from 46 waves of thread parallelism.
//
// v4.f32 needs a 16-byte-aligned address.  That holds structurally rather than by
// luck: the gain base is checked 16-byte aligned and lane offsets are multiples of
// eight floats, and ``cos_sin`` is contiguous with a row of ``rotary_dim`` floats
// where the predicate forces ``rotary_dim % 16 == 0``, so every row start, every
// eight-float lane offset and ``half_rotary`` are all multiples of four floats.
__device__ __forceinline__ float4 ld_reuse4(const float* p) {
  float4 r;
  asm("ld.global.nc.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
      : "l"(p));
  return r;
}

__device__ __forceinline__ void ld_reuse8(const float* p, float* out) {
  const float4 a = ld_reuse4(p);
  const float4 b = ld_reuse4(p + 4);
  out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
  out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}

__device__ __forceinline__ int64_t read_index(const void* p, int is_int64,
                                             int64_t i) {
  return is_int64 ? static_cast<const int64_t*>(p)[i]
                  : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

// One warp owns one head row; one lane owns one 16-byte unit of it.  A row is
// therefore read and written as a single coalesced transaction, and the whole
// grid is (token, row block).
//
// Rows are laid out q first, then k, then v, so a block of ``rows_per_block``
// consecutive rows is usually all one kind and needs only one of the two gain
// vectors.  ``threadIdx.y`` selects the row and ``blockDim.x`` is exactly one
// warp, so ``row`` -- and hence every early return below -- is warp-uniform.
// That is what makes the full-warp shuffle mask honest: any warp that reaches the
// exchange has all 32 lanes present.
template <typename Codec>
__global__ void qk_norm_rope_store_kernel(
    const Unit* __restrict__ q_gate, const Unit* __restrict__ k_src,
    const Unit* __restrict__ v_src, Unit* __restrict__ q_out,
    Unit* __restrict__ k_cache, Unit* __restrict__ v_cache,
    const float* __restrict__ q_gain, const float* __restrict__ k_gain,
    const float* __restrict__ cos_sin, const void* __restrict__ positions,
    const void* __restrict__ slot_mapping, const int64_t qkv_row_units,
    const int64_t q_out_row_units, const int64_t cos_sin_row_floats,
    const int units_per_row, const int num_q_heads, const int num_kv_heads,
    const int head_dim, const int rotary_dim, const int half_rotary,
    const int lane_dist, const float eps, const int rows_per_block,
    const int num_rows, const int page_size, const int page_shift,
    const int pos_is_int64, const int slot_is_int64) {
  const int row = blockIdx.y * rows_per_block + threadIdx.y;
  if (row >= num_rows) return;
  const int token = blockIdx.x;
  const int lane = threadIdx.x;
  // Narrower rows than a full warp leave the tail lanes with nothing to move.
  // They stay in the warp for the reduction and the shuffle -- contributing zero
  // and consuming nothing -- rather than returning, which would break the mask.
  const bool lane_holds_data = lane < units_per_row;

  int kind;  // 0 = q, 1 = k, 2 = v
  int local;
  const Unit* src;
  if (row < num_q_heads) {
    kind = 0;
    local = row;
    src = q_gate + token * qkv_row_units +
          static_cast<int64_t>(local) * 2 * units_per_row;
  } else if (row < num_q_heads + num_kv_heads) {
    kind = 1;
    local = row - num_q_heads;
    src = k_src + token * qkv_row_units +
          static_cast<int64_t>(local) * units_per_row;
  } else {
    kind = 2;
    local = row - num_q_heads - num_kv_heads;
    src = v_src + token * qkv_row_units +
          static_cast<int64_t>(local) * units_per_row;
  }

  Pack in;
  if (lane_holds_data) in.u4 = ld_stream(src + lane);

  // The paged destination, needed by k and v rows only.  A negative slot is a
  // padded or unscheduled token: it is skipped, exactly as the frozen store skips
  // it, and the token's q output is still produced.
  int64_t cache_unit = 0;
  bool store_to_cache = false;
  if (kind != 0) {
    const int64_t slot = read_index(slot_mapping, slot_is_int64, token);
    if (slot >= 0) {
      int64_t block, sibling;
      if (page_shift >= 0) {
        block = slot >> page_shift;
        sibling = slot & (static_cast<int64_t>(page_size) - 1);
      } else {
        block = slot / page_size;
        sibling = slot - block * page_size;
      }
      // 64-bit throughout: num_blocks * num_kv_heads * page_size * head_dim
      // passes 2^31 well inside what the contract allows for num_blocks.
      cache_unit = ((block * num_kv_heads + local) * page_size + sibling) *
                   static_cast<int64_t>(units_per_row);
      store_to_cache = true;
    }
  }

  if (kind == 2) {
    // V is stored raw: the baseline hands the store a view straight out of qkv,
    // with no norm and no rotation.
    if (store_to_cache && lane_holds_data) v_cache[cache_unit + lane] = in.u4;
    return;
  }

  float xf[kLaneElems];
  float partial = 0.0f;
  if (lane_holds_data) {
#pragma unroll
    for (int j = 0; j < kLaneElems; ++j) {
      xf[j] = Codec::up(in.h[j]);
      partial += xf[j] * xf[j];
    }
  } else {
#pragma unroll
    for (int j = 0; j < kLaneElems; ++j) xf[j] = 0.0f;
  }

  // A shuffle butterfly over the whole warp: no shared memory, no barrier, and
  // every lane leaves holding the total.  The summation order differs from
  // Triton's block tree, which is the deviation the module docstring names.
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    partial += __shfl_xor_sync(kFullWarp, partial, offset);
  }
  const float inv_rms = rsqrt_approx(partial / static_cast<float>(head_dim) + eps);

  // The gain vectors are 1 KiB re-read by every block in the grid, so they are
  // left in global memory and served out of L1/L2 rather than staged through
  // shared memory: the barrier that staging needs would have to be reached by
  // warps that have already returned above, and the traffic it saves is on-chip
  // request bandwidth only.
  const float* gain = (kind == 0) ? q_gain : k_gain;

  // Round to the storage dtype and widen back before rotating.  This is not an
  // optimisation to skip: it is what the vendored kernel does, and the round trip
  // exists there to match the unfused store-then-rotate reference.
  uint16_t xr[kLaneElems];
  float xn[kLaneElems];
  if (lane_holds_data) {
    float gv[kLaneElems];
    ld_reuse8(gain + lane * kLaneElems, gv);
#pragma unroll
    for (int j = 0; j < kLaneElems; ++j) {
      xr[j] = Codec::down(xf[j] * inv_rms * gv[j]);
      xn[j] = Codec::up(xr[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < kLaneElems; ++j) {
      xr[j] = 0;
      xn[j] = 0.0f;
    }
  }

  // NeoX pairs element i with element i + half_rotary.  With kLaneElems elements
  // per lane and half_rotary a multiple of kLaneElems, the partner of lane L's
  // element j is lane (L + lane_dist)'s element j -- and because lane_dist is a
  // power of two and the rotary region starts at lane 0, XOR gives that partner
  // in both directions.  Every lane executes the exchange so the mask covers the
  // whole warp; only the rotary lanes consume the result.
  //
  // Both the gain and the cos/sin slice are read at their point of use, not
  // hoisted above the reduction.  NCU attributes 52% of this kernel's
  // long-scoreboard stall to the cos/sin loads here and 8% to the gain, so
  // hoisting them into registers to overlap with the reduction is the obvious
  // move -- and measured, it is a regression where it counts.  The cache policy on
  // those same two reads (``ld_reuse8`` above) is what actually fixed the stall;
  // it spends registers too, and still wins.  Both variants
  // built into one process and benchmarked interleaved on one device (the only
  // fair way here: unlocked clocks move absolute numbers 1.7x between leases)
  // give bit-identical output and, at N=16384 / R=4, 251.4 us point-of-use
  // against 275.9 us hoisted.  The 24 extra live registers cost more occupancy
  // than the added memory-level parallelism buys, so this kernel's latency is
  // covered better by resident warps than by per-warp ILP.  The hoist does win
  // 1.19x at N=60 (11.7 us -> 9.8 us), which changes nothing: there the kernel is
  // 3 us of a 66 us host-bound call and its device time is entirely hidden.
  float partner[kLaneElems];
#pragma unroll
  for (int j = 0; j < kLaneElems; ++j) {
    partner[j] = __shfl_xor_sync(kFullWarp, xn[j], lane_dist);
  }

  Pack out;
  const int first_elem = lane * kLaneElems;
  if (first_elem < rotary_dim) {
    // Both partner lanes read the *same* cos/sin slice: element index
    // kLaneElems * (lane mod lane_dist) + j, never the cache's sine half by
    // accident.  ``lane & (lane_dist - 1)`` is that modulo, lane_dist being a
    // power of two.
    const float* cs = cos_sin + read_index(positions, pos_is_int64, token) *
                                    cos_sin_row_floats;
    const float* cos_p = cs + (lane & (lane_dist - 1)) * kLaneElems;
    float cv[kLaneElems];
    float sv[kLaneElems];
    ld_reuse8(cos_p, cv);
    ld_reuse8(cos_p + half_rotary, sv);
    if (lane < lane_dist) {
#pragma unroll
      for (int j = 0; j < kLaneElems; ++j) {
        out.h[j] = Codec::down(xn[j] * cv[j] - partner[j] * sv[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kLaneElems; ++j) {
        out.h[j] = Codec::down(xn[j] * cv[j] + partner[j] * sv[j]);
      }
    }
  } else {
    // Pass-through tail: normalized, unrotated.  ``xr`` is already the rounded
    // storage value, so nothing is rounded twice.
#pragma unroll
    for (int j = 0; j < kLaneElems; ++j) out.h[j] = xr[j];
  }

  if (kind == 0) {
    if (lane_holds_data) {
      q_out[token * q_out_row_units +
            static_cast<int64_t>(local) * units_per_row + lane] = out.u4;
    }
  } else if (store_to_cache && lane_holds_data) {
    k_cache[cache_unit + lane] = out.u4;
  }
}

// ``out *= sigmoid(gate)``, with the gate read strided in place out of qkv at head
// stride 2 * head_dim and offset head_dim.  A head's gate is still a contiguous
// head_dim run, so nothing about coalescing changes; what disappears is the
// separate gate buffer the Triton baseline writes and immediately reads back.
//
// ``AtenRounding`` selects which baseline branch is being reproduced.  With
// prefills present the baseline is aten -- ``sigmoid`` rounds to the storage dtype
// and ``mul`` rounds the product again -- and decode-only is the baseline's own
// Triton kernel, which keeps the sigmoid in fp32 through the multiply.
template <typename Codec, bool AtenRounding>
__global__ void gate_mul_kernel(Unit* __restrict__ out,
                                const Unit* __restrict__ gate,
                                const int64_t out_row_units,
                                const int64_t qkv_row_units,
                                const int units_per_row, const int num_q_heads,
                                const int heads_per_block) {
  const int head = blockIdx.y * heads_per_block + threadIdx.y;
  if (head >= num_q_heads) return;
  const int unit = threadIdx.x;
  const int64_t token = blockIdx.x;

  Unit* dst = out + token * out_row_units +
              static_cast<int64_t>(head) * units_per_row + unit;
  Pack o;
  o.u4 = *dst;
  Pack g;
  g.u4 = ld_stream(gate + token * qkv_row_units +
                   static_cast<int64_t>(head) * 2 * units_per_row + unit);

#pragma unroll
  for (int j = 0; j < kLaneElems; ++j) {
    float s = 1.0f / (1.0f + expf(-Codec::up(g.h[j])));
    if (AtenRounding) s = Codec::up(Codec::down(s));
    o.h[j] = Codec::down(Codec::up(o.h[j]) * s);
  }
  *dst = o.u4;
}

// ``cu_k = [0, cumsum(seq_lens[nd:])]`` and, when there are decodes to skip,
// ``cu_q = query_start_loc[nd:] - query_start_loc[nd]`` -- in one launch instead of
// the baseline's ``sub`` + ``zeros`` + ``cumsum`` + slice-assign.
//
// The accumulator is 64-bit and the store truncates to int32, which is what the
// baseline does: ``torch.cumsum`` promotes an int32 input to int64 and only the
// assignment into the int32 destination narrows it.  ``seq_lens`` is truncated to
// int32 *before* accumulating, because the baseline's ``.to(torch.int32)`` runs
// before its cumsum.
__global__ void prefill_cu_seqlens_kernel(
    const void* __restrict__ seq_lens, const int seq_is_int64,
    const void* __restrict__ query_start_loc, const int qsl_is_int64,
    int32_t* __restrict__ cu_k, int32_t* __restrict__ cu_q,
    const int num_prefills, const int decode_offset, const int write_cu_q,
    const int nthreads) {
  extern __shared__ int64_t scan[];
  const int t = threadIdx.x;
  int64_t v = 0;
  if (t < num_prefills) {
    v = static_cast<int64_t>(static_cast<int32_t>(
        read_index(seq_lens, seq_is_int64, decode_offset + t)));
  }
  scan[t] = v;
  __syncthreads();
  // Hillis-Steele inclusive scan.  Every thread runs every step, so the two
  // barriers per step separate the read of a neighbour from the write over it.
  for (int offset = 1; offset < nthreads; offset <<= 1) {
    const int64_t add = (t >= offset) ? scan[t - offset] : static_cast<int64_t>(0);
    __syncthreads();
    scan[t] += add;
    __syncthreads();
  }
  if (t == 0) cu_k[0] = 0;
  if (t < num_prefills) cu_k[t + 1] = static_cast<int32_t>(scan[t]);
  if (write_cu_q && t <= num_prefills) {
    const int64_t here =
        read_index(query_start_loc, qsl_is_int64, decode_offset + t);
    const int64_t base = read_index(query_start_loc, qsl_is_int64, decode_offset);
    cu_q[t] = static_cast<int32_t>(here - base);
  }
}

}  // namespace


void qk_norm_rope_store(const at::Tensor& qkv, at::Tensor q_out,
                        at::Tensor k_cache, at::Tensor v_cache,
                        const at::Tensor& q_gain, const at::Tensor& k_gain,
                        const at::Tensor& cos_sin_cache,
                        const at::Tensor& positions,
                        const at::Tensor& slot_mapping, double eps,
                        int64_t num_q_heads, int64_t num_kv_heads,
                        int64_t head_dim, int64_t rotary_dim,
                        int64_t page_size, int64_t rows_per_block) {
  const int64_t num_tokens = q_out.size(0);
  const int64_t units_per_row = head_dim / kLaneElems;
  const int64_t half_rotary = rotary_dim / 2;
  const int64_t lane_dist = half_rotary / kLaneElems;
  const int64_t num_rows = num_q_heads + 2 * num_kv_heads;

  // Invariants of the lane geometry rather than of the input: the host predicate
  // has already refused anything that fails them, so reaching one is a bug here.
  TORCH_CHECK(num_tokens > 0, "no tokens");
  TORCH_CHECK(qkv.element_size() == 2, "qkv must be a 2-byte float dtype");
  TORCH_CHECK(head_dim % kLaneElems == 0 && units_per_row >= 1 &&
                  units_per_row <= kWarp,
              "head_dim ", head_dim, " is not 1..", kWarp, " units of ",
              kLaneElems);
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0,
              "rotary_dim ", rotary_dim, " out of range for head_dim ", head_dim);
  TORCH_CHECK(half_rotary % kLaneElems == 0 && lane_dist >= 1 &&
                  (lane_dist & (lane_dist - 1)) == 0,
              "rotary_dim ", rotary_dim,
              " does not map onto a power-of-two lane distance");
  TORCH_CHECK(page_size >= 1 && page_size <= std::numeric_limits<int>::max(),
              "page_size out of range: ", page_size);
  TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(), "too many tokens");
  TORCH_CHECK(rows_per_block >= 1, "rows_per_block must be positive");

  const at::cuda::CUDAGuard guard(qkv.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  const auto* base = static_cast<const uint16_t*>(qkv.data_ptr());
  const auto* k_base = base + num_q_heads * 2 * head_dim;
  const auto* v_base = k_base + num_kv_heads * head_dim;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(base) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(k_base) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(v_base) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(q_out.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(k_cache.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(v_cache.data_ptr()) % 16 == 0,
              "a 16-byte access needs 16-byte-aligned bases");
  TORCH_CHECK(qkv.stride(0) % kLaneElems == 0,
              "qkv row stride ", qkv.stride(0), " breaks 16-byte alignment");

  int page_shift = -1;
  if ((page_size & (page_size - 1)) == 0) {
    page_shift = 0;
    for (int64_t p = page_size; p > 1; p >>= 1) ++page_shift;
  }

  const int rpb = static_cast<int>(std::min<int64_t>(rows_per_block, num_rows));
  const dim3 block(kWarp, static_cast<unsigned>(rpb));
  const dim3 grid(static_cast<unsigned>(num_tokens),
                  static_cast<unsigned>((num_rows + rpb - 1) / rpb));
  const int pos_is_int64 = positions.scalar_type() == at::kLong;
  const int slot_is_int64 = slot_mapping.scalar_type() == at::kLong;

#define LAUNCH_FUSION(Codec)                                                   \
  qk_norm_rope_store_kernel<Codec><<<grid, block, 0, stream>>>(                \
      reinterpret_cast<const Unit*>(base),                                    \
      reinterpret_cast<const Unit*>(k_base),                                  \
      reinterpret_cast<const Unit*>(v_base),                                  \
      reinterpret_cast<Unit*>(q_out.data_ptr()),                              \
      reinterpret_cast<Unit*>(k_cache.data_ptr()),                            \
      reinterpret_cast<Unit*>(v_cache.data_ptr()),                            \
      q_gain.data_ptr<float>(), k_gain.data_ptr<float>(),                     \
      cos_sin_cache.data_ptr<float>(), positions.data_ptr(),                  \
      slot_mapping.data_ptr(), qkv.stride(0) / kLaneElems,                    \
      q_out.stride(0) / kLaneElems, cos_sin_cache.stride(0),                  \
      static_cast<int>(units_per_row), static_cast<int>(num_q_heads),          \
      static_cast<int>(num_kv_heads), static_cast<int>(head_dim),              \
      static_cast<int>(rotary_dim), static_cast<int>(half_rotary),             \
      static_cast<int>(lane_dist), static_cast<float>(eps), rpb,               \
      static_cast<int>(num_rows), static_cast<int>(page_size), page_shift,     \
      pos_is_int64, slot_is_int64)

  if (qkv.scalar_type() == at::kBFloat16) {
    LAUNCH_FUSION(BF16);
  } else if (qkv.scalar_type() == at::kHalf) {
    LAUNCH_FUSION(FP16);
  } else {
    TORCH_CHECK(false, "unsupported dtype ", qkv.scalar_type());
  }
#undef LAUNCH_FUSION
  AT_CUDA_CHECK(cudaGetLastError());
}


void gate_mul(at::Tensor out, const at::Tensor& qkv, int64_t num_q_heads,
              int64_t head_dim, int64_t aten_rounding) {
  const int64_t num_tokens = out.size(0);
  const int64_t units_per_row = head_dim / kLaneElems;
  TORCH_CHECK(num_tokens > 0, "no tokens");
  TORCH_CHECK(out.element_size() == 2 && qkv.element_size() == 2,
              "gate_mul needs 2-byte float tensors");
  TORCH_CHECK(head_dim % kLaneElems == 0 && units_per_row >= 1 &&
                  units_per_row <= kGateThreads,
              "head_dim ", head_dim, " does not fit the block geometry");
  TORCH_CHECK(num_tokens <= std::numeric_limits<int>::max(), "too many tokens");

  const at::cuda::CUDAGuard guard(out.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  const auto* gate_base =
      static_cast<const uint16_t*>(qkv.data_ptr()) + head_dim;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(gate_base) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "a 16-byte access needs 16-byte-aligned bases");
  TORCH_CHECK(qkv.stride(0) % kLaneElems == 0 && out.stride(0) % kLaneElems == 0,
              "row strides break 16-byte alignment");

  const int heads_per_block = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(num_q_heads, kGateThreads / units_per_row)));
  const dim3 block(static_cast<unsigned>(units_per_row),
                   static_cast<unsigned>(heads_per_block));
  const dim3 grid(
      static_cast<unsigned>(num_tokens),
      static_cast<unsigned>((num_q_heads + heads_per_block - 1) / heads_per_block));

#define LAUNCH_GATE(Codec, ATEN)                                              \
  gate_mul_kernel<Codec, ATEN><<<grid, block, 0, stream>>>(                    \
      reinterpret_cast<Unit*>(out.data_ptr()),                                \
      reinterpret_cast<const Unit*>(gate_base), out.stride(0) / kLaneElems,   \
      qkv.stride(0) / kLaneElems, static_cast<int>(units_per_row),            \
      static_cast<int>(num_q_heads), heads_per_block)

  if (out.scalar_type() == at::kBFloat16) {
    if (aten_rounding) { LAUNCH_GATE(BF16, true); }
    else               { LAUNCH_GATE(BF16, false); }
  } else if (out.scalar_type() == at::kHalf) {
    if (aten_rounding) { LAUNCH_GATE(FP16, true); }
    else               { LAUNCH_GATE(FP16, false); }
  } else {
    TORCH_CHECK(false, "unsupported dtype ", out.scalar_type());
  }
#undef LAUNCH_GATE
  AT_CUDA_CHECK(cudaGetLastError());
}


void prefill_cu_seqlens(const at::Tensor& seq_lens,
                        const at::Tensor& query_start_loc, at::Tensor cu_k,
                        at::Tensor cu_q, int64_t num_prefills,
                        int64_t decode_offset, int64_t write_cu_q) {
  TORCH_CHECK(num_prefills >= 1 && num_prefills < kMaxScanThreads,
              "num_prefills ", num_prefills, " is outside the single-block scan");
  TORCH_CHECK(cu_k.numel() == num_prefills + 1, "cu_k must hold num_prefills + 1");
  TORCH_CHECK(!write_cu_q || cu_q.numel() == num_prefills + 1,
              "cu_q must hold num_prefills + 1");
  TORCH_CHECK(cu_k.scalar_type() == at::kInt &&
                  (!write_cu_q || cu_q.scalar_type() == at::kInt),
              "the cumulative vectors must be int32");

  int nthreads = 32;
  while (nthreads < num_prefills + 1) nthreads <<= 1;

  const at::cuda::CUDAGuard guard(cu_k.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  prefill_cu_seqlens_kernel<<<1, nthreads, nthreads * sizeof(int64_t), stream>>>(
      seq_lens.data_ptr(), seq_lens.scalar_type() == at::kLong,
      query_start_loc.data_ptr(), query_start_loc.scalar_type() == at::kLong,
      cu_k.data_ptr<int32_t>(), cu_q.data_ptr<int32_t>(),
      static_cast<int>(num_prefills), static_cast<int>(decode_offset),
      static_cast<int>(write_cu_q), nthreads);
  AT_CUDA_CHECK(cudaGetLastError());
}
"""


def _target_arch() -> str:
    """The single architecture to compile for.

    Asks the visible device so the extension is never built for an architecture it
    will not run on; falls back to Blackwell when no device is visible, which is
    the case for a plain import outside a GPU lease.
    """
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}.{minor}"
    except Exception:  # noqa: BLE001 -- a probe failure must not stop the build
        pass
    return "10.0"


def _build_extension():
    """Compile at import, so ``forward`` has no build branch to take.

    The bench's no-new-threads guard wraps only the candidate's timing loop, and
    its stall watchdog allows 600 s, so a build here is both invisible to the
    measurement and comfortably inside the budget -- but only because the
    architecture is pinned to the visible device.  The ambient list names six,
    which would be compiled six times over.
    """
    global BACKEND, BUILD_ERROR, ARCH_LIST_USED

    cached = sys.modules.get(_EXT_MODULE_NAME)
    if cached is not None:
        BACKEND = "cuda"
        ARCH_LIST_USED = getattr(cached, "_arch_list_used", _target_arch())
        return cached

    from torch.utils.cpp_extension import load_inline

    arch = _target_arch()
    previous_extensions_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    EXTENSION_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(EXTENSION_DIR)
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        module = load_inline(
            name=_EXT_MODULE_NAME,
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["qk_norm_rope_store", "gate_mul", "prefill_cu_seqlens"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, then asserted on by the gate
        BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    else:
        BACKEND = "cuda"
        ARCH_LIST_USED = arch
        module._arch_list_used = arch
        sys.modules[_EXT_MODULE_NAME] = module
        return module
    finally:
        if AMBIENT_ARCH_LIST == "__unset__":
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = AMBIENT_ARCH_LIST
        if previous_extensions_dir is None:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
        else:
            os.environ["TORCH_EXTENSIONS_DIR"] = previous_extensions_dir


_ext = _build_extension()


#: Raw ``cudaStream_t`` of the current stream, as an int.  The public
#: ``torch.cuda.current_stream(device).stream_id`` says the same thing but
#: constructs a ``Stream`` object to do it, which measures 1.97 us here against
#: 0.051 us -- 3% of the whole host budget this module is trying to reach, spent
#: on a value used only as a dictionary key.  This is the accessor Inductor's own
#: generated code uses, so it is stable in practice; the public form is kept as a
#: fallback in case a build does not expose it.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_stream_id(device: torch.device) -> int:
    """Identity of the stream the caller is queueing on.

    A host-side query on the caller's own stream: it queues nothing and
    synchronises nothing.  Two *live* streams always have distinct handles, which
    is what the scratch key needs.

    The one gap, named rather than papered over: ``cudaStreamDestroy`` returns
    without waiting for queued work, so a stream destroyed with work in flight can
    have its handle recycled by a new stream while the old work still runs, and
    the new stream would inherit the old stream's scratch entry without any
    ordering between them.  This is unreachable through torch's own API -- pooled
    streams from ``torch.cuda.Stream`` are not destroyed for the life of the
    process -- and using ``stream_id`` instead would not fix it, since that id
    tracks the same handle lifecycle.  It would take an externally created stream,
    destroyed non-empty, to reach.
    """
    if _raw_stream is not None:
        return _raw_stream(device.index or 0)
    return torch.cuda.current_stream(device).stream_id


def _is_dense_rows(t: torch.Tensor, shape: tuple[int, int, int],
                   dtype: torch.dtype) -> bool:
    """Is *t* exactly the buffer the baseline would have allocated and copied into?

    Checked on the tensor's observed dtype, shape, strides and alignment rather
    than on the assumption that every route through the frozen attention wrapper
    returns the layout this one was measured to return.
    """
    return (t is not None and t.dtype is dtype and tuple(t.shape) == shape
            and t.stride() == (shape[1] * shape[2], shape[2], 1)
            and t.data_ptr() % 16 == 0)


@triton.jit
def _gate_mul_inplace_kernel(
    out_ptr,
    gate_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    out = tl.load(out_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    gate = 1.0 / (1.0 + tl.exp(-gate))
    tl.store(out_ptr + offsets, out * gate, mask=mask)


def _gate_mul_inplace(out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block = 1024
    _gate_mul_inplace_kernel[(triton.cdiv(n_elements, block),)](
        out,
        gate,
        n_elements,
        BLOCK=block,
    )
    return out


class Qwen3NextAttention(nn.Module):
    """Full attention with per-head QK-norm, partial RoPE, output gating, and KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int,
        rms_norm_eps: float = 1e-6,
        reduce_output: bool = True,
    ):
        super().__init__()
        tp = _tp_size()
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads // tp
        self.num_kv_heads = num_key_value_heads // tp if num_key_value_heads % tp == 0 else num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        # QKV projection: Q outputs 2x heads (Q + gate)
        self.qkv_proj = QKVParallelLinear(
            hidden_size, head_dim,
            num_attention_heads * 2,  # doubled for output gate
            num_key_value_heads,
        )

        # ``reduce_output=False`` defers the all-reduce to the decoder layer's
        # next norm, which fuses the two.
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim, hidden_size,
            reduce_results=reduce_output,
        )

        # Per-head QK norms (GemmaRMSNorm)
        self.q_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=rms_norm_eps)

        # Qwen3-Next's full-attention layers use head_dim=256.  vLLM 0.26 runs
        # them on FlashInfer with an HND cache, so follow the same per-device
        # backend selection the generic ``Attention`` layer uses.  FlashAttention
        # is not a substitute here: FA4's SM100 head_dim=256 forward rejects
        # seqused_k/seqused_q, which the paged decode path requires.
        attn_cfg = get_attn_backend_config()
        self.kv_layout = attn_cfg.kv_layout
        self._use_trtllm = attn_cfg.use_trtllm
        self._fused_qk_rope_gate = True
        self._norm_gain_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._use_custom_op = False
        self._layer_name = ""
        self.rotary_emb = None
        if self._use_trtllm:
            from ..L1.flashinfer_decode import TRTLLMDecode
            from ..L1.flashinfer_prefill import TRTLLMPrefill

            self.store_kvcache = StoreKVCacheHND(page_size=attn_cfg.block_size)
            self.flash_attn_prefill = TRTLLMPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = TRTLLMDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
        else:
            self.store_kvcache = StoreKVCache()
            self.flash_attn_prefill = FlashAttnPrefill(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            self.flash_attn_decode = FlashAttnDecode(
                self.num_heads, self.num_kv_heads, self.head_dim,
            )

        # --- Derived once, because at this launch count a per-call multiply and
        # --- attribute walk are both measurable.
        self._q_gate_size = self.num_heads * 2 * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        self._out_size = self.num_heads * self.head_dim
        self._page_size = attn_cfg.block_size if self._use_trtllm else 0
        self._out_shape = (self.num_heads, self.head_dim)

        # Validity of the gain vectors, recorded when they are materialized.
        self._gain_ok = False
        self._gain_device: torch.device | None = None

        # Per-(device, stream, prefill count) int32 scratch for the cumulative
        # sequence vectors.  Add-only; see ``_cu_seqlens_scratch``.
        self._cu_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

        # Whatever about the fused geometry is decided by the constructor rather
        # than by a call.  Held as the rejection reason itself so the hot path
        # neither recomputes it nor formats a string.
        self._static_rejection = self._static_fast_path_rejection()

    def _static_fast_path_rejection(self) -> str | None:
        if _ext is None:
            return "extension-not-built"
        if not self._use_trtllm:
            # The fusion writes an HND cache directly.  The non-trtllm backend
            # keeps an NHD cache and its own store, which this kernel cannot
            # express, so that whole backend stays on the baseline composition.
            return "backend-not-trtllm"
        if self.head_dim % _LANE_ELEMS or not (
                _LANE_ELEMS <= self.head_dim <= _MAX_HEAD_DIM):
            # One warp covers one head row with one 16-byte access per lane, so a
            # row must be between 1 and 32 whole 16-byte units.
            return "head-dim-off-lane-geometry"
        if self.num_heads < 1 or self.num_kv_heads < 1 or self._page_size < 1:
            return "degenerate-head-or-page-count"
        return None

    def set_trtllm_workspace(self, workspace: torch.Tensor) -> None:
        """Adopt the engine's single shared trtllm-gen workspace.

        Without this each layer keeps the 512 MiB buffer it allocated in
        ``__init__``; Qwen3-Next has one MHA layer per 4 decoder layers, so
        that would waste several GiB.
        """
        if self._use_trtllm:
            self.flash_attn_decode._workspace = workspace
            self.flash_attn_prefill._workspace = workspace

    def _norm_gains(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``1 + weight`` for the QK norms, in fp32, materialized once.

        vLLM recomputes ``q_norm.weight.float() + 1.0`` per call and lets
        Inductor hoist it; in eager that would be two extra launches on every
        one of the 12 attention layers. The values are constants after weight
        loading, so caching them is exact -- and it has to happen here rather
        than in ``__init__``, because the harness calls ``load_state_dict``
        after construction and a precomputed gain would go stale.

        Whether the result is something the fused kernel can index is settled
        here too, once, instead of on every call.
        """
        cache = self._norm_gain_cache
        if cache is None:
            q_gain = self.q_norm.weight.float() + 1.0
            k_gain = self.k_norm.weight.float() + 1.0
            cache = (q_gain, k_gain)
            self._norm_gain_cache = cache
            self._gain_ok = (
                q_gain.is_cuda and k_gain.is_cuda
                and q_gain.device == k_gain.device
                and q_gain.is_contiguous() and k_gain.is_contiguous()
                and q_gain.numel() == self.head_dim
                and k_gain.numel() == self.head_dim
                and q_gain.data_ptr() % 16 == 0 and k_gain.data_ptr() % 16 == 0
            )
            self._gain_device = q_gain.device if self._gain_ok else None
        return cache

    # -- The guard ------------------------------------------------------------

    def _fast_path_rejection(self, qkv, positions, rotary_emb, md, k_cache,
                             v_cache, N) -> str | None:
        """Everything the fused path promises, re-checked on every call.

        Re-checked rather than memoized because the bench's ``_ShiftingPool``
        hands every contiguous input a different ``data_ptr`` on each of its 60
        timed iterations, so no answer keyed on a pointer stays valid.  Kept cheap
        because on a ~90 us host budget the predicate is itself a measurable
        stage.  Returns ``None`` when the fused path applies, otherwise the name
        of the precondition that rejected it.

        Nothing here reads a device value.  In particular slot range and slot
        distinctness are not checked -- the baseline does not check them either,
        and a device read would cost more than the whole path it is guarding.
        """
        if N <= 0:
            return "empty-batch"
        if not self._fused_qk_rope_gate:
            return "fused-norm-disabled"
        if rotary_emb is None or positions is None:
            return "rotary-missing"
        if not getattr(rotary_emb, "is_neox_style", False):
            return "rotary-not-neox"

        head_dim = self.head_dim
        rotary_dim = rotary_emb.head_dim
        if not (2 <= rotary_dim <= head_dim) or rotary_dim % 16:
            # A multiple of 16 is what makes rotary_dim / 2 a whole number of
            # 8-element lanes, which is what lets the NeoX partner live in
            # another lane's element *j* rather than straddling two lanes.
            return "rotary-dim-off-lane-geometry"
        lane_dist = rotary_dim // (2 * _LANE_ELEMS)
        if lane_dist & (lane_dist - 1):
            # The partner exchange is one ``__shfl_xor_sync`` at a fixed distance,
            # and XOR only equals "+ lane_dist" for lanes below it when that
            # distance is a power of two.  Anything else would rotate against the
            # wrong element, so it is refused rather than approximated.
            return "rotary-lane-distance-not-power-of-two"

        cos_sin = rotary_emb.cos_sin_cache
        if (cos_sin.dtype is not torch.float32 or cos_sin.dim() != 2
                or cos_sin.shape[1] != rotary_dim
                or not cos_sin.is_contiguous()
                or cos_sin.data_ptr() % 16):
            return "cos-sin-cache-layout"

        dtype = qkv.dtype
        if dtype not in _HALF_DTYPES:
            return "dtype-not-2-byte-float"
        device = qkv.device
        if device.type != "cuda":
            return "not-cuda"
        if cos_sin.device != device:
            return "cos-sin-cache-device"
        if (qkv.dim() != 2 or qkv.stride(1) != 1
                or qkv.stride(0) % _LANE_ELEMS or qkv.data_ptr() % 16):
            return "qkv-layout"
        if qkv.shape[1] < self._q_gate_size + 2 * self._kv_size:
            return "qkv-too-narrow"

        if k_cache is None or v_cache is None:
            return "cache-missing"
        if (k_cache.dtype is not dtype or v_cache.dtype is not dtype
                or k_cache.device != device or v_cache.device != device):
            return "cache-dtype-or-device"
        if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
            return "cache-shape"
        if (k_cache.shape[1] != self.num_kv_heads
                or k_cache.shape[2] != self._page_size
                or k_cache.shape[3] != head_dim):
            return "cache-not-hnd-geometry"
        if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
            return "cache-not-contiguous"
        if k_cache.data_ptr() == v_cache.data_ptr():
            # Both caches are marked __restrict__ in the kernel.
            return "cache-aliased"
        if k_cache.data_ptr() % 16 or v_cache.data_ptr() % 16:
            return "cache-misaligned"

        if (positions.dtype not in _INT_DTYPES or positions.dim() != 1
                or positions.numel() != N or not positions.is_contiguous()
                or positions.device != device):
            return "positions-layout"
        slots = md.slot_mapping
        if slots is None:
            return "slot-mapping-missing"
        if (slots.dtype not in _INT_DTYPES or slots.dim() != 1
                or not slots.is_contiguous() or slots.device != device):
            return "slot-mapping-layout"
        if slots.numel() < N:
            # Refused rather than handled, so the frozen store raises its own
            # error on the fallback and the observable failure stays identical.
            return "slot-mapping-too-short"

        if self._norm_gain_cache is None:
            self._norm_gains()
        if not self._gain_ok or self._gain_device != device:
            return "norm-gains-layout"

        nd = md.num_decodes
        np_ = md.num_prefills
        if np_ > 0:
            qsl = md.query_start_loc
            seqs = md.seq_lens
            if qsl is None or seqs is None:
                return "prefill-metadata-missing"
            if (qsl.dtype not in _INT_DTYPES or seqs.dtype not in _INT_DTYPES
                    or qsl.dim() != 1 or seqs.dim() != 1
                    or not qsl.is_contiguous() or not seqs.is_contiguous()
                    or qsl.device != device or seqs.device != device):
                return "prefill-metadata-layout"
            if qsl.numel() != nd + np_ + 1 or seqs.numel() != nd + np_:
                # Both lengths are exact, not lower bounds.  The cumulative
                # vectors this module builds hold exactly ``np_ + 1`` entries,
                # which is what the baseline's ``query_start_loc[nd:]`` slice
                # holds -- any other length and the two stop agreeing.  A
                # *surplus* ``seq_lens`` matters just as much as a short one: the
                # scan would read the first ``np_`` prefill lengths and silently
                # ignore the tail, where the baseline slices ``seq_lens[nd:]``,
                # gets more elements than its ``np_ + 1`` destination holds, and
                # raises.  Succeeding where the reference raises is exactly the
                # kind of divergence this predicate exists to prevent.
                return "prefill-metadata-length"
        return None

    # -- Entry points ---------------------------------------------------------

    def forward(self, hidden_states, rotary_emb=None, positions=None,
                state_manager=None):
        if rotary_emb is not None:
            self.rotary_emb = rotary_emb
        if self._use_custom_op:
            return torch.ops.fastkernels.qwen3_next_attention(
                hidden_states, positions, self._layer_name,
            )
        return self.forward_impl(hidden_states, positions, state_manager)

    def forward_impl(self, hidden_states, positions=None, state_manager=None):
        global LAST_REJECTION
        md = get_context().kda_metadata
        if state_manager is None:
            state_manager = get_context().kda_state
        rotary_emb = self.rotary_emb
        if md is None or state_manager is None:
            raise RuntimeError(
                "Qwen3NextAttention requires engine-managed KV state and metadata",
            )

        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        N = x.shape[0]
        qkv = self.qkv_proj(x)

        layer_idx = self.layer_idx
        k_cache = state_manager.k_cache[layer_idx]
        v_cache = state_manager.v_cache[layer_idx]

        rejection = self._static_rejection
        if rejection is None:
            rejection = self._fast_path_rejection(
                qkv, positions, rotary_emb, md, k_cache, v_cache, N,
            )
        if rejection is None:
            LAST_REJECTION = None
            return self._forward_fused(
                qkv, positions, rotary_emb, md, k_cache, v_cache, N,
            )
        LAST_REJECTION = rejection
        return self._forward_reference(
            qkv, positions, rotary_emb, md, k_cache, v_cache, N,
        )

    # -- The fused route ------------------------------------------------------

    def _cu_seqlens_scratch(self, device, np_):
        """Persistent int32 scratch for the two cumulative vectors.

        Keyed by the current stream, not just by the instance, for the reason the
        frozen ``TRTLLMPrefill._packed_kv_buffer`` gives: work issued on one stream
        is ordered against itself, so a buffer reused across calls on the same
        stream is safe, while two calls on *different* streams have no such
        ordering and one call's scan could overwrite what the other's attention is
        still reading.

        Keyed by the prefill count as well, so an entry is only ever *added* --
        never replaced, never evicted.  The obvious alternative, one buffer per
        stream grown on demand, is a lifetime hole: a CUDA graph captured while the
        smaller buffer was live records that buffer's address, and a later eager
        call with more prefills would drop the only reference to it, freeing
        storage the graph still replays into.  A mapping that only grows cannot do
        that.  The cost is bounded and small -- the single-block scan caps the
        prefill count, so one stream holds at most ``_SCAN_MAX_PREFILLS`` entries
        of ``8 * (np_ + 1)`` bytes, under 8 MiB even if every count were used.

        Both returned views are exactly ``np_ + 1`` elements and every element of
        both is written by the scan on every call, so nothing another call left
        behind is observable through them.
        """
        key = (device.type, device.index, _current_stream_id(device), np_)
        got = self._cu_scratch.get(key)
        if got is None:
            buf = torch.zeros(2 * (np_ + 1), dtype=torch.int32, device=device)
            # The views own a reference to the buffer, so it stays alive for as
            # long as anything -- an eager call or a captured graph -- can reach it.
            got = (buf[:np_ + 1], buf[np_ + 1:])
            self._cu_scratch[key] = got
        return got

    def _prefill_cu_seqlens(self, md, nd, np_, device):
        """``cu_q`` and ``cu_k`` for the prefill half of the batch, in one launch.

        The baseline builds these with four operations -- a subtract, a ``zeros``,
        a ``cumsum`` and a slice-assign -- costing 44.0 us of host time at N=60,
        more than the attention call they feed.  One scan produces both, and it
        accumulates in 64-bit before narrowing to int32 because that is what the
        baseline does: ``torch.cumsum`` promotes an int32 input to int64 and only
        the assignment into the int32 destination narrows it.

        ``cu_q`` is written by the kernel even when there are no decodes to skip.
        The draft's shortcut there was to pass ``query_start_loc`` straight
        through, since subtracting its first element is the identity -- but that
        first element is a *device* value and no host-side check can confirm it is
        zero.  Producing the shifted vector needs no extra launch, the scan being
        already in flight, so the assumption is simply not made.

        A cumulative KV length past 2^31 wraps rather than raising, and it wraps
        the same way the baseline does: ``dst_int32[1:] = cumsum_int64`` narrows
        modularly rather than refusing, so accumulating in 64-bit and storing
        int32 reproduces the baseline's result there too.  ``tests/test_metadata.py``
        pins that with ``seq_lens`` chosen to overflow the cumulative sum while
        staying individually representable.
        """
        if np_ > _SCAN_MAX_PREFILLS:
            # Above the single-block scan, take the baseline's own formulation
            # rather than a multi-block scan this phase has no measurement for.
            LAST_PATH["prefill_cu_seqlens"] = "aten"
            qsl = md.query_start_loc
            cu_q = qsl[nd:] - qsl[nd]
            if cu_q.dtype is not torch.int32:
                cu_q = cu_q.to(torch.int32)
            cu_k = torch.zeros(np_ + 1, dtype=torch.int32, device=device)
            cu_k[1:] = torch.cumsum(md.seq_lens[nd:].to(torch.int32), dim=0)
            return cu_q, cu_k

        cu_k, cu_q = self._cu_seqlens_scratch(device, np_)
        _ext.prefill_cu_seqlens(
            md.seq_lens, md.query_start_loc, cu_k, cu_q, np_, nd, 1,
        )
        LAST_PATH["prefill_cu_seqlens"] = "fused"
        return cu_q, cu_k

    def _forward_fused(self, qkv, positions, rotary_emb, md, k_cache, v_cache, N):
        num_heads = self.num_heads
        head_dim = self.head_dim
        dtype = qkv.dtype
        device = qkv.device
        path = LAST_PATH

        q_gain, k_gain = self._norm_gain_cache
        q_out = torch.empty(N, num_heads, head_dim, device=device, dtype=dtype)
        # QK-RMSNorm, partial RoPE and the paged K/V store, in one pass over qkv.
        # K's normalized value goes straight into the cache -- nothing downstream
        # reads it anywhere else, because both attention branches read K from the
        # cache -- and the gate is left where it already is.
        _ext.qk_norm_rope_store(
            qkv, q_out, k_cache, v_cache, q_gain, k_gain,
            rotary_emb.cos_sin_cache, positions, md.slot_mapping,
            self.q_norm.variance_epsilon, num_heads, self.num_kv_heads,
            head_dim, rotary_emb.head_dim, self._page_size, _ROWS_PER_BLOCK,
        )
        path["qk_norm_rope_store"] = "fused"

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        decode_out = None
        if nd > 0:
            seqs = md.seq_lens[:nd]
            if seqs.dtype is not torch.int32:
                seqs = seqs.to(torch.int32)
            decode_out = self.flash_attn_decode(
                q_out[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=seqs,
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        prefill_out = None
        if np_ > 0:
            cu_q, cu_k = self._prefill_cu_seqlens(md, nd, np_, device)
            prefill_out = self.flash_attn_prefill(
                # ``[0:]`` is the whole tensor; the frozen wrapper makes both
                # contiguous anyway, so the slice is pure host cost.
                q_out if ndt == 0 else q_out[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=md.block_tables if nd == 0 else md.block_tables[nd:],
            )
        else:
            path["prefill_cu_seqlens"] = "none"

        # A single-branch batch already holds its whole result in a
        # (N, num_heads, head_dim) contiguous buffer the attention wrapper
        # allocated, which is bit-for-bit the buffer the baseline allocates and
        # then copies into -- 52.5 us of Memcpy DtoD at N=16384 for nothing.  The
        # shortcut is taken on the returned tensor's observed dtype, shape,
        # strides and alignment, not on the assumption that every route through
        # the wrapper returns that form.  A genuinely mixed batch still stages.
        shape = (N, num_heads, head_dim)
        if nd == 0 and ndt == 0 and _is_dense_rows(prefill_out, shape, dtype):
            out = prefill_out
            path["attn_output"] = "direct"
        elif np_ == 0 and ndt == N and _is_dense_rows(decode_out, shape, dtype):
            out = decode_out
            path["attn_output"] = "direct"
        else:
            out = torch.empty(N, num_heads, head_dim, device=device, dtype=dtype)
            if decode_out is not None:
                out[:ndt] = decode_out
            if prefill_out is not None:
                out[ndt:] = prefill_out
            path["attn_output"] = "staged"

        # ``out *= sigmoid(gate)`` with the gate read strided in place out of qkv.
        # The rounding follows whichever baseline branch is being replaced: aten's
        # double rounding whenever prefills are present, the baseline's own Triton
        # kernel's fp32 sigmoid when the batch is decode-only.  The mutation is
        # in place, which the bench's ``torch.no_grad()`` makes free of any
        # version-counter concern, and ``out`` is owned either by this call or by
        # the attention wrapper that just returned it.
        _ext.gate_mul(out, qkv, num_heads, head_dim, 1 if np_ > 0 else 0)
        path["gate_mul"] = "fused"

        return self.o_proj(out.reshape(N, self._out_size))

    # -- The exact fallback ---------------------------------------------------

    def _forward_reference(self, qkv, positions, rotary_emb, md, k_cache,
                           v_cache, N):
        """The baseline composition, verbatim, for everything the guard rejects.

        Kept as its own body rather than as a partially shared path: this is the
        thing every fused result is measured against, so a shortcut taken here
        would move the reference rather than the candidate.
        """
        path = LAST_PATH
        path["qk_norm_rope_store"] = "reference"
        path["prefill_cu_seqlens"] = "reference"
        path["gate_mul"] = "reference"
        path["attn_output"] = "reference"

        q_gate_size = self.num_heads * 2 * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q_gate, k, v = qkv.split([q_gate_size, kv_size, kv_size], dim=-1)

        use_fused = (
            self._fused_qk_rope_gate
            and rotary_emb is not None
            and positions is not None
            and getattr(rotary_emb, "is_neox_style", False)
        )
        if use_fused:
            q_gain, k_gain = self._norm_gains()
            q, k, gate = _vllm_fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                q_gain,
                k_gain,
                rotary_emb.cos_sin_cache,
                positions.reshape(-1),
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_emb.head_dim,
            )
            q = q.view(N, self.num_heads, self.head_dim)
            k = k.view(N, self.num_kv_heads, self.head_dim)
            gate = gate.view(N, self.num_heads, self.head_dim)
        else:
            # Split Q and gate
            q_gate = q_gate.view(N, self.num_heads, 2 * self.head_dim)
            q = q_gate[:, :, :self.head_dim].contiguous()
            gate = q_gate[:, :, self.head_dim:].contiguous()

            k = k.view(N, self.num_kv_heads, self.head_dim)

            # Per-head QK-norm (applied before RoPE)
            q = self.q_norm(q.reshape(-1, self.head_dim)).view(
                N, self.num_heads, self.head_dim)
            k = self.k_norm(k.reshape(-1, self.head_dim)).view(
                N, self.num_kv_heads, self.head_dim)

            # Partial RoPE (only rotates first rotary_dim dimensions)
            if rotary_emb is not None and positions is not None:
                pos_flat = (
                    positions.reshape(-1) if positions.dim() > 1 else positions
                )
                rotary_dim = rotary_emb.head_dim
                q_rot, q_pass = (
                    q[..., :rotary_dim].contiguous(), q[..., rotary_dim:],
                )
                k_rot, k_pass = (
                    k[..., :rotary_dim].contiguous(), k[..., rotary_dim:],
                )
                q_rot, k_rot = rotary_emb(pos_flat, q_rot, k_rot)
                q = torch.cat([q_rot, q_pass], dim=-1)
                k = torch.cat([k_rot, k_pass], dim=-1)

        v = v.view(N, self.num_kv_heads, self.head_dim)

        self.store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        out = torch.empty(
            N,
            self.num_heads,
            self.head_dim,
            device=qkv.device,
            dtype=qkv.dtype,
        )

        nd = md.num_decodes
        ndt = md.num_decode_tokens
        np_ = md.num_prefills

        if nd > 0:
            out[:ndt] = self.flash_attn_decode(
                q[:ndt],
                k_cache,
                v_cache,
                cache_seqlens=md.seq_lens[:nd].to(torch.int32),
                block_table=md.block_tables[:nd],
                softmax_scale=self.scaling,
                causal=True,
                max_seq_len=md.max_seq_len,
            )

        if np_ > 0:
            cu_pf = (md.query_start_loc[nd:] - md.query_start_loc[nd]).to(
                torch.int32,
            )
            seqs_k = md.seq_lens[nd:]
            cu_k_pf = torch.zeros(np_ + 1, dtype=torch.int32, device=q.device)
            cu_k_pf[1:] = torch.cumsum(seqs_k.to(torch.int32), dim=0)
            out[ndt:] = self.flash_attn_prefill(
                q[ndt:],
                k_cache,
                v_cache,
                cu_seqlens_q=cu_pf,
                cu_seqlens_k=cu_k_pf,
                max_seqlen_q=md.max_query_len,
                max_seqlen_k=md.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                block_table=md.block_tables[nd:],
            )

        # Output gating: o * sigmoid(gate). The Triton path is faster in
        # the captured decode graph; PyTorch's vectorized path is better for
        # large prefill chunks.
        if np_ == 0:
            o = _gate_mul_inplace(out, gate)
        else:
            o = out * torch.sigmoid(gate)

        # Output projection
        o = o.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(o)
