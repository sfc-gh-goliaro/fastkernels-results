"""Oasis DiT blocks -- the whole forward as one CUDA graph over collapsed glue.

The four compute calls (``OasisSpatialAxialAttention``,
``OasisTemporalAxialAttention`` and two ``OasisMLP``) are the frozen L2 winners.
What this file owns is the *glue* around them -- and, once that was small enough
to matter, the *dispatch* of the whole chain.

Two rounds, in the order they were worth doing:

1. The block-level glue, ~40 launches down to six (see "Six launches replace
   them" below).  That took the window from 641 us to 137 us at T=6.
2. The entire forward captured into one CUDA graph per frame count, so a call is
   **one** ``cudaGraphLaunch`` instead of 22 kernel launches.  At T=2 the eager
   chain's window was 139 us against 93 us of device time -- 46 us of it was the
   GPU waiting for Python -- and the graph deletes essentially all of that.  The
   three pointers that move between calls (the caller's ``x`` and ``c``, and the
   returned tensor, which may not alias) are handled by making the chain's own
   first and last kernels the retargetable endpoints; see
   "Retargetable CUDA-graph endpoints".

**This operator is host-bound, not device-bound.**  Measured on B200 at the
captured shapes (``x:fp16[1, T, 9, 16, 1024]``, T=2..6, ``c:fp16[1, T, 1024]``,
so 1.7 MB of activation at most): the reference composition issues 56 device
ops per call whose *total device time is 206 us*, inside a scored window of
641 us.  The GPU is idle five-sixths of the time waiting for Python.  Splitting
the 734 us of wall time per call by pieces:

    adaLN x2 (silu -> linear -> chunk)     68 us
    s_norm + _modulate, x4 sites          256 us
    _gate + residual add, x4 sites        122 us
    -------------------------------------------
    glue                                  446 us   (61%)
    s_attn + t_attn + 2x s_mlp            139 us
    other (module dispatch, ...)         ~150 us

So the lever is *ops issued*, and it pays twice -- once in host dispatch, once
in the 2.048 us device quantum every launch costs.  The reference spends 40 of
its 56 ops on glue:

* ``_modulate``/``_gate`` call ``Tensor.repeat`` (12 per forward) which at these
  shapes is pure waste -- batch is 1, so ``x.shape[0] // shift.shape[0] == 1``
  and every ``repeat`` is a full copy of data that already broadcasts;
* ``1 + scale`` is its own launch, the multiply and the add are two more;
* both ``adaLN_modulation`` branches recompute ``SiLU(c)`` on the *same* ``c``.

Six launches replace them:

1. ``_adaln_gemv`` -- one M=T GEMV with SiLU fused into its A-load prologue,
   over a **merged** ``[12*H, H]`` weight, emitting all twelve shift/scale/gate
   vectors at once.  Replaces 4 ops (2 SiLU + 2 addmm).  The merge is done to
   the two parameters' *own storage* (see ``_merge_pair``), the same trick
   ``L2.oasis_mlp._kmajor`` uses for its K-major rewrite, so there is no second
   copy to invalidate and an in-place weight update cannot be missed.
2-5. ``_glue`` -- one row-wise pass that does *gate + residual add + the next
   modulated LayerNorm together*.  Both are row-wise over the 1024-wide last
   axis, so a program that already holds ``x + g*y`` in registers can reduce it
   and emit the modulated norm from the same registers.  That fuses ~9 reference
   ops into 1 and drops the intermediate round trip; the four sites plus the
   final gate-only pass are 5 launches, versus 36.

Everything is issued through the compiled kernels' own C launchers (see
``_Launcher``) rather than ``kernel[grid](...)``, which re-specializes and
re-hashes every argument -- ~12 us of Python per launch, more than the kernel.
The launches are chained with programmatic dependent launch so an adjacent pair
does not each pay a full inter-kernel gap.

**The captured ``x`` is not contiguous.**  It arrives with stride
``[.., 147456, 16, 1, 144]`` -- physically ``[1, T, C, 9, 16]`` viewed as
``[1, T, 9, 16, C]`` -- so the 1024-wide row this operator normalizes is a
stride-144 gather, and every frozen L2 component's fast path (all of which
require ``is_contiguous()``) would refuse it.  The reference chain gets away
with this because ``s_norm1``'s output is contiguous, so only the first norm and
the four residual adds ever touch the strided tensor.  The eager fused path
therefore takes the residual input's strides as *constexpr*: the two sites whose
residual is the block input get x's real strides, the other three get ``XC = 1``
and fold back to the vectorized contiguous form.  The captured path does not have
to: its prologue endpoint has to read every element of ``x`` anyway, so it
de-swizzles on the way in and *every* site takes the vectorized form -- worth
~4.8 us at T=6 for no extra node.

Getting any of this wrong is invisible rather than loud.  An earlier revision
required ``x.is_contiguous()`` and silently ran the reference path for all five
captured shapes, scoring exactly the parent's number; this round the same thing
happened again from a 64-byte argument-blob limit, with a clean 5/5 PASSED and
no message anywhere.  Hence ``dev/armed.py`` (per-path call counts over the real
timing loop), the ``static_assert``s on the endpoint argument structs, and
``OASIS_BLOCK_STRICT=1``.

**What the graph changes, and what it does not.**  Captured verbatim the chain
*loses* 25 us, all of it in the launches carrying a programmatic-dependent-launch
pair: ``_glue`` x5 goes 22.4 -> 47.0 us and ``L2.oasis_mlp``'s activation pass
goes 6.2 -> 10.8 us, i.e. ~2-5 us per ``griddepcontrol`` wait.  Inside a graph
the dependency is an edge and the wait buys nothing, so ``PDL`` is a constexpr on
both kernels here and the two MLPs' GEMMs are issued from ``_mlp`` below (the
same cuBLAS calls on the same K-majored weights, with a PDL-free copy of L2's
activation pass) rather than through ``OasisMLP.forward``.  With that out, the
graph replays at the eager chain's own device time and keeps the host saving.
The eager path keeps PDL, where it is worth ~0.9 us.

Two things measured as *not* levers, so that the next round does not re-spend the
time: the adaLN projection is at a hardware floor (its 25.2 MB weight stream
takes 13.3 us cold, and cuBLAS ``addmm``, five hand-tuned tile shapes and four
split-K factors all land on that same number -- a 25.2 MB *copy* on this box
measures 15.4 us under the same L2 flush), and the glue kernels are at theirs
(they move 1.7 MB per pass, which is less than one HBM latency's worth of traffic
on this device, so ~2 TB/s is the ceiling and only *fewer passes* can help).

The reference ``_modulate``/``_gate`` composition is retained verbatim and is
what runs for any shape, dtype, layout or grad-enabled call the fused path does
not cover, and for any call where the launcher could not be bound; the eager
fused chain is what runs before the graph is captured, for a grad-enabled or
capturing caller, and for a call whose pointers the endpoints cannot be pointed
at.
"""

from __future__ import annotations

import hashlib as _hashlib
import os as _os

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as _raw_stream
from torch._C import _cuda_getDevice as _cur_device

try:
    # Programmatic dependent launch (sm_90+): lets a kernel's blocks be
    # scheduled while its predecessor drains, so adjacent glue launches do not
    # each pay a full inter-kernel gap.
    from triton.language.extra.cuda import gdc_launch_dependents as _gdc_trigger
    from triton.language.extra.cuda import gdc_wait as _gdc_wait
    _PDL = True
except ImportError:                      # pre-3.5 Triton: run without PDL
    _PDL = False
if not _PDL:

    @triton.jit
    def _gdc_trigger():
        pass

    @triton.jit
    def _gdc_wait():
        pass

from ..L1.gelu import _gelu_fast, _gelu_fast_tanh
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.oasis_rotary import OasisRotaryEmbedding
from ..L1.silu import SiLU
from ..L2.oasis_mlp import OasisMLP
from ..L2.oasis_spatial_axial_attention import OasisSpatialAxialAttention
from ..L2.oasis_temporal_axial_attention import OasisTemporalAxialAttention

# Widest row kept entirely in registers; above this the fused paths are off.
_MAX_N = 16384
_FAST_DTYPES = (torch.float16, torch.bfloat16)

# Launch shapes, measured on B200 at the captured H=1024 / N=12288.
#
# ``_GLUE_WARPS`` is not a lever: over {1, 2, 4, 8} x {gate, norm} x {strided,
# contiguous residual} the whole grid lands inside 0.4 us, under the
# measurement's own spread.  ``_GEMV_BN`` is: the row tile has to be wide enough
# that the MMA's 16xBN output can be split across the warps, and with a **cold
# L2** -- which is what the scored loop sees, since the 25 MB weight is read
# once per call -- BN=16 costs 28.7 us against 18.5 us at BN=64 (BN=128 ties,
# BN=32 is 26.6).  BK=128 over BK=256 is worth another 2 us at BN=64.  See
# ITERATIONS.md for the full sweep, including why widening further does nothing.
_GLUE_WARPS = 4
_GEMV_BN = 64
_GEMV_BK = 128
_GEMV_WARPS = 2
# tl.dot's minimum M; the 2..6 real rows are masked into it.
_GEMV_MP = 16

_MISS = object()          # "never planned" vs. a memoized refusal (None)


# ---------------------------------------------------------------------------
# Retargetable CUDA-graph endpoints
# ---------------------------------------------------------------------------
# The whole block forward is captured into one graph per frame count, so the
# per-call cost is one ``cudaGraphLaunch`` instead of 22 kernel launches.  Three
# things move between calls and cannot be baked into the graph: the caller's
# ``x``/``c`` (the harness' shifting pool hands a new ``c`` address every
# iteration) and the returned tensor, which must be a fresh allocation because a
# ``forward`` may not alias its output across calls.
#
# Bracketing the replay with ``cudaMemcpyAsync`` costs ~3.7 us of CPU each --
# three driver launches for a call whose whole host budget is ~10 us -- and a
# copy node inside the graph is no better on the device side, because a node here
# costs ~2 us however little it does.  So there are no copies at all: the two
# kernels that *have* to touch these buffers anyway (the prologue that
# de-swizzles ``x`` and applies the projection's SiLU, and the final gate +
# residual add) are the endpoints, and their pointer is rewritten on the
# instantiated graph with
# ``cudaGraphExecKernelNodeSetParams`` (~0.5 us, and documented to affect only
# future launches, so it is safe against a replay still in flight).  This is the
# mechanism ``L2.oasis_spatial_axial_attention`` uses for its own graph, rebuilt
# here because this level needs three live endpoints rather than two.
#
# Node identification is exact rather than positional: each slot is a distinct
# template instantiation, so its host function pointer is a unique key.  Note
# that ``cudaGraphKernelNodeGetParams`` *fails* on the cuBLAS, Triton and nvrtc
# nodes in this graph (they are launched through the driver API), so its error
# must be swallowed *and* cleared or a stale ``cudaErrorInvalidDeviceFunction``
# surfaces later on an unrelated ATen call.
_EXT_CPP = r"""
#include <torch/extension.h>

void l3_ep_reset();
void l3_ep_prologue(int64_t xdst, int64_t xsrc, int64_t cdst, int64_t csrc,
                    int64_t dev, int64_t t, int64_t s, int64_t w, int64_t h,
                    int64_t xt, int64_t xh, int64_t xw, int64_t xc, bool fp16);
void l3_ep_gate(int64_t out, int64_t xp, int64_t yv, int64_t g, int64_t slot,
                int64_t dev, int64_t t, int64_t s, int64_t h, int64_t ers,
                bool fp16);
int64_t l3_plan_build(int64_t graph_ptr, int64_t exec_ptr, int64_t dev);
void l3_plan_run(int64_t plan, int64_t p0, int64_t p1, int64_t p2);
void l3_plan_free(int64_t plan);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ep_reset", &l3_ep_reset);
  m.def("ep_prologue", &l3_ep_prologue);
  m.def("ep_gate", &l3_ep_gate);
  m.def("plan_build", &l3_plan_build);
  m.def("plan_run", &l3_plan_run);
  m.def("plan_free", &l3_plan_free);
}
"""

_EXT_CU = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace {

constexpr int L3_SLOTS = 3;          // 0 = x in, 1 = c in, 2 = out
constexpr size_t L3_MAX_ARGS = 128;

// bf16 has no implicit conversion to float in device code; fp16 does.  One
// overload pair keeps every kernel below dtype-generic.
__device__ __forceinline__ float l3_f(const __half v) { return __half2float(v); }
__device__ __forceinline__ float l3_f(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void l3_st(__half& d, float v) { d = __float2half_rn(v); }
__device__ __forceinline__ void l3_st(__nv_bfloat16& d, float v) { d = __float2bfloat16(v); }

// The prologue endpoint's two moving source pointers: the caller's ``x`` (slot 0)
// and the caller's conditioning vector (slot 1), both read by one kernel.
struct L3PrologueArgs {
  const void* src;
  void* dst;
  const void* csrc;
  void* cdst;
  int S, H, W, cn;
  long long xt;
  int xh, xw, xc;
};

struct L3GateArgs {
  const void* xp;
  const void* yv;
  const void* g;
  void* out;
  long long nvec;
  int S, H, ers;
};

static_assert(sizeof(L3PrologueArgs) <= L3_MAX_ARGS, "prologue args too large");
static_assert(sizeof(L3GateArgs) <= L3_MAX_ARGS, "gate args too large");

struct L3Ep {
  cudaGraphNode_t node = nullptr;
  void* func = nullptr;
  dim3 grid{1, 1, 1};
  dim3 block{1, 1, 1};
  unsigned shmem = 0;
  size_t args_size = 0;
  size_t ptr_offset = 0;
  // A node can carry more than one moving pointer -- the prologue reads both
  // ``x`` and ``c`` -- so a slot either *owns* its node or borrows another
  // slot's and names only its own offset into that node's argument blob.
  int owner = -1;
  bool used = false;
  alignas(16) unsigned char args[L3_MAX_ARGS] = {};
};

L3Ep g_rec[L3_SLOTS];

bool ep_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &st) != cudaSuccess) { cudaGetLastError(); return false; }
  return st == cudaStreamCaptureStatusActive;
}

// Called only while a capture is in flight: the same launchers run during
// warm-up, where the launch is a real one and there is no node to remember.
void ep_record(int slot, void* func, dim3 grid, dim3 block, unsigned shmem,
               const void* args, size_t size, size_t offset, int owner) {
  TORCH_CHECK(size <= L3_MAX_ARGS, "endpoint argument blob too large");
  L3Ep& e = g_rec[slot];
  e.node = nullptr;
  e.func = func;
  e.grid = grid;
  e.block = block;
  e.shmem = shmem;
  e.args_size = size;
  e.ptr_offset = offset;
  e.owner = owner;
  e.used = true;
  std::memcpy(e.args, args, size);
}

// ---------------------------------------------------------------------------
// Endpoint kernels that do the chain's own first and last work, so that reading
// the caller's `x`/`c` and writing the caller's output costs no extra node.
// A Triton or cuBLAS node cannot be retargeted -- `cudaGraphKernelNodeGetParams`
// refuses driver-launched nodes -- so the three kernels that touch a moving
// pointer are the three written here.
// ---------------------------------------------------------------------------

// The captured `x` is channels-first: stride [.., 147456, 16, 1, 144] for
// [1, T, 9, 16, 1024], i.e. physically [1, T, C, 9, 16].  The two glue sites
// whose residual is the block input would therefore read a stride-144 gather,
// which costs ~2.4 us per site at T=6.  This endpoint has to exist anyway -- it
// is what makes `x`'s pointer patchable -- so it de-swizzles rather than copying:
// out along x's contiguous spatial axis, in along `dst`'s contiguous channel
// axis, both in 16-byte vectors through a padded shared tile, after which every
// site reads the vectorized contiguous form.
//
// SiLU of the 12 KB conditioning vector rides along in the same node.  A graph
// node costs ~2 us on this box however little it does (measured: 2.04 us for a
// kernel that moved 24 KB), and the two endpoints are unordered with respect to
// each other, so they share one launch: one extra row of tile-blocks carries the
// SiLU and slot 1 patches its pointer inside slot 0's node.
constexpr int L3_TS = 64;            // spatial positions per tile
constexpr int L3_TC = 64;            // channels per tile
constexpr int L3_TPAD = L3_TS + 8;   // shared-row padding, against bank conflicts
constexpr int L3_PBX = 8;            // threads along the vectorized axis, 8 each
constexpr int L3_PBY = 32;

template <typename T>
__device__ __forceinline__ void l3_silu_all(const L3PrologueArgs& a, long long lane,
                                            long long stride) {
  for (long long i = lane; i < a.cn; i += stride) {
    const float v = l3_f(((const T*)a.csrc)[i]);
    // `x * (1/(1+exp(-x)))` in fp32 with a single round back down, which is what
    // ATen's fp16 SiLU does and what the fused GEMV's prologue reproduced.
    l3_st(((T*)a.cdst)[i], __fmul_rn(v, __frcp_rn(__fadd_rn(1.0f, expf(-v)))));
  }
}

template <typename T, int SLOT>
__global__ void l3_prologue_kernel(L3PrologueArgs a) {
  __shared__ T tile[L3_TC][L3_TPAD];
  const int t = blockIdx.z;
  const int tx = threadIdx.x;                   // 8 contiguous elements each
  const int ty = threadIdx.y;
  const long long nthread = (long long)blockDim.x * blockDim.y;
  if (blockIdx.y == gridDim.y - 1) {            // the SiLU rides in the last row
    const long long lane = ((long long)t * gridDim.x + blockIdx.x) * nthread
                           + ty * blockDim.x + tx;
    l3_silu_all<T>(a, lane, (long long)gridDim.z * gridDim.x * nthread);
    return;
  }
  const int s0 = blockIdx.x * L3_TS;
  const int c0 = blockIdx.y * L3_TC;
  const T* src = (const T*)a.src + (long long)t * a.xt;
  T* dst = (T*)a.dst + ((long long)t * a.S) * a.H;
  const int ss = s0 + tx * 8;
  for (int c = ty; c < L3_TC && c0 + c < a.H; c += blockDim.y) {
    const T* row = src + (long long)(c0 + c) * a.xc;
    if (ss + 8 <= a.S) {
      *(uint4*)&tile[c][tx * 8] = *(const uint4*)(row + ss);
    } else {
      for (int k = 0; k < 8; ++k)
        if (ss + k < a.S) tile[c][tx * 8 + k] = row[ss + k];
    }
  }
  __syncthreads();
  const int cw = tx * 8;
  for (int sl = ty; sl < L3_TS && s0 + sl < a.S; sl += blockDim.y) {
    T* out = dst + (long long)(s0 + sl) * a.H + c0 + cw;
    if (c0 + cw + 8 <= a.H) {
      uint4 ov;
      T* os = (T*)&ov;
#pragma unroll
      for (int k = 0; k < 8; ++k) os[k] = tile[cw + k][sl];
      *(uint4*)out = ov;
    } else {
      for (int k = 0; k < 8; ++k)
        if (c0 + cw + k < a.H) out[k] = tile[cw + k][sl];
    }
  }
}

// The general layout: any strides, one thread per element, with the SiLU folded
// in the same way.  Correct for an `x` whose spatial axes are not one contiguous
// run; never reached by the captured shapes, but a fused path that quietly does
// not arm is worse than a slow one (see the module docstring).
template <typename T, int SLOT>
__global__ void l3_gather_kernel(L3PrologueArgs a) {
  const int t = blockIdx.y;
  if (blockIdx.x == 0 && t == 0)
    l3_silu_all<T>(a, threadIdx.x, blockDim.x);
  const long long n = (long long)a.S * a.H;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int sp = (int)(i / a.H);
  const int ch = (int)(i % a.H);
  const T* src = (const T*)a.src + (long long)t * a.xt + (sp / a.W) * a.xh
                 + (sp % a.W) * a.xw + (long long)ch * a.xc;
  ((T*)a.dst)[(long long)t * n + i] = *src;
}

// The last residual: `out = xp + g * yv`, straight into the caller's fresh
// output buffer.  This is the output endpoint, so the graph never needs a
// copy-out and the returned tensor still aliases nothing.
template <typename T, int SLOT>
__global__ void l3_gate_kernel(L3GateArgs a) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= a.nvec) return;
  const long long e0 = i * 8;                       // 8 halves per thread
  const int ch = (int)(e0 % a.H);
  const int t = (int)(e0 / ((long long)a.S * a.H));
  const uint4 xv = ((const uint4*)a.xp)[i];
  const uint4 yv = ((const uint4*)a.yv)[i];
  const uint4 gv = *(const uint4*)((const T*)a.g + (long long)t * a.ers + ch);
  const T* xs = (const T*)&xv;
  const T* ys = (const T*)&yv;
  const T* gs = (const T*)&gv;
  uint4 ov;
  T* os = (T*)&ov;
#pragma unroll
  for (int k = 0; k < 8; ++k)
    l3_st(os[k], __fadd_rn(l3_f(xs[k]), __fmul_rn(l3_f(gs[k]), l3_f(ys[k]))));
  ((uint4*)a.out)[i] = ov;
}

struct L3Plan {
  cudaGraphExec_t exec = nullptr;
  L3Ep ep[L3_SLOTS];
  c10::DeviceIndex dev = 0;
  const void* cur[L3_SLOTS] = {nullptr, nullptr, nullptr};
};

void ep_set(L3Plan* p, int slot, const void* ptr) {
  L3Ep& e = p->ep[p->ep[slot].owner];       // the slot that owns the node
  std::memcpy(e.args + p->ep[slot].ptr_offset, &ptr, sizeof(void*));
  void* kp[1] = {e.args};
  cudaKernelNodeParams np{};
  np.func = e.func;
  np.gridDim = e.grid;
  np.blockDim = e.block;
  np.sharedMemBytes = e.shmem;
  np.kernelParams = kp;
  np.extra = nullptr;
  AT_CUDA_CHECK(cudaGraphExecKernelNodeSetParams(p->exec, e.node, &np));
}

cudaGraphNode_t find_node(const std::vector<cudaGraphNode_t>& nodes, void* func) {
  cudaGraphNode_t hit = nullptr;
  int count = 0;
  for (cudaGraphNode_t nd : nodes) {
    cudaGraphNodeType t;
    if (cudaGraphNodeGetType(nd, &t) != cudaSuccess) { cudaGetLastError(); continue; }
    if (t != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams kp{};
    // Fails on the driver-launched (cuBLAS / Triton / nvrtc) nodes; clear it or
    // the stale error resurfaces on an unrelated call.
    if (cudaGraphKernelNodeGetParams(nd, &kp) != cudaSuccess) { cudaGetLastError(); continue; }
    if (kp.func == func) { ++count; hit = nd; }
  }
  TORCH_CHECK(count == 1, "endpoint kernel matched ", count, " graph nodes");
  return hit;
}

}  // namespace

void l3_ep_reset() {
  for (int i = 0; i < L3_SLOTS; ++i) g_rec[i].used = false;
}

// Launch a 16-byte-vector copy and, if a capture is in flight, remember
// everything needed to rewrite this node's endpoint pointer later.
#define L3_DISPATCH_SLOT(KERN, ARGS, GRID, BLOCK, SHM, T, SLOT, FUNC)      \
  do {                                                                     \
    switch (SLOT) {                                                        \
      case 0: KERN<T, 0><<<GRID, BLOCK, SHM, stream>>>(ARGS);               \
              FUNC = (void*)&KERN<T, 0>; break;                            \
      case 1: KERN<T, 1><<<GRID, BLOCK, SHM, stream>>>(ARGS);               \
              FUNC = (void*)&KERN<T, 1>; break;                            \
      default: KERN<T, 2><<<GRID, BLOCK, SHM, stream>>>(ARGS);              \
               FUNC = (void*)&KERN<T, 2>; break;                           \
    }                                                                      \
  } while (0)

void l3_ep_prologue(int64_t xdst, int64_t xsrc, int64_t cdst, int64_t csrc,
                    int64_t dev, int64_t t, int64_t s, int64_t w, int64_t h,
                    int64_t xt, int64_t xh, int64_t xw, int64_t xc, bool fp16) {
  TORCH_CHECK(t > 0 && s > 0 && h > 0 && w > 0, "bad prologue extents");
  L3PrologueArgs a;
  a.src = reinterpret_cast<const void*>((uintptr_t)xsrc);
  a.dst = reinterpret_cast<void*>((uintptr_t)xdst);
  a.csrc = reinterpret_cast<const void*>((uintptr_t)csrc);
  a.cdst = reinterpret_cast<void*>((uintptr_t)cdst);
  a.S = (int)s; a.H = (int)h; a.W = (int)w; a.cn = (int)(t * h);
  a.xt = xt; a.xh = (int)xh; a.xw = (int)xw; a.xc = (int)xc;
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)dev);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* func = nullptr;
  dim3 grid, block;
  // The tiled path needs x's spatial axes to be one contiguous run (that is what
  // `xw == 1 && xh == w` says) and every 16-byte vector it touches to be aligned,
  // which given a 16-byte-aligned base means the row strides it steps by must be
  // multiples of 8 elements.  Anything else takes the scalar gather.
  if (xw == 1 && xh == w && (xc % 8) == 0 && (xt % 8) == 0 && (h % 8) == 0
      && (s % 8) == 0) {
    grid = dim3((unsigned)((s + L3_TS - 1) / L3_TS),
                (unsigned)((h + L3_TC - 1) / L3_TC) + 1u, (unsigned)t);
    block = dim3(L3_PBX, L3_PBY);
    if (fp16) L3_DISPATCH_SLOT(l3_prologue_kernel, a, grid, block, 0, __half, 0, func);
    else      L3_DISPATCH_SLOT(l3_prologue_kernel, a, grid, block, 0, __nv_bfloat16, 0, func);
  } else {
    const int threads = 256;
    grid = dim3((unsigned)((s * h + threads - 1) / threads), (unsigned)t);
    block = dim3(threads);
    if (fp16) L3_DISPATCH_SLOT(l3_gather_kernel, a, grid, block, 0, __half, 0, func);
    else      L3_DISPATCH_SLOT(l3_gather_kernel, a, grid, block, 0, __nv_bfloat16, 0, func);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  if (!ep_capturing(stream)) return;
  // One node, two moving pointers: slot 1 borrows slot 0's node and names only
  // its own offset into that node's argument blob.
  ep_record(0, func, grid, block, 0, &a, sizeof(a),
            offsetof(L3PrologueArgs, src), 0);
  ep_record(1, func, grid, block, 0, &a, sizeof(a),
            offsetof(L3PrologueArgs, csrc), 0);
}

void l3_ep_gate(int64_t out, int64_t xp, int64_t yv, int64_t g, int64_t slot,
                int64_t dev, int64_t t, int64_t s, int64_t h, int64_t ers,
                bool fp16) {
  TORCH_CHECK(slot >= 0 && slot < L3_SLOTS, "bad endpoint slot");
  TORCH_CHECK(h % 8 == 0, "gate endpoint needs a row width divisible by 8");
  TORCH_CHECK((out % 16) == 0 && (xp % 16) == 0 && (yv % 16) == 0 && (g % 16) == 0,
              "gate endpoint needs 16-byte-aligned buffers");
  L3GateArgs a;
  a.xp = reinterpret_cast<const void*>((uintptr_t)xp);
  a.yv = reinterpret_cast<const void*>((uintptr_t)yv);
  a.g = reinterpret_cast<const void*>((uintptr_t)g);
  a.out = reinterpret_cast<void*>((uintptr_t)out);
  a.nvec = (long long)t * s * h / 8;
  a.S = (int)s; a.H = (int)h; a.ers = (int)ers;
  const c10::cuda::CUDAGuard guard((c10::DeviceIndex)dev);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  void* func = nullptr;
  const int threads = 256;
  dim3 grid((unsigned)((a.nvec + threads - 1) / threads));
  dim3 block(threads);
  if (fp16) L3_DISPATCH_SLOT(l3_gate_kernel, a, grid, block, 0, __half, (int)slot, func);
  else      L3_DISPATCH_SLOT(l3_gate_kernel, a, grid, block, 0, __nv_bfloat16, (int)slot, func);
  AT_CUDA_CHECK(cudaGetLastError());
  if (!ep_capturing(stream)) return;
  ep_record((int)slot, func, grid, block, 0, &a, sizeof(a),
            offsetof(L3GateArgs, out), (int)slot);
}

int64_t l3_plan_build(int64_t graph_ptr, int64_t exec_ptr, int64_t dev) {
  TORCH_CHECK(graph_ptr != 0 && exec_ptr != 0, "null graph handles");
  cudaGraph_t g = reinterpret_cast<cudaGraph_t>(graph_ptr);
  size_t n = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &n));
  TORCH_CHECK(n > 0, "captured graph has no nodes");
  std::vector<cudaGraphNode_t> nodes(n);
  AT_CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &n));
  auto* p = new L3Plan();
  p->exec = reinterpret_cast<cudaGraphExec_t>(exec_ptr);
  p->dev = (c10::DeviceIndex)dev;
  try {
    const c10::cuda::CUDAGuard guard(p->dev);
    for (int slot = 0; slot < L3_SLOTS; ++slot) {
      TORCH_CHECK(g_rec[slot].used, "capture did not record endpoint ", slot);
      p->ep[slot] = g_rec[slot];
      const int owner = p->ep[slot].owner;
      TORCH_CHECK(owner >= 0 && owner <= slot, "endpoint ", slot, " has no owner");
      p->ep[slot].node = (owner == slot) ? find_node(nodes, g_rec[slot].func)
                                         : p->ep[owner].node;
      // Prove the driver accepts an exec-level parameter update before anything
      // relies on it: re-set the endpoint to the pointer it already holds.
      const void* q;
      std::memcpy(&q, p->ep[owner].args + p->ep[slot].ptr_offset, sizeof(void*));
      ep_set(p, slot, q);
      p->cur[slot] = q;
    }
  } catch (...) {
    delete p;
    l3_ep_reset();
    throw;
  }
  l3_ep_reset();
  return (int64_t)(uintptr_t)p;
}

// One pybind call per forward: patch whichever endpoints moved, then launch.
void l3_plan_run(int64_t plan, int64_t p0, int64_t p1, int64_t p2) {
  auto* p = reinterpret_cast<L3Plan*>(plan);
  TORCH_CHECK(p != nullptr, "null plan");
  const int64_t in[L3_SLOTS] = {p0, p1, p2};
  const c10::cuda::CUDAGuard guard(p->dev);
  for (int s = 0; s < L3_SLOTS; ++s) {
    const void* q = reinterpret_cast<const void*>((uintptr_t)in[s]);
    if (q != p->cur[s]) {
      ep_set(p, s, q);
      p->cur[s] = q;
    }
  }
  AT_CUDA_CHECK(cudaGraphLaunch(p->exec, at::cuda::getCurrentCUDAStream()));
}

void l3_plan_free(int64_t plan) { delete reinterpret_cast<L3Plan*>(plan); }
"""


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    # Pin the arch list to the device actually present: torch's default list is
    # six architectures, which costs minutes per build for no benefit.  The tag
    # carries the arch so a cached .so is never reused across GPUs.
    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}.{cap[1]}"
    tag = _hashlib.md5((_EXT_CPP + _EXT_CU + arch).encode()).hexdigest()[:10]
    prev = _os.environ.get("TORCH_CUDA_ARCH_LIST")
    _os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        return load_inline(
            name=f"oasis_block_graph_{tag}",
            cpp_sources=_EXT_CPP,
            cuda_sources=_EXT_CU,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    finally:
        if prev is None:
            _os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            _os.environ["TORCH_CUDA_ARCH_LIST"] = prev


_EXT = None
if torch.cuda.is_available() and not _os.environ.get("OASIS_BLOCK_NO_EXT"):
    try:
        _EXT = _build_ext()
    except Exception:                                   # noqa: BLE001
        _EXT = None

# Set OASIS_BLOCK_NO_GRAPH=1 to keep the eager fused chain (NCU attribution).
_USE_GRAPH = (_EXT is not None
              and not _os.environ.get("OASIS_BLOCK_NO_GRAPH")
              and hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph")
              and hasattr(torch.cuda.CUDAGraph, "raw_cuda_graph_exec"))
# Eager calls per shape before the graph is captured.  The first arms every lazy
# thing the frozen components own (the temporal plan's nvrtc compile and cuBLASLt
# heuristic, the spatial rope tables and Triton attention JIT, cuBLAS workspace
# growth, the L2 K-major weight rewrites); nothing lazy may happen inside a
# capture.
_WARM_CALLS = 2
# Re-raise instead of falling back, for when a capture is expected to work.
_STRICT = bool(_os.environ.get("OASIS_BLOCK_STRICT"))
# L2.oasis_mlp's own launch shape for the activation pass, re-used verbatim.
_MLP_BLOCK = 4096
_MLP_WARPS = 16
# Run the two MLPs' GEMMs from here (same cuBLAS calls, PDL-free activation)
# instead of through ``L2.OasisMLP.forward``.
_MLP_INLINE = not _os.environ.get("OASIS_BLOCK_NO_MLP_INLINE")


# ---------------------------------------------------------------------------
# Reference glue (the fallback path, unchanged from the baseline)
# ---------------------------------------------------------------------------
def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def _adaln_gemv(A, W, BIAS, Y,
                M: tl.constexpr, K: tl.constexpr, LDY: tl.constexpr,
                MP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                HAS_BIAS: tl.constexpr, PDL: tl.constexpr,
                SILU: tl.constexpr):
    """``y[m, n] = sum_k silu(a[m, k]) * w[n, k] + bias[n]``, BN outputs/program.

    M is the frame count (2..6), so this is a weight-streaming problem: 25 MB of
    merged ``[12*H, H]`` weight against 12 KB of activation and 150 KB of
    result.  The M axis is padded to ``MP = 16`` and masked, which costs nothing
    -- an MMA is orders of magnitude cheaper here than the weight stream -- and
    buys an accumulator small enough (``[16, BN]`` fp32) that the only thing
    resident is the ``[BK, BN]`` weight tile currently in flight.  BN is what
    matters, and not for reuse: below 64 the ``[16, BN]`` MMA output cannot be
    split across the warps, so Triton replicates the operands and each warp
    re-reads the tile.

    SiLU is folded into the A-load prologue: each program recomputes it on the
    few activation elements it needs, which removes a whole launch.  It is
    rounded back to the input dtype before the MMA, so the product is bit-for-bit
    what the reference's separate ``F.silu`` would have fed to cuBLAS, and the
    fp32 accumulate matches cuBLAS' own.
    """
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = tl.arange(0, MP)
    mm = rm < M
    # Every load below is of data written before the predecessor kernel, so the
    # wait goes ahead of all of them; the win is having the blocks already
    # resident when the predecessor's tail drains.  ``PDL`` is off for the
    # captured variant: a ``griddepcontrol`` pair costs ~5 us per launch when
    # replayed from a graph (measured) and buys nothing there, because the graph
    # already knows the dependency.
    if PDL:
        _gdc_wait()
    ap = A + rm[:, None] * K + tl.arange(0, BK)[None, :]
    wp = W + rn[None, :].to(tl.int64) * K + tl.arange(0, BK)[:, None]
    acc = tl.zeros([MP, BN], tl.float32)
    for _ in range(K // BK):
        if SILU:
            av = tl.load(ap, mask=mm[:, None], other=0.0).to(tl.float32)
            a16 = (av * tl.sigmoid(av)).to(A.dtype.element_ty)
        else:
            # The captured path's ``c`` endpoint already emitted SiLU(c) on its
            # way into the static buffer, so the 192 programs do not each
            # recompute it.
            a16 = tl.load(ap, mask=mm[:, None], other=0.0)
        w = tl.load(wp, eviction_policy="evict_first")
        acc = tl.dot(a16, w, acc)
        ap += BK
        wp += BK
    if HAS_BIAS:
        acc += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * LDY + rn[None, :], acc.to(Y.dtype.element_ty),
             mask=mm[:, None])
    # Release the first glue kernel; it still waits on this grid before reading
    # what we stored, so the trigger only buys it an earlier start.
    if PDL:
        _gdc_trigger()


@triton.jit
def _gelu_ip(X, TANH: tl.constexpr, BLOCK: tl.constexpr):
    """``L2.oasis_mlp``'s in-place GELU pass, minus the PDL wait.

    The frozen module launches its own copy with ``launch_pdl=True`` and a
    ``gdc_wait``, which costs ~2 us *per launch* when the chain is replayed from
    a graph (measured: the two passes go 6.2 -> 11.2 us at T=6).  Inside the
    graph the dependency is an edge, so the wait buys nothing.  Everything else
    -- the fitted quintic / half-angle tanh formulations, imported from ``L1``
    rather than copied, the in-place pass over ``fc1``'s private output, and the
    4096/16-warp launch shape L2 swept for a hidden tensor ``fc2`` consumes
    immediately -- is the frozen winner's, so the numerics are bit-identical.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + off).to(tl.float32)
    y = _gelu_fast_tanh(x) if TANH else _gelu_fast(x)
    tl.store(X + off, y.to(X.dtype.element_ty))


@triton.jit
def _glue(XP, YV, XO, HO, G, SH, SC,
          S: tl.constexpr, W: tl.constexpr, N: tl.constexpr,
          ERS: tl.constexpr, EPS: tl.constexpr,
          XT: tl.constexpr, XH: tl.constexpr, XW: tl.constexpr, XC: tl.constexpr,
          B0: tl.constexpr, B1: tl.constexpr, TWO: tl.constexpr, MASK1: tl.constexpr,
          GATE: tl.constexpr, NORM: tl.constexpr, LATE: tl.constexpr,
          PDL: tl.constexpr):
    """One row-wise pass: ``xo = xp + g*yv`` then ``ho = modnorm(xo)``.

    grid is ``(S, T)`` -- one program per (spatial token, frame), so the frame
    index that selects the modulation vector is a grid coordinate and the
    per-frame ``[H]`` vectors are read as a plain pointer offset.  Nothing is
    ever materialized: no ``repeat`` (batch is 1 and ``c`` broadcasts over the
    9x16 spatial tokens), no ``1 + scale``, no chunk view.

    ``GATE`` / ``NORM`` select the site: the first norm has no residual to fold
    in, the last residual has no norm after it, and the three interior sites do
    both -- which is the point, since the value the reduction needs is exactly
    the value the residual add just produced, and it is already in registers.

    ``XT/XH/XW/XC`` are the *residual* input's strides, because the captured
    ``x`` is not contiguous: it arrives as ``stride [.., 147456, 16, 1, 144]``,
    i.e. physically ``[1, T, C, 9, 16]`` viewed as ``[1, T, 9, 16, C]``, so the
    1024-wide row this kernel reduces is a stride-144 gather.  Passing the
    strides as constexpr keeps one kernel for both layouts: the two sites that
    read the block input get the real strides and the ones that read this
    module's own buffers get ``XC = 1``, which folds back into the vectorized
    contiguous form.  Everything written (``XO``, ``HO``) is contiguous.

    The row is covered by one or two power-of-two tiles summing to exactly N
    (the frozen L1 ``_layer_norm_fwd`` trick) rather than a masked
    ``next_pow2`` tile that would idle lanes.  Statistics are two reduction
    trees over registers -- mean, then variance about it -- which is exact
    without the shifted-one-pass dance the frozen kernels need, because here
    the row is never reloaded.
    """
    sp = tl.program_id(0)
    t = tl.program_id(1)
    obase = (t * S + sp).to(tl.int64) * N
    xbase = t.to(tl.int64) * XT + (sp // W) * XH + (sp % W) * XW
    eo = t * ERS
    if PDL and not LATE:
        _gdc_wait()
    c0 = tl.arange(0, B0)
    if GATE:
        v0 = (tl.load(XP + xbase + c0 * XC).to(tl.float32)
              + tl.load(G + eo + c0).to(tl.float32)
              * tl.load(YV + obase + c0, eviction_policy="evict_first").to(tl.float32))
        tl.store(XO + obase + c0, v0.to(XO.dtype.element_ty))
    else:
        v0 = tl.load(XP + xbase + c0 * XC).to(tl.float32)
    if TWO:
        c1 = B0 + tl.arange(0, B1)
        if MASK1:
            m1 = c1 < N
            if GATE:
                v1 = (tl.load(XP + xbase + c1 * XC, mask=m1, other=0.0).to(tl.float32)
                      + tl.load(G + eo + c1, mask=m1, other=0.0).to(tl.float32)
                      * tl.load(YV + obase + c1, mask=m1, other=0.0).to(tl.float32))
                tl.store(XO + obase + c1, v1.to(XO.dtype.element_ty), mask=m1)
            else:
                v1 = tl.load(XP + xbase + c1 * XC, mask=m1, other=0.0).to(tl.float32)
        else:
            if GATE:
                v1 = (tl.load(XP + xbase + c1 * XC).to(tl.float32)
                      + tl.load(G + eo + c1).to(tl.float32)
                      * tl.load(YV + obase + c1).to(tl.float32))
                tl.store(XO + obase + c1, v1.to(XO.dtype.element_ty))
            else:
                v1 = tl.load(XP + xbase + c1 * XC).to(tl.float32)
    if NORM:
        inv_n: tl.constexpr = 1.0 / N
        tot = tl.sum(v0, axis=0)
        if TWO:
            tot += tl.sum(v1, axis=0)
        mu = tot * inv_n
        d0 = v0 - mu
        sq = tl.sum(d0 * d0, axis=0)
        if TWO:
            d1 = v1 - mu
            if MASK1:
                # Padding lanes must contribute 0 to the variance, so they are
                # zeroed after the shift rather than loaded as ``other=0``.
                d1 = tl.where(m1, d1, 0.0)
            sq += tl.sum(d1 * d1, axis=0)
        rstd = 1.0 / tl.sqrt(sq * inv_n + EPS)
        # Only the modulation vectors come from the predecessor GEMV, so on the
        # first site the whole row load and reduction above is the prefetch
        # window and the wait sits here, as late as it can go.
        if PDL and LATE:
            _gdc_wait()
        h0 = (d0 * rstd) * (1.0 + tl.load(SC + eo + c0).to(tl.float32)) \
            + tl.load(SH + eo + c0).to(tl.float32)
        tl.store(HO + obase + c0, h0.to(HO.dtype.element_ty))
        if TWO:
            if MASK1:
                h1 = (d1 * rstd) * (1.0 + tl.load(SC + eo + c1, mask=m1).to(tl.float32)) \
                    + tl.load(SH + eo + c1, mask=m1).to(tl.float32)
                tl.store(HO + obase + c1, h1.to(HO.dtype.element_ty), mask=m1)
            else:
                h1 = (d1 * rstd) * (1.0 + tl.load(SC + eo + c1).to(tl.float32)) \
                    + tl.load(SH + eo + c1).to(tl.float32)
                tl.store(HO + obase + c1, h1.to(HO.dtype.element_ty))
    if PDL:
        _gdc_trigger()


# ---------------------------------------------------------------------------
# Launch plumbing
# ---------------------------------------------------------------------------
def _tile_split(n: int):
    """Cover exactly *n* columns with one or two power-of-two tiles."""
    b0 = 1 << (n.bit_length() - 1)
    rem = n - b0
    if rem == 0:
        return b0, 1, False, False
    b1 = triton.next_power_of_2(rem)
    return b0, b1, True, b1 != rem


def _gemv_k_tile(k: int):
    """The largest measured-good K chunk that divides *k* exactly, or ``None``.

    The reduction is walked in equal power-of-two chunks rather than one masked
    ``next_pow2`` tile, so no MMA lane is ever spent on padding.
    """
    for bk in (_GEMV_BK, 512, 128, 64, 32, 16):
        if bk <= k and k % bk == 0:
            return (bk,)
    return None


class _Launcher:
    """A compiled Triton kernel's own C launcher plus its invariant args.

    ``kernel[grid](...)`` re-binds and re-specializes every argument, hashes
    them and rebuilds the launch metadata on each call -- ~12 us of Python.
    Everything the binder derives is invariant here (every argument is constexpr
    except the pointers) except Triton's *pointer-alignment* specialization,
    hence the ``& 15`` guards at the bind sites.

    ``CompiledKernel.run`` is the ``CudaLauncher``, whose ``__call__`` defines a
    closure and makes two scratch-allocation calls per launch; with both scratch
    sizes 0 that is pure overhead, so we hold its ``.launch`` (the generated C
    entry point) and pass the arguments ``__call__`` would have inserted.  Every
    piece is fetched defensively: if a future Triton reshapes it, or a kernel
    turns out to need scratch (ours never do), ``bind`` fails and the caller
    keeps to the reference composition.
    """

    __slots__ = ("run", "pre", "dev")

    def __init__(self):
        self.run = None
        self.pre = ()
        self.dev = -1

    def bind(self, kern, device) -> bool:
        launcher = None if kern is None else kern.run
        raw = getattr(launcher, "launch", None)
        if (raw is None
                or getattr(launcher, "global_scratch_size", None) != 0
                or getattr(launcher, "profile_scratch_size", None) != 0):
            return False
        self.run = raw
        self.pre = (
            kern.function,
            launcher.launch_cooperative_grid, launcher.launch_pdl,
            None, None,                      # global / profile scratch
            kern.packed_metadata,
            None, None, None,                # launch metadata, 2 hooks
        )
        self.dev = device
        return True


class _Site:
    """One compiled ``_glue`` variant plus the byte offsets it reads ``e`` at."""

    __slots__ = ("cargs", "grid_s", "g_off", "sh_off", "sc_off", "_l", "pdl")

    def __init__(self, hidden: int, eps: float, s: int, w: int, xstride, tile,
                 gate: bool, norm: bool, late: bool, offs, esize: int,
                 pdl: bool = True):
        self.cargs = ((s, w, hidden, 12 * hidden, eps) + tuple(xstride) + tile
                      + (gate, norm, late, pdl))
        self.pdl = pdl
        self.grid_s = s
        g, sh, sc = offs
        self.g_off = g * hidden * esize
        self.sh_off = sh * hidden * esize
        self.sc_off = sc * hidden * esize
        self._l = _Launcher()

    def setup(self, t: int, xp, yv, xo, ho, e) -> bool:
        """Compile through the supported path, then memoize the C launcher."""
        flat = e.reshape(-1)
        es = e.element_size()
        kern = _glue[(self.grid_s, t)](
            xp, yv, xo, ho,
            flat[self.g_off // es:], flat[self.sh_off // es:],
            flat[self.sc_off // es:],
            *self.cargs, num_warps=_GLUE_WARPS, launch_pdl=self.pdl,
        )
        ptrs = [xp.data_ptr(), yv.data_ptr(), xo.data_ptr(), ho.data_ptr(),
                e.data_ptr()]
        if any(p & 15 for p in ptrs):
            return False
        return self._l.bind(kern, e.get_device())

    def __call__(self, t, stream, xp, yv, xo, ho, ep):
        l = self._l
        l.run(self.grid_s, t, 1, stream, *l.pre,
              xp, yv, xo, ho, ep + self.g_off, ep + self.sh_off,
              ep + self.sc_off, *self.cargs)


class _Gemv:
    """The merged SiLU + ``[12*H, H]`` adaLN projection, one launch per call."""

    __slots__ = ("k", "n", "_grid", "_base", "_per_t", "pdl")

    def __init__(self, hidden: int, pdl: bool = True, silu: bool = True):
        self.pdl = pdl
        self.k = hidden
        self.n = 12 * hidden
        kt = _gemv_k_tile(hidden)
        if kt is None or not (0 < hidden <= _MAX_N):
            self._base = None
            return
        bn = next((c for c in (_GEMV_BN, 32, 16) if self.n % c == 0), None)
        if bn is None:
            self._base = None
            return
        self._grid = self.n // bn
        # (K, LDY, MP, BN, BK, HAS_BIAS, PDL, SILU) -- M is prepended per frame.
        self._base = (hidden, self.n, _GEMV_MP, bn, kt[0], True, pdl, silu)
        self._per_t: dict = {}

    def setup(self, t: int, a, w, b, y) -> bool:
        if self._base is None:
            return False
        if t in self._per_t:
            return True
        cargs = (t,) + self._base
        kern = _adaln_gemv[(self._grid,)](
            a, w, b, y, *cargs, num_warps=_GEMV_WARPS, launch_pdl=self.pdl,
        )
        if any(p & 15 for p in (a.data_ptr(), w.data_ptr(), b.data_ptr(),
                                y.data_ptr())):
            return False
        l = _Launcher()
        if not l.bind(kern, y.get_device()):
            return False
        self._per_t[t] = (l, cargs)
        return True

    def __call__(self, t, stream, ap, wp, bp, yp):
        l, cargs = self._per_t[t]
        l.run(self._grid, 1, 1, stream, *l.pre, ap, wp, bp, yp, *cargs)


class _Plan:
    """Everything resolved for one (shape, dtype, device): buffers + launchers."""

    __slots__ = ("t", "dev", "xshape", "xstride", "cshape", "dtype", "nel", "sites",
                 "sw", "tw", "sb", "tb", "wgap", "bgap", "wp", "bp",
                 "e", "hb", "b1", "b2", "gemv", "g", "ntry",
                 "s", "w", "xc", "cb", "xst", "goff", "hb2", "mlp")


class _GPlan:
    """One captured whole-block graph plus the endpoints patched per call."""

    __slots__ = ("plan", "graph", "sout", "p", "oshape", "dtype", "dev", "wsig",
)

    def free(self):
        if self.plan:
            _EXT.plan_free(self.plan)
        self.plan = 0


class _Bail(Exception):
    """A submodule returned something the fused chain cannot consume."""


def _merge_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    """Rewrite *a* and *b* so *b*'s storage directly follows *a*'s.

    Same ``Parameter`` objects, same shapes, dtypes, devices and values -- only
    the storage they view changes, exactly like ``L2.oasis_mlp._kmajor``.  The
    two ``[6*H, H]`` adaLN weights then *are* one ``[12*H, H]`` matrix and one
    GEMV emits all twelve modulation vectors.  Because this is the parameters'
    own storage and not a cached copy, a later in-place weight update
    (``load_state_dict``) lands in the merged buffer and cannot be missed, and
    ``state_dict()`` still returns the same two tensors with the same values.
    """
    n = a.shape[0]
    buf = torch.empty((n + b.shape[0],) + tuple(a.shape[1:]),
                      dtype=a.dtype, device=a.device)
    with torch.no_grad():
        buf[:n].copy_(a)
        buf[n:].copy_(b)
    a.data = buf[:n]
    b.data = buf[n:]


def _ptr(y: torch.Tensor, nel: int, dtype: torch.dtype) -> int:
    """The data pointer of a compute output the glue kernel may read/overwrite.

    The frozen L2 components all return a fresh contiguous buffer of the shape
    they were handed, so this never fires; it is here because the fused chain
    hands the pointer straight to a kernel compiled for Triton's
    divisible-by-16 specialization, and "never fires" is not a thing to assume
    about somebody else's return value.
    """
    if (y.dtype is not dtype or y.numel() != nel or not y.is_contiguous()
            or y.data_ptr() & 15):
        raise _Bail
    return y.data_ptr()


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        is_causal: bool = True,
        spatial_rotary_emb: OasisRotaryEmbedding,
        temporal_rotary_emb: OasisRotaryEmbedding,
    ):
        super().__init__()
        self.s_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_attn = OasisSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.s_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.s_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.t_norm1 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_attn = OasisTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=temporal_rotary_emb,
            is_causal=is_causal,
        )
        self.t_norm2 = LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.t_mlp = OasisMLP(
            hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            approximate_tanh=True,
        )
        self.t_adaLN_modulation = nn.Sequential(
            SiLU(),
            Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # --- fused-path state -------------------------------------------------
        self._hidden = hidden_size
        # The two projections' parameter dicts, so the per-call "has a weight
        # been replaced?" guard is a dict lookup rather than a trip through
        # nn.Module.__getattr__ (a Python function costing more than the guard).
        self._sp = self.s_adaLN_modulation[1]._parameters
        self._tp = self.t_adaLN_modulation[1]._parameters
        # Two variants of every fused kernel: with the programmatic-dependent-
        # launch pair for the eager chain, and without it for the captured one
        # (a ``griddepcontrol`` wait costs ~5 us per launch when replayed from a
        # graph and buys nothing there -- the graph owns the dependency).
        self._gemv = (_Gemv(hidden_size, _PDL, True),
                      _Gemv(hidden_size, False, False))
        self._sitecache: dict = {}      # (tokens, pdl, strides) -> 5 _Sites
        self._last: _Plan | None = None
        self._plans: dict = {}
        # Every parameter this block's graph bakes a pointer to, as (owning
        # ``_parameters`` dict, key) pairs: a captured graph reads the weights
        # *through* those pointers, so an in-place update (``load_state_dict``)
        # is honoured automatically, but a *replaced* Parameter -- ``.half()``,
        # ``.to(device)``, an assignment -- has to force a re-capture.
        self._wrefs = tuple((m._parameters, k) for m in self.modules()
                            for k, v in m._parameters.items() if v is not None)
        # The two rotary tables are derived from these and cached inside the
        # frozen attentions on (identity, version), so an in-place edit of one
        # invalidates a table whose pointer the graph holds.
        self._freqs = (self.s_attn.rotary_emb.freqs, self.t_attn.rotary_emb.freqs)

    # -- reference composition (fallback) ----------------------------------
    def _forward_reference(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = (
            self.s_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.s_attn(_modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + _gate(self.s_mlp(_modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = (
            self.t_adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + _gate(self.t_attn(_modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + _gate(self.t_mlp(_modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)
        return x

    # -- graph path --------------------------------------------------------
    def _wsig(self):
        """Identity of every weight pointer the captured graph depends on."""
        try:
            return (tuple([d[k].data_ptr() for d, k in self._wrefs]),
                    self._freqs[0]._version, self._freqs[1]._version)
        except Exception:                                   # noqa: BLE001
            return None

    def _mlp(self, spec, xin):
        """``fc1 -> GELU -> fc2`` -- the frozen L2 winner's own three ops.

        Identical cuBLAS calls on identical (already K-majored) weights and the
        same in-place activation formulation; the only difference is that the
        activation launch carries no PDL wait, which is worth ~2 us per pass
        inside a graph.
        """
        # ``torch._addmm_activation`` would fold bias+GELU into fc1's cuBLASLt
        # epilogue and delete this pass entirely.  L2 measured that losing ~3 us
        # eager, because cuBLASLt only serves the epilogue from cutlass3x_sm100
        # kernels; re-measured inside the graph, where a dispatch slot is only a
        # node, it is still a wash to slightly worse at every T.  Not taken.
        w1t, b1, w2t, b2, tanh, grid = spec
        hid = torch.addmm(b1, xin, w1t)
        _gelu_ip[(grid,)](hid, tanh, _MLP_BLOCK, num_warps=_MLP_WARPS,
                          launch_pdl=False)
        return torch.addmm(b2, hid, w2t)

    def _capture_body(self, gp: _Plan, sx, sc, sout):
        """The captured chain, bracketed by the three retargetable endpoints.

        Every kernel whose data pointer moves between calls is one of the three
        written in this file, because a Triton, cuBLAS or nvrtc node cannot be
        retargeted (``cudaGraphKernelNodeGetParams`` refuses driver-launched
        nodes).  So the endpoints are not copies: the block's *own* first and
        last work is what reads the caller's ``x``/``c`` and writes the caller's
        output.

        * slots 0 and 1 share one prologue node.  It de-swizzles the
          channels-first block input -- it has to touch every element anyway, so
          it lands them contiguous and both sites whose residual is the block
          input drop their stride-144 gather -- and applies the projection's SiLU
          on the way through.  Two moving source pointers, one launch, because a
          node costs ~2 us here however little it does.
        * slot 2 is the last gate + residual add, straight into the caller's
          fresh output tensor -- so nothing aliases and there is no copy-out.

        The projection is the one piece of this chain that is *not* on the
        critical path -- only ``shift0``/``scale0`` are needed before the first
        norm -- but putting its other ten vectors on a second captured branch
        *loses* 5-8 us rather than hiding 11 (see ITERATIONS.md): right after the
        harness' L2 flush the memory system is still draining 132 MB of dirty
        lines, so there is no spare bandwidth for a second stream to use and the
        fork buys only its own extra nodes.
        """
        ex = _EXT
        dev, t, h, s = gp.dev, gp.t, self._hidden, gp.s
        fp16 = gp.dtype is torch.float16
        xt, xh, xw, xcs = gp.xst
        st = _raw_stream(dev)
        e, hb, b1, b2, xc, cb = gp.e, gp.hb, gp.b1, gp.b2, gp.xc, gp.cb
        ep = e.data_ptr()
        nel, dt = gp.nel, gp.dtype
        ex.ep_prologue(xc.data_ptr(), sx.data_ptr(), cb.data_ptr(),
                       sc.data_ptr(), dev, t, s, gp.w, h, xt, xh, xw, xcs, fp16)
        # The projection is one launch over the merged [12*H, H] weight.  cuBLAS
        # on the same merged weight was re-measured inside the graph, where r1's
        # reason for declining it (an extra dispatch slot of host time) no longer
        # applies, and it ties exactly: 13.34 us against 13.31 cold-L2.  Both are
        # at a hardware floor -- see ITERATIONS.md.
        gp.gemv(t, st, cb.data_ptr(), gp.wp, gp.bp, ep)
        site = gp.sites
        xcp, hp, p1, p2 = xc.data_ptr(), hb.data_ptr(), b1.data_ptr(), b2.data_ptr()
        site[0](t, st, xcp, 0, 0, hp, ep)
        y = self.s_attn(hb)
        site[1](t, st, xcp, _ptr(y, nel, dt), p1, hp, ep)
        mlp = gp.mlp
        y = self._mlp(mlp[0], gp.hb2) if mlp else self.s_mlp(hb)
        site[2](t, st, p1, _ptr(y, nel, dt), p2, hp, ep)
        y = self.t_attn(hb)
        site[3](t, st, p2, _ptr(y, nel, dt), p1, hp, ep)
        y = self._mlp(mlp[1], gp.hb2) if mlp else self.t_mlp(hb)
        ex.ep_gate(sout.data_ptr(), p1, _ptr(y, nel, dt), ep + gp.goff, 2, dev,
                   t, s, h, 12 * h, fp16)

    def _capture(self, p: _Plan, x: torch.Tensor, c: torch.Tensor):
        """Capture the whole forward for this shape, or return ``False``.

        Everything lazy has already run: the caller only gets here after
        ``_WARM_CALLS`` eager calls, which is what arms the temporal attention's
        nvrtc compile and cuBLASLt heuristic, the spatial attention's rope tables
        and Triton attention JIT (through its own capture's warm-up), cuBLAS
        workspace growth for all six GEMMs, and ``L2.oasis_mlp``'s in-place
        K-major weight rewrite.  None of those may happen inside a capture.
        """
        dev = p.dev
        gp = self._make_plan(x, c, pdl=False, share=p, graph=True)
        if gp is None:
            return False
        if (x.data_ptr() | c.data_ptr()) & 15:
            return False
        sout = torch.empty(p.xshape, dtype=p.dtype, device=x.device)
        wsig = self._wsig()
        if wsig is None:
            return False
        args = (gp, x, c, sout)
        # Three calls on a side stream, then rejoin: under capture the spatial
        # attention takes its non-graph path (it refuses to replay its own graph
        # while a capture is in flight) and the PDL-free kernels are new, so this
        # is the first time this exact body has run.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._capture_body(*args)
        torch.cuda.current_stream().wait_stream(side)
        _EXT.ep_reset()
        graph = torch.cuda.CUDAGraph(keep_graph=True)   # node handles must survive
        with torch.cuda.graph(graph):
            self._capture_body(*args)
        graph.instantiate()
        plan = _EXT.plan_build(graph.raw_cuda_graph(), graph.raw_cuda_graph_exec(),
                               dev)
        g = _GPlan()
        g.plan = plan
        g.graph = graph          # owns the cudaGraph_t / cudaGraphExec_t
        g.sout, g.p = sout, gp
        g.oshape = p.xshape
        g.dtype = p.dtype
        g.dev = dev
        g.wsig = wsig
        return g

    def _forward_graph(self, g: _GPlan, x: torch.Tensor, c: torch.Tensor):
        """One driver launch: patch whichever endpoints moved, then replay."""
        xp = x.data_ptr()
        cp = c.data_ptr()
        out = torch.empty(g.oshape, dtype=g.dtype, device=x.device)
        op = out.data_ptr()
        if (xp | cp | op) & 15:     # the endpoint copies are 16-byte vectorized
            return None
        _EXT.plan_run(g.plan, xp, cp, op)
        return out

    # -- fused path --------------------------------------------------------
    def _weights_ok(self, p: _Plan) -> bool:
        """Re-validate the merged adaLN storage; refresh the plan's addresses.

        The merge is an invariant about two parameters' *addresses*, so it is
        checked rather than assumed: ``load_state_dict`` writes through it, but
        ``.half()`` / ``.to(device)`` / a reassigned Parameter all replace the
        storage and must send the call back through ``_build``.
        """
        sp, tp = self._sp, self._tp
        sw = sp.get("weight")
        tw = tp.get("weight")
        sb = sp.get("bias")
        tb = tp.get("bias")
        if sw is not p.sw or tw is not p.tw or sb is not p.sb or tb is not p.tb:
            return False
        if sw.dtype is not p.dtype:
            return False
        wp = sw.data_ptr()
        if (tw.data_ptr() - wp != p.wgap
                or not sw.is_contiguous() or not tw.is_contiguous()):
            return False
        bp = sb.data_ptr()
        if tb.data_ptr() - bp != p.bgap:
            return False
        p.wp = wp
        p.bp = bp
        return True

    def _plan_for(self, x: torch.Tensor, c: torch.Tensor):
        """The fused plan for this call, or ``None`` for the reference path.

        The captured ``x`` is *not* contiguous -- it is channels-first, see the
        ``_glue`` docstring -- so its layout is part of the plan's identity
        rather than a requirement: the kernel is compiled against whatever
        strides this shape arrived with, and a call whose strides differ has to
        replan.  ``c`` is contiguous in every capture and is required to be, so
        the GEMV can walk it with a single row stride.
        """
        if not c.is_contiguous() or torch.is_grad_enabled():
            return None
        p = self._last
        if (p is not None
                and x.shape == p.xshape
                and x.stride() == p.xstride
                and c.shape == p.cshape
                and x.dtype is p.dtype
                and x.get_device() == p.dev
                and _cur_device() == p.dev
                and self._weights_ok(p)):
            return p
        return self._build(x, c)

    def _build(self, x: torch.Tensor, c: torch.Tensor):
        """Resolve (or refuse) the fused plan for this call's shape and dtype.

        Compiles every kernel through the supported ``kernel[grid](...)`` path,
        binds the C launchers, and caches the scratch buffers.  A refusal is
        memoized under the same (shape, strides, dtype, device) key, so a
        declined call never re-runs this.  If a launcher cannot be bound -- a
        future Triton reshaping ``CompiledKernel.run``, a misaligned pointer --
        the plan is refused rather than half-built, and the reference
        composition runs instead.
        """
        key = (x.shape, x.stride(), c.shape, x.dtype, x.get_device())
        hit = self._plans.get(key, _MISS)
        if hit is not _MISS:
            if hit is None:
                return None
            # Shape, strides, dtype and device are all in the key, so the
            # only thing left that can invalidate a hit is the adaLN storage
            # having moved.
            if _cur_device() == hit.dev and self._weights_ok(hit):
                self._last = hit
                return hit
            if type(hit.g) is _GPlan:
                hit.g.free()
            del self._plans[key]        # the weights moved: replan this shape
        p = None
        try:
            p = self._make_plan(x, c)
        except Exception:                                   # noqa: BLE001
            p = None
        if len(self._plans) >= 16:
            for old in self._plans.values():
                if old is not None and type(old.g) is _GPlan:
                    old.g.free()
            self._plans.clear()
        self._plans[key] = p
        self._last = p
        return p

    def _make_plan(self, x: torch.Tensor, c: torch.Tensor, pdl: bool = True,
                   share: "_Plan | None" = None, graph: bool = False):
        h = self._hidden
        pdl = pdl and _PDL
        gemv = self._gemv[0 if pdl else 1]
        if (torch.is_grad_enabled()
                or x.dtype not in _FAST_DTYPES
                or c.dtype is not x.dtype
                or not x.is_cuda
                or x.ndim != 5
                or c.ndim != 3
                or x.shape[0] != 1
                or c.shape[0] != 1
                or x.shape[1] != c.shape[1]
                or x.shape[4] != h
                or c.shape[2] != h
                or not c.is_contiguous()
                or gemv._base is None):
            return None
        sp, tp = self._sp, self._tp
        sw, tw = sp.get("weight"), tp.get("weight")
        sb, tb = sp.get("bias"), tp.get("bias")
        if (sw is None or tw is None or sb is None or tb is None
                or sw.dtype is not x.dtype or tw.dtype is not x.dtype
                or sb.dtype is not x.dtype or tb.dtype is not x.dtype
                or sw.shape != (6 * h, h) or tw.shape != (6 * h, h)
                or sb.shape != (6 * h,) or tb.shape != (6 * h,)
                or sw.get_device() != x.get_device()):
            return None
        esize = x.element_size()
        wgap = 6 * h * h * esize
        bgap = 6 * h * esize
        if (not sw.is_contiguous() or not tw.is_contiguous()
                or tw.data_ptr() - sw.data_ptr() != wgap):
            _merge_pair(sw, tw)
        if tb.data_ptr() - sb.data_ptr() != bgap:
            _merge_pair(sb, tb)
        if (tw.data_ptr() - sw.data_ptr() != wgap
                or tb.data_ptr() - sb.data_ptr() != bgap
                or not sw.is_contiguous() or not tw.is_contiguous()
                or not sb.is_contiguous() or not tb.is_contiguous()):
            return None

        t = int(x.shape[1])
        s = int(x.shape[2]) * int(x.shape[3])
        w0 = int(x.shape[3])
        p = _Plan()
        p.t = t
        p.dev = x.get_device()
        p.xshape = x.shape
        p.xstride = x.stride()
        p.cshape = c.shape
        p.dtype = x.dtype
        p.nel = x.numel()
        p.sw, p.tw, p.sb, p.tb = sw, tw, sb, tb
        p.wgap, p.bgap = wgap, bgap
        p.wp, p.bp = sw.data_ptr(), sb.data_ptr()
        p.gemv = gemv
        p.s = s
        p.w = w0
        p.xst = (int(x.stride(1)), int(x.stride(2)), int(x.stride(3)), int(x.stride(4)))
        p.goff = 11 * h * esize
        p.hb2 = None
        p.mlp = None
        p.g = 0                 # eager calls so far / the captured graph / False
        p.ntry = 0
        if graph:
            # The de-swizzled block input and SiLU(c), both written by endpoint
            # kernels so that reading the caller's buffers costs no extra node.
            p.xc = torch.empty((t, s, h), dtype=x.dtype, device=x.device)
            p.cb = torch.empty((t, h), dtype=x.dtype, device=x.device)
        else:
            p.xc = p.cb = None
        if share is not None:
            # The captured twin differs from the eager plan only in its kernels,
            # so it runs on the same scratch; nothing else touches these while a
            # replay is in flight (same stream, one call at a time).
            p.e, p.hb, p.b1, p.b2 = share.e, share.hb, share.b1, share.b2
        else:
            # ``empty_like`` would inherit x's channels-first layout, and every
            # frozen L2 component wants a contiguous row-major input.
            p.e = torch.empty((t, 12 * h), dtype=x.dtype, device=x.device)
            p.hb = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            p.b1 = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            p.b2 = torch.empty(x.shape, dtype=x.dtype, device=x.device)

        p.hb2 = p.hb.view(-1, h)
        if graph and _MLP_INLINE:
            # (fc1 weight, fc1 bias, fc2 weight, fc2 bias, tanh) per MLP, or
            # ``None`` if anything about it is not what the inline pass expects
            # -- then ``L2.OasisMLP.forward`` runs, which handles it all.
            mlp = []
            nhid = 0
            for m in (self.s_mlp, self.t_mlp):
                w1 = getattr(getattr(m, "fc1", None), "weight", None)
                b1 = getattr(getattr(m, "fc1", None), "bias", None)
                w2 = getattr(getattr(m, "fc2", None), "weight", None)
                b2 = getattr(getattr(m, "fc2", None), "bias", None)
                act = getattr(m, "act", None)
                tanh = getattr(act, "_tanh", None)
                ok = (w1 is not None and b1 is not None and w2 is not None
                      and b2 is not None and type(tanh) is bool
                      and w1.ndim == 2 and w2.ndim == 2
                      and w1.dtype is x.dtype and w2.dtype is x.dtype
                      and b1.dtype is x.dtype and b2.dtype is x.dtype
                      and w1.shape[1] == h and w2.shape[0] == h
                      and w1.shape[0] == w2.shape[1]
                      # ``L2._kmajor`` has already rewritten both weights in
                      # place (it runs on the first eager call, and the graph is
                      # only captured after two), so ``w.t()`` is contiguous and
                      # cuBLAS answers with the nvjet NNT kernel rather than TNT.
                      and w1.stride(0) == 1 and w2.stride(0) == 1
                      and (t * s * w1.shape[0]) % _MLP_BLOCK == 0)
                if not ok:
                    mlp = None
                    break
                nhid = t * s * int(w1.shape[0])
                mlp.append((w1.t(), b1, w2.t(), b2, tanh, nhid // _MLP_BLOCK))
            p.mlp = tuple(mlp) if mlp else None
        w = w0
        xs = (x.stride(1), x.stride(2), x.stride(3), x.stride(4))
        cs = (s * h, w * h, h, 1)       # this module's own contiguous buffers
        skey = (s, w, xs, pdl, graph)
        sites = self._sitecache.get(skey)
        if sites is None:
            tile = _tile_split(h)
            eps = 1e-6
            late = pdl
            # (gate, norm, wait-late), the residual input's strides, and the
            # (gate, shift, scale) column indices each site reads out of the
            # 12-vector projection.  The first two sites take the block input as
            # their residual, so they get x's real (channels-first) strides; the
            # other three read buffers allocated above.
            if graph:
                # The de-swizzle endpoint has already made the block input
                # contiguous, so every site takes the vectorized form, and the
                # last residual is the output endpoint rather than a site.
                sites = (
                    _Site(h, eps, s, w, cs, tile, False, True, late, (0, 0, 1), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, True, False, (2, 3, 4), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, True, False, (5, 6, 7), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, True, False, (8, 9, 10), esize, pdl),
                )
            else:
                sites = (
                    _Site(h, eps, s, w, xs, tile, False, True, late, (0, 0, 1), esize, pdl),
                    _Site(h, eps, s, w, xs, tile, True, True, False, (2, 3, 4), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, True, False, (5, 6, 7), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, True, False, (8, 9, 10), esize, pdl),
                    _Site(h, eps, s, w, cs, tile, True, False, False, (11, 0, 1), esize, pdl),
                )
            self._sitecache[skey] = sites
        p.sites = sites
        # Compile + bind on private scratch; the launches below run on garbage
        # and their results are discarded, which is why they touch nothing the
        # caller owns.
        if not gemv.setup(t, (p.cb if graph else c.reshape(t, h)), p.sw, p.sb, p.e):
            return None
        for i, site in enumerate(sites):
            # In the eager plan sites 0/1 address their residual with x's own
            # (channels-first) strides, so they are warmed on x itself; the warm
            # launch only ever reads it.
            xp = p.b1 if (graph or i >= 2) else x
            if not site.setup(t, xp, p.b2, p.b1, p.hb, p.e):
                return None
        return p

    def _forward_fused(self, p: _Plan, x: torch.Tensor, c: torch.Tensor):
        st = _raw_stream(p.dev)
        e, hb, b1, b2 = p.e, p.hb, p.b1, p.b2
        ep = e.data_ptr()
        t, nel, dt = p.t, p.nel, p.dtype
        s = p.sites
        xp = x.data_ptr()
        cp = c.data_ptr()
        if (xp | cp) & 15:      # the kernels are specialized on 16B alignment
            raise _Bail
        # (1) SiLU + the merged 12*H projection: all twelve vectors, one launch.
        p.gemv(t, st, cp, p.wp, p.bp, ep)
        hp = hb.data_ptr()
        p1 = b1.data_ptr()
        p2 = b2.data_ptr()
        # (2) modulated LayerNorm of the block input -- no residual to fold yet.
        s[0](t, st, xp, 0, 0, hp, ep)
        y = self.s_attn(hb)
        # (3)-(5) each fold one residual in and emit the next modulated norm from
        # the same registers.  ``xp`` is the strided block input at (3); after
        # that the residual is b1/b2, alternating so nothing is overwritten
        # while it is still the residual.
        s[1](t, st, xp, _ptr(y, nel, dt), p1, hp, ep)
        y = self.s_mlp(hb)
        s[2](t, st, p1, _ptr(y, nel, dt), p2, hp, ep)
        y = self.t_attn(hb)
        s[3](t, st, p2, _ptr(y, nel, dt), p1, hp, ep)
        y = self.t_mlp(hb)
        # (6) the last residual, written in place into the MLP's own output.
        yp = _ptr(y, nel, dt)
        s[4](t, st, p1, yp, yp, 0, ep)
        return y

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        p = self._plan_for(x, c)
        if p is None:
            return self._forward_reference(x, c)
        try:
            g = p.g
            if type(g) is _GPlan:
                if torch.cuda.is_current_stream_capturing():
                    # Somebody is capturing *us*; replaying a graph inside a
                    # capture would bake in this call's endpoint pointers.  The
                    # eager chain below is capturable, and this graph stays.
                    pass
                elif self._wsig() == g.wsig:
                    out = self._forward_graph(g, x, c)
                    if out is not None:
                        return out
                else:
                    # A Parameter was replaced (a cast, a device move, an
                    # assignment): the captured pointers are stale.  Bounded
                    # retries, so a caller that replaces weights every call
                    # degrades to the eager chain instead of re-capturing forever.
                    g.free()
                    p.g = 0 if p.ntry < 4 else False
            elif type(g) is int:
                if g < _WARM_CALLS:
                    p.g = g + 1
                elif _USE_GRAPH and not torch.cuda.is_current_stream_capturing():
                    p.ntry += 1
                    try:
                        p.g = self._capture(p, x, c)
                    except Exception:                       # noqa: BLE001
                        # A failed capture is *silent* by design -- the eager
                        # chain is correct -- which is exactly how a fast path
                        # goes missing.  `OASIS_BLOCK_STRICT=1` makes it loud.
                        if _STRICT:
                            raise
                        p.g = False
                    if type(p.g) is _GPlan:
                        out = self._forward_graph(p.g, x, c)
                        if out is not None:
                            return out
                else:
                    p.g = False
            return self._forward_fused(p, x, c)
        except _Bail:
            pass
        return self._forward_reference(x, c)
