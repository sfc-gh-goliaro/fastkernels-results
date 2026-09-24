"""Attention with pair bias for AlphaFold3.

AttentionPairBias: Used in PairFormer and diffusion transformer. Uses a single
    layer_norm_a for both Q and K (AdaLN or LayerNorm).
CrossAttentionPairBias: Used in atom attention (sequence-local). Uses separate
    layer_norm_a_q and layer_norm_a_k, no layer_norm_z.

Reference: openfold3/core/model/layers/attention_pair_bias.py

At the captured shapes (16 tokens, or 368 atoms in 12 sequence-local blocks)
both operators are entirely launch-bound: the reference composition issues ~45
tiny kernels whose combined arithmetic is a few tens of microseconds' worth of
work on a B200. Each forward here is instead a *single* fused kernel -- the
layer norms, the AdaLN conditioning, the pair-bias projection, q/k/v/gate,
softmax attention and the output projection are phases of one launch separated
by a device-wide barrier. Weight folding (LayerNorm scale into the following
linear, 1/sqrt(c_hidden) into linear_q) happens once, lazily, on the host.

The module structure and parameter names are unchanged, so a state_dict from
the reference implementation loads as-is; anything the fused path does not
cover falls back to the reference composition.
"""

from __future__ import annotations

import math
import os
import threading

import torch
import torch.nn as nn

from ..L1.sigmoid import Sigmoid
from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from .alphafold3_swiglu import AdaLN
from .alphafold3_of3_attention import OF3Attention


def _permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# ---------------------------------------------------------------------------
# Fused CUDA extension (built on first use; None if unavailable).
# ---------------------------------------------------------------------------
_EXT = None
_EXT_TRIED = False
def _phase_mask() -> int:
    """Debug knob: which fused phases to run (all of them by default)."""
    try:
        v = int(os.environ.get("AF3_PHASES", "31"))
    except ValueError:
        return 31
    return v if 1 <= v <= 1023 else 31


_PHASES = _phase_mask()
_EXT_LOCK = threading.Lock()

_CPP_DECLS = """
#include <torch/extension.h>
#include <vector>
int64_t apb_plan(at::Tensor Wb, at::Tensor Wf, at::Tensor an, at::Tensor qkvg,
                 at::Tensor zb, at::Tensor oo, at::Tensor ctr,
                 std::vector<int64_t> iv, double inf, double eps);
at::Tensor apb_run(at::Tensor a, at::Tensor z, c10::optional<at::Tensor> s,
                   c10::optional<at::Tensor> mask, int64_t h, int64_t ph);
int64_t cap_plan(at::Tensor Wb, at::Tensor Wf, at::Tensor aq, at::Tensor ak,
                 at::Tensor qkvg, at::Tensor zb, at::Tensor oo, at::Tensor ctr,
                 std::vector<int64_t> iv, double inf, double eps);
at::Tensor cap_run(at::Tensor a, at::Tensor z, at::Tensor s,
                   c10::optional<at::Tensor> mask, int64_t h, int64_t ph);
"""


def _load_ext():
    global _EXT, _EXT_TRIED
    with _EXT_LOCK:
        if _EXT_TRIED:
            return _EXT
        _EXT_TRIED = True
        try:
            if not torch.cuda.is_available():
                return None
            major, minor = torch.cuda.get_device_capability()
            if major < 8:  # mma.m16n8k16.bf16 needs sm_80+
                return None
            # Build for the local architecture only: the ambient arch list has
            # seven targets, which turns a 40 s build into a ten-minute one.
            prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            try:
                from torch.utils.cpp_extension import load_inline
                _EXT = load_inline(
                    name="af3_apb_fused_v1",
                    cpp_sources=_CPP_DECLS,
                    cuda_sources=_CUDA_SRC,
                    functions=["apb_plan", "apb_run", "cap_plan", "cap_run"],
                    extra_cuda_cflags=["-O3", "--use_fast_math"],
                    verbose=False,
                )
            finally:
                if prev is None:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = prev
        except Exception:
            _EXT = None
        return _EXT


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _swizzle_b(W: torch.Tensor) -> torch.Tensor:
    """Permute a [Nout, K] linear weight into mma.m16n8k16 B-fragment order.

    Result is flat; the fragment pair a lane needs for k-tiles (2kp, 2kp+1) of
    output tile nt lives at element ((nt * KP + kp) * 32 + lane) * 8, so the
    kernel reads it with one 16-byte load and a warp covers 512 contiguous
    bytes.
    """
    Nout, K = W.shape
    Kp = ((K + 31) // 32) * 32
    if Kp != K:
        padded = torch.zeros(Nout, Kp, dtype=W.dtype, device=W.device)
        padded[:, :K] = W
        W = padded
    NT, KP = Nout // 8, Kp // 32
    dev = W.device
    ln = torch.arange(32, device=dev)
    g = (ln >> 2).view(1, 1, 32, 1)
    t2 = ((ln & 3) * 2).view(1, 1, 32, 1)
    e = torch.arange(8, device=dev).view(1, 1, 1, 8)
    nt = torch.arange(NT, device=dev).view(NT, 1, 1, 1)
    kp = torch.arange(KP, device=dev).view(1, KP, 1, 1)
    row = (nt * 8 + g).expand(NT, KP, 32, 8).reshape(-1)
    col = (kp * 32 + (e // 4) * 16 + t2 + (e % 2) + (e % 4 // 2) * 8)
    col = col.expand(NT, KP, 32, 8).reshape(-1)
    return W[row, col].reshape(-1)


class _Packer:
    """Concatenates flattened weight blocks, remembering element offsets."""

    def __init__(self, dtype, device):
        self.parts: list[torch.Tensor] = []
        self.n = 0
        self.dtype = dtype
        self.device = device

    def add(self, t: torch.Tensor) -> int:
        pad = _align8(self.n) - self.n
        if pad:
            self.parts.append(torch.zeros(pad, dtype=self.dtype, device=self.device))
            self.n += pad
        off = self.n
        flat = t.reshape(-1).to(self.dtype)
        self.parts.append(flat)
        self.n += flat.numel()
        return off

    def addb(self, t: torch.Tensor) -> int:
        """Add a linear weight in mma B-fragment order."""
        return self.add(_swizzle_b(t))

    def tensor(self) -> torch.Tensor:
        if not self.parts:
            return torch.zeros(8, dtype=self.dtype, device=self.device)
        return torch.cat(self.parts).contiguous()


_CUDA_SRC = r"""
// ---------------------------------------------------------------------------
// Fused AlphaFold3 attention-with-pair-bias kernels.
//
// Both operators are launch-bound at the captured shapes (16 or 368 rows), so
// each forward is a *single* kernel: the phases that would otherwise be
// separate launches are separated by a device-wide barrier instead.
//
// The barrier uses a monotonically increasing per-slot counter plus a
// host-side generation number, so it never needs to be reset (a reset races
// with blocks still spinning). Every block must be resident for it to
// terminate, so the host clamps the grid with the occupancy API.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cstdlib>

using bf16 = __nv_bfloat16;
#define MAXCZ 16   // CrossAttentionPairBias c_z; keeps zv[] in registers
typedef unsigned long long u64;

#define FULLM 0xffffffffu

__device__ __forceinline__ float b2f(const bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(const float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULLM, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULLM, v, o));
  return v;
}

// Device-wide barrier; slot must be distinct per barrier within one launch.
// mode: bits 0..4 enable phases (debug), bit 8 skips barriers, bit 9 uses
// __nanosleep backoff in the spin.
//
// Each launch consumes exactly one generation of every
// slot, so the tickets are monotonic and never need resetting -- resetting
// races with blocks that are still spinning. Waiters poll a *release* word
// written once per barrier instead of the ticket counter itself: polling a
// stable line is an L2 hit, whereas polling the counter fights every other
// block's atomic for ownership of that line. Ticket and release words sit on
// separate 128-byte lines.
__device__ __forceinline__ void gbar(u64* ctr, int slot, unsigned nb, u64 gen, int mode) {
  if (mode & 256) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    u64* tick = ctr + (size_t)slot * 32;
    u64* rel = tick + 16;
    const u64 want = gen + 1ull;
    if (atomicAdd(tick, 1ull) + 1ull == (u64)nb * want) {
      atomicExch(rel, want);
    } else if (mode & 512) {
      volatile u64* r = rel;
      while (*r < want) __nanosleep(24);
    } else {
      volatile u64* r = rel;
      while (*r < want) { }
    }
  }
  __syncthreads();
}

// Block-wide sum of two values; every thread gets the result. sh needs 2*nw floats.
__device__ __forceinline__ void blk_sum2(float& x, float& y, float* sh, int nw) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  x = warp_sum(x); y = warp_sum(y);
  if (lane == 0) { sh[warp] = x; sh[nw + warp] = y; }
  __syncthreads();
  float sx = 0.f, sy = 0.f;
  for (int i = 0; i < nw; ++i) { sx += sh[i]; sy += sh[nw + i]; }
  __syncthreads();
  x = sx; y = sy;
}

// ---------------------------------------------------------------------------
// One warp: acc[T][4] += A[16][K] * B[nbase + 8t .. ][K]^T   (bf16 tensor core)
// A: 16 rows in shared memory, row stride lda (bf16 elements), K padded to 16.
// B: row-major [Nout][K], row stride ldb. Works for a global or shared B.
// acc layout matches mma.m16n8k16: (row g, col 2t2), (g, 2t2+1), (g+8, ...),
// where g = lane>>2 and t2 = (lane&3)*2.
// ---------------------------------------------------------------------------
template <int T>
__device__ __forceinline__ void warp_mma16(const bf16* __restrict__ As, int lda,
                                           const bf16* __restrict__ Bs, int ldb,
                                           int K, float acc[T][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t2 = (lane & 3) * 2;
  const bf16* ar0 = As + (size_t)g * lda + t2;
  const bf16* ar1 = As + (size_t)(g + 8) * lda + t2;
  const bf16* br = Bs + (size_t)g * ldb + t2;
  for (int k0 = 0; k0 < K; k0 += 16) {
    const uint32_t a0 = *(const uint32_t*)(ar0 + k0);
    const uint32_t a1 = *(const uint32_t*)(ar1 + k0);
    const uint32_t a2 = *(const uint32_t*)(ar0 + k0 + 8);
    const uint32_t a3 = *(const uint32_t*)(ar1 + k0 + 8);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const bf16* bp = br + (size_t)t * 8 * ldb + k0;
      const uint32_t b0 = *(const uint32_t*)(bp);
      const uint32_t b1 = *(const uint32_t*)(bp + 8);
      asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(acc[t][0]), "+f"(acc[t][1]), "+f"(acc[t][2]), "+f"(acc[t][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
}

#define MMA16(acc, a0, a1, a2, a3, b0, b1)                                     \
  asm(                                                                         \
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                    \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                \
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])                  \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// ---------------------------------------------------------------------------
// Swizzled-B GEMM. Every weight matrix is permuted on the host into
// mma-fragment order, so the two B registers a lane needs for a pair of
// k-tiles are 16 contiguous bytes: one uint4 per lane, i.e. 512 contiguous
// bytes per warp instruction (four full sectors) instead of eight scattered
// 32-byte ones. With only a handful of warps of work per SM these phases are
// bound by bytes-in-flight, not by arithmetic, so load width and unroll depth
// dominate everything else.
//
// Swizzled layout of B = [Nout][K], KP = ceil(K/32) slabs:
//   element ((nt * KP + kp) * 32 + lane) * 8 + e
// nt = 8-row output tile, kp = 32-wide k slab, e in [0,8) =
//   {reg0.lo, reg0.hi, reg1.lo, reg1.hi} for k-tile 2kp, then for 2kp+1.
// ---------------------------------------------------------------------------
#define AFRAGS(kb)                                                             \
  const uint32_t a0 = *(const uint32_t*)(ar0 + (kb));                          \
  const uint32_t a1 = *(const uint32_t*)(ar1 + (kb));                          \
  const uint32_t a2 = *(const uint32_t*)(ar0 + (kb) + 8);                      \
  const uint32_t a3 = *(const uint32_t*)(ar1 + (kb) + 8);                      \
  const uint32_t a4 = *(const uint32_t*)(ar0 + (kb) + 16);                     \
  const uint32_t a5 = *(const uint32_t*)(ar1 + (kb) + 16);                     \
  const uint32_t a6 = *(const uint32_t*)(ar0 + (kb) + 24);                     \
  const uint32_t a7 = *(const uint32_t*)(ar1 + (kb) + 24)

// One 16x8 output tile from one swizzled matrix.
// ---------------------------------------------------------------------------
// k-split GEMM. A single warp walking a whole tile's k range has to wait on
// its own loads round after round, which caps a phase at ~0.3 TB/s here. So
// KSPLIT warps share one output tile, each taking KP/KSPLIT slabs, and every
// slab a warp needs is issued before the first mma: one latency round per
// phase instead of one per slab group. The partial accumulators are summed
// through shared memory afterwards.
// ---------------------------------------------------------------------------
#define KSPLIT 4
// MS = compile-time bound on KP / KSPLIT; sizing it per operator keeps the
// staged uint4 registers off the stack.

#define LOADV(dst, B, u)                                                       \
  dst[u] = *(const uint4*)((B) + ((size_t)nt * KP + k0 + u) * 256 + lane * 8)

#define AFRAG_MMA(acc, v, u)                                                   \
  MMA16(acc, a0, a1, a2, a3, v[u].x, v[u].y);                                  \
  MMA16(acc, a4, a5, a6, a7, v[u].z, v[u].w)

template <int MS>
__device__ __forceinline__ void mma_rng1(const bf16* __restrict__ As, int lda,
                                         const bf16* __restrict__ B, int KP,
                                         int nt, int k0, int n, float* acc) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  const bf16* ar0 = As + (size_t)g * lda + t2;
  const bf16* ar1 = As + (size_t)(g + 8) * lda + t2;
  uint4 v[MS];
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) LOADV(v, B, u);
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) {
    AFRAGS((k0 + u) * 32);
    AFRAG_MMA(acc, v, u);
  }
}

template <int MS>
__device__ __forceinline__ void mma_rng2(const bf16* __restrict__ As, int lda,
                                         const bf16* __restrict__ B0,
                                         const bf16* __restrict__ B1, int KP,
                                         int nt, int k0, int n, float* acc0,
                                         float* acc1) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  const bf16* ar0 = As + (size_t)g * lda + t2;
  const bf16* ar1 = As + (size_t)(g + 8) * lda + t2;
  uint4 v0[MS], v1[MS];
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) { LOADV(v0, B0, u); LOADV(v1, B1, u); }
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) {
    AFRAGS((k0 + u) * 32);
    AFRAG_MMA(acc0, v0, u);
    AFRAG_MMA(acc1, v1, u);
  }
}

// Two independent (A, B) pairs -- used for "project and gate" epilogues where
// the gate reads a different activation than the projection.
template <int MS>
__device__ __forceinline__ void mma_rng1x2(const bf16* __restrict__ As0, int lda0,
                                           const bf16* __restrict__ B0, int KP0,
                                           const bf16* __restrict__ As1, int lda1,
                                           const bf16* __restrict__ B1, int KP1,
                                           int nt, int k0f, float* acc0, float* acc1) {
  const int n0 = KP0 / KSPLIT, n1 = KP1 / KSPLIT;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  uint4 v0[MS], v1[MS];
  {
    const int KP = KP0, k0 = k0f * n0;
#pragma unroll
    for (int u = 0; u < MS; ++u) if (u < n0) LOADV(v0, B0, u);
  }
  {
    const int KP = KP1, k0 = k0f * n1;
#pragma unroll
    for (int u = 0; u < MS; ++u) if (u < n1) LOADV(v1, B1, u);
  }
  {
    const bf16* ar0 = As0 + (size_t)g * lda0 + t2;
    const bf16* ar1 = As0 + (size_t)(g + 8) * lda0 + t2;
    const int k0 = k0f * n0;
#pragma unroll
    for (int u = 0; u < MS; ++u) if (u < n0) {
      AFRAGS((k0 + u) * 32);
      AFRAG_MMA(acc0, v0, u);
    }
  }
  {
    const bf16* ar0 = As1 + (size_t)g * lda1 + t2;
    const bf16* ar1 = As1 + (size_t)(g + 8) * lda1 + t2;
    const int k0 = k0f * n1;
#pragma unroll
    for (int u = 0; u < MS; ++u) if (u < n1) {
      AFRAGS((k0 + u) * 32);
      AFRAG_MMA(acc1, v1, u);
    }
  }
}

template <int MS>
__device__ __forceinline__ void mma_rng4(const bf16* __restrict__ As, int lda,
                                         const bf16* __restrict__ B0,
                                         const bf16* __restrict__ B1,
                                         const bf16* __restrict__ B2,
                                         const bf16* __restrict__ B3, int KP,
                                         int nt, int k0, int n, float* c0,
                                         float* c1, float* c2, float* c3) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t2 = (lane & 3) * 2;
  const bf16* ar0 = As + (size_t)g * lda + t2;
  const bf16* ar1 = As + (size_t)(g + 8) * lda + t2;
  uint4 v0[MS], v1[MS], v2[MS], v3[MS];
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) {
    LOADV(v0, B0, u); LOADV(v1, B1, u); LOADV(v2, B2, u); LOADV(v3, B3, u);
  }
#pragma unroll
  for (int u = 0; u < MS; ++u) if (u < n) {
    AFRAGS((k0 + u) * 32);
    AFRAG_MMA(c0, v0, u);
    AFRAG_MMA(c1, v1, u);
    AFRAG_MMA(c2, v2, u);
    AFRAG_MMA(c3, v3, u);
  }
}

// Sum NS accumulator sets over the KSPLIT warps sharing a tile. Lead warps
// (kseg == 0) hold the totals afterwards. red needs NS * blockDim.x * 4 floats.
template <int NS>
__device__ __forceinline__ void ks_reduce(float* red, float** sets) {
  const int tid = threadIdx.x, nt = blockDim.x;
#pragma unroll
  for (int k = 0; k < NS; ++k) {
    float4* slot = (float4*)red + (size_t)k * nt + tid;
    *slot = make_float4(sets[k][0], sets[k][1], sets[k][2], sets[k][3]);
  }
  __syncthreads();
  if (((tid >> 5) & (KSPLIT - 1)) == 0) {
#pragma unroll
    for (int k = 0; k < NS; ++k) {
      const float4* base = (const float4*)red + (size_t)k * nt + tid;
#pragma unroll
      for (int u = 1; u < KSPLIT; ++u) {
        const float4 o = base[u * 32];
        sets[k][0] += o.x; sets[k][1] += o.y; sets[k][2] += o.z; sets[k][3] += o.w;
      }
    }
  }
}

// Stage rows [r0, r0+16) of a row-major [*, K] bf16 matrix into an mma A tile.
// Copies 16 bytes per thread per step and only zeroes what the copy does not
// cover: the scalar version cost ~600 instructions per thread and this kernel
// is instruction-bound (ncu: DRAM 1.6%, issue every 6.1 cycles).
__device__ __forceinline__ void stage_tile(bf16* dst, int lda, const bf16* src,
                                           int srcRowStride, int r0, int rows, int K) {
  const int tid = threadIdx.x, nt = blockDim.x;
  const int nr = min(16, max(rows - r0, 0));
  if (((K | srcRowStride) & 7) == 0) {
    const int K8 = K >> 3;
    for (int i = tid; i < nr * K8; i += nt) {
      const int r = i / K8, c8 = (i - r * K8) << 3;
      *(uint4*)(dst + r * lda + c8) =
          *(const uint4*)(src + (size_t)(r0 + r) * srcRowStride + c8);
    }
  } else {
    for (int i = tid; i < nr * K; i += nt) {
      const int r = i / K, c = i - r * K;
      dst[r * lda + c] = src[(size_t)(r0 + r) * srcRowStride + c];
    }
  }
  for (int i = tid; i < (16 - nr) * lda; i += nt) dst[nr * lda + i] = f2b(0.f);
  if (lda > K)
    for (int i = tid; i < nr * (lda - K); i += nt) {
      const int r = i / (lda - K), c = K + i - r * (lda - K);
      dst[r * lda + c] = f2b(0.f);
    }
  __syncthreads();
}

// ===========================================================================
// AttentionPairBias  (N tokens <= 16, C = H*D channels)
// ===========================================================================
struct APBArgs {
  const bf16* a;
  const bf16* z;
  const bf16* s;
  const bf16* mask;
  const bf16* Wb;
  const float* Wf;
  bf16* out;
  bf16* an;
  bf16* qkvg;
  float* zb;
  bf16* oo;
  u64* ctr;
  u64 gen;
  int N, C, H, D, Cz, Cs, ada, ldaC, ldaCs, KPc, KPs;
  float inf, eps;
  int ph;
  int oQKVG, oWo, oWzF, oWga, oWsa, oWao;
  int fbq, fbzF, fsWzF, flnaw, flnab, fbga, fbao;
};

__global__ __launch_bounds__(512, 1) void apb_kernel(const APBArgs A) {
  extern __shared__ char smem[];
  const int tid = threadIdx.x, nt = blockDim.x, nw = nt >> 5;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x, nb = gridDim.x;
  const int N = A.N, C = A.C, H = A.H, D = A.D, Cz = A.Cz, Cs = A.Cs;
  const int gw = bid * nw + warp, gnw = nb * nw;
  // block-major task index: when there are fewer tasks than warps this
  // spreads them over every SM instead of filling a few blocks.
  const int gwb = bid + nb * warp;
  const int kseg = warp & (KSPLIT - 1), tslot = warp / KSPLIT;
  const int TPB = nw / KSPLIT;

  // ---------------- phase 1: row stats, s-norm, AdaLN-conditioned a --------
  float* ast = (float*)smem;                    // [2][16] mean, rstd of a rows
  bf16* SNs = (bf16*)(ast + 32);                // normalized s, mma A layout
  if (A.ph & 1) {
    for (int r = warp; r < N; r += nw) {
      const bf16* q = A.a + (size_t)r * C;
      float s1 = 0.f, s2 = 0.f;
      for (int c = lane; c < C; c += 32) {
        const float v = b2f(q[c]);
        s1 += v; s2 += v * v;
      }
      s1 = warp_sum(s1); s2 = warp_sum(s2);
      if (lane == 0) {
        const float mu = s1 / C;
        ast[r] = mu;
        ast[16 + r] = rsqrtf(fmaxf(s2 / C - mu * mu, 0.f) + A.eps);
      }
    }
    if (A.ada) {
      for (int i = tid; i < 16 * A.ldaCs; i += nt) SNs[i] = f2b(0.f);
      __syncthreads();
      for (int r = warp; r < N; r += nw) {
        const bf16* q = A.s + (size_t)r * Cs;
        float s1 = 0.f, s2 = 0.f;
        for (int c = lane; c < Cs; c += 32) {
          const float v = b2f(q[c]);
          s1 += v; s2 += v * v;
        }
        s1 = warp_sum(s1); s2 = warp_sum(s2);
        const float mu = s1 / Cs;
        const float rstd = rsqrtf(fmaxf(s2 / Cs - mu * mu, 0.f) + A.eps);
        for (int c = lane; c < Cs; c += 32)
          SNs[r * A.ldaCs + c] = f2b((b2f(q[c]) - mu) * rstd);
      }
    }
    __syncthreads();
  }
  if (!(A.ph & 2)) { } else if (A.ada) {
    float* red = (float*)(SNs + 16 * A.ldaCs);
    const int ntl = C / 8;
    const int nks = A.KPs / KSPLIT;
    const int nit = (ntl + TPB - 1) / TPB;
    for (int it = bid; it < nit; it += nb) {
      const int t = it * TPB + tslot;
      float ag[4] = {0.f, 0.f, 0.f, 0.f};
      float as[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl)
        mma_rng2<6>(SNs, A.ldaCs, A.Wb + A.oWga, A.Wb + A.oWsa, A.KPs, t,
                 kseg * nks, nks, ag, as);
      float* sets[2] = {ag, as};
      ks_reduce<2>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = g + (u >> 1) * 8;
          const int c = t * 8 + t2 + (u & 1);
          if (r >= N) continue;
          const float lna = (b2f(A.a[(size_t)r * C + c]) - ast[r]) * ast[16 + r];
          const float gate = sigmoidf_(ag[u] + A.Wf[A.fbga + c]);
          A.an[(size_t)r * C + c] = f2b(gate * (lna + as[u]));
        }
      }
      __syncthreads();
    }
  } else {
    for (int i = bid * nt + tid; i < N * C; i += nb * nt) {
      const int r = i / C, c = i - r * C;
      const float v = (b2f(A.a[i]) - ast[r]) * ast[16 + r];
      A.an[i] = f2b(v * A.Wf[A.flnaw + c] + A.Wf[A.flnab + c]);
    }
  }
  gbar(A.ctr, 0, nb, A.gen, A.ph);

  // ---------------- phase 2: pair bias + q, k, v, gate projections --------
  if (A.ph & 4) {
    // zb[h][q][k] = layer_norm_z(z) . linear_z + mask bias   (folded weights).
    // One warp per (pair, head): the lane holds Cz/32 of the z row, so the dot
    // is a handful of FMAs plus one shuffle reduction. Doing a whole 128-step
    // dot on 16 lanes instead cost ~640 instructions per pair.
    float* WzT = (float*)smem;                  // [H][Cz], transposed so lanes
    for (int i = tid; i < Cz * H; i += nt) {    // read consecutive addresses
      const int c = i / H, h = i - c * H;
      WzT[h * Cz + c] = b2f(A.Wb[A.oWzF + i]);
    }
    __syncthreads();
    const int nzt = N * N * H;
    for (int task = gwb; task < nzt; task += gnw) {
      const int p = task / H, h = task - p * H;
      const bf16* zr = A.z + (size_t)p * Cz;
      float s1 = 0.f, s2 = 0.f, dot = 0.f;
      for (int c = lane; c < Cz; c += 32) {
        const float v = b2f(zr[c]);
        s1 += v; s2 += v * v;
        dot += v * WzT[h * Cz + c];
      }
      s1 = warp_sum(s1); s2 = warp_sum(s2); dot = warp_sum(dot);
      if (lane == 0) {
        const float mu = s1 / Cz;
        const float rstd = rsqrtf(fmaxf(s2 / Cz - mu * mu, 0.f) + A.eps);
        float v = rstd * (dot - mu * A.Wf[A.fsWzF + h]) + A.Wf[A.fbzF + h];
        const int k = p - (p / N) * N;
        if (A.mask) v += A.inf * (b2f(A.mask[k]) - 1.f);
        A.zb[(size_t)h * N * N + p] = v;
      }
    }
    __syncthreads();
    bf16* ANs = (bf16*)smem;
    float* red = (float*)(ANs + 16 * A.ldaC);
    stage_tile(ANs, A.ldaC, A.an, C, 0, N, C);
    const int per = C / 8, ntl = 4 * per;
    const int nks = A.KPc / KSPLIT;
    const int nit = (ntl + TPB - 1) / TPB;
    for (int it = bid; it < nit; it += nb) {
      const int t = it * TPB + tslot;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl)
        mma_rng1<6>(ANs, A.ldaC, A.Wb + A.oQKVG, A.KPc, t, kseg * nks, nks, acc);
      float* sets[1] = {acc};
      ks_reduce<1>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int m = t / per, jt = t - m * per;
        const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = g + (u >> 1) * 8;
          const int c = jt * 8 + t2 + (u & 1);
          if (r >= N) continue;
          float v = acc[u];
          if (m == 0) v += A.Wf[A.fbq + c];
          A.qkvg[((size_t)m * N + r) * C + c] = f2b(v);
        }
      }
      __syncthreads();
    }
  }
  gbar(A.ctr, 1, nb, A.gen, A.ph);

  // ---------------- phase 3: attention, one block per head ----------------
  if (A.ph & 8) {
    // rows padded by one word so the scores loop (each thread walks a whole
    // row of K) is bank-conflict free.
    const int ldd = D + 1;
    float* Qs = (float*)smem;
    float* Ks = Qs + N * ldd;
    float* Vs = Ks + N * ldd;
    float* Gs = Vs + N * ldd;
    float* sc = Gs + N * ldd;
    for (int h = bid; h < H; h += nb) {
      __syncthreads();
      for (int i = tid; i < N * D; i += nt) {
        const int r = i / D, d = i - r * D;
        const size_t base = (size_t)r * C + h * D + d;
        const int o = r * ldd + d;
        Qs[o] = b2f(A.qkvg[base]);
        Ks[o] = b2f(A.qkvg[(size_t)N * C + base]);
        Vs[o] = b2f(A.qkvg[(size_t)2 * N * C + base]);
        Gs[o] = b2f(A.qkvg[(size_t)3 * N * C + base]);
      }
      __syncthreads();
      for (int i = tid; i < N * N; i += nt) {
        const int q = i / N, k = i - q * N;
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += Qs[q * ldd + d] * Ks[k * ldd + d];
        sc[i] = acc + A.zb[((size_t)h * N + q) * N + k];
      }
      __syncthreads();
      for (int q = warp; q < N; q += nw) {
        float m = -INFINITY, sm = 0.f;
        for (int k = lane; k < N; k += 32) m = fmaxf(m, sc[q * N + k]);
        m = warp_max(m);
        for (int k = lane; k < N; k += 32) {
          const float e = __expf(sc[q * N + k] - m);
          sc[q * N + k] = e;
          sm += e;
        }
        sm = warp_sum(sm);
        const float inv = 1.f / sm;
        for (int k = lane; k < N; k += 32) sc[q * N + k] *= inv;
      }
      __syncthreads();
      for (int i = tid; i < N * D; i += nt) {
        const int r = i / D, d = i - r * D;
        const int o = r * ldd + d;
        float acc = 0.f;
        for (int k = 0; k < N; ++k) acc += sc[r * N + k] * Vs[k * ldd + d];
        A.oo[(size_t)r * C + h * D + d] = f2b(acc * sigmoidf_(Gs[o]));
      }
    }
  }
  gbar(A.ctr, 2, nb, A.gen, A.ph);

  // ---------------- phase 4: output projection + AdaLN-out gate -----------
  if (A.ph & 16) {
    bf16* OOs = (bf16*)smem;
    bf16* Ss = OOs + 16 * A.ldaC;
    float* red = (float*)(Ss + 16 * A.ldaCs);
    stage_tile(OOs, A.ldaC, A.oo, C, 0, N, C);
    if (A.ada) stage_tile(Ss, A.ldaCs, A.s, Cs, 0, N, Cs);
    const int ntl = C / 8;
    const int nks = A.KPc / KSPLIT;
    const int nit = (ntl + TPB - 1) / TPB;
    for (int it = bid; it < nit; it += nb) {
      const int t = it * TPB + tslot;
      float ao[4] = {0.f, 0.f, 0.f, 0.f};
      float ag[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl) {
        if (A.ada)
          mma_rng1x2<6>(OOs, A.ldaC, A.Wb + A.oWo, A.KPc,
                     Ss, A.ldaCs, A.Wb + A.oWao, A.KPs, t, kseg, ao, ag);
        else
          mma_rng1<6>(OOs, A.ldaC, A.Wb + A.oWo, A.KPc, t, kseg * nks, nks, ao);
      }
      float* sets[2] = {ao, ag};
      if (A.ada) ks_reduce<2>(red, sets);
      else ks_reduce<1>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = g + (u >> 1) * 8;
          const int c = t * 8 + t2 + (u & 1);
          if (r >= N) continue;
          float v = ao[u];
          if (A.ada) v *= sigmoidf_(ag[u] + A.Wf[A.fbao + c]);
          A.out[(size_t)r * C + c] = f2b(v);
        }
      }
      __syncthreads();
    }
  }
}

// ===========================================================================
// CrossAttentionPairBias (sequence-local blocked attention over Na atoms)
// ===========================================================================
struct CAPArgs {
  const bf16* a;
  const bf16* z;
  const bf16* s;
  const bf16* mask;
  const bf16* Wb;
  const float* Wf;
  bf16* out;
  bf16* aq;
  bf16* ak;
  bf16* qkvg;
  float* zb;
  bf16* oo;
  float* sstat;
  float* astat;
  u64* ctr;
  u64 gen;
  int Na, P, NB, nq, nk, C, H, D, Cz, Cs;
  int ldaC, ldaCs, ldD, ldK, Dp, KPc, KPs;
  float inf, eps;
  int ph;
  int oQKVG, oWo, oWzT, oWgq, oWsq, oWgk, oWsk, oWao;
  int fbq, fbgq, fbgk, fbao;
};

// key gather indices of one sequence-local block (mirrors _get_block_key_indices)
__device__ __forceinline__ void key_index(int b, int j, int nq, int nk, int n_real,
                                          int& idx, bool& invalid) {
  const int base = nq / 2 + b * nq - nk / 2;
  const int under = max(0, -base);
  const int over = max(0, base + nk - 1 - (n_real - 1));
  const int shift = under > 0 ? under : -over;
  const int fin = base + j + shift;
  invalid = (fin < 0) || (fin >= n_real);
  idx = min(max(fin, 0), max(n_real - 1, 0));
}

__global__ __launch_bounds__(512, 1) void cap_kernel(const CAPArgs A) {
  extern __shared__ char smem[];
  const int tid = threadIdx.x, nt = blockDim.x, nw = nt >> 5;
  const int lane = tid & 31, warp = tid >> 5;
  const int bid = blockIdx.x, nb = gridDim.x;
  const int Na = A.Na, P = A.P, NB = A.NB, nq = A.nq, nk = A.nk;
  const int C = A.C, H = A.H, D = A.D, Cz = A.Cz, Cs = A.Cs;
  const int gw = bid * nw + warp, gnw = nb * nw;
  // block-major task index: when there are fewer tasks than warps this
  // spreads them over every SM instead of filling a few blocks.
  const int gwb = bid + nb * warp;
  const int kseg = warp & (KSPLIT - 1), tslot = warp / KSPLIT;
  const int TPB = nw / KSPLIT;
  const int MT = P / 16, NT = C / 8;

  // ---------------- phase 0: row norms of s and a, pair bias --------------
  if (A.ph & 1) {
    float* Wzs = (float*)smem;                  // WzT[Cz][H]
    for (int i = tid; i < Cz * H; i += nt) Wzs[i] = b2f(A.Wb[A.oWzT + i]);
    __syncthreads();
    // one warp per (block, query, 32-key group), walked block-major so the work
    // lands on every SM rather than filling the low blocks first.
    const int jg = nk / 32;
    for (int t = gwb; t < P * jg; t += gnw) {
      {
        // zb[b][h][i][j] = linear_z(z)  for one (b, i). z is the largest input
        // here (1.5 MB), so it is pulled in 16-byte chunks; the little
        // [Cz][H] weight is broadcast from shared memory.
        const int bi = t / jg, jbase = (t - bi * jg) * 32;
        const int b = bi / nq, i = bi - b * nq;
        const bf16* zr = A.z + ((size_t)b * nq + i) * nk * Cz;
        {
          const int j = jbase + lane;
          const bf16* zp = zr + (size_t)j * Cz;
          float zv[MAXCZ];
#pragma unroll
          for (int c8 = 0; c8 < MAXCZ; c8 += 8) {
            if (c8 < Cz) {
              const uint4 w = *(const uint4*)(zp + c8);
              const bf16* wb = (const bf16*)&w;
#pragma unroll
              for (int e = 0; e < 8; ++e) zv[c8 + e] = b2f(wb[e]);
            }
          }
          float acc[8];
#pragma unroll
          for (int h = 0; h < 8; ++h) acc[h] = 0.f;
#pragma unroll
          for (int c = 0; c < MAXCZ; ++c) {
            if (c >= Cz) break;
#pragma unroll
            for (int h = 0; h < 8; ++h)
              if (h < H) acc[h] += zv[c] * Wzs[c * H + h];
          }
          for (int h = 0; h < H; ++h)
            A.zb[(((size_t)b * H + h) * nq + i) * nk + j] = acc[h];
        }
      }
    }
  }
  __syncthreads();

  // ---------------- phase 1: AdaLN for query and key rows ----------------
  if (A.ph & 2) {
    bf16* LNs = (bf16*)smem;                    // normalized s, mma A layout
    bf16* LNa = LNs + 16 * A.ldaCs;             // normalized a, plain [16][C]
    float* red = (float*)(LNa + 16 * C);
    const int ntl = MT * NT;
    const int nks = A.KPs / KSPLIT;
    const int nit = (ntl + TPB - 1) / TPB;
    int staged = -1;
    for (int it = bid; it < nit; it += nb) {
      const int mt = (it * TPB) / NT;          // uniform: TPB divides NT
      if (mt != staged) {
        staged = mt;
        const int r0 = mt * 16;
        for (int i = tid; i < 16 * A.ldaCs; i += nt) LNs[i] = f2b(0.f);
        __syncthreads();
        for (int r = warp; r < 16; r += nw) {
          const int row = r0 + r;
          if (row >= Na) {
            for (int c = lane; c < C; c += 32) LNa[r * C + c] = f2b(0.f);
            continue;
          }
          const bf16* sp = A.s + (size_t)row * Cs;
          float s1 = 0.f, s2 = 0.f;
          for (int c = lane; c < Cs; c += 32) {
            const float v = b2f(sp[c]);
            s1 += v; s2 += v * v;
          }
          s1 = warp_sum(s1); s2 = warp_sum(s2);
          const float smu = s1 / Cs;
          const float srs = rsqrtf(fmaxf(s2 / Cs - smu * smu, 0.f) + A.eps);
          for (int c = lane; c < Cs; c += 32)
            LNs[r * A.ldaCs + c] = f2b((b2f(sp[c]) - smu) * srs);
          const bf16* ap = A.a + (size_t)row * C;
          float a1 = 0.f, a2 = 0.f;
          for (int c = lane; c < C; c += 32) {
            const float v = b2f(ap[c]);
            a1 += v; a2 += v * v;
          }
          a1 = warp_sum(a1); a2 = warp_sum(a2);
          const float amu = a1 / C;
          const float ars = rsqrtf(fmaxf(a2 / C - amu * amu, 0.f) + A.eps);
          for (int c = lane; c < C; c += 32)
            LNa[r * C + c] = f2b((b2f(ap[c]) - amu) * ars);
        }
        __syncthreads();
      }
      const int t = it * TPB + tslot;
      float agq[4] = {0.f, 0.f, 0.f, 0.f}, asq[4] = {0.f, 0.f, 0.f, 0.f};
      float agk[4] = {0.f, 0.f, 0.f, 0.f}, ask[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl)
        mma_rng4<2>(LNs, A.ldaCs, A.Wb + A.oWgq, A.Wb + A.oWsq, A.Wb + A.oWgk,
                 A.Wb + A.oWsk, A.KPs, t - mt * NT, kseg * nks, nks,
                 agq, asq, agk, ask);
      float* sets[4] = {agq, asq, agk, ask};
      ks_reduce<4>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int g = lane >> 2, t2 = (lane & 3) * 2;
        const int nt8 = t - mt * NT;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = mt * 16 + g + (u >> 1) * 8;
          const int c = nt8 * 8 + t2 + (u & 1);
          if (r >= P) continue;
          const float lna = b2f(LNa[(r - mt * 16) * C + c]);
          A.aq[(size_t)r * C + c] =
              f2b(sigmoidf_(agq[u] + A.Wf[A.fbgq + c]) * (lna + asq[u]));
          A.ak[(size_t)r * C + c] =
              f2b(sigmoidf_(agk[u] + A.Wf[A.fbgk + c]) * (lna + ask[u]));
        }
      }
      __syncthreads();
    }
  }
  gbar(A.ctr, 0, nb, A.gen, A.ph);

  // ---------------- phase 2: q, gate from a_q; k, v from a_k -------------
  if (A.ph & 4) {
    bf16* AQs = (bf16*)smem;
    bf16* AKs = AQs + 16 * A.ldaC;
    float* red = (float*)(AKs + 16 * A.ldaC);
    const int ntl = MT * NT;
    const int nks = A.KPc / KSPLIT;
    const int nit = (ntl + TPB - 1) / TPB;
    const int per = C / 8;
    const bf16* Wqk = A.Wb + A.oQKVG;
    const size_t mstride = (size_t)per * A.KPc * 256;
    int staged = -1;
    for (int it = bid; it < nit; it += nb) {
      const int mt = (it * TPB) / NT;
      if (mt != staged) {
        stage_tile(AQs, A.ldaC, A.aq, C, mt * 16, P, C);
        stage_tile(AKs, A.ldaC, A.ak, C, mt * 16, P, C);
        staged = mt;
      }
      const int t = it * TPB + tslot;
      const int nt8 = t - mt * NT;
      float aq_[4] = {0.f, 0.f, 0.f, 0.f}, ag_[4] = {0.f, 0.f, 0.f, 0.f};
      float ak_[4] = {0.f, 0.f, 0.f, 0.f}, av_[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl) {
        mma_rng2<2>(AQs, A.ldaC, Wqk, Wqk + 3 * mstride, A.KPc, nt8,
                 kseg * nks, nks, aq_, ag_);
        mma_rng2<2>(AKs, A.ldaC, Wqk + mstride, Wqk + 2 * mstride, A.KPc, nt8,
                 kseg * nks, nks, ak_, av_);
      }
      float* sets[4] = {aq_, ag_, ak_, av_};
      ks_reduce<4>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = mt * 16 + g + (u >> 1) * 8;
          const int c = nt8 * 8 + t2 + (u & 1);
          if (r >= P) continue;
          const size_t o = (size_t)r * C + c;
          A.qkvg[o] = f2b(aq_[u] + A.Wf[A.fbq + c]);
          A.qkvg[(size_t)P * C + o] = f2b(ak_[u]);
          A.qkvg[(size_t)2 * P * C + o] = f2b(av_[u]);
          A.qkvg[(size_t)3 * P * C + o] = f2b(ag_[u]);
        }
      }
      __syncthreads();
    }
  }
  gbar(A.ctr, 1, nb, A.gen, A.ph);

  // ---------------- phase 3: blocked attention, one block per (block, head)
  if (A.ph & 8) {
    bf16* Qs = (bf16*)smem;                      // [16][ldD]
    bf16* Ks = Qs + 16 * A.ldD;                  // [nk][ldD]
    bf16* Vt = Ks + nk * A.ldD;                  // [D][ldK]
    bf16* Ps = Vt + D * A.ldK;                   // [16][ldK]
    float* sc = (float*)(Ps + 16 * A.ldK);       // [16][nk]
    int* kidx = (int*)(sc + 16 * nk);            // [nk]
    float* kval = (float*)(kidx + nk);           // [nk]
    float* shr = kval + nk;                      // [2*nw]

    // n_real = sum(mask); identical in every block
    float nrf = 0.f, dummy = 0.f;
    if (A.mask) {
      for (int c = tid; c < Na; c += nt) nrf += b2f(A.mask[c]);
      blk_sum2(nrf, dummy, shr, nw);
    } else {
      nrf = (float)Na;
    }
    const int n_real = (int)lrintf(nrf);

    // (block, head, 16-query tile): splitting the query dimension doubles the
    // number of blocks with work, which is what this phase is short of.
    const int qmt = nq / 16;
    for (int task = bid; task < NB * H * qmt; task += nb) {
      const int qt = task % qmt, rest = task / qmt;
      const int b = rest / H, h = rest - b * H;
      const int q0 = qt * 16;
      __syncthreads();
      for (int j = tid; j < nk; j += nt) {
        int idx; bool inv;
        key_index(b, j, nq, nk, n_real, idx, inv);
        kidx[j] = idx;
        kval[j] = inv ? 0.f : (A.mask ? b2f(A.mask[idx]) : 1.f);
      }
      __syncthreads();   // publish kidx before the gather below reads it
      // No zero fill beyond that: the mma k range is exactly D, so the lda
      // padding is never read.
      for (int i = tid; i < 16 * D; i += nt) {
        const int r = i / D, d = i - r * D;
        const int gr = b * nq + q0 + r;
        Qs[r * A.ldD + d] = A.qkvg[(size_t)gr * C + h * D + d];
      }
      for (int i = tid; i < nk * D; i += nt) {
        const int j = i / D, d = i - j * D;
        const size_t src = (size_t)kidx[j] * C + h * D + d;
        Ks[j * A.ldD + d] = A.qkvg[(size_t)P * C + src];
        Vt[d * A.ldK + j] = A.qkvg[(size_t)2 * P * C + src];
      }
      __syncthreads();
      // scores
      {
        for (int t = warp; t < nk / 8; t += nw) {
          float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
          warp_mma16<1>(Qs, A.ldD, Ks + (size_t)t * 8 * A.ldD, A.ldD, A.Dp, acc);
          const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int i = g + (u >> 1) * 8;
            const int j = t * 8 + t2 + (u & 1);
            const int gr = b * nq + q0 + i;
            const float mq = (gr < Na) ? (A.mask ? b2f(A.mask[gr]) : 1.f) : 0.f;
            sc[i * nk + j] =
                acc[0][u] + A.zb[(((size_t)b * H + h) * nq + q0 + i) * nk + j]
                + A.inf * (mq * kval[j] - 1.f);
          }
        }
      }
      __syncthreads();
      // softmax over keys
      for (int i = warp; i < 16; i += nw) {
        float m = -INFINITY;
        for (int j = lane; j < nk; j += 32) m = fmaxf(m, sc[i * nk + j]);
        m = warp_max(m);
        float sm = 0.f;
        for (int j = lane; j < nk; j += 32) {
          const float e = __expf(sc[i * nk + j] - m);
          sc[i * nk + j] = e;
          sm += e;
        }
        sm = warp_sum(sm);
        const float inv = 1.f / sm;
        for (int j = lane; j < nk; j += 32) Ps[i * A.ldK + j] = f2b(sc[i * nk + j] * inv);
      }
      __syncthreads();
      // context
      {
        for (int t = warp; t < D / 8; t += nw) {
          float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
          warp_mma16<1>(Ps, A.ldK, Vt + (size_t)t * 8 * A.ldK, A.ldK, nk, acc);
          const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int i = g + (u >> 1) * 8;
            const int d = t * 8 + t2 + (u & 1);
            const size_t o = (size_t)(b * nq + q0 + i) * C + h * D + d;
            A.oo[o] = f2b(acc[0][u] * sigmoidf_(b2f(A.qkvg[(size_t)3 * P * C + o])));
          }
        }
      }
    }
  }
  gbar(A.ctr, 2, nb, A.gen, A.ph);

  // ---------------- phase 4: output projection + AdaLN-out gate ----------
  if (A.ph & 16) {
    bf16* OOs = (bf16*)smem;
    bf16* Ss = OOs + 16 * A.ldaC;
    float* red = (float*)(Ss + 16 * A.ldaCs);
    const int MTo = (Na + 15) / 16;
    const int ntl = MTo * NT;
    const int nit = (ntl + TPB - 1) / TPB;
    int staged = -1;
    for (int it = bid; it < nit; it += nb) {
      const int mt = (it * TPB) / NT;
      if (mt != staged) {
        stage_tile(OOs, A.ldaC, A.oo, C, mt * 16, P, C);
        stage_tile(Ss, A.ldaCs, A.s, Cs, mt * 16, Na, Cs);
        staged = mt;
      }
      const int t = it * TPB + tslot;
      const int nt8 = t - mt * NT;
      float ao[4] = {0.f, 0.f, 0.f, 0.f}, ag[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < ntl)
        mma_rng1x2<2>(OOs, A.ldaC, A.Wb + A.oWo, A.KPc,
                   Ss, A.ldaCs, A.Wb + A.oWao, A.KPs, nt8, kseg, ao, ag);
      float* sets[2] = {ao, ag};
      ks_reduce<2>(red, sets);
      if (kseg == 0 && t < ntl) {
        const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int r = mt * 16 + g + (u >> 1) * 8;
          const int c = nt8 * 8 + t2 + (u & 1);
          if (r >= Na) continue;
          A.out[(size_t)r * C + c] =
              f2b(ao[u] * sigmoidf_(ag[u] + A.Wf[A.fbao + c]));
        }
      }
      __syncthreads();
    }
  }
}

// ===========================================================================
// Host side: one plan per (operator, shape); one kernel launch per forward.
// ===========================================================================
#define CDIV(x, y) (((x) + (y) - 1) / (y))

// Shared-memory row stride for a 16-row A tile of K columns: K is rounded up to
// a multiple of 16 for the mma k-loop, then padded by 8 bf16 so the eight
// lane-groups of a warp land on distinct banks.
static inline int ld_for(int K) {
  const int Kp = CDIV(K, 32) * 32;
  return CDIV(Kp, 64) * 64 + 8;
}

// Grid width / block size are tunable so the phase mix can be swept.
static inline int env_int(const char* name, int dflt) {
  const char* v = getenv(name);
  if (!v || !*v) return dflt;
  const int x = atoi(v);
  return x > 0 ? x : dflt;
}

// Largest grid whose blocks are all guaranteed co-resident. The device-wide
// barrier deadlocks if a block has not been scheduled, so never exceed this.
static int resident_grid(const void* fn, int threads, int smem, int want) {
  int dev = 0, sms = 1, per_sm = 1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, threads, smem)
      != cudaSuccess || per_sm < 1) {
    per_sm = 1;
  }
  const int cap = sms * per_sm;
  int nb = want < cap ? want : cap;
  return nb < 1 ? 1 : nb;
}

struct APBPlan {
  APBArgs a;
  at::Tensor Wb, Wf, an, qkvg, zb, oo, ctr;
  int nb = 0, nthreads = 256, smem = 0;
  u64 gen = 0;
};
struct CAPPlan {
  CAPArgs a;
  at::Tensor Wb, Wf, aq, ak, qkvg, zb, oo, ctr;
  int nb = 0, nthreads = 256, smem = 0;
  u64 gen = 0;
};
static std::vector<APBPlan*> gAPB;
static std::vector<CAPPlan*> gCAP;

int64_t apb_plan(at::Tensor Wb, at::Tensor Wf, at::Tensor an, at::Tensor qkvg,
                 at::Tensor zb, at::Tensor oo, at::Tensor ctr,
                 std::vector<int64_t> iv, double inf, double eps) {
  auto* p = new APBPlan();
  p->Wb = Wb; p->Wf = Wf; p->an = an; p->qkvg = qkvg;
  p->zb = zb; p->oo = oo; p->ctr = ctr;
  APBArgs& A = p->a;
  A.Wb = (const bf16*)Wb.data_ptr();
  A.Wf = Wf.data_ptr<float>();
  A.an = (bf16*)an.data_ptr();
  A.qkvg = (bf16*)qkvg.data_ptr();
  A.zb = zb.data_ptr<float>();
  A.oo = (bf16*)oo.data_ptr();
  A.ctr = (u64*)ctr.data_ptr();
  int i = 0;
  A.N = (int)iv[i++]; A.C = (int)iv[i++]; A.H = (int)iv[i++]; A.D = (int)iv[i++];
  A.Cz = (int)iv[i++]; A.Cs = (int)iv[i++]; A.ada = (int)iv[i++];
  A.oQKVG = (int)iv[i++]; A.oWo = (int)iv[i++]; A.oWzF = (int)iv[i++];
  A.oWga = (int)iv[i++]; A.oWsa = (int)iv[i++]; A.oWao = (int)iv[i++];
  A.fbq = (int)iv[i++]; A.fbzF = (int)iv[i++]; A.fsWzF = (int)iv[i++];
  A.flnaw = (int)iv[i++]; A.flnab = (int)iv[i++]; A.fbga = (int)iv[i++];
  A.fbao = (int)iv[i++];
  A.inf = (float)inf; A.eps = (float)eps;
  A.ldaC = ld_for(A.C); A.ldaCs = ld_for(A.Cs);
  A.KPc = CDIV(A.C, 32); A.KPs = CDIV(A.Cs, 32);
  p->nthreads = env_int("AF3_APB_THREADS", 512);
  const int nw = p->nthreads / 32;
  p->nb = env_int("AF3_APB_BLOCKS", 148);
  const size_t red = (size_t)2 * p->nthreads * 4 * sizeof(float);
  const size_t s1 = (size_t)16 * A.ldaCs * sizeof(bf16) + red;
  const size_t s2 = (size_t)16 * A.ldaC * sizeof(bf16) + red;
  const size_t s3 = (size_t)(4 * A.N * (A.D + 1) + A.N * A.N) * sizeof(float);
  const size_t s4 = (size_t)16 * (A.ldaC + A.ldaCs) * sizeof(bf16) + red;
  // phase 2 reuses the tile region for the folded linear_z weight
  const size_t s0 = (size_t)(A.Cz * A.H) * sizeof(float);
  size_t sm = std::max(std::max(s0, s1 + 32 * sizeof(float)),
                       std::max(s2, std::max(s3, s4)));
  p->smem = (int)sm;
  cudaFuncSetAttribute((void*)apb_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       p->smem);
  p->nb = resident_grid((const void*)apb_kernel, p->nthreads, p->smem, p->nb);
  gAPB.push_back(p);
  return (int64_t)gAPB.size() - 1;
}

at::Tensor apb_run(at::Tensor a, at::Tensor z, c10::optional<at::Tensor> s,
                   c10::optional<at::Tensor> mask, int64_t h, int64_t ph) {
  APBPlan* p = gAPB[(size_t)h];
  APBArgs A = p->a;
  A.a = (const bf16*)a.data_ptr();
  A.z = (const bf16*)z.data_ptr();
  A.s = s.has_value() ? (const bf16*)s->data_ptr() : nullptr;
  A.mask = mask.has_value() ? (const bf16*)mask->data_ptr() : nullptr;
  auto out = at::empty_like(a);
  A.out = (bf16*)out.data_ptr();
  A.ph = (int)ph;
  A.gen = (ph & 256) ? p->gen : p->gen++;
  apb_kernel<<<p->nb, p->nthreads, p->smem, c10::cuda::getCurrentCUDAStream()>>>(A);
  return out;
}

int64_t cap_plan(at::Tensor Wb, at::Tensor Wf, at::Tensor aq, at::Tensor ak,
                 at::Tensor qkvg, at::Tensor zb, at::Tensor oo, at::Tensor ctr,
                 std::vector<int64_t> iv, double inf, double eps) {
  auto* p = new CAPPlan();
  p->Wb = Wb; p->Wf = Wf; p->aq = aq; p->ak = ak;
  p->qkvg = qkvg; p->zb = zb; p->oo = oo; p->ctr = ctr;
  CAPArgs& A = p->a;
  A.Wb = (const bf16*)Wb.data_ptr();
  A.Wf = Wf.data_ptr<float>();
  A.aq = (bf16*)aq.data_ptr();
  A.ak = (bf16*)ak.data_ptr();
  A.qkvg = (bf16*)qkvg.data_ptr();
  A.zb = zb.data_ptr<float>();
  A.oo = (bf16*)oo.data_ptr();
  A.ctr = (u64*)ctr.data_ptr();
  int i = 0;
  A.Na = (int)iv[i++]; A.P = (int)iv[i++]; A.NB = (int)iv[i++];
  A.nq = (int)iv[i++]; A.nk = (int)iv[i++]; A.C = (int)iv[i++];
  A.H = (int)iv[i++]; A.D = (int)iv[i++]; A.Cz = (int)iv[i++]; A.Cs = (int)iv[i++];
  A.oQKVG = (int)iv[i++]; A.oWo = (int)iv[i++]; A.oWzT = (int)iv[i++];
  A.oWgq = (int)iv[i++]; A.oWsq = (int)iv[i++]; A.oWgk = (int)iv[i++];
  A.oWsk = (int)iv[i++]; A.oWao = (int)iv[i++];
  A.fbq = (int)iv[i++]; A.fbgq = (int)iv[i++]; A.fbgk = (int)iv[i++];
  A.fbao = (int)iv[i++];
  A.inf = (float)inf; A.eps = (float)eps;
  A.ldaC = ld_for(A.C); A.ldaCs = ld_for(A.Cs);
  A.ldD = ld_for(A.D); A.ldK = ld_for(A.nk);
  A.Dp = CDIV(A.D, 16) * 16;
  A.KPc = CDIV(A.C, 32); A.KPs = CDIV(A.Cs, 32);
  p->nthreads = env_int("AF3_CAP_THREADS", 512);
  const int nw = p->nthreads / 32;
  p->nb = env_int("AF3_CAP_BLOCKS", 148);
  const size_t red = (size_t)4 * p->nthreads * 4 * sizeof(float);
  const size_t s1 = (size_t)16 * (A.ldaCs + A.C) * sizeof(bf16) + red;
  const size_t s2 = (size_t)2 * 16 * A.ldaC * sizeof(bf16) + red;
  const size_t s4 = (size_t)16 * (A.ldaC + A.ldaCs) * sizeof(bf16) + red;
  size_t s3 = (size_t)(16 * A.ldD + A.nk * A.ldD + A.D * A.ldK + 16 * A.ldK)
              * sizeof(bf16);
  s3 += (size_t)16 * A.nk * sizeof(float) + (size_t)A.nk * (sizeof(int) + sizeof(float))
        + (size_t)4 * nw * sizeof(float);
  const size_t s0 = (size_t)(A.Cz * A.H) * sizeof(float);
  s3 = std::max(s3, s0);
  size_t sm = std::max(std::max(s1, s2), std::max(s3, s4));
  p->smem = (int)sm;
  cudaFuncSetAttribute((void*)cap_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       p->smem);
  p->nb = resident_grid((const void*)cap_kernel, p->nthreads, p->smem, p->nb);
  gCAP.push_back(p);
  return (int64_t)gCAP.size() - 1;
}

at::Tensor cap_run(at::Tensor a, at::Tensor z, at::Tensor s,
                   c10::optional<at::Tensor> mask, int64_t h, int64_t ph) {
  CAPPlan* p = gCAP[(size_t)h];
  CAPArgs A = p->a;
  A.a = (const bf16*)a.data_ptr();
  A.z = (const bf16*)z.data_ptr();
  A.s = (const bf16*)s.data_ptr();
  A.mask = mask.has_value() ? (const bf16*)mask->data_ptr() : nullptr;
  auto out = at::empty_like(a);
  A.out = (bf16*)out.data_ptr();
  A.ph = (int)ph;
  A.gen = (ph & 256) ? p->gen : p->gen++;
  cap_kernel<<<p->nb, p->nthreads, p->smem, c10::cuda::getCurrentCUDAStream()>>>(A);
  return out;
}

"""


def _invalidate_plans(module, incompatible_keys):
    """Weights changed -> the folded/packed copies are stale."""
    module._plans.clear()
    module._fused = None


def _hook_state_dict(module):
    """Drop cached plans when a state_dict is loaded, if torch supports it."""
    for name in ("register_load_state_dict_post_hook",
                 "_register_load_state_dict_post_hook"):
        fn = getattr(type(module), name, None)
        if fn is None:
            continue
        try:
            fn(module, _invalidate_plans)
            return
        except Exception:
            pass


class AttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Attention with pair bias.

    When use_ada_layer_norm is True, uses two separate AdaLN instances
    (layer_norm_a_q, layer_norm_a_k) for query and key normalization,
    plus a linear_ada_out for output gating.

    Reference: openfold3/core/model/layers/attention_pair_bias.py

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 128,
        c_hidden: int = 32,
        no_heads: int = 4,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm

        if use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a = LayerNorm(c_q)

        self.layer_norm_z = LayerNorm(
            c_z, create_offset=not use_ada_layer_norm,
        )
        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._plans: dict = {}
        self._fused = None
        _hook_state_dict(self)

    # -- fused path ---------------------------------------------------------
    def _make_plan(self, a, z, s, mask):
        ext = _load_ext()
        if ext is None:
            return None
        C, H, D = self.c_q, self.mha.no_heads, self.mha.c_hidden
        Cz = self.c_z
        ada = bool(self.use_ada_layer_norm)
        Cs = self.c_s if ada else 8
        if self.mha.linear_g is None or H * D != C or C % 8 or H > 32 or Cz > 256:
            return None
        if a.dtype != torch.bfloat16 or not a.is_cuda or a.dim() < 2:
            return None
        N = a.shape[-2]
        if N > 16 or a.shape[-1] != C or a.numel() != N * C:
            return None
        if not (a.is_contiguous() and z.is_contiguous()):
            return None
        if z.dtype != torch.bfloat16 or z.numel() != N * N * Cz:
            return None
        if ada:
            if s is None or s.dtype != torch.bfloat16 or not s.is_contiguous():
                return None
            if s.numel() != N * self.c_s:
                return None
        if mask is not None:
            if (mask.dtype != torch.bfloat16 or not mask.is_contiguous()
                    or mask.numel() != N):
                return None
        eps = self.layer_norm_z.eps
        inner = ([self.layer_norm_a.layer_norm_a, self.layer_norm_a.layer_norm_s]
                 if ada else [self.layer_norm_a])
        if any(abs(m.eps - eps) > 0 for m in inner):
            return None

        dev = self.mha.linear_q.weight.device
        bf, f32 = torch.bfloat16, torch.float32
        pb, pf = _Packer(bf, dev), _Packer(f32, dev)
        scale = 1.0 / math.sqrt(D)
        m = self.mha
        oQKVG = pb.addb(torch.cat([m.linear_q.weight.float() * scale,
                                  m.linear_k.weight.float(),
                                  m.linear_v.weight.float(),
                                  m.linear_g.weight.float()], dim=0))
        oWo = pb.addb(m.linear_o.weight.float())
        # layer_norm_z scale/offset folded into linear_z
        lnzw = self.layer_norm_z.weight.float()
        lnzb = (self.layer_norm_z.bias.float() if self.layer_norm_z.bias is not None
                else torch.zeros(Cz, device=dev))
        wz = self.linear_z.weight.float()                       # [H, Cz]
        wzf = (wz * lnzw.unsqueeze(0)).t().contiguous()         # [Cz, H]
        oWzF = pb.add(wzf)
        fbq = pf.add(m.linear_q.bias.float() * scale)
        fbzF = pf.add((wz * lnzb.unsqueeze(0)).sum(-1))
        fsWzF = pf.add(wzf.sum(0))
        oWga = oWsa = oWao = 0
        flnaw = flnab = fbga = fbao = 0
        if ada:
            # layer_norm_s scale folded into AdaLN's two linears
            lnsw = self.layer_norm_a.layer_norm_s.weight.float()
            oWga = pb.addb(self.layer_norm_a.linear_g.weight.float() * lnsw)
            oWsa = pb.addb(self.layer_norm_a.linear_s.weight.float() * lnsw)
            oWao = pb.addb(self.linear_ada_out.weight.float())
            fbga = pf.add(self.layer_norm_a.linear_g.bias.float())
            fbao = pf.add(self.linear_ada_out.bias.float())
        else:
            flnaw = pf.add(self.layer_norm_a.weight.float())
            flnab = pf.add(self.layer_norm_a.bias.float())

        def wsb(n):
            return torch.empty(max(n, 8), dtype=bf, device=dev)

        def wsf(n):
            return torch.empty(max(n, 8), dtype=f32, device=dev)

        an, oo = wsb(N * C), wsb(N * C)
        qkvg = wsb(4 * N * C)
        zb = wsf(H * N * N)
        ctr = torch.zeros(256, dtype=torch.int64, device=dev)
        iv = [N, C, H, D, Cz, Cs, int(ada), oQKVG, oWo, oWzF, oWga, oWsa, oWao,
              fbq, fbzF, fsWzF, flnaw, flnab, fbga, fbao]
        self._keep = (pb.tensor(), pf.tensor(), an, qkvg, zb, oo, ctr)
        return ext.apb_plan(self._keep[0], self._keep[1], an, qkvg, zb, oo,
                            ctr, iv, float(self.inf), float(eps))

    def _dispatch(self, a, z, s, mask):
        key = (tuple(a.shape), tuple(z.shape), s is None, mask is None)
        plans = self._plans
        if key in plans:
            h = plans[key]
        else:
            try:
                h = self._make_plan(a, z, s, mask)
            except Exception:
                h = None
            plans[key] = h
        if h is None:
            return self._reference(a, z, s, mask)
        self._fused = (a.shape, z.shape, s is not None, h)
        return _EXT.apb_run(a, z, s, mask, h, _PHASES)

    # -- reference composition (fallback) -----------------------------------
    def _prep_bias(
        self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        batch_dims = a.shape[:-2]
        mask = mask.expand((*batch_dims, -1))

        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        z = self.layer_norm_z(z)
        z = self.linear_z(z)
        z = _permute_final_dims(z, [2, 0, 1])
        biases.append(z)

        return biases

    def _reference(self, a, z, s, mask):
        biases = self._prep_bias(a=a, z=z, mask=mask)
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)
        a = self.mha(q_x=a, kv_x=a, biases=biases)
        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a
        return a

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N, C_q] token/atom-level embedding
            z:    [*, N, N, C_z] pair embedding
            s:    [*, N, C_s] single embedding (for AdaLN)
            mask: [*, N] mask

        Returns:
            [*, N, C_q] attention update
        """
        f = self._fused
        if (f is not None and a.shape == f[0] and z.shape == f[1]
                and (s is not None) == f[2]):
            return _EXT.apb_run(a, z, s, mask, f[3], _PHASES)
        return self._dispatch(a, z, s, mask)


class CrossAttentionPairBias(nn.Module):
    """AF3 Algorithm 24: Cross-attention with pair bias for atom attention.

    Uses separate layer_norm_a_q and layer_norm_a_k for query/key, and
    does NOT apply layer_norm_z (pair bias goes through linear_z directly).
    Handles sequence-local blocked inputs.

    Reference: openfold3/core/model/layers/attention_pair_bias.py CrossAttentionPairBias

    Args:
        c_q: Input dimension of query/key/value
        c_s: Single activation channel dimension (for AdaLN)
        c_z: Pair activation channel dimension
        c_hidden: Per-head hidden dimension
        no_heads: Number of attention heads
        use_ada_layer_norm: Whether to use AdaLN-Zero conditioning
        n_query: Block size for queries
        n_key: Block size for keys
        gating: Whether to gate output
        inf: Large constant for masking
    """

    def __init__(
        self,
        c_q: int,
        c_k: int = 0,
        c_v: int = 0,
        c_s: int = 0,
        c_z: int = 16,
        c_hidden: int = 16,
        no_heads: int = 4,
        use_ada_layer_norm: bool = True,
        n_query: int | None = None,
        n_key: int | None = None,
        gating: bool = True,
        inf: float = 1e9,
    ):
        super().__init__()
        c_k = c_k or c_q
        c_v = c_v or c_q

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf
        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(c_a=c_q, c_s=c_s)
            self.layer_norm_a_k = AdaLN(c_a=c_q, c_s=c_s)
            self.linear_ada_out = Linear(c_s, c_q, bias=True)
        else:
            self.layer_norm_a_q = LayerNorm(c_q)
            self.layer_norm_a_k = LayerNorm(c_q)

        self.linear_z = Linear(c_z, no_heads, bias=False)

        self.sigmoid = Sigmoid()

        self.mha = OF3Attention(
            c_q=c_q, c_k=c_k, c_v=c_v,
            c_hidden=c_hidden, no_heads=no_heads, gating=gating,
            q_bias=True,
        )

        self._plans: dict = {}
        self._fused = None
        _hook_state_dict(self)

    # -- fused path ---------------------------------------------------------
    def _make_plan(self, a, z, s, mask):
        ext = _load_ext()
        if ext is None:
            return None
        C, H, D = self.c_q, self.mha.no_heads, self.mha.c_hidden
        Cz, Cs = self.c_z, self.c_s
        nq, nk = self.n_query, self.n_key
        if not self.use_ada_layer_norm or self.mha.linear_g is None:
            return None
        if nq is None or nk is None or nq % 16 or nk % 16 or H * D != C:
            return None
        if C % 8 or D % 8 or H > 8 or Cz > 16 or Cz % 8:
            return None
        if a.dtype != torch.bfloat16 or not a.is_cuda or a.dim() < 2:
            return None
        Na = a.shape[-2]
        if a.shape[-1] != C or a.numel() != Na * C or not a.is_contiguous():
            return None
        NB = (Na + nq - 1) // nq
        P = NB * nq
        if s is None or s.dtype != torch.bfloat16 or not s.is_contiguous():
            return None
        if s.numel() != Na * Cs:
            return None
        if (z.dtype != torch.bfloat16 or not z.is_contiguous()
                or z.numel() != NB * nq * nk * Cz):
            return None
        if mask is not None:
            if (mask.dtype != torch.bfloat16 or not mask.is_contiguous()
                    or mask.numel() != Na):
                return None
        eps = self.layer_norm_a_q.layer_norm_a.eps
        for m in (self.layer_norm_a_q, self.layer_norm_a_k):
            if abs(m.layer_norm_a.eps - eps) > 0 or abs(m.layer_norm_s.eps - eps) > 0:
                return None

        dev = self.mha.linear_q.weight.device
        bf, f32 = torch.bfloat16, torch.float32
        pb, pf = _Packer(bf, dev), _Packer(f32, dev)
        scale = 1.0 / math.sqrt(D)
        m = self.mha
        oQKVG = pb.addb(torch.cat([m.linear_q.weight.float() * scale,
                                  m.linear_k.weight.float(),
                                  m.linear_v.weight.float(),
                                  m.linear_g.weight.float()], dim=0))
        oWo = pb.addb(m.linear_o.weight.float())
        oWzT = pb.add(self.linear_z.weight.float().t().contiguous())
        lnq = self.layer_norm_a_q.layer_norm_s.weight.float()
        lnk = self.layer_norm_a_k.layer_norm_s.weight.float()
        oWgq = pb.addb(self.layer_norm_a_q.linear_g.weight.float() * lnq)
        oWsq = pb.addb(self.layer_norm_a_q.linear_s.weight.float() * lnq)
        oWgk = pb.addb(self.layer_norm_a_k.linear_g.weight.float() * lnk)
        oWsk = pb.addb(self.layer_norm_a_k.linear_s.weight.float() * lnk)
        oWao = pb.addb(self.linear_ada_out.weight.float())
        fbq = pf.add(m.linear_q.bias.float() * scale)
        fbgq = pf.add(self.layer_norm_a_q.linear_g.bias.float())
        fbgk = pf.add(self.layer_norm_a_k.linear_g.bias.float())
        fbao = pf.add(self.linear_ada_out.bias.float())

        def wsb(n):
            return torch.empty(max(n, 8), dtype=bf, device=dev)

        aq, ak, oo = wsb(P * C), wsb(P * C), wsb(P * C)
        qkvg = wsb(4 * P * C)
        zb = torch.empty(NB * H * nq * nk, dtype=f32, device=dev)
        ctr = torch.zeros(256, dtype=torch.int64, device=dev)
        iv = [Na, P, NB, nq, nk, C, H, D, Cz, Cs, oQKVG, oWo, oWzT,
              oWgq, oWsq, oWgk, oWsk, oWao, fbq, fbgq, fbgk, fbao]
        self._keep = (pb.tensor(), pf.tensor(), aq, ak, qkvg, zb, oo, ctr)
        return ext.cap_plan(self._keep[0], self._keep[1], aq, ak, qkvg,
                            zb, oo, ctr, iv, float(self.inf), float(eps))

    def _dispatch(self, a, z, s, mask):
        key = (tuple(a.shape), tuple(z.shape), s is None, mask is None)
        plans = self._plans
        if key in plans:
            h = plans[key]
        else:
            try:
                h = self._make_plan(a, z, s, mask)
            except Exception:
                h = None
            plans[key] = h
        if h is None:
            return self._reference(a, z, s, mask)
        self._fused = (a.shape, z.shape, h)
        return _EXT.cap_run(a, z, s, mask, h, _PHASES)

    # -- reference composition (fallback) -----------------------------------
    def _reference(self, a, z, s, mask):
        from .alphafold3_atom_attention import (
            _convert_single_rep_to_blocks, _apply_block_indices,
        )

        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        a_query, a_key, block_mask = _convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
        )

        mask_bias = (self.inf * (block_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        z_bias = self.linear_z(z)
        z_bias = _permute_final_dims(z_bias, [2, 0, 1])
        biases.append(z_bias)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = _apply_block_indices(
                ql=s, n_query=self.n_query, n_key=self.n_key, atom_mask=mask,
            )
            a_q = self.layer_norm_a_q(a_query, s_q)
            a_k = self.layer_norm_a_k(a_key, s_k)
        else:
            a_q = self.layer_norm_a_q(a_query)
            a_k = self.layer_norm_a_k(a_key)

        a_out = self.mha(q_x=a_q, kv_x=a_k, biases=biases)

        a_out = a_out.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a_out = self.sigmoid(self.linear_ada_out(s)) * a_out

        return a_out

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            a:    [*, N_atom, C_q] atom-level embedding
            z:    [*, N_blocks, n_key, n_key, C_z] blocked pair embedding
            s:    [*, N_atom, C_s] single embedding (for AdaLN)
            mask: [*, N_atom] mask

        Returns:
            [*, N_atom, C_q] attention update
        """
        f = self._fused
        if f is not None and s is not None and a.shape == f[0] and z.shape == f[1]:
            return _EXT.cap_run(a, z, s, mask, f[2], _PHASES)
        return self._dispatch(a, z, s, mask)
