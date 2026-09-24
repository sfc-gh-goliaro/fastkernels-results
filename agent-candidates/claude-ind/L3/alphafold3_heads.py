"""Auxiliary prediction heads for AlphaFold3 -- single fused CUDA launch.

Distogram, pLDDT, PAE, PDE and ExperimentallyResolved all read the same two
tensors (``s``, ``z``) and are independent of each other, so the baseline's
~19 kernels (five LayerNorms' worth of casts, five GEMMs, two transpose-adds)
are almost pure launch overhead at the captured size (16 tokens, 256 pair
entries).  ``AuxiliaryHeads.forward`` here issues exactly one kernel.

Two algebraic rewrites make that possible:

* Each ``LayerNorm`` affine is folded into the ``Linear`` that consumes it,
  once, at the first forward::

      linear(ln(x))[c] = sum_k ((x_k - mu) * rstd * w_k + b_k) * W[c,k]
                       = dot(xhat, W'[c]) + bias'[c]

  with ``W'[c,k] = w_k * W[c,k]`` and ``bias'[c] = sum_k b_k * W[c,k]``.  The
  kernel therefore only needs the bare normalization ``xhat``, and the pLDDT
  and ExperimentallyResolved weights concatenate into one column block even
  though they normalize with different affines.

* ``logits + logits.transpose(-2, -3)`` (distogram, PDE) is pushed through the
  linear onto its input: ``z[i,j] @ W + z[j,i] @ W == (z[i,j] + z[j,i]) @ W``,
  so the symmetrization needs no second pass over the output.

``PairformerEmbedding`` is constructed by the baseline but never called by
``forward``; it is omitted here (``load_state_dict(strict=False)`` ignores the
extra keys) so nothing is built that the op does not use.

Reference: openfold3/core/model/heads/prediction_heads.py
           openfold3/core/model/heads/head_modules.py AuxiliaryHeadsAllAtom
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


__targets__ = ["AuxiliaryHeads"]

_TCS = 16          # must match TCS in the kernel source

_CUDA_SRC = r"""
// Fused AlphaFold3 auxiliary-heads kernel.
//
// One launch produces all five head outputs.  The captured problem is tiny
// (16 tokens, 256 pair entries, ~14 MFMA total), so the baseline's ~19
// separate kernels are almost pure overhead -- and once they are fused what is
// left is *memory latency*, not arithmetic: with the benchmark's L2 flush in
// front of every iteration, each round trip that something waits on costs more
// than the whole GEMM.  Hence the two structural choices here:
//
//   * every dot product runs on the bf16 tensor cores (one m16n8k16 MMA
//     replaces 2048 FFMAs and reads its operands from shared memory once per
//     16x8 tile rather than once per output), and
//   * every global load a block needs -- weight tile, input rows, column bias
//     -- is issued up front into registers, so the block waits on one round
//     trip instead of one per phase.
//
// Per-head math, with each LayerNorm's affine folded into the following
// Linear weight on the host (W'[c][k] = w[k]*W[c][k], bias'[c] = sum_k
// b[k]*W[c][k]), so every head is a plain GEMM plus a per-column bias:
//
//   distogram[i,j] = (z[i,j] + z[j,i]) @ Wd^T
//   pae[i,j]       = xz[i,j] @ Wpae'^T + bpae'
//   pde[i,j]       = (xz[i,j] + xz[j,i]) @ Wpde'^T + 2*bpde'
//   plddt[j]       = xs[j] @ Wpl'^T + bpl'
//   expres[j]      = xs[j] @ Wex'^T + bex'
//
// xz / xs are the bare normalizations (x - mean) * rstd; folding the affine
// out is what lets the pLDDT and ExperimentallyResolved weights -- which
// normalize with different affines -- share one column block, and it removes
// the symmetrization's second pass: z[i,j] @ W + z[j,i] @ W == (z[i,j] +
// z[j,i]) @ W, so the transpose-add lands on the 128-wide input instead of on
// the output.
//
// Blocks 0..3N-1 own one (head, i) row of the pair heads: 16 rows x 64 bins,
// eight 16x8 tiles, each split KSPZ ways along k.  The rest own a TCS-column
// tile of the concatenated pLDDT | ExperimentallyResolved weight: two 16x8
// tiles, each split KSPLIT ways.  Either way every warp gets a k-slice, which
// both shortens the MMA accumulator chain and gives the scheduler more warps
// to switch between while the loads land.
#include <cuda_bf16.h>

#define MROWS 16            // rows per MMA tile; the fast path needs N <= MROWS
#define CZ   128            // c_z
#define CS   384            // c_s
#define NBZ  64             // pair-head bins (distogram / pae / pde)
#define TCS  16             // single-head column tile per block
#define NTHREADS 512        // 16 warps: one per token row, and enough of them
                            // that the k-splits keep the load latency covered
#define NWARPS (NTHREADS / 32)
#define KSPLIT (NWARPS / (TCS / 8))   // k-split of the single-head GEMM
#define KCS (CS / KSPLIT)
#define NTZ (NBZ / 8)                 // pair-head 16x8 tiles per (head, i)
#define KSPZ (NWARPS / NTZ)           // k-split of the pair GEMM
#define KCZ (CZ / KSPZ)
#define WTILE (TCS * CS / 8)          // weight uint4s per single-head tile
#define ZTILE (NBZ * CZ / 8)
#define WSTG ((WTILE + NTHREADS - 1) / NTHREADS)   // ... staged per thread
#define ZSTG ((ZTILE + NTHREADS - 1) / NTHREADS)

// Padded k strides.  4 banks of skew per row (KP/2 == 4 mod 32) puts the MMA
// fragment loads -- row gid, k quad tig -- on 32 distinct banks.
#define KPZ 136
#define KPS 392

static_assert(NWARPS >= MROWS, "the row transform gives each warp one row");

__device__ __forceinline__ unsigned lds32(const __nv_bfloat16 *p) {
  return *(const unsigned *)p;
}

__device__ __forceinline__ void mma_m16n8k16(float *d, const unsigned *a,
                                             const unsigned *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One warp's 16x8 tile, accumulated over k in [kbeg, kend).
__device__ __forceinline__ void gemm_tile(float *d, const __nv_bfloat16 *Ash,
                                          const __nv_bfloat16 *Bsh, int kp,
                                          int kbeg, int kend, int lane) {
  const int gid = lane >> 2, tig = (lane & 3) * 2;
  const __nv_bfloat16 *a0 = Ash + gid * kp + tig;
  const __nv_bfloat16 *a1 = a0 + 8 * kp;
  const __nv_bfloat16 *b0 = Bsh + gid * kp + tig;
  for (int kb = kbeg; kb < kend; kb += 16) {
    unsigned a[4], b[2];
    a[0] = lds32(a0 + kb);
    a[1] = lds32(a1 + kb);
    a[2] = lds32(a0 + kb + 8);
    a[3] = lds32(a1 + kb + 8);
    b[0] = lds32(b0 + kb);
    b[1] = lds32(b0 + kb + 8);
    mma_m16n8k16(d, a, b);
  }
}

__device__ __forceinline__ void unpack4(uint2 v, float *o) {
  const __nv_bfloat162 *h = (const __nv_bfloat162 *)&v;
  o[0] = __low2float(h[0]);
  o[1] = __high2float(h[0]);
  o[2] = __low2float(h[1]);
  o[3] = __high2float(h[1]);
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16 *p, const float *v) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]);
  __nv_bfloat162 hi = __floats2bfloat162_rn(v[2], v[3]);
  uint2 packed;
  packed.x = *(const unsigned *)&lo;
  packed.y = *(const unsigned *)&hi;
  *(uint2 *)p = packed;
}

// Warp-wide mean / rstd of a row spread 4-per-lane across the warp.
__device__ __forceinline__ void row_stats(const float *a, int n, int cnt,
                                          float eps, float &mean, float &rstd) {
  float s = 0.f, q = 0.f;
  for (int t = 0; t < cnt; ++t) {
    s += a[t];
    q = fmaf(a[t], a[t], q);
  }
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, d);
    q += __shfl_xor_sync(0xffffffff, q, d);
  }
  const float inv = 1.0f / n;
  mean = s * inv;
  rstd = rsqrtf(fmaxf(q * inv - mean * mean, 0.f) + eps);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1) fk_af3_heads(
    const __nv_bfloat16 *__restrict__ sin_,
    const __nv_bfloat16 *__restrict__ zin,
    const __nv_bfloat16 *__restrict__ Wz,  // [3][NBZ][CZ]
    const float *__restrict__ Bz,          // [3][NBZ]
    const __nv_bfloat16 *__restrict__ Ws,  // [NSP][CS]  (rows padded to TCS)
    const float *__restrict__ Bs,          // [NSP]
    __nv_bfloat16 *__restrict__ o_dist,
    __nv_bfloat16 *__restrict__ o_plddt,
    __nv_bfloat16 *__restrict__ o_pae,
    __nv_bfloat16 *__restrict__ o_pde,
    __nv_bfloat16 *__restrict__ o_exp,
    int N, int NS, int NPL, float eps) {
  extern __shared__ char smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int gid = lane >> 2, tig = (lane & 3) * 2;
  const int nzb = 3 * N;
  float d[4] = {0.f, 0.f, 0.f, 0.f};

  if (blockIdx.x < nzb) {
    // ================= pair heads (distogram / pae / pde) ==============
    const int h = blockIdx.x / N;
    const int i = blockIdx.x - h * N;
    const int nbase = (warp % NTZ) * 8;   // NTZ tiles cover the 64 bins
    const int kz = warp / NTZ;
    __nv_bfloat16 *Ash = (__nv_bfloat16 *)smem;        // [MROWS][KPZ]
    __nv_bfloat16 *Bsh = Ash + MROWS * KPZ;            // [NBZ][KPZ]
    float *red = (float *)(Bsh + (size_t)NBZ * KPZ);   // [NTHREADS][4]
    const __nv_bfloat16 *zi = zin + (size_t)i * N * CZ + lane * 4;
    const __nv_bfloat16 *zj = zin + (size_t)i * CZ + lane * 4;

    // ---- every global load this block needs, issued back to back ----
    uint4 wb[ZSTG];
#pragma unroll
    for (int q = 0; q < ZSTG; ++q) {
      const int e = tid + q * NTHREADS;
      if (e < ZTILE) {
        const int c = e / (CZ / 8), kc = e - c * (CZ / 8);
        wb[q] = *(const uint4 *)(Wz + ((size_t)h * NBZ + c) * CZ + kc * 8);
      }
    }
    // One row per warp (NWARPS >= MROWS >= N): row ``warp`` of this i-slice,
    // clamped so an idle warp still reads in range and is zeroed below.
    const int j = warp;
    const bool live = j < N;
    const int jl = live ? j : 0;
    const uint2 ra = *(const uint2 *)(zi + (size_t)jl * CZ);
    uint2 rb = ra;
    if (h != 1) rb = *(const uint2 *)(zj + (size_t)jl * N * CZ);
    const float2 bias = *(const float2 *)(Bz + h * NBZ + nbase + tig);

#pragma unroll
    for (int q = 0; q < ZSTG; ++q) {
      const int e = tid + q * NTHREADS;
      if (e < ZTILE) {
        const int c = e / (CZ / 8), kc = e - c * (CZ / 8);
        *(uint4 *)(Bsh + c * KPZ + kc * 8) = wb[q];
      }
    }
    if (j < MROWS) {
      float a[4], b[4], v[4];
      unpack4(ra, a);
      unpack4(rb, b);
      if (h == 0) {
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = a[q] + b[q];
      } else {
        float ma, ra_;
        row_stats(a, CZ, 4, eps, ma, ra_);
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = (a[q] - ma) * ra_;
        if (h == 2) {
          float mb, rb_;
          row_stats(b, CZ, 4, eps, mb, rb_);
#pragma unroll
          for (int q = 0; q < 4; ++q) v[q] += (b[q] - mb) * rb_;
        }
      }
      if (!live) {
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = 0.f;
      }
      store_bf16x4(Ash + j * KPZ + lane * 4, v);
    }
    __syncthreads();

    gemm_tile(d, Ash, Bsh + (size_t)nbase * KPZ, KPZ, kz * KCZ,
              kz * KCZ + KCZ, lane);
#if KSPZ > 1
    *(float4 *)(red + tid * 4) = make_float4(d[0], d[1], d[2], d[3]);
    __syncthreads();
    if (warp >= NTZ) return;
#pragma unroll
    for (int p = 1; p < KSPZ; ++p) {
      const float4 o = *(const float4 *)(red + (tid + p * NTZ * 32) * 4);
      d[0] += o.x; d[1] += o.y; d[2] += o.z; d[3] += o.w;
    }
#endif
    __nv_bfloat16 *out = (h == 0) ? o_dist : (h == 1 ? o_pae : o_pde);
    __nv_bfloat162 r0 = __floats2bfloat162_rn(d[0] + bias.x, d[1] + bias.y);
    __nv_bfloat162 r1 = __floats2bfloat162_rn(d[2] + bias.x, d[3] + bias.y);
    if (gid < N)
      *(unsigned *)(out + ((size_t)i * N + gid) * NBZ + nbase + tig) =
          *(const unsigned *)&r0;
    if (gid + 8 < N)
      *(unsigned *)(out + ((size_t)i * N + gid + 8) * NBZ + nbase + tig) =
          *(const unsigned *)&r1;
  } else {
    // ================= single heads (pLDDT + exp. resolved) ============
    const int c0 = (blockIdx.x - nzb) * TCS;
    const int ntile = warp & (TCS / 8 - 1);
    const int ks = warp / (TCS / 8);
    const int nbase = ntile * 8;
    __nv_bfloat16 *Ash = (__nv_bfloat16 *)smem;        // [MROWS][KPS]
    __nv_bfloat16 *Bsh = Ash + MROWS * KPS;            // [TCS][KPS]
    float *red = (float *)(Bsh + (size_t)TCS * KPS);   // [NTHREADS][4]
    const __nv_bfloat16 *sp = sin_ + lane * 4;

    // ---- every global load this block needs, issued back to back ----
    uint4 wb[WSTG];
#pragma unroll
    for (int q = 0; q < WSTG; ++q) {
      const int e = tid + q * NTHREADS;
      if (e < WTILE) {
        const int c = e / (CS / 8), kc = e - c * (CS / 8);
        wb[q] = *(const uint4 *)(Ws + (size_t)(c0 + c) * CS + kc * 8);
      }
    }
    const int j = warp;                 // one token row per warp
    const bool live = j < N;
    const int jl = live ? j : 0;
    uint2 rs[3];
#pragma unroll
    for (int u = 0; u < 3; ++u)
      rs[u] = *(const uint2 *)(sp + (size_t)jl * CS + u * 128);
    const float2 bias = *(const float2 *)(Bs + c0 + nbase + tig);

#pragma unroll
    for (int q = 0; q < WSTG; ++q) {
      const int e = tid + q * NTHREADS;
      if (e < WTILE) {
        const int c = e / (CS / 8), kc = e - c * (CS / 8);
        *(uint4 *)(Bsh + c * KPS + kc * 8) = wb[q];
      }
    }
    if (j < MROWS) {
      float a[12];
#pragma unroll
      for (int u = 0; u < 3; ++u) unpack4(rs[u], a + u * 4);
      float ma, ra;
      row_stats(a, CS, 12, eps, ma, ra);
      if (!live) ra = 0.f;
#pragma unroll
      for (int u = 0; u < 3; ++u) {
        float v[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = (a[u * 4 + q] - ma) * ra;
        store_bf16x4(Ash + j * KPS + u * 128 + lane * 4, v);
      }
    }
    __syncthreads();

    gemm_tile(d, Ash, Bsh + (size_t)nbase * KPS, KPS, ks * KCS,
              ks * KCS + KCS, lane);

#if KSPLIT > 1
    *(float4 *)(red + tid * 4) = make_float4(d[0], d[1], d[2], d[3]);
    __syncthreads();
    if (warp >= TCS / 8) return;
#pragma unroll
    for (int p = 1; p < KSPLIT; ++p) {
      const float4 o = *(const float4 *)(red + (tid + p * (TCS / 8) * 32) * 4);
      d[0] += o.x; d[1] += o.y; d[2] += o.z; d[3] += o.w;
    }
#endif
    d[0] += bias.x; d[1] += bias.y; d[2] += bias.x; d[3] += bias.y;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int j = gid + (q >> 1) * 8;
      const int c = c0 + nbase + tig + (q & 1);
      if (j < N && c < NS) {
        const __nv_bfloat16 val = __float2bfloat16(d[q]);
        if (c < NPL) o_plddt[(size_t)j * NPL + c] = val;
        else o_exp[(size_t)j * (NS - NPL) + (c - NPL)] = val;
      }
    }
  }
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

py::dict af3_heads(at::Tensor s, at::Tensor z, at::Tensor Wz, at::Tensor Bz,
                   at::Tensor Ws, at::Tensor Bs, int64_t ns, int64_t npl,
                   double eps) {
  const int N = (int)s.size(1);
  const int NS = (int)ns;
  const int NPL = (int)npl;
  auto opt = at::TensorOptions().dtype(at::kBFloat16).device(s.device());
  auto dist = at::empty({1, N, N, NBZ}, opt);
  auto pae = at::empty({1, N, N, NBZ}, opt);
  auto pde = at::empty({1, N, N, NBZ}, opt);
  auto plddt = at::empty({1, N, NPL}, opt);
  auto expres = at::empty({1, N, NS - NPL}, opt);

  const size_t zb = (MROWS * KPZ + (size_t)NBZ * KPZ) * 2 + NTHREADS * 16;
  const size_t sb = (MROWS * KPS + (size_t)TCS * KPS) * 2 + NTHREADS * 16;
  const size_t shm = zb > sb ? zb : sb;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute((const void *)fk_af3_heads,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 180000);
    attr_set = true;
  }
  const int grid = 3 * N + (NS + TCS - 1) / TCS;
  fk_af3_heads<<<grid, NTHREADS, shm, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16 *)s.data_ptr(), (const __nv_bfloat16 *)z.data_ptr(),
      (const __nv_bfloat16 *)Wz.data_ptr(), (const float *)Bz.data_ptr(),
      (const __nv_bfloat16 *)Ws.data_ptr(), (const float *)Bs.data_ptr(),
      (__nv_bfloat16 *)dist.data_ptr(), (__nv_bfloat16 *)plddt.data_ptr(),
      (__nv_bfloat16 *)pae.data_ptr(), (__nv_bfloat16 *)pde.data_ptr(),
      (__nv_bfloat16 *)expres.data_ptr(), N, NS, NPL, (float)eps);
  py::dict out;
  out["distogram_logits"] = dist;
  out["plddt_logits"] = plddt;
  out["pae_logits"] = pae;
  out["pde_logits"] = pde;
  out["experimentally_resolved_logits"] = expres;
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
py::dict af3_heads(at::Tensor s, at::Tensor z, at::Tensor Wz, at::Tensor Bz,
                   at::Tensor Ws, at::Tensor Bs, int64_t ns, int64_t npl,
                   double eps);
"""

_EXT = None
_EXT_FAILED = False


def _pin_build_arch() -> None:
    """Pin the JIT build to the local arch, as ``infra.cuda_ext`` does.

    The ambient ``TORCH_CUDA_ARCH_LIST`` spans sm_75..sm_120; the bf16
    ``mma.m16n8k16`` this kernel is built on needs sm_80 or higher, and
    compiling six architectures for one local GPU is wasted build time anyway.
    """
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _ext():
    """JIT-build (once, cached on disk) the fused-heads extension.

    Returns ``None`` if the build is not possible here (no toolchain, or a GPU
    older than the sm_80 the bf16 MMA needs); the caller then runs the eager
    reference path.
    """
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _pin_build_arch()
            from torch.utils.cpp_extension import load_inline
            _EXT = load_inline(
                name="fk_af3_aux_heads",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["af3_heads"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        except Exception:  # noqa: BLE001 - fall back to the reference path
            _EXT_FAILED = True
    return _EXT


class DistogramHead(nn.Module):
    """Predicts inter-residue distance distribution.

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of distance bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class PLDDTHead(nn.Module):
    """Predicts per-atom pLDDT confidence (PerResidueLDDTAllAtom).

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of pLDDT bins
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 50, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class PAEHead(nn.Module):
    """Predicts Predicted Aligned Error (PAE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PAE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(z))


class PDEHead(nn.Module):
    """Predicts Predicted Distance Error (PDE).

    Args:
        c_z: Pair embedding channel dimension
        no_bins: Number of PDE bins
    """

    def __init__(self, c_z: int, no_bins: int = 64):
        super().__init__()
        self.layer_norm = LayerNorm(c_z)
        self.linear = Linear(c_z, no_bins, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.linear(self.layer_norm(z))
        logits = logits + logits.transpose(-2, -3)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """Predicts per-atom experimental resolution confidence.

    Args:
        c_s: Single embedding channel dimension
        no_bins: Number of bins (2 for resolved/not resolved)
        max_atoms_per_token: Maximum atoms per token (23 for all-atom)
    """

    def __init__(self, c_s: int, no_bins: int = 2, max_atoms_per_token: int = 23):
        super().__init__()
        self.no_bins = no_bins
        self.max_atoms_per_token = max_atoms_per_token
        self.layer_norm = LayerNorm(c_s)
        self.linear = Linear(c_s, max_atoms_per_token * no_bins, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(self.layer_norm(s))


class AuxiliaryHeads(nn.Module):
    """All auxiliary prediction heads for AF3.

    Args:
        c_s: Single embedding channel dimension
        c_z: Pair embedding channel dimension
        c_s_input: Input single rep dimension (for PairformerEmbedding)
        max_atoms_per_token: Max atoms per token (23 for all-atom)
    """

    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_input: int = 449,
        max_atoms_per_token: int = 23,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token,
        )
        # Folded weights, built on the first forward (weight loading and any
        # dtype/device cast complete before then, exactly as ``L1.LayerNorm``
        # assumes for its own fp32 affine cache).  Re-derived if the storage
        # a weight points at changes (a ``.to(dtype)`` / ``.to(device)``).
        self._packed = None
        self._packed_key = None

    # -- fused path -------------------------------------------------------
    def _fold(self):
        """Pack the five heads' weights with their LayerNorm affines folded in.

        Both packs stay in the ``[out, in]`` layout the MMA's column-major B
        operand wants, and ``Ws``' rows are zero-padded up to a whole column
        tile so every block stages a full tile.
        """
        d, pl, pa, pd, ex = (self.distogram, self.plddt, self.pae, self.pde,
                             self.experimentally_resolved)
        f = torch.float32
        w_d = d.linear.weight.to(f)
        w_pa = pa.linear.weight.to(f) * pa.layer_norm.weight.to(f)
        w_pd = pd.linear.weight.to(f) * pd.layer_norm.weight.to(f)
        wz = torch.stack([w_d, w_pa, w_pd]).contiguous().to(torch.bfloat16)
        bz = torch.stack([
            torch.zeros(w_d.shape[0], device=w_d.device, dtype=f),
            pa.linear.weight.to(f) @ pa.layer_norm.bias.to(f),
            2.0 * (pd.linear.weight.to(f) @ pd.layer_norm.bias.to(f)),
        ]).contiguous()
        ws = torch.cat([
            pl.linear.weight.to(f) * pl.layer_norm.weight.to(f),
            ex.linear.weight.to(f) * ex.layer_norm.weight.to(f),
        ])
        bs = torch.cat([
            pl.linear.weight.to(f) @ pl.layer_norm.bias.to(f),
            ex.linear.weight.to(f) @ ex.layer_norm.bias.to(f),
        ])
        ns = ws.shape[0]
        nsp = -(-ns // _TCS) * _TCS          # whole column tiles for the grid
        ws_p = torch.zeros(nsp, ws.shape[1], device=ws.device, dtype=f)
        ws_p[:ns] = ws
        bs_p = torch.zeros(nsp, device=bs.device, dtype=f)
        bs_p[:ns] = bs
        self._packed = (wz, bz, ws_p.to(torch.bfloat16), bs_p, ns,
                        pl.linear.weight.shape[0], float(pl.layer_norm.eps))

    def _fast_ok(self, s: torch.Tensor, z: torch.Tensor) -> bool:
        return (
            self.c_z == 128 and self.c_s == 384
            and s.dtype == torch.bfloat16 and z.dtype == torch.bfloat16
            and s.is_cuda and s.dim() == 3 and z.dim() == 4
            and s.shape[0] == 1 and z.shape[0] == 1
            and z.shape[1] == z.shape[2] == s.shape[1]
            and 1 <= s.shape[1] <= 16   # one MMA row tile
            and s.is_contiguous() and z.is_contiguous()
            and self.distogram.linear.weight.shape[0] == 64
            and self.pae.linear.weight.shape[0] == 64
            and self.pde.linear.weight.shape[0] == 64
            and self.pae.layer_norm.eps == self.pde.layer_norm.eps
            == self.plddt.layer_norm.eps
            == self.experimentally_resolved.layer_norm.eps
        )

    def _forward_ref(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "distogram_logits": self.distogram(z),
            "plddt_logits": self.plddt(s),
            "pae_logits": self.pae(z),
            "pde_logits": self.pde(z),
            "experimentally_resolved_logits": self.experimentally_resolved(s),
        }

    def forward(
        self, s: torch.Tensor, z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ext = _ext()
        if ext is None or not self._fast_ok(s, z):
            return self._forward_ref(s, z)
        w = self.plddt.linear.weight
        key = (w.data_ptr(), w.dtype, self.pae.linear.weight.data_ptr())
        if self._packed_key != key:
            self._fold()
            self._packed_key = key
        wz, bz, ws, bs, ns, npl, eps = self._packed
        return ext.af3_heads(s, z, wz, bz, ws, bs, ns, npl, eps)
