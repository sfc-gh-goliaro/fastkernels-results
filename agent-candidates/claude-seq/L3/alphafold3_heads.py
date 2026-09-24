"""Auxiliary prediction heads for AlphaFold3 -- all five heads in one launch.

Distogram, pLDDT, PAE, PDE, ExperimentallyResolved confidence heads that
produce binned logits from single and pair representations.  The
PairformerEmbedding refines s/z before confidence heads.

Reference: openfold3/core/model/heads/prediction_heads.py
           openfold3/core/model/heads/head_modules.py AuxiliaryHeadsAllAtom

Why this is one kernel
----------------------
The captured shape is tiny -- ``s: [1, 16, 384]``, ``z: [1, 16, 16, 128]`` -- so
the whole operator is 27 MFLOP, which a B200 retires in well under a
microsecond.  What the baseline actually spends its time on is *dispatch*: five
head modules, each a LayerNorm plus a Linear, plus two transpose-adds for the
symmetrized heads.  That is eleven eager ops, and measured on this machine one
LayerNorm call alone costs 17us of Python/launch time against a 2.9us
empty-launch floor -- the whole forward measures ~145us with barely any of it
spent computing.

So the deliverable is a single ``__global__`` function covering every head, and
a ``forward`` whose only work is one pybind call.  The kernel's grid carries two
kinds of block:

* **Pair blocks** own one *unordered* token pair ``{i, j}``.  ``distogram`` and
  ``pde`` are symmetrized (``logits + logits.transpose(-2, -3)``), so their
  output at ``(i, j)`` and ``(j, i)`` is the same value; computing the raw
  logits for both directions in one block lets it write both entries and halves
  the pair-head MACs versus one block per ordered pair.  ``pae`` is not
  symmetric, so both directions are kept separately.
* **Single blocks** own one token row and a slice of the output columns.
  ``plddt`` (1150 bins) and ``experimentally_resolved`` (46) read the same row
  but normalize it with different LayerNorm affines, so the block normalizes
  once, stages *both* affine results in shared memory, and each thread picks
  the one its column belongs to.

Both halves split their reduction dimension across several threads and join the
partial sums with warp shuffles.  That is not about arithmetic -- there is
barely any -- but about having enough resident warps to hide the weight loads,
which is what this kernel is actually limited by; as a bonus it makes
consecutive threads read consecutive 16B weight chunks, so those loads coalesce.

Numerics follow the baseline's *intermediates*, not just its formula: the
LayerNorm result is rounded to bfloat16 in shared memory before it is fed to
the dot product, and the symmetrized heads round each raw logit to bfloat16
before adding, exactly as the separate LayerNorm -> Linear -> add op chain does.
Only the accumulation order inside the dot product differs from cuBLAS.

The host wrapper is written against the same clock: the five output tensors come
from the CUDA allocator directly rather than through the ATen dispatcher, and
the result dict is built in C++, because at this size the forward's CPU time and
its GPU time are the same order of magnitude.

The CUDA source is inlined below so this file is the whole deliverable; it is
JIT-compiled once per machine and cached by ``torch.utils.cpp_extension``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear


__targets__ = ["AuxiliaryHeads"]


_CUDA_SRC = r"""// All five AlphaFold3 auxiliary heads in a single launch.
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

namespace {

using bf = __nv_bfloat16;

__device__ __forceinline__ float f2(bf v) { return __bfloat162float(v); }
__device__ __forceinline__ bf tb(float v) { return __float2bfloat16_rn(v); }

// bfloat16 is the top half of a float32, so widening a packed pair is two
// integer ops and no conversion instruction -- and, unlike indexing a local
// ``bf16[8]`` view of a uint4, it never puts the vector on the stack.
__device__ __forceinline__ float lo_bf(unsigned int v) {
  return __uint_as_float(v << 16);
}
__device__ __forceinline__ float hi_bf(unsigned int v) {
  return __uint_as_float(v & 0xffff0000u);
}

// Eight MACs of one 16B weight chunk against one 16B chunk of a staged
// activation row, split over two accumulator chains so the FMA pipeline is not
// waiting on its own result.
//
// Both operands arrive as packed bfloat16 pairs.  Staging the activations as
// bfloat16 rather than float is lossless -- every value written there has
// already been rounded to bfloat16 to match the baseline's intermediate -- and
// it halves the shared-memory loads, which is the pipe this kernel leans on
// hardest (ncu: L1TEX the top utilization at 61%).
__device__ __forceinline__ void mac8(float &e, float &o, const uint4 a,
                                     const uint4 w) {
  e = fmaf(lo_bf(a.x), lo_bf(w.x), e);
  o = fmaf(hi_bf(a.x), hi_bf(w.x), o);
  e = fmaf(lo_bf(a.y), lo_bf(w.y), e);
  o = fmaf(hi_bf(a.y), hi_bf(w.y), o);
  e = fmaf(lo_bf(a.z), lo_bf(w.z), e);
  o = fmaf(hi_bf(a.z), hi_bf(w.z), o);
  e = fmaf(lo_bf(a.w), lo_bf(w.w), e);
  o = fmaf(hi_bf(a.w), hi_bf(w.w), o);
}

__device__ __forceinline__ uint4 lds16(const bf *p) {
  return *reinterpret_cast<const uint4 *>(p);
}

// Sum four values across the block and leave the total in every thread.  The
// leading barrier makes the routine safe to call repeatedly with the same
// scratch (a previous call's readers are done) and also publishes whatever the
// caller staged in shared memory before the reduction.
template <int TPB>
__device__ __forceinline__ void block_red4(float &a, float &b, float &c,
                                           float &d, float *sm) {
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    a += __shfl_xor_sync(0xffffffffu, a, off);
    b += __shfl_xor_sync(0xffffffffu, b, off);
    c += __shfl_xor_sync(0xffffffffu, c, off);
    d += __shfl_xor_sync(0xffffffffu, d, off);
  }
  constexpr int NW = TPB / 32;
  __syncthreads();
  if (NW == 1) return;
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) {
    sm[wid] = a;
    sm[NW + wid] = b;
    sm[2 * NW + wid] = c;
    sm[3 * NW + wid] = d;
  }
  __syncthreads();
  float A = 0.f, B = 0.f, C = 0.f, D = 0.f;
#pragma unroll
  for (int k = 0; k < NW; ++k) {
    A += sm[k];
    B += sm[NW + k];
    C += sm[2 * NW + k];
    D += sm[3 * NW + k];
  }
  a = A; b = B; c = C; d = D;
}

// Launch geometry.  Everything here is about *parallelism*, not arithmetic: the
// whole operator is 27 MFLOP, which the GPU retires in well under a
// microsecond, so what the kernel is actually fighting is memory latency with
// too few warps to hide it.  Both halves therefore split their reduction
// dimension across several threads (`ZK` / `SG`) and join the partials with
// warp shuffles.  That multiplies the resident warps, shortens every thread's
// dependency chain, and -- because consecutive threads then take *consecutive*
// 16B weight chunks -- makes the weight reads fully coalesced.  The values
// below were swept against the scorer's own timer; the sweep is flat within
// ~1us either side, which is also this machine's measurement granularity.
constexpr int TPB = 384;   // 12 warps
constexpr int ZK = 4;      // threads per (pair head, bin)
constexpr int ZTY = TPB / ZK;
constexpr int ZCH = 4;     // weight chunks a thread keeps in flight
constexpr int SG = 8;      // threads per plddt/experimentally-resolved column
constexpr int SWPT = TPB / SG;   // columns per single-head block
constexpr int SCH = 6;
// Left to itself ptxas spends 63 registers here, which caps the SM at two
// resident blocks.  Asking for three costs 7 registers (56, still no spill) and
// buys a third block's worth of warps to hide the weight loads behind; a sweep
// run in one process preferred >= 3 by about a microsecond, which is roughly
// this machine's measurement granularity, so treat it as a free hint rather
// than a large win.
constexpr int MIN_BLOCKS = 3;

// Blocks [0, NZB) run the pair heads, the rest the single-rep heads.
//
//   WZ  [3][NBZ][CZ]  distogram, pde, pae output projections
//   LNZ [4][CZ]       pde.gamma, pde.beta, pae.gamma, pae.beta
//   WS  [NT][CS]      plddt rows (NP of them) then experimentally_resolved
//   LNS [4][CS]       plddt.gamma, plddt.beta, er.gamma, er.beta
__global__ __launch_bounds__(TPB, MIN_BLOCKS) void af3_heads_kernel(
    const bf *__restrict__ S, const bf *__restrict__ Z,
    const bf *__restrict__ WZ, const bf *__restrict__ LNZ,
    const bf *__restrict__ WS, const bf *__restrict__ LNS,
    bf *__restrict__ O_DIS, bf *__restrict__ O_PAE, bf *__restrict__ O_PDE,
    bf *__restrict__ O_PLD, bf *__restrict__ O_ER,
    int N, int CZ, int NBZ, int CS, int NP, int NT, int R,
    int NZB, int CC, float epsz, float epss) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x;
  const int blk = blockIdx.x;

  if (blk < NZB) {
    // ----- pair heads: distogram, pde (both symmetrized) and pae -----
    //
    // One block per *unordered* pair {i, j}: distogram and pde are symmetric in
    // the output, so computing both raw directions here lets the block write
    // (i, j) and (j, i) itself and halves the pair-head MACs.  The triangular
    // index is inverted with a float sqrt plus a fixup, which is cheaper than
    // spending half the block slots on a j < i early-out.
    const int tri = (N * (N + 1)) >> 1;
    const int b = blk / tri;
    const int p = blk - b * tri;
    const float fn = 2.0f * (float)N + 1.0f;
    int i = (int)((fn - sqrtf(fmaxf(fn * fn - 8.0f * (float)p, 0.f))) * 0.5f);
    i = min(max(i, 0), N - 1);
    while (i > 0 && i * N - ((i * (i - 1)) >> 1) > p) --i;
    while (i + 1 < N && (i + 1) * N - (((i + 1) * i) >> 1) <= p) ++i;
    const int j = p - (i * N - ((i * (i - 1)) >> 1)) + i;

    bf *sh = reinterpret_cast<bf *>(smem);   // x1 | x2 | pde(x1) | pde(x2) | pae(x1) | pae(x2)
    float *red = reinterpret_cast<float *>(smem + 12 * CZ);
    const bool diag = (i == j);
    const bf *z1 = Z + ((long)(b * N + i) * N + j) * CZ;
    const bf *z2 = Z + ((long)(b * N + j) * N + i) * CZ;

    float s1 = 0.f, q1 = 0.f, s2 = 0.f, q2 = 0.f;
    for (int k = tid; k < CZ; k += TPB) {
      const bf b1 = z1[k];
      const bf b2 = diag ? b1 : z2[k];
      const float v1 = f2(b1);
      const float v2 = f2(b2);
      sh[k] = b1;                 // the distogram head reads z unnormalized
      sh[CZ + k] = b2;
      s1 += v1; q1 += v1 * v1;
      s2 += v2; q2 += v2 * v2;
    }
    block_red4<TPB>(s1, q1, s2, q2, red);

    const float inv = 1.0f / (float)CZ;
    const float m1 = s1 * inv;
    const float r1 = rsqrtf(fmaxf(q1 * inv - m1 * m1, 0.f) + epsz);
    const float m2 = s2 * inv;
    const float r2 = rsqrtf(fmaxf(q2 * inv - m2 * m2, 0.f) + epsz);
    for (int k = tid; k < CZ; k += TPB) {
      const float n1 = (f2(sh[k]) - m1) * r1;
      const float n2 = (f2(sh[CZ + k]) - m2) * r2;
      const float pw = f2(LNZ[k]), pb = f2(LNZ[CZ + k]);
      const float aw = f2(LNZ[2 * CZ + k]), ab = f2(LNZ[3 * CZ + k]);
      sh[2 * CZ + k] = tb(n1 * pw + pb);
      sh[3 * CZ + k] = tb(n2 * pw + pb);
      sh[4 * CZ + k] = tb(n1 * aw + ab);
      sh[5 * CZ + k] = tb(n2 * aw + ab);
    }
    __syncthreads();

    const int nk = CZ >> 3;
    const long oij = ((long)(b * N + i) * N + j) * NBZ;
    const long oji = ((long)(b * N + j) * N + i) * NBZ;
    const int kp = tid % ZK, ty = tid / ZK;
    // The trip count is made uniform across the block, and a thread with no
    // task of its own still walks a (harmless) row: the ZK partial sums are
    // joined with full-warp shuffles, so every lane has to reach them.
    const int ntask = 3 * NBZ;
    const int nit = (ntask + ZTY - 1) / ZTY;
    for (int it = 0; it < nit; ++it) {
      const int t0 = ty + it * ZTY;
      const bool live = (t0 < ntask);
      int t = live ? t0 : 0, g = 0;
      if (t >= NBZ) { g = 1; t -= NBZ; }
      if (t >= NBZ) { g = 2; t -= NBZ; }
      const int n = t;
      const uint4 *w4 =
          reinterpret_cast<const uint4 *>(WZ + (long)(g * NBZ + n) * CZ);
      const bf *v1p = sh + (g == 0 ? 0 : (g == 1 ? 2 * CZ : 4 * CZ));
      const bf *v2p = v1p + CZ;
      float e1 = 0.f, o1 = 0.f, e2 = 0.f, o2 = 0.f;
      for (int k0 = kp; k0 < nk; k0 += ZK * ZCH) {
        uint4 w[ZCH];
#pragma unroll
        for (int u = 0; u < ZCH; ++u) {
          const int kk = k0 + ZK * u;
          if (kk < nk) w[u] = w4[kk];
        }
#pragma unroll
        for (int u = 0; u < ZCH; ++u) {
          const int kk = k0 + ZK * u;
          if (kk < nk) {
            mac8(e1, o1, lds16(v1p + kk * 8), w[u]);
            mac8(e2, o2, lds16(v2p + kk * 8), w[u]);
          }
        }
      }
      float a1 = e1 + o1, a2 = e2 + o2;
#pragma unroll
      for (int off = 1; off < ZK; off <<= 1) {
        a1 += __shfl_xor_sync(0xffffffffu, a1, off);
        a2 += __shfl_xor_sync(0xffffffffu, a2, off);
      }
      if (kp == 0 && live) {
        if (g == 2) {
          O_PAE[oij + n] = tb(a1);
          if (!diag) O_PAE[oji + n] = tb(a2);
        } else {
          // The baseline adds two bfloat16 logit tensors, so each raw logit is
          // rounded before the add.
          const bf o = tb(f2(tb(a1)) + f2(tb(a2)));
          bf *dst = (g == 0) ? O_DIS : O_PDE;
          dst[oij + n] = o;
          if (!diag) dst[oji + n] = o;
        }
      }
    }
    return;
  }

  // ----- single-rep heads: plddt and experimentally_resolved -----
  //
  // Both read the same token row but normalize it with different affines, so
  // the block normalizes once and stages *both* results in shared memory; each
  // thread then picks the one its output column belongs to.
  const int sb = blk - NZB;
  const int cc = sb % CC;
  const int r = sb / CC;
  bf *sh = reinterpret_cast<bf *>(smem);   // plddt-affine row | er-affine row
  float *red = reinterpret_cast<float *>(smem + 4 * CS);
  const float invs = 1.0f / (float)CS;
  {
    const bf *x = S + (long)r * CS;
    float sa = 0.f, qa = 0.f, d0 = 0.f, d1 = 0.f;
    for (int k = tid; k < CS; k += TPB) {
      const bf b = x[k];
      const float v = f2(b);
      sh[k] = b;
      sa += v;
      qa += v * v;
    }
    block_red4<TPB>(sa, qa, d0, d1, red);
    const float mean = sa * invs;
    const float rstd = rsqrtf(fmaxf(qa * invs - mean * mean, 0.f) + epss);
    for (int k = tid; k < CS; k += TPB) {
      const float nv = (f2(sh[k]) - mean) * rstd;
      sh[k] = tb(nv * f2(LNS[k]) + f2(LNS[CS + k]));
      sh[CS + k] = tb(nv * f2(LNS[2 * CS + k]) + f2(LNS[3 * CS + k]));
    }
  }
  __syncthreads();

  const int g = tid % SG, c = tid / SG;
  const int c0 = cc * SWPT + c;
  const bool live = (c0 < NT);
  const int col = live ? c0 : 0;   // the SG partials are joined warp-wide below
  const bool is_pld = (col < NP);
  const bf *base = sh + (is_pld ? 0 : CS);
  const uint4 *w4 = reinterpret_cast<const uint4 *>(WS + (long)col * CS);
  const int nk = CS >> 3;
  float e0 = 0.f, o0 = 0.f;
  for (int k0 = g; k0 < nk; k0 += SG * SCH) {
    uint4 w[SCH];
#pragma unroll
    for (int u = 0; u < SCH; ++u) {
      const int kk = k0 + SG * u;
      if (kk < nk) w[u] = w4[kk];
    }
#pragma unroll
    for (int u = 0; u < SCH; ++u) {
      const int kk = k0 + SG * u;
      if (kk < nk) mac8(e0, o0, lds16(base + kk * 8), w[u]);
    }
  }
  float acc = e0 + o0;
#pragma unroll
  for (int off = 1; off < SG; off <<= 1)
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (g == 0 && live) {
    if (is_pld) O_PLD[(long)r * NP + col] = tb(acc);
    else O_ER[(long)r * (NT - NP) + (col - NP)] = tb(acc);
  }
}

}  // namespace

// ``at::empty`` routes through the ATen dispatcher, which measured ~1.05us per
// output here -- 5.2us of the forward's ~12us of CPU time, against a GPU that
// has only ~12us of work to do.  The CUDA factory underneath it does the same
// allocation without the dispatch.
static inline at::Tensor new_out(at::IntArrayRef size, const at::TensorOptions &opt) {
  return at::Tensor(at::detail::empty_cuda(size, opt));
}

py::dict af3_heads(const at::Tensor &s, const at::Tensor &z,
                   const at::Tensor &wz, const at::Tensor &lnz,
                   const at::Tensor &ws, const at::Tensor &lns,
                   int64_t np, double epsz, double epss) {
  TORCH_CHECK(s.is_cuda() && z.is_cuda(), "af3_heads: cuda tensors required");
  TORCH_CHECK(s.scalar_type() == at::kBFloat16 && z.scalar_type() == at::kBFloat16,
              "af3_heads: bfloat16 required");
  TORCH_CHECK(s.is_contiguous() && z.is_contiguous(), "af3_heads: contiguous required");
  TORCH_CHECK(s.dim() >= 2 && z.dim() >= 3, "af3_heads: bad rank");

  const int CS = (int)s.size(-1);
  const int CZ = (int)z.size(-1);
  const int N = (int)z.size(-2);
  TORCH_CHECK((int)z.size(-3) == N, "af3_heads: z is not square");
  TORCH_CHECK(CS % 8 == 0 && CZ % 8 == 0, "af3_heads: channels must be a multiple of 8");
  const int NBZ = (int)wz.size(1);
  const int NT = (int)ws.size(0);
  const long B = z.numel() / ((long)N * N * CZ);
  const int R = (int)(s.numel() / CS);
  TORCH_CHECK((long)R == B * N, "af3_heads: s and z token counts disagree");
  TORCH_CHECK(np > 0 && np < NT, "af3_heads: bad plddt split");
  TORCH_CHECK(wz.dim() == 3 && wz.size(0) == 3 && (int)wz.size(2) == CZ
                  && lnz.dim() == 2 && lnz.size(0) == 4 && (int)lnz.size(1) == CZ
                  && ws.dim() == 2 && (int)ws.size(1) == CS
                  && lns.dim() == 2 && lns.size(0) == 4 && (int)lns.size(1) == CS,
              "af3_heads: packed weights do not match the activations");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(s));
  const auto opt = s.options();
  at::DimVector zs(z.sizes().begin(), z.sizes().end());
  zs.back() = NBZ;
  at::DimVector ps(s.sizes().begin(), s.sizes().end());
  ps.back() = np;
  at::DimVector es(s.sizes().begin(), s.sizes().end());
  es.back() = NT - np;

  at::Tensor o_dis = new_out(zs, opt);
  at::Tensor o_pae = new_out(zs, opt);
  at::Tensor o_pde = new_out(zs, opt);
  at::Tensor o_pld = new_out(ps, opt);
  at::Tensor o_er = new_out(es, opt);

  const int NZB = (int)(B * ((N * (N + 1)) / 2));
  const int CC = (NT + SWPT - 1) / SWPT;
  const int grid = NZB + R * CC;
  // bf16 staging (12*CZ or 4*CS bytes, both 16B-aligned since the channel
  // counts are multiples of 8) plus the fp32 reduction scratch.
  const size_t shb = (size_t)std::max(12 * CZ, 4 * CS) + 4 * (TPB / 32) * sizeof(float);

  af3_heads_kernel<<<grid, TPB, shb,
                     at::cuda::getCurrentCUDAStream()>>>(
      (const bf *)s.data_ptr(), (const bf *)z.data_ptr(),
      (const bf *)wz.data_ptr(), (const bf *)lnz.data_ptr(),
      (const bf *)ws.data_ptr(), (const bf *)lns.data_ptr(),
      (bf *)o_dis.data_ptr(), (bf *)o_pae.data_ptr(), (bf *)o_pde.data_ptr(),
      (bf *)o_pld.data_ptr(), (bf *)o_er.data_ptr(),
      N, CZ, NBZ, CS, (int)np, NT, R, NZB, CC, (float)epsz, (float)epss);

  // Built here rather than in Python: the caller's only remaining work is the
  // call itself.
  py::dict out;
  out["distogram_logits"] = o_dis;
  out["plddt_logits"] = o_pld;
  out["pae_logits"] = o_pae;
  out["pde_logits"] = o_pde;
  out["experimentally_resolved_logits"] = o_er;
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("af3_heads", &af3_heads, "fused AlphaFold3 auxiliary heads");
}
"""

_EXT = None
_FUSED = None
_LOADED = False


def _pin_arch() -> None:
    """Build for the local arch only (the ambient list has six)."""
    override = os.environ.get("FASTKERNELS_CUDA_ARCH_LIST")
    if override is not None:
        if override.strip():
            os.environ["TORCH_CUDA_ARCH_LIST"] = override
        return
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001 - no device: leave the ambient list alone
        return
    suffix = "a" if major in (9, 10, 12) else ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _load() -> None:
    global _EXT, _FUSED, _LOADED
    _LOADED = True
    try:
        from torch.utils.cpp_extension import load_inline

        _pin_arch()
        _EXT = load_inline(
            name="fk_l3_af3_heads_fused",
            cpp_sources="",
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=False,
        )
        _FUSED = _EXT.af3_heads
    except Exception:  # noqa: BLE001 - no nvcc / no GPU: run the eager path
        _EXT = None
        _FUSED = None


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

    Outputs max_atoms_per_token * no_bins logits per token.

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


class PairformerEmbedding(nn.Module):
    """Confidence head PairformerEmbedding.

    Refines pair representation using predicted atom positions before
    confidence heads (PAE, PDE, pLDDT, experimentally resolved).

    Reference: openfold3/core/model/heads/prediction_heads.py PairformerEmbedding

    Args:
        c_s_input: Input single rep dimension
        c_z: Pair rep dimension
        c_s: Single rep dimension
        no_distance_bins: Number of distance bins
        pairformer_kwargs: Config for pairformer stack
    """

    def __init__(
        self,
        c_s_input: int = 449,
        c_z: int = 128,
        c_s: int = 384,
        no_distance_bins: int = 39,
        pairformer_no_blocks: int = 4,
        pairformer_c_hidden_pair_bias: int = 24,
        pairformer_no_heads_pair_bias: int = 16,
        pairformer_c_hidden_mul: int = 128,
        pairformer_c_hidden_pair_att: int = 32,
        pairformer_no_heads_pair: int = 4,
        pairformer_transition_n: int = 4,
        pairformer_pair_dropout: float = 0.0,
    ):
        super().__init__()
        from ..L3.alphafold3_pairformer import PairFormerStack

        self.linear_i = Linear(c_s_input, c_z, bias=False)
        self.linear_j = Linear(c_s_input, c_z, bias=False)
        self.linear_distance = Linear(no_distance_bins, c_z, bias=False)

        self.pairformer_stack = PairFormerStack(
            c_s=c_s,
            c_z=c_z,
            c_hidden_pair_bias=pairformer_c_hidden_pair_bias,
            no_heads_pair_bias=pairformer_no_heads_pair_bias,
            c_hidden_mul=pairformer_c_hidden_mul,
            c_hidden_pair_att=pairformer_c_hidden_pair_att,
            no_heads_pair=pairformer_no_heads_pair,
            no_blocks=pairformer_no_blocks,
            transition_n=pairformer_transition_n,
            pair_dropout=pairformer_pair_dropout,
        )

    def forward(
        self,
        si_input: torch.Tensor,
        zij: torch.Tensor,
        s: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zij = (
            zij
            + self.linear_i(si_input)[..., :, None, :]
            + self.linear_j(si_input)[..., None, :, :]
        )

        s, zij = self.pairformer_stack(
            s=s, z=zij, single_mask=single_mask, pair_mask=pair_mask,
        )
        return s, zij


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
        self.pairformer_embedding = PairformerEmbedding(
            c_s_input=c_s_input,
            c_z=c_z,
            c_s=c_s,
        )
        self.distogram = DistogramHead(c_z, no_bins=64)
        self.plddt = PLDDTHead(c_s, no_bins=50, max_atoms_per_token=max_atoms_per_token)
        self.pae = PAEHead(c_z, no_bins=64)
        self.pde = PDEHead(c_z, no_bins=64)
        self.experimentally_resolved = ExperimentallyResolvedHead(
            c_s, no_bins=2, max_atoms_per_token=max_atoms_per_token,
        )
        if not _LOADED:
            _load()
        # Packed kernel arguments, built on first forward and dropped whenever
        # the parameters are replaced (``load_state_dict`` / ``.to()``).
        self._packed = None
        if hasattr(self, "register_load_state_dict_post_hook"):
            self.register_load_state_dict_post_hook(_drop_packed)

    def _apply(self, *args, **kwargs):
        self._packed = None
        return super()._apply(*args, **kwargs)

    def _pack(self):
        """Contiguous kernel arguments, or None if the fused path cannot run."""
        if _FUSED is None:
            return None
        dis, pae, pde = self.distogram, self.pae, self.pde
        pld, er = self.plddt, self.experimentally_resolved
        lins = (dis.linear, pae.linear, pde.linear, pld.linear, er.linear)
        norms = (pae.layer_norm, pde.layer_norm, pld.layer_norm, er.layer_norm)
        if any(l.bias is not None for l in lins):
            return None
        if any(n.weight is None or n.bias is None for n in norms):
            return None
        ws = [l.weight for l in lins] + [n.weight for n in norms]
        ws += [n.bias for n in norms]
        if any(w.dtype is not torch.bfloat16 or not w.is_cuda for w in ws):
            return None
        if dis.linear.weight.shape != pde.linear.weight.shape:
            return None
        if dis.linear.weight.shape != pae.linear.weight.shape:
            return None
        if pae.layer_norm.eps != pde.layer_norm.eps:
            return None
        if pld.layer_norm.eps != er.layer_norm.eps:
            return None
        with torch.no_grad():
            wz = torch.stack(
                [dis.linear.weight, pde.linear.weight, pae.linear.weight]
            ).contiguous()
            lnz = torch.stack([
                pde.layer_norm.weight, pde.layer_norm.bias,
                pae.layer_norm.weight, pae.layer_norm.bias,
            ]).contiguous()
            wsm = torch.cat([pld.linear.weight, er.linear.weight]).contiguous()
            lns = torch.stack([
                pld.layer_norm.weight, pld.layer_norm.bias,
                er.layer_norm.weight, er.layer_norm.bias,
            ]).contiguous()
        return (wz, lnz, wsm, lns, pld.linear.weight.shape[0],
                float(pae.layer_norm.eps), float(pld.layer_norm.eps))

    def _eager(self, s: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
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
        packed = self._packed
        if packed is None:
            if _FUSED is None or torch.is_grad_enabled():
                return self._eager(s, z)
            packed = self._packed = self._pack()
            if packed is None:
                return self._eager(s, z)
        elif torch.is_grad_enabled():
            # The kernel returns leaf tensors; a training-mode caller needs the
            # autograd-capable path.
            return self._eager(s, z)
        try:
            # Shape/dtype/contiguity are validated inside the kernel wrapper, and
            # it returns the result dict itself; a call it cannot serve raises
            # and falls back here.
            return _FUSED(s, z, *packed)
        except Exception:  # noqa: BLE001 - unsupported layout: eager path
            return self._eager(s, z)


def _drop_packed(module, incompatible_keys):
    module._packed = None
