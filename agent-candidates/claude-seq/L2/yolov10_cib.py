"""YOLOv10 CIB (Compact Inverted Block) -- the whole block in one CUDA launch.

The captured shapes are tiny ([1|4, 128, 20, 20]): the five ``YOLOConv`` layers
plus the ``YOLORepVGGDW`` branch add up to ~128 MMAC, well under a microsecond
of arithmetic, while the eager baseline spends ~225us dispatching ~19 cuDNN /
BatchNorm / SiLU kernels.  Everything here is aimed at that: fold each
BatchNorm (eval-mode, so a per-channel affine) into its convolution, fold the
RepVGGDW 3x3 branch into the centre of the 7x7 kernel, and run all five stages
inside a *single* persistent kernel whose stages are separated by a grid-wide
barrier instead of a kernel launch.

Stage map (per image, NCHW, P = H*W positions):

  A  t1 = silu(dw3x3(x))         C1 planes, channel-paired half2 FMA
  B  t2 = silu(pw(t1))           GEMM [C2 x C1] @ [C1 x P], wmma tensor cores
  C  t3 = silu(dw7x7(t2))        C2 planes (7x7 = fused 7x7 + centred 3x3)
  D  t4 = silu(pw(t3))           GEMM [C1 x C2] @ [C2 x P]
  E  out = x + silu(dw3x3(t4))

The depthwise stages stage one 20x20 plane *pair* in shared memory and run two
channels at a time in the two halves of a ``half2``, so one ``hfma2`` retires
two taps.  The pointwise stages are plain GEMMs in the native NCHW layout: for
one image ``t[c][p]`` already is a K-major (K x P) matrix, so the activation
tile needs no transpose to feed a ``wmma`` ``matrix_b`` fragment.

Anything the kernel does not cover (other channel counts, non-``lk`` blocks, no
residual, non-fp16, CPU tensors) falls through to the baseline submodules.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv
from .yolov10_repvggdw import YOLORepVGGDW

# Must mirror NT / KC in the kernel source below.
NT_ = 80
KC_ = 128

_CUDA_SRC = r"""
#include <torch/extension.h>

#include <mutex>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>

// ---------------------------------------------------------------------------
// Tiling.  MT x NT is the pointwise output tile (out channels x positions), KC
// the reduction chunk.  MT is deliberately small: at these sizes the stages are
// latency-bound, so what matters is that every stage has at least one work item
// per resident block (~500) rather than arithmetic reuse -- the global traffic
// is a couple of MB either way.  The +8 row paddings keep the shared tiles
// bank-conflict free and 32B aligned, which is what wmma's load/store_matrix_sync
// requires.  __launch_bounds__ pins 4 blocks/SM so the grid can cover the widest
// stage (N*C2/2 items) in a single round.
// ---------------------------------------------------------------------------
#define NTHREADS 256
#define MT 16
#define NT 80
#define KC 128
#define NFRAG (NT / 16)
#define MFRAG (MT / 16)
#define NACC ((MFRAG * NFRAG + 7) / 8)
#define LDA (KC + 8)
#define LDB (NT + 8)
#define SH_PW ((MT * LDA + KC * LDB) * (int)sizeof(__half))
#define SH_EPI (MT * NT * (int)sizeof(float) + MT * (int)sizeof(__half))

struct Params {
  const __half* __restrict__ x;
  __half* __restrict__ out;
  __half* __restrict__ t1;
  __half* __restrict__ t2;
  __half* __restrict__ t3;
  __half* __restrict__ t4;
  const __half2* __restrict__ w0;   // [C1/2][9]   channel-paired
  const __half2* __restrict__ b0;   // [C1/2]
  const __half*  __restrict__ w1;   // [C2][C1]
  const __half*  __restrict__ b1;   // [C2]
  const __half2* __restrict__ w2;   // [C2/2][49]
  const __half2* __restrict__ b2;   // [C2/2]
  const __half*  __restrict__ w3;   // [C1][C2]
  const __half*  __restrict__ b3;   // [C1]
  const __half2* __restrict__ w4;   // [C1/2][9]
  const __half2* __restrict__ b4;   // [C1/2]
  unsigned long long* bar;
  int N, C1, C2, H, W, P;
};

__device__ __forceinline__ float silu(float v) {
  return v / (1.f + __expf(-v));
}

// n / d for small non-negative n: a float reciprocal plus two fixups.  A
// runtime 32-bit divide is an out-of-line call on this device and these
// quotients sit in the innermost loops.
__device__ __forceinline__ int fdiv(int n, int d, float rcp) {
  int q = (int)((float)n * rcp);
  q -= (q * d > n);
  q += ((q + 1) * d <= n);
  return q;
}

// Grid-wide barrier: a monotonic 64-bit arrival counter (so it survives any
// number of launches); release/acquire ordering rather than a full fence.
__device__ __forceinline__ void gbar(unsigned long long* c, unsigned long long nb) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long old, v;
    asm volatile("atom.add.release.gpu.u64 %0, [%1], 1;" : "=l"(old) : "l"(c) : "memory");
    const unsigned long long t = (old / nb + 1ull) * nb;
    do {
      asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(c) : "memory");
    } while (v < t);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Depthwise KS x KS + bias + SiLU (+ optional residual).
//
// A work item is (image, channel pair): the pair's two planes are staged
// interleaved in a zero-bordered half2 tile, so one LDS.32 plus one hfma2
// retires a tap for *both* channels.  A thread owns two horizontally adjacent
// positions, which share KS-1 of their KS taps per row, so a row of taps costs
// KS+1 loads instead of 2*KS, the two accumulators keep the fma chain from
// serialising, and the pair of outputs stores as one half2.  The taps live in
// shared memory rather than registers: a 7x7 kernel is 49 half2 and holding
// that in registers starved the rest of the (fused, five-stage) kernel.
// ---------------------------------------------------------------------------
template <int KS, bool RES>
__device__ void dw_stage(const Params& p, const __half* __restrict__ in,
                         __half* __restrict__ out, const __half2* __restrict__ wt,
                         const __half2* __restrict__ bs, int Cch,
                         const __half* __restrict__ res, __half2* sh) {
  constexpr int PAD = KS / 2;
  constexpr int NW = KS * KS;
  const int SW = p.W + 2 * PAD;
  const int SN = (p.H + 2 * PAD) * SW;
  const int CP = Cch >> 1;
  const int items = p.N * CP;
  const int WP = p.W >> 1;                 // position pairs per row
  const int NP = p.P >> 1;                 // position pairs per plane
  const __half2 zero = __float2half2_rn(0.f);
  const int tid = threadIdx.x;
  __half2* shw = sh;
  __half2* sht = sh + NW;
  const float rcp_wp = 1.f / (float)WP;
  const float rcp_sw = 1.f / (float)SW;
  const float rcp_cp = 1.f / (float)CP;

  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int n = fdiv(it, CP, rcp_cp);
    const int cp = it - n * CP;
    const size_t base = (size_t)(n * Cch + 2 * cp) * p.P;
    const __half* i0 = in + base;
    const __half* i1 = i0 + p.P;

    if (tid < NW) shw[tid] = wt[cp * NW + tid];
    for (int s = tid; s < SN; s += NTHREADS) {
      const int sy = fdiv(s, SW, rcp_sw);
      const int h = sy - PAD, w = s - sy * SW - PAD;
      __half2 v = zero;
      if ((unsigned)h < (unsigned)p.H && (unsigned)w < (unsigned)p.W) {
        const int q = h * p.W + w;
        v = __halves2half2(__ldcg(i0 + q), __ldcg(i1 + q));
      }
      sht[s] = v;
    }
    const __half2 bb = bs[cp];
    __syncthreads();

    __half* o0 = out + base;
    __half* o1 = o0 + p.P;
    const __half* r0 = RES ? res + base : nullptr;
    for (int j = tid; j < NP; j += NTHREADS) {
      const int h = fdiv(j, WP, rcp_wp);
      const int w = (j - h * WP) << 1;
      const __half2* sp = sht + h * SW + w;
      __half2 ac0 = zero, ac1 = zero;
#pragma unroll
      for (int ky = 0; ky < KS; ++ky) {
        __half2 v[KS + 1];
#pragma unroll
        for (int i = 0; i <= KS; ++i) v[i] = sp[ky * SW + i];
#pragma unroll
        for (int kx = 0; kx < KS; ++kx) {
          const __half2 ww = shw[ky * KS + kx];
          ac0 = __hfma2(ww, v[kx], ac0);
          ac1 = __hfma2(ww, v[kx + 1], ac1);
        }
      }
      ac0 = __hadd2(ac0, bb);
      ac1 = __hadd2(ac1, bb);
      __half2 y0 = __halves2half2(__float2half(silu(__low2float(ac0))),
                                  __float2half(silu(__low2float(ac1))));
      __half2 y1 = __halves2half2(__float2half(silu(__high2float(ac0))),
                                  __float2half(silu(__high2float(ac1))));
      const int q = h * p.W + w;
      if (RES) {
        y0 = __hadd2(y0, __ldcg((const __half2*)(r0 + q)));
        y1 = __hadd2(y1, __ldcg((const __half2*)(r0 + p.P + q)));
        *(__half2*)(o0 + q) = y0;
        *(__half2*)(o1 + q) = y1;
      } else {
        __stcg((__half2*)(o0 + q), y0);
        __stcg((__half2*)(o1 + q), y1);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Pointwise (1x1) conv + bias + SiLU as a per-image GEMM
//     out[M][P] = Wt[M][K] @ in[K][P]
// on tensor cores.  For one image ``t[c][p]`` already is a K-major (K x P)
// matrix, so both operands stage into shared memory with plain 128-bit
// contiguous loads and no transpose.  A work item is (image, MT-channel tile,
// NT-position tile); the tile's 16x16 fragments are dealt round-robin to the
// eight warps, and each splits k over two accumulators so the mma chain is not
// serialised on one of them.  MT is deliberately small: these stages are
// latency-bound, so having an item per resident block beats arithmetic reuse
// (measured: MT=16 is 1.4x faster than MT=64 and 2x faster than MT=128).
// ---------------------------------------------------------------------------
__device__ void pw_stage(const Params& p, const __half* __restrict__ Wt,
                         const __half* __restrict__ bias,
                         const __half* __restrict__ in, __half* __restrict__ out,
                         int M, int K, char* raw) {
  using namespace nvcuda::wmma;
  __half* sh_a = (__half*)raw;
  __half* sh_b = sh_a + MT * LDA;
  float* sh_c = (float*)raw;

  const int nt = p.P / NT;
  const int mt = M / MT;
  const int items = p.N * mt * nt;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const float rcp_nt = 1.f / (float)nt;
  const float rcp_mt = 1.f / (float)mt;

  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int r1 = fdiv(it, nt, rcp_nt);
    const int pi = it - r1 * nt;
    const int n = fdiv(r1, mt, rcp_mt);
    const int m0 = (r1 - n * mt) * MT, p0 = pi * NT;

    fragment<accumulator, 16, 16, 16, float> ac[NACC][2];
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
      fill_fragment(ac[j][0], 0.f);
      fill_fragment(ac[j][1], 0.f);
    }
    const __half* ap = Wt + (size_t)m0 * K;
    const __half* bp = in + (size_t)n * K * p.P + p0;

    for (int k0 = 0; k0 < K; k0 += KC) {
      __syncthreads();
      for (int i = tid; i < MT * (KC / 8); i += NTHREADS) {
        const int r = i / (KC / 8), c8 = (i - r * (KC / 8)) << 3;
        *(int4*)(sh_a + r * LDA + c8) =
            *(const int4*)(ap + (size_t)r * K + k0 + c8);
      }
      for (int i = tid; i < KC * (NT / 8); i += NTHREADS) {
        const int r = i / (NT / 8);
        const int c8 = (i - r * (NT / 8)) << 3;
        *(int4*)(sh_b + r * LDB + c8) =
            __ldcg((const int4*)(bp + (size_t)(k0 + r) * p.P + c8));
      }
      __syncthreads();

#pragma unroll
      for (int j = 0; j < NACC; ++j) {
        const int f = warp + (j << 3);
        if (f < MFRAG * NFRAG) {
          const int mi = f / NFRAG;
          const __half* sa = sh_a + (mi * 16) * LDA;
          const __half* sb = sh_b + (f - mi * NFRAG) * 16;
#pragma unroll
          for (int kk = 0; kk + 1 < KC / 16; kk += 2) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> f0, f1;
            fragment<matrix_b, 16, 16, 16, __half, row_major> g0, g1;
            load_matrix_sync(f0, sa + kk * 16, LDA);
            load_matrix_sync(g0, sb + (kk * 16) * LDB, LDB);
            load_matrix_sync(f1, sa + (kk + 1) * 16, LDA);
            load_matrix_sync(g1, sb + ((kk + 1) * 16) * LDB, LDB);
            mma_sync(ac[j][0], f0, g0, ac[j][0]);
            mma_sync(ac[j][1], f1, g1, ac[j][1]);
          }
        }
      }
    }

    __syncthreads();
#pragma unroll
    for (int j = 0; j < NACC; ++j) {
      const int f = warp + (j << 3);
      if (f < MFRAG * NFRAG) {
        const int mi = f / NFRAG;
#pragma unroll
        for (int i = 0; i < ac[j][0].num_elements; ++i)
          ac[j][0].x[i] += ac[j][1].x[i];
        store_matrix_sync(sh_c + (mi * 16) * NT + (f - mi * NFRAG) * 16,
                          ac[j][0], NT, mem_row_major);
      }
    }
    __syncthreads();

    // The tile's MT biases are read MT*NT/NTHREADS times each; stage them once.
    __half* sh_bias = (__half*)(sh_c + MT * NT);
    if (tid < MT) sh_bias[tid] = bias[m0 + tid];
    __syncthreads();
    for (int i = tid; i < MT * NT; i += NTHREADS) {
      const int r = i / NT;
      __stcg(out + (size_t)(n * M + m0 + r) * p.P + p0 + (i - r * NT),
             __float2half(silu(sh_c[i] + __half2float(sh_bias[r]))));
    }
  }
}

__global__ __launch_bounds__(NTHREADS, 4) void cib_kernel(Params p) {
  extern __shared__ __align__(32) char raw[];
  const unsigned long long nb = gridDim.x;

  dw_stage<3, false>(p, p.x, p.t1, p.w0, p.b0, p.C1, nullptr, (__half2*)raw);
  gbar(p.bar, nb);
  pw_stage(p, p.w1, p.b1, p.t1, p.t2, p.C2, p.C1, raw);
  gbar(p.bar, nb);
  dw_stage<7, false>(p, p.t2, p.t3, p.w2, p.b2, p.C2, nullptr, (__half2*)raw);
  gbar(p.bar, nb);
  pw_stage(p, p.w3, p.b3, p.t3, p.t4, p.C1, p.C2, raw);
  gbar(p.bar, nb);
  dw_stage<3, true>(p, p.t4, p.out, p.w4, p.b4, p.C1, p.x, (__half2*)raw);
}

// ---------------------------------------------------------------------------
// Host side: a plan holds everything but the input/output pointers, so the
// per-call Python path is one pybind call with two arguments.
// ---------------------------------------------------------------------------
struct Plan {
  Params p;
  int grid;
  int shmem;
};

static std::vector<Plan> g_plans;
static std::mutex g_plan_mu;

static void set_shmem(int shmem) {
  static int done = 0;
  if (shmem > 48 * 1024 && !done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((const void*)cib_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    done = 1;
  }
}

static int max_grid(int shmem) {
  int per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, (const void*)cib_kernel, NTHREADS, shmem));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int g = per_sm * sms;
  return g > 0 ? g : sms;
}

int64_t cib_prepare(torch::Tensor wblob, torch::Tensor ws, torch::Tensor bar,
                    std::vector<int64_t> off, int64_t N, int64_t C1, int64_t C2,
                    int64_t H, int64_t W) {
  const at::cuda::CUDAGuard guard(wblob.device());
  Plan pl{};
  Params& p = pl.p;
  const __half* wb = (const __half*)wblob.data_ptr();
  __half* w = (__half*)ws.data_ptr();
  const int P = (int)(H * W);
  p.N = (int)N; p.C1 = (int)C1; p.C2 = (int)C2;
  p.H = (int)H; p.W = (int)W; p.P = P;
  p.t1 = w;
  p.t2 = p.t1 + (size_t)N * C1 * P;
  p.t3 = p.t2 + (size_t)N * C2 * P;
  p.t4 = p.t3 + (size_t)N * C2 * P;
  p.w0 = (const __half2*)(wb + off[0]);
  p.b0 = (const __half2*)(wb + off[1]);
  p.w1 = wb + off[2];
  p.b1 = wb + off[3];
  p.w2 = (const __half2*)(wb + off[4]);
  p.b2 = (const __half2*)(wb + off[5]);
  p.w3 = wb + off[6];
  p.b3 = wb + off[7];
  p.w4 = (const __half2*)(wb + off[8]);
  p.b4 = (const __half2*)(wb + off[9]);
  p.bar = (unsigned long long*)bar.data_ptr();

  const int dw3 = (int)(((H + 2) * (W + 2) + 9) * sizeof(__half2));
  const int dw7 = (int)(((H + 6) * (W + 6) + 49) * sizeof(__half2));
  int shmem = SH_PW > SH_EPI ? SH_PW : SH_EPI;
  shmem = shmem > dw3 ? shmem : dw3;
  shmem = shmem > dw7 ? shmem : dw7;
  pl.shmem = shmem;

  const int items = (int)(N * (C2 >> 1));       // stage C, the widest
  set_shmem(shmem);
  const int cap = max_grid(shmem);
  pl.grid = items < cap ? items : cap;
  TORCH_CHECK(pl.grid > 0, "empty grid");

  std::lock_guard<std::mutex> lk(g_plan_mu);
  g_plans.push_back(pl);
  return (int64_t)g_plans.size() - 1;
}

torch::Tensor cib_forward(torch::Tensor x, int64_t plan) {
  const Plan& pl = g_plans[plan];
  torch::Tensor out = torch::empty_like(x);
  Params p = pl.p;
  p.x = (const __half*)x.data_ptr();
  p.out = (__half*)out.data_ptr();
  cib_kernel<<<pl.grid, NTHREADS, pl.shmem,
               at::cuda::getCurrentCUDAStream()>>>(p);
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
int64_t cib_prepare(torch::Tensor, torch::Tensor, torch::Tensor,
                    std::vector<int64_t>, int64_t, int64_t, int64_t, int64_t,
                    int64_t);
torch::Tensor cib_forward(torch::Tensor, int64_t);
"""

_EXT = None
_EXT_FAILED = False


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}{'a' if major >= 9 else ''}"


def _ext():
    """The compiled extension, or None if it cannot be built here."""
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline

            arch = _arch_list()
            if arch:
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
            tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
            _EXT = load_inline(
                name=f"fk_yolov10_cib_{tag}",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["cib_prepare", "cib_forward"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "-use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-relaxed-constexpr",
                ],
                verbose=False,
            )
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _fuse_bn(conv, bn):
    """Eval-mode Conv-BN as a single (weight, bias) pair, in fp32."""
    w = conv.weight.float()
    if bn is None:
        b = conv.bias.float() if conv.bias is not None else w.new_zeros(w.shape[0])
        return w, b
    inv = torch.rsqrt(bn.running_var.float() + bn.eps)
    s = bn.weight.float() * inv
    b = bn.bias.float() - bn.running_mean.float() * s
    if conv.bias is not None:
        b = b + s * conv.bias.float()
    return w * s.view(-1, *([1] * (w.dim() - 1))), b


def _conv_bn(yc: YOLOConv):
    return _fuse_bn(yc.conv, None if getattr(yc, "_is_fused", False) else yc.bn)


def _pair(t: torch.Tensor) -> torch.Tensor:
    """[C, K] -> [C/2, K, 2]: adjacent channels in the halves of a half2."""
    c, k = t.shape
    return t.view(c // 2, 2, k).permute(0, 2, 1).contiguous()


class YOLOCIB(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            YOLOConv(c1, c1, 3, g=c1),
            YOLOConv(c1, 2 * c_, 1),
            YOLOConv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else YOLORepVGGDW(2 * c_),
            YOLOConv(2 * c_, c2, 1),
            YOLOConv(c2, c2, 3, g=c2),
        )
        self.add = shortcut and c1 == c2
        self._lk = lk
        self._c0 = self.cv1[0]
        self._fast = None       # None = untested, False = use the baseline path
        self._fn = None         # bound kernel entry point once a plan exists
        self._plan = None
        self._pn = -1           # batch size the plan was built for
        self._pf = None         # cv1[0]._is_fused when it was built
        self._keep = None       # keeps the plan's device buffers alive

    # -- fast path --------------------------------------------------------
    def _eligible(self, x: torch.Tensor) -> bool:
        """Whether the fused kernel covers this block and this input.

        The kernel assumes the captured CIB shape: an ``lk`` block with a
        residual, fp16 NCHW activations, channel counts that are whole
        multiples of the pointwise reduction chunk, a position count that is a
        whole multiple of the position tile, and an even width (a thread owns
        two adjacent positions in the depthwise stages).
        """
        if not (self._lk and self.add and x.is_cuda and x.is_contiguous()):
            return False
        if x.dim() != 4 or x.dtype != torch.float16:
            return False
        c1 = self.cv1[0].conv.weight.shape[0]
        c2 = self.cv1[1].conv.weight.shape[0]
        if x.shape[1] != c1 or c1 % KC_ != 0 or c2 % KC_ != 0:
            return False
        if (x.shape[2] * x.shape[3]) % NT_ != 0 or x.shape[3] % 2 != 0:
            return False
        if self.cv1[0].conv.weight.shape[-1] != 3 or self.cv1[4].conv.weight.shape[-1] != 3:
            return False
        rep = self.cv1[2]
        if rep.conv.conv.weight.shape[-1] != 7:
            return False
        if any(p.dtype != torch.float16 for p in self.parameters()):
            return False
        return _ext() is not None

    @torch.no_grad()
    def _build_plan(self, x: torch.Tensor) -> int:
        dev = x.device
        c1 = self.cv1[0].conv.weight.shape[0]
        c2 = self.cv1[1].conv.weight.shape[0]
        n, _, h, w = x.shape
        p = h * w

        w0, b0 = _conv_bn(self.cv1[0])
        w1, b1 = _conv_bn(self.cv1[1])
        w3, b3 = _conv_bn(self.cv1[3])
        w4, b4 = _conv_bn(self.cv1[4])
        rep = self.cv1[2]
        if getattr(rep, "_is_fused", False):
            w2, b2 = _conv_bn(rep.conv)
        else:
            wa, ba = _conv_bn(rep.conv)
            wb, bb = _conv_bn(rep.conv1)
            w2 = wa + nn.functional.pad(wb, [2, 2, 2, 2])
            b2 = ba + bb

        parts = [
            _pair(w0.reshape(c1, 9)),
            b0.view(c1 // 2, 2),
            w1.reshape(c2, c1),
            b1,
            _pair(w2.reshape(c2, 49)),
            b2.view(c2 // 2, 2),
            w3.reshape(c1, c2),
            b3,
            _pair(w4.reshape(c1, 9)),
            b4.view(c1 // 2, 2),
        ]
        offs, flat, pos = [], [], 0
        for t in parts:
            offs.append(pos)
            f = t.reshape(-1).to(device=dev, dtype=torch.float16)
            pad = (-f.numel()) % 16
            if pad:
                f = torch.cat([f, f.new_zeros(pad)])
            flat.append(f)
            pos += f.numel()
        blob = torch.cat(flat)
        ws = torch.empty(n * (2 * c1 + 2 * c2) * p, dtype=torch.float16, device=dev)
        bar = torch.zeros(1, dtype=torch.int64, device=dev)
        self._keep = (blob, ws, bar)
        return _ext().cib_prepare(blob, ws, bar, offs, n, c1, c2, h, w)

    def _setup(self, x: torch.Tensor) -> torch.Tensor:
        """First call for a given batch size / weight state (or the slow path)."""
        if self._fast is None:
            self._fast = self._eligible(x)
        if self._fast:
            self._plan = self._build_plan(x)
            self._pn = x.shape[0]
            self._pf = self._c0._is_fused
            self._fn = _ext().cib_forward
            return self._fn(x, self._plan)
        y = self.cv1(x)
        return x + y if self.add else y

    def _load_from_state_dict(self, *args, **kwargs):
        self._fast = None       # weights changed: refold them on the next call
        self._fn = None
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = self._fn
        if fn is not None and x.shape[0] == self._pn and self._c0._is_fused == self._pf:
            return fn(x, self._plan)
        return self._setup(x)

