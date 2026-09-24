"""YOLOv10 Spatial Pyramid Pooling - Fast (fused CUDA implementation).

The baseline runs ~10 separate CUDA kernels (conv, batch-norm, SiLU, three
max-pools, cat, conv, batch-norm, SiLU) on a tiny 20x20 feature map, so it is
entirely launch-overhead bound.  This version folds the whole block into three
hand-written kernels chained with Programmatic Dependent Launch:

  1. ``cv1``: 1x1 conv (a 128x1600x256 GEMM) with the BatchNorm affine folded in
     and SiLU applied in the epilogue, writing channels 0..127 of a scratch
     buffer.
  2. the 5x5 max-pool cascade, computed separably (row pass then column pass)
     in shared memory, filling channels 128..511 of the same buffer -- which is
     exactly the ``cat((x, y1, y2, y3), 1)`` the baseline materialises.
  3. ``cv2``: 1x1 conv (a 256x1600x512 GEMM) over that buffer, again with
     BatchNorm folded in and SiLU in the epilogue.

Both GEMMs use wmma m16n16k16 tensor-core fragments fed from shared memory
staged with ``cp.async``, 32x32 warp tiles (one fragment load per mma) and
split-K across warps for occupancy.  ``forward`` falls back to the reference
implementation for any shape/dtype/device the kernels do not cover.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..L1.max_pool2d import MaxPool2d
from .yolov10_conv import YOLOConv

_CPP = r'''

#include <torch/extension.h>
at::Tensor sppf_forward(at::Tensor x, at::Tensor W1, at::Tensor sb1,
                        at::Tensor W2, at::Tensor sb2, at::Tensor tc,
                        int64_t npixp);
'''

_CU = r'''

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define PP 400
#define HH 20
#define WW 20

__device__ __forceinline__ void cpa16(void* sm, const void* gm) {
  unsigned s = (unsigned)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gm));
}
#define CPAC()  asm volatile("cp.async.commit_group;\n" ::)
#define CPAW0() asm volatile("cp.async.wait_group 0;\n" ::)

// ---------------------------------------------------------------- generic GEMM+silu
// B: [KTOT][bstr] starting at pixel p0 ; A: [MTOT][KTOT]
// out: TC[m][dstr] (flat pix) if FLATD, else OUT[n][MTOT][PP]
template <int KTOT, int MTOT, int MT, int NT, int WM, int WN, int KSP, int FLATD>
struct Cg {
  static constexpr int NWT = (MT/WM)*(NT/WN);
  static constexpr int NW = NWT*KSP, NTH = NW*32;
  static constexpr int AP = KTOT+8, BP = NT+8, DP = NT+8;
  static constexpr int SA = MT*AP, SB = KTOT*BP;
  static constexpr int KW = KTOT/KSP, AM = WM/16, BN = WN/16;
  static constexpr int SM = ((SA+SB)*2 > KSP*MT*DP*4) ? (SA+SB)*2 : KSP*MT*DP*4;
};

template <int KTOT, int MTOT, int MT, int NT, int WM, int WN, int KSP, int FLATD>
__global__ __launch_bounds__(Cg<KTOT,MTOT,MT,NT,WM,WN,KSP,FLATD>::NTH)
void gemm_silu(const __half* __restrict__ Bg, const __half* __restrict__ Ag,
               const float* __restrict__ sb, __half* __restrict__ Dg,
               int ntiles, int bstr, int dstr, int npix, int pdl) {
  if (pdl) cudaGridDependencySynchronize();
  using C = Cg<KTOT,MTOT,MT,NT,WM,WN,KSP,FLATD>;
  extern __shared__ __half sm[];
  __half* As = sm;
  __half* Bs = sm + C::SA;
  float* Ds = (float*)sm;
  const int b = blockIdx.x;
  const int mt = b % (MTOT/MT);
  int rest = b / (MTOT/MT);
  const int nt = FLATD ? (rest % ntiles) : rest;
  const int n  = FLATD ? (rest / ntiles) : 0;
  const int m0 = mt*MT, p0 = nt*NT;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int g = warp / C::NWT, wt = warp % C::NWT;
  const int wm = (wt % (MT/WM))*WM, wn = (wt / (MT/WM))*WN;
  const __half* Ab = Ag + (size_t)m0*KTOT;
  const __half* Bb = Bg + (FLATD ? (size_t)n*KTOT*PP : 0) + p0;
#pragma unroll
  for (int i = tid; i < MT*(KTOT/8); i += C::NTH) {
    int r = i/(KTOT/8), c = (i%(KTOT/8))*8;
    cpa16(&As[r*C::AP + c], Ab + (size_t)r*KTOT + c);
  }
#pragma unroll
  for (int i = tid; i < KTOT*(NT/8); i += C::NTH) {
    int r = i/(NT/8), c = (i%(NT/8))*8;
    cpa16(&Bs[r*C::BP + c], Bb + (size_t)r*bstr + c);
  }
  CPAC(); CPAW0(); __syncthreads();

  wmma::fragment<wmma::accumulator,16,16,16,float> acc[C::AM][C::BN];
#pragma unroll
  for (int i = 0; i < C::AM; i++)
#pragma unroll
    for (int j = 0; j < C::BN; j++) wmma::fill_fragment(acc[i][j], 0.f);
  {
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> fa[C::AM];
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> fb[C::BN];
#pragma unroll
    for (int kk = 0; kk < C::KW; kk += 16) {
      const int ko = g*C::KW + kk;
#pragma unroll
      for (int i = 0; i < C::AM; i++)
        wmma::load_matrix_sync(fa[i], &As[(wm+i*16)*C::AP + ko], C::AP);
#pragma unroll
      for (int j = 0; j < C::BN; j++)
        wmma::load_matrix_sync(fb[j], &Bs[ko*C::BP + wn + j*16], C::BP);
#pragma unroll
      for (int i = 0; i < C::AM; i++)
#pragma unroll
        for (int j = 0; j < C::BN; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < C::AM; i++)
#pragma unroll
    for (int j = 0; j < C::BN; j++)
      wmma::store_matrix_sync(&Ds[(g*MT + wm + i*16)*C::DP + wn + j*16], acc[i][j],
                              C::DP, wmma::mem_row_major);
  __syncthreads();
  const float* sc = sb; const float* bi = sb + MTOT;
#pragma unroll
  for (int i = tid; i < MT*(NT/8); i += C::NTH) {
    const int r = i/(NT/8), c = (i%(NT/8))*8;
    const float s = sc[m0+r], bb = bi[m0+r];
    __half2 o[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float v0 = Ds[r*C::DP + c+2*j], v1 = Ds[r*C::DP + c+2*j+1];
#pragma unroll
      for (int q = 1; q < KSP; q++) {
        v0 += Ds[(q*MT+r)*C::DP + c+2*j];
        v1 += Ds[(q*MT+r)*C::DP + c+2*j+1];
      }
      v0 = v0*s + bb; v1 = v1*s + bb;
      v0 = __fdividef(v0, 1.f + __expf(-v0));
      v1 = __fdividef(v1, 1.f + __expf(-v1));
      o[j] = __floats2half2_rn(v0, v1);
    }
    if (FLATD) {
      *(uint4*)(Dg + (size_t)(m0+r)*dstr + (size_t)n*PP + p0 + c) = *(uint4*)o;
    } else {
      const int pix = p0 + c;
      if (pix < npix) {
        const int nn = pix/PP, pq = pix - nn*PP;
        *(uint4*)(Dg + (size_t)nn*MTOT*PP + (size_t)(m0+r)*PP + pq) = *(uint4*)o;
      }
    }
  }
}

// ---------------------------------------------------------------- pooling
// y1 = pool5(t), y2 = pool5(y1) = pool9(t), y3 = pool5(y2) = pool13(t).
// Max over a rectangle is separable, so each level is one row pass + one column
// pass of the corresponding radius -- no cascade, hence only two barriers.
#define GP 2
#define NTHP (GP*PP)

template <int RAD>
__device__ __forceinline__ __half rmax(const __half* p, int w) {
  __half m = p[0];
#pragma unroll
  for (int d = 1; d <= RAD; d++) {
    m = __hmax(m, p[w - d < 0 ? -w : -d]);
    m = __hmax(m, p[w + d > WW - 1 ? (WW - 1 - w) : d]);
  }
  return m;
}
template <int RAD>
__device__ __forceinline__ __half cmax(const __half* p, int h) {
  __half m = p[0];
#pragma unroll
  for (int d = 1; d <= RAD; d++) {
    m = __hmax(m, p[(h - d < 0 ? -h : -d) * WW]);
    m = __hmax(m, p[(h + d > HH - 1 ? (HH - 1 - h) : d) * WW]);
  }
  return m;
}

__global__ __launch_bounds__(NTHP) void k2_pool(__half* __restrict__ TC, int dstr, int pdl) {
  if (pdl) cudaGridDependencySynchronize();
  const int b = blockIdx.x;
  const int cg = b % (128/GP);
  const int n = b / (128/GP);
  const int c0 = cg*GP;
  const int t = threadIdx.x;
  const int c = t / PP;
  const int hw = t - c*PP;
  const int h = hw/WW, w = hw - h*WW;
  __shared__ __half st[NTHP], s1[NTHP], s2[NTHP], s3[NTHP];
  __half* base = TC + (size_t)n*PP;
  st[t] = base[(size_t)(c0+c)*dstr + hw];
  __syncthreads();
  const __half* p = st + t;
  s1[t] = rmax<2>(p, w);
  s2[t] = rmax<4>(p, w);
  s3[t] = rmax<6>(p, w);
  __syncthreads();
  __half* d = base + (size_t)(128 + c0 + c)*dstr + hw;
  d[0] = cmax<2>(s1 + t, h);
  d[(size_t)128*dstr] = cmax<4>(s2 + t, h);
  d[(size_t)256*dstr] = cmax<6>(s3 + t, h);
}

template <typename F, typename... Args>
static void lpdl(F kern, dim3 g, dim3 bk, int smb, cudaStream_t s, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = bk; cfg.dynamicSmemBytes = smb; cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, kern, args...);
}

// cv1: K=256, M=128, block 32x80, warp 32x16, 2-way split-K, flat dest
#define KCV1 256,128,32,80,32,16,2,1
// cv2: K=512, M=256, block 32x64, warp 32x32, 4-way split-K, image dest
#define KCV2 512,256,32,64,32,32,4,0
using G1 = Cg<KCV1>;
using G2 = Cg<KCV2>;

static bool g_init = false;
static void init_once() {
  if (g_init) return;
  cudaFuncSetAttribute((void*)gemm_silu<KCV1>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, G1::SM);
  cudaFuncSetAttribute((void*)gemm_silu<KCV2>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, G2::SM);
  g_init = true;
}

at::Tensor sppf_forward(at::Tensor x, at::Tensor W1, at::Tensor sb1,
                        at::Tensor W2, at::Tensor sb2, at::Tensor tc,
                        int64_t npixp) {
  init_once();
  const int n = (int)x.size(0);
  const int npix = n * PP;
  const int dstr = (int)npixp;
  auto out = at::empty({n, 256, HH, WW}, x.options());
  auto s = at::cuda::getCurrentCUDAStream();
  const __half* X = (const __half*)x.data_ptr();
  __half* TCp = (__half*)tc.data_ptr();
  __half* O = (__half*)out.data_ptr();
  const __half* w1 = (const __half*)W1.data_ptr();
  const __half* w2 = (const __half*)W2.data_ptr();
  const float* p1 = sb1.data_ptr<float>();
  const float* p2 = sb2.data_ptr<float>();

  gemm_silu<KCV1><<<dim3(n * (PP / 80) * (128 / 32)), G1::NTH, G1::SM, s>>>(
      X, w1, p1, TCp, PP / 80, PP, dstr, npix, 0);
  lpdl(k2_pool, dim3(n * (128 / GP)), dim3(NTHP), 0, s, TCp, dstr, 1);
  lpdl(gemm_silu<KCV2>, dim3((dstr / 64) * (256 / 32)), dim3(G2::NTH), G2::SM, s,
       (const __half*)TCp, w2, p2, O, 0, dstr, 0, npix, 1);
  return out;
}
'''

_MOD = None
_BUILD_FAILED = False


def _ext():
    """Build (once) and return the fused CUDA extension, or None if unavailable."""
    global _MOD, _BUILD_FAILED
    if _MOD is None and not _BUILD_FAILED:
        try:
            from torch.utils.cpp_extension import load_inline
            major, _ = torch.cuda.get_device_capability()
            arch = f"-arch=sm_{major}0a" if major >= 9 else f"-arch=sm_{major}0"
            _MOD = load_inline(
                "fk_yolo_sppf_fused",
                cpp_sources=[_CPP],
                cuda_sources=[_CU],
                functions=["sppf_forward"],
                verbose=False,
                extra_cuda_cflags=[
                    "-O3", arch,
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                ],
            )
        except Exception:
            _BUILD_FAILED = True
            _MOD = None
    return _MOD


def _fold_bn(conv: nn.Module, bn: nn.Module | None, nout: int):
    """Return (weight[nout, cin] fp16, [scale, bias] fp32) with BN folded in."""
    w = conv.weight.reshape(nout, -1).contiguous()
    if bn is None:
        scale = torch.ones(nout, device=w.device, dtype=torch.float32)
        bias = (conv.bias.float() if conv.bias is not None
                else torch.zeros(nout, device=w.device, dtype=torch.float32))
    else:
        scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        bias = bn.bias.float() - scale * bn.running_mean.float()
        if conv.bias is not None:
            bias = bias + scale * conv.bias.float()
    return w, torch.cat([scale, bias]).contiguous()


class YOLOSPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = YOLOConv(c1, c_, 1, 1)
        self.cv2 = YOLOConv(c_ * 4, c2, 1, 1)
        self.m = MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self._c1, self._c2, self._k = c1, c2, k
        self._packed = None
        self._packed_tag = None
        self._scratch = {}

    # -- reference path (kept for shapes/dtypes the fused kernels do not cover)
    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

    def _weights(self):
        conv1, conv2 = self.cv1.conv, self.cv2.conv
        tag = (conv1.weight.data_ptr(), conv2.weight.data_ptr(),
               conv1.weight.dtype, conv1.weight._version, conv2.weight._version)
        if self._packed is None or self._packed_tag != tag:
            w1, sb1 = _fold_bn(conv1, getattr(self.cv1, "bn", None), self._c1 // 2)
            w2, sb2 = _fold_bn(conv2, getattr(self.cv2, "bn", None), self._c2)
            self._packed = (w1, sb1, w2, sb2)
            self._packed_tag = tag
        return self._packed

    def _tc(self, npixp: int, device, dtype):
        buf = self._scratch.get(npixp)
        if buf is None or buf.device != device:
            buf = torch.zeros(512 * npixp, device=device, dtype=dtype)
            self._scratch[npixp] = buf
        return buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dtype is torch.float16 and x.is_cuda and x.dim() == 4
                and x.is_contiguous() and x.shape[1] == 256 and x.shape[2] == 20
                and x.shape[3] == 20 and self._c1 == 256 and self._c2 == 256
                and self._k == 5 and not self.training):
            ext = _ext()
            if ext is not None:
                try:
                    w1, sb1, w2, sb2 = self._weights()
                    npixp = ((x.shape[0] * 400 + 63) // 64) * 64
                    tc = self._tc(npixp, x.device, x.dtype)
                    return ext.sppf_forward(x, w1, sb1, w2, sb2, tc, npixp)
                except Exception:
                    pass
        return self._reference(x)
