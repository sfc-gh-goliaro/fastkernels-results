"""YOLOv10 SCDown (spatial channel downsampling) block.

The baseline runs two YOLOConv blocks -- a 1x1 conv + BN + SiLU, then a 3x3
stride-2 depthwise conv + BN -- as five eager ops. At the captured sizes that is
a few hundred microseconds of Python and launch overhead wrapped around about a
microsecond of arithmetic, so the win is to do the whole block in one kernel: the
fp16 intermediate never reaches HBM and there is a single launch to pay for.
Both BatchNorms are folded into their conv weights once, at setup.

Anything that is not a captured (shape, dtype, layout) combination falls back to
the baseline eager path.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from .yolov10_conv import YOLOConv

_CPP = r"""
#include <torch/extension.h>
at::Tensor scdown_fused(at::Tensor x, at::Tensor w1, at::Tensor b1,
                        at::Tensor w2, at::Tensor b2);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Fused YOLOv10 SCDown:  y = BN2(dwconv3x3_s2( SiLU(BN1(pwconv1x1(x))) ))
//
// Both BatchNorms are folded into their conv weights on the host, so the kernel
// computes
//   t[c,i,j] = silu( sum_k w1[c,k] * x[k,i,j] + b1[c] )       (1x1 over C1)
//   y[c,p,q] = b2[c] + sum_{u,v} w2[c,u,v] * t[c,2p+u-1,2q+v-1]
// treating t as zero outside the image: cv2 pads its *input*, so an out-of-image
// tap contributes 0, not silu(b1).
//
// A block owns CTB output channels of an OHT-output-row band spanning the full
// image width of one image. The band needs PR = 2*OHT+1 input rows; a full-width
// row band of an NCHW tensor is contiguous, so staging it is a flat 16B copy.
// The staged band is then reused by CTB/CTG channel chunks, which is what keeps
// x from being re-read once per channel tile.
//
//   stage    Xs[C1][NP] (GEMM B, pixel-major) + As[CTB][C1] + folded biases,
//            all via cp.async so the whole band is in flight at once
//   per chunk
//     gemm   m16n16k16 wmma over all of C1. The accumulator is consumed in
//            registers (bias + SiLU + fp16 convert) and written to Ts with one
//            4-byte store per two pixels: the fragment holds column-adjacent
//            pairs, which are adjacent pixels of one image row because W is even.
//     dw     3x3 stride-2 depthwise out of Ts, two output columns per thread so
//            the shared tap column is loaded once and the pair leaves as one
//            4-byte store; w2/b2 are hoisted into registers per channel.
//
// Ts rows carry two zero halves in front of the data (t[c,r,j] lives at index
// 2+j) so the q=0 tap at column -1 reads zero without a branch and every 2-pixel
// access stays 4-byte aligned. The per-channel Ts stride is padded so that the
// 8 accumulator row groups of a warp tile all 32 banks.
// ---------------------------------------------------------------------------

template <int C1_, int C2_, int H_, int W_, int CTB_, int CTG_, int OHT_, int NW_>
struct Cfg {
  static constexpr int C1 = C1_, C2 = C2_, H = H_, W = W_;
  static constexpr int CTB = CTB_, CTG = CTG_, OHT = OHT_, NWARPS = NW_;
  static constexpr int NCG = CTB_ / CTG_;
  static constexpr int NTHREADS = NW_ * 32;
  static constexpr int OH = (H_ + 1) / 2, OW = (W_ + 1) / 2;
  static constexpr int OWP = OW / 2;
  static constexpr int PR = 2 * OHT_ + 1;
  static constexpr int NPIX = PR * W_;
  static constexpr int NP = (NPIX + 15) / 16 * 16;
  static constexpr int LDB = NP + 8;
  static constexpr int LDA = C1_ + 8;
  static constexpr int TSW = W_ + 4;
  static constexpr int TSC = PR * TSW + ((8 - PR * TSW) % 64 + 64) % 64;
  static constexpr int MT = CTG_ / 16, NT = NP / 16, KT = C1_ / 16;
  static constexpr int VPR = W_ / 8;
  static constexpr int VPB = PR * VPR;
  static constexpr int NV = C1_ * VPB;
  static constexpr int PERT = (NV + NW_ * 32 - 1) / (NW_ * 32);
  static constexpr int NCB = C2_ / CTB_;
  static constexpr int NRB = (OH + OHT_ - 1) / OHT_;
  static constexpr int TPC = NTHREADS >= CTG_ ? NTHREADS / CTG_ : 1;
  static constexpr int SM_HALF = C1_ * LDB + CTB_ * LDA + CTG_ * TSC;
  static constexpr int SM_FLOAT = CTB_ * 12 + CTB_;  // w2|b2 rows of 12, then b1
  static constexpr int SMEM = SM_HALF * 2 + SM_FLOAT * 4;
  static constexpr bool OK = (C2_ % CTB_ == 0) && (CTB_ % CTG_ == 0) &&
                             (NW_ % (CTG_ / 16) == 0) &&
                             (CTG_ % 16 == 0) && (W_ % 8 == 0) && (OW % 2 == 0) &&
                             (SMEM <= 227 * 1024);
};

__device__ __forceinline__ float silu(float v) {
  return __fdividef(v, 1.0f + __expf(-v));
}

template <typename C>
__global__ __launch_bounds__(C::NTHREADS) void scdown_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ w1,
    const float* __restrict__ b1,
    const float* __restrict__ w2,
    const float* __restrict__ b2,
    __half* __restrict__ out) {
  constexpr int C1 = C::C1, C2 = C::C2, H = C::H, W = C::W;
  constexpr int CTB = C::CTB, CTG = C::CTG, NCG = C::NCG;
  constexpr int OHT = C::OHT, OH = C::OH, OW = C::OW, OWP = C::OWP;
  constexpr int PR = C::PR, NPIX = C::NPIX, NP = C::NP;
  constexpr int LDB = C::LDB, LDA = C::LDA, TSW = C::TSW, TSC = C::TSC;
  constexpr int MT = C::MT, NT = C::NT, KT = C::KT;
  constexpr int VPR = C::VPR, VPB = C::VPB, TPC = C::TPC;
  constexpr int NV = C::NV, PERT = C::PERT;
  constexpr int NTHREADS = C::NTHREADS, NWARPS = C::NWARPS;
  constexpr int HW = H * W;

  extern __shared__ __align__(16) char smem_raw[];
  __half* Xs = reinterpret_cast<__half*>(smem_raw);
  __half* As = Xs + C1 * LDB;
  __half* Ts = As + CTB * LDA;
  // w2s rows are padded to 12 floats with b2 in slot 9, so the depthwise picks
  // up all ten per-channel constants in three float4 loads instead of ten.
  float* w2s = reinterpret_cast<float*>(Ts + CTG * TSC);
  float* b1s = w2s + CTB * 12;

  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  const int cb = bid % C::NCB;      // channel block fastest: neighbours share x
  bid /= C::NCB;
  const int rb = bid % C::NRB;
  const int n = bid / C::NRB;

  const int oh0 = rb * OHT;
  const int ir0 = 2 * oh0 - 1;
  const int c00 = cb * CTB;
  const __half* xb = x + (size_t)n * C1 * HW;

  // ---- stage ------------------------------------------------------------
  if (ir0 >= 0 && ir0 + PR <= H) {          // interior band: flat vector copy
    const __half* src = xb + ir0 * W;
#pragma unroll
    for (int u = 0; u < PERT; ++u) {
      const int i = tid + u * NTHREADS;
      if (PERT * NTHREADS == NV || i < NV) {
        const int c = i / VPB, r = i - c * VPB;
        __pipeline_memcpy_async(Xs + c * LDB + r * 8,
                                src + (size_t)c * HW + r * 8, 16);
      }
    }
  } else {                                  // top/bottom band: row-clamped
#pragma unroll
    for (int u = 0; u < PERT; ++u) {
      const int i = tid + u * NTHREADS;
      if (PERT * NTHREADS == NV || i < NV) {
        const int c = i / VPB, r = i - c * VPB;
        const int lr = r / VPR, v = r - lr * VPR;
        const int ir = ir0 + lr;
        const bool in = (unsigned)ir < (unsigned)H;
        __pipeline_memcpy_async(Xs + c * LDB + r * 8,
                                xb + (size_t)c * HW + (in ? ir : 0) * W + v * 8,
                                16, in ? 0 : 16);
      }
    }
  }
  for (int i = tid; i < CTB * (C1 / 8); i += NTHREADS) {
    const int m = i / (C1 / 8), v = i - m * (C1 / 8);
    __pipeline_memcpy_async(As + m * LDA + v * 8,
                            w1 + (size_t)(c00 + m) * C1 + v * 8, 16);
  }
  for (int i = tid; i < CTG * PR; i += NTHREADS) {       // leading zero taps
    const int c = i / PR, r = i - c * PR;
    *reinterpret_cast<__half2*>(Ts + c * TSC + r * TSW) = __half2half2(__float2half(0.f));
  }
  for (int i = tid; i < CTB * 9; i += NTHREADS) {
    const int m = i / 9, e = i - m * 9;
    w2s[m * 12 + e] = w2[(size_t)c00 * 9 + i];
  }
  for (int i = tid; i < CTB; i += NTHREADS) {
    b1s[i] = b1[c00 + i];
    w2s[i * 12 + 9] = b2[c00 + i];
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  const int r0 = lane >> 2;            // accumulator row (channel) offset
  const int lq = (lane & 3) << 1;      // accumulator column (pixel) offset

  for (int g = 0; g < NCG; ++g) {
    if (g) __syncthreads();            // previous chunk's depthwise done with Ts
    const int cg = g * CTG;

    // ---- 1x1 conv, epilogue straight out of the accumulator --------------
    // NWARPS % MT == 0, so a warp's m-tile is fixed: its A fragments and its two
    // bias values are loaded once instead of once per output tile.
    {
      const int mt = warp % MT;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[KT];
#pragma unroll
      for (int k = 0; k < KT; ++k)
        wmma::load_matrix_sync(fa[k], As + (cg + mt * 16) * LDA + k * 16, LDA);
      const int ra = mt * 16 + r0, rbb = ra + 8;
      const float ba = b1s[cg + ra], bb = b1s[cg + rbb];
      __half* ta = Ts + ra * TSC;
      __half* tb = Ts + rbb * TSC;
      for (int nt = warp / MT; nt < NT; nt += NWARPS / MT) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < KT; ++k) {
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb;
          wmma::load_matrix_sync(fb, Xs + (k * 16) * LDB + nt * 16, LDB);
          wmma::mma_sync(acc, fa[k], fb, acc);
        }
        // acc.x[4*nb+j] is at row r0 + 8*(j>=2), col 8*nb + lq + (j&1).
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
          const int pix = nt * 16 + nb * 8 + lq;
          if (NP > NPIX && pix >= NPIX) continue;
          const int lr = pix / W, j = pix - lr * W;
          const bool valid = (unsigned)(ir0 + lr) < (unsigned)H;
          const float* a = &acc.x[4 * nb];
          __half2 va = valid ? __floats2half2_rn(silu(a[0] + ba), silu(a[1] + ba))
                             : __half2half2(__float2half(0.f));
          __half2 vb = valid ? __floats2half2_rn(silu(a[2] + bb), silu(a[3] + bb))
                             : __half2half2(__float2half(0.f));
          *reinterpret_cast<__half2*>(ta + lr * TSW + 2 + j) = va;
          *reinterpret_cast<__half2*>(tb + lr * TSW + 2 + j) = vb;
        }
      }
    }
    __syncthreads();

    // ---- 3x3 stride-2 depthwise, two output columns per thread -----------
    __half* ob = out + ((size_t)n * C2 + c00 + cg) * OH * OW;
    for (int lc = tid / TPC; lc < CTG; lc += NTHREADS / TPC) {
      const float4 wa = *reinterpret_cast<const float4*>(w2s + (cg + lc) * 12);
      const float4 wb = *reinterpret_cast<const float4*>(w2s + (cg + lc) * 12 + 4);
      const float4 wc = *reinterpret_cast<const float4*>(w2s + (cg + lc) * 12 + 8);
      const float wv[9] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w, wc.x};
      const float bv = wc.y;
      const __half* tc = Ts + lc * TSC;
      __half* oc = ob + (size_t)lc * OH * OW;
      for (int o = tid % TPC; o < OHT * OWP; o += TPC) {
        const int m = o % OWP, lo = o / OWP;
        const int p = oh0 + lo;
        if (p >= OH) continue;
        float s0 = bv, s1 = bv;
        const __half* tp = tc + (2 * lo) * TSW + 4 * m + 1;
#pragma unroll
        for (int u = 0; u < 3; ++u) {
          const __half* r = tp + u * TSW;
          const float v0 = __half2float(r[0]);
          const float2 v12 = __half22float2(*reinterpret_cast<const __half2*>(r + 1));
          const float2 v34 = __half22float2(*reinterpret_cast<const __half2*>(r + 3));
          s0 = fmaf(v0, wv[3 * u], s0);
          s0 = fmaf(v12.x, wv[3 * u + 1], s0);
          s0 = fmaf(v12.y, wv[3 * u + 2], s0);
          s1 = fmaf(v12.y, wv[3 * u], s1);
          s1 = fmaf(v34.x, wv[3 * u + 1], s1);
          s1 = fmaf(v34.y, wv[3 * u + 2], s1);
        }
        *reinterpret_cast<__half2*>(oc + p * OW + 2 * m) = __floats2half2_rn(s0, s1);
      }
    }
  }
}

// ---------------------------------------------------------------------------
template <typename C>
static void launch(const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1,
                   const at::Tensor& w2, const at::Tensor& b2, at::Tensor& out, int N) {
  static bool inited = false;
  if (!inited) {
    if (C::SMEM > 48 * 1024)
      cudaFuncSetAttribute(scdown_kernel<C>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    inited = true;
  }
  scdown_kernel<C><<<dim3(N * C::NCB * C::NRB), C::NTHREADS, C::SMEM,
                     at::cuda::getCurrentCUDAStream()>>>(
      (const __half*)x.data_ptr(), (const __half*)w1.data_ptr(), b1.data_ptr<float>(),
      w2.data_ptr<float>(), b2.data_ptr<float>(), (__half*)out.data_ptr());
}

// Tile shapes chosen by a sweep over the captured shapes (dev/cmp.py). The
// batched cases have enough row bands to afford the wider 512-thread tile; the
// single-image cases need the narrow one to keep the grid big enough to fill the
// machine.
//     id   CTB  CTG  OHT  NWARPS
#define CFG_LIST(S)          \
  Y(0, S, 32, 32, 1, 16)     \
  Y(1, S, 16, 16, 1,  8)

static int64_t pick_cfg(int N) { return N > 1 ? 0 : 1; }

at::Tensor scdown_fused(at::Tensor x, at::Tensor w1, at::Tensor b1, at::Tensor w2,
                        at::Tensor b2) {
  const int N = x.size(0), C1 = x.size(1), H = x.size(2), W = x.size(3);
  const int C2 = w1.size(0);
  auto out = at::empty({N, C2, (H + 1) / 2, (W + 1) / 2}, x.options());
  const int64_t cfg = pick_cfg(N);

  bool ok = false;
#define Y(id, S, ctb, ctg, oht, nw)                                          \
  if (cfg == (id)) {                                                         \
    using T = Cfg<S::A, S::B, S::C, S::D, ctb, ctg, oht, nw>;                \
    if (T::OK) { launch<T>(x, w1, b1, w2, b2, out, N); ok = true; }          \
  }
  struct SA { enum { A = 128, B = 256, C = 40, D = 40 }; };
  struct SB { enum { A = 64, B = 128, C = 80, D = 80 }; };
  if (C1 == 128 && C2 == 256 && H == 40 && W == 40) { CFG_LIST(SA) }
  else if (C1 == 64 && C2 == 128 && H == 80 && W == 80) { CFG_LIST(SB) }
#undef Y
  TORCH_CHECK(ok, "scdown_fused: unsupported shape (C1=", C1, " C2=", C2,
              " H=", H, " W=", W, ")");
  return out;
}
"""

_EXT = None


def _ext():
    """Build (once) and return the fused-kernel extension, or None if unavailable."""
    global _EXT
    if _EXT is None:
        try:
            from torch.utils.cpp_extension import load_inline

            build_dir = Path(
                os.environ.get(
                    "FK_SCDOWN_BUILD_DIR",
                    str(Path(__file__).resolve().parents[2] / ".fk_build" / "scdown"),
                )
            )
            build_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")
            _EXT = load_inline(
                name="fk_yolov10_scdown",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                functions=["scdown_fused"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                build_directory=str(build_dir),
                verbose=False,
            )
        except Exception:  # pragma: no cover - no nvcc / unsupported arch
            if os.environ.get("FK_SCDOWN_DEBUG"):
                raise
            _EXT = False
    return _EXT


# (C1, C2, H, W) combinations the kernel is instantiated for.
_SHAPES = {(128, 256, 40, 40), (64, 128, 80, 80)}
_CHANNELS = {(c1, c2) for c1, c2, _, _ in _SHAPES}


class YOLOSCDown(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self._fast = None  # None = not probed yet, False = fall back, else params

    @staticmethod
    @torch.no_grad()
    def _fold(conv, bn, out_ch: int):
        """Weight and bias of *conv* with its BatchNorm folded in, as float32."""
        w = conv.weight.detach().float()
        b = (
            conv.bias.detach().float()
            if conv.bias is not None
            else torch.zeros(out_ch, device=w.device, dtype=torch.float32)
        )
        if bn is not None:
            s = bn.weight.detach().float() / torch.sqrt(
                bn.running_var.detach().float() + bn.eps
            )
            w = w * s.reshape(-1, *([1] * (w.dim() - 1)))
            b = (b - bn.running_mean.detach().float()) * s + bn.bias.detach().float()
        return w, b

    @torch.no_grad()
    def _prepare(self):
        """Probe the fused path and, if it applies, fold the weights for it."""
        cv1, cv2 = self.cv1, self.cv2
        w = cv1.conv.weight
        c2, c1 = w.shape[0], w.shape[1]
        if (c1, c2) not in _CHANNELS:   # don't pay a build for a shape we can't run
            self._fast = False
            return False
        ext = _ext()
        ok = (
            ext
            and w.is_cuda
            and w.dtype == torch.float16
            and tuple(w.shape[2:]) == (1, 1)
            and tuple(cv1.conv.stride) == (1, 1)
            and tuple(cv1.conv.padding) == (0, 0)
            and cv1.conv.groups == 1
            and isinstance(cv1.act, type(YOLOConv.default_act))
            and tuple(cv2.conv.weight.shape) == (c2, 1, 3, 3)
            and tuple(cv2.conv.stride) == (2, 2)
            and tuple(cv2.conv.padding) == (1, 1)
            and tuple(cv2.conv.dilation) == (1, 1)
            and cv2.conv.groups == c2
            and isinstance(cv2.act, nn.Identity)
        )
        if not ok:
            self._fast = False
            return False
        w1, b1 = self._fold(cv1.conv, getattr(cv1, "bn", None), c2)
        w2, b2 = self._fold(cv2.conv, getattr(cv2, "bn", None), c2)
        self._fast = (
            w1.reshape(c2, c1).to(torch.float16).contiguous(),
            b1.contiguous(),
            w2.reshape(c2, 9).contiguous(),
            b2.contiguous(),
            ext.scdown_fused,
        )
        self._shape_ok = {(c1, c2, h, w_) for (a, b, h, w_) in _SHAPES if (a, b) == (c1, c2)}
        return self._fast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self._fast
        if p is None:
            p = self._prepare()
        if (
            p is not False
            and x.dtype == torch.float16
            and x.is_cuda
            and x.dim() == 4
            and x.is_contiguous()
            and (x.size(1), p[0].size(0), x.size(2), x.size(3)) in self._shape_ok
        ):
            return p[4](x, p[0], p[1], p[2], p[3])
        return self.cv2(self.cv1(x))
