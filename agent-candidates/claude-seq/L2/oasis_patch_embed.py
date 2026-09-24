"""Oasis 2D patch embedding -- fused implicit-GEMM CUDA kernel (Blackwell sm_100).

The baseline runs ``F.conv2d`` with kernel == stride and then *permutes* the
NCHW result into [B, PH, PW, C] (or flattens it to [B, PH*PW, C]).  Both halves
are replaced by one launch: a hand-written NCHW implicit GEMM that gathers each
patch on the fly and writes the embedding straight out in the permuted layout.
For the captured shapes that is worth far more than the arithmetic -- they are
tiny (M <= 864 patches with K = 64 for the 2x2 latent stem), cuDNN spends most
of its time in per-call setup, and on this machine every extra kernel in the
stream costs ~3.5us.

See the CUDA source below for the GEMM decomposition, the shared-memory
pipeline, and the TF32 precision policy that keeps this numerically
interchangeable with the cuDNN reference.
"""

from __future__ import annotations

import hashlib
import os

import torch
import torch.nn as nn

from ..L1.conv2d import Conv2d

_CUDA_SRC = r"""
// Fused Oasis patch embedding: conv2d(stride == kernel) + the NHWC relayout in
// a single launch.  The convolution is an implicit GEMM
//
//     Out[m, n] = bias[n] + sum_k A[m, k] * Wk[k, n]
//     m = (b*PH + ph)*PW + pw      (patch id, PH*PW patches per image)
//     k = (c*R + r)*S + s          (weight [Cout, Cin, R, S] == [N, K], and Wk
//                                   is that transposed to k-major, see below)
//     A[m, k] = x[b, c, ph*R + r, pw*S + s]
//
// so the result lands straight in the [B, PH, PW, Cout] / [B, PH*PW, Cout]
// layout the baseline builds by permuting a conv output -- no transpose kernel
// and no im2col workspace.  The gather is written as x[abase[m] + koff[k]]:
// abase depends only on the patch and koff only on the weight index, so both
// hoist out of the inner loops (koff for the whole K is built once into shared
// memory at kernel entry, keeping the integer divides off the k loop).
//
// One launch matters as much as the arithmetic: the captured shapes are tiny
// (M <= 864 with K = 64 for the latent stem) and on this machine each extra
// kernel in the stream costs ~3.5us, more than the kernel itself.
//
// The weight is pre-staged once, host side, into k-major [K, N] order (and,
// in TF32 mode, pre-rounded) -- it is a constant across calls.  That makes the
// B tile a contiguous float4 copy instead of a strided transpose-on-load, and
// leaves the activation gather as the only scattered access.
//
// Precision.  torch.backends.cudnn.allow_tf32 defaults to True, so the
// reference F.conv2d computes this fp32 convolution in TF32 whenever cuDNN
// picks a tensor-core kernel -- ~2.6e-4 absolute away from an exact fp32 dot
// product, far outside the fp32 comparison bound in either direction.  To stay
// numerically interchangeable with the reference, the operands are rounded to
// TF32 (round-to-nearest-even, as cuDNN's operand convert does) while the
// products and accumulation stay fp32, which reproduces a TF32 tensor-core
// GEMM to ~2.5e-7.  Where the reference is exact fp32 instead, the same mma is
// used three times on a hi/lo operand split (see FK_FRAG).  The caller decides
// per call which mode applies.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define FK_DIVUP(a, b) (((a) + (b) - 1) / (b))

// Round to TF32 (10 explicit mantissa bits), ties to even.
__device__ __forceinline__ float to_tf32(float v) {
  unsigned u = __float_as_uint(v);
  unsigned low = u & 0x1FFFu;
  unsigned hi = u >> 13;
  hi += (low > 0x1000u) | ((low == 0x1000u) & (hi & 1u));
  return __uint_as_float(hi << 13);
}

template <bool TF32>
__device__ __forceinline__ float cvt(float v) {
  return TF32 ? to_tf32(v) : v;
}

__device__ __forceinline__ void mma_tf32(float* d, const unsigned* a,
                                         const unsigned* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// BM x BN block tile, BK-deep k-tile, split over WPM x WPN warps; each warp
// covers its WM x WN patch with NM x NN m16n8k8 tensor-core tiles.
//
// Shared memory holds both tiles k-major (As[BK][BM], Bs[BK][BN]) and double
// buffered, so one barrier per k-tile is enough: the next tile is prefetched
// into registers, consumed into the idle buffer, and the barrier only orders
// that buffer's fill against its use.  Fragments are in turn register double
// buffered, so slice kb+8's shared loads issue while the tensor core is still
// working on slice kb -- with so few warps per scheduler there is no other warp
// to cover that latency.
template <int BM, int BN, int BK, int WPM, int WPN, bool TF32>
__global__ __launch_bounds__(WPM * WPN * 32)
void pe_kernel(const float* __restrict__ X, const float* __restrict__ Wk,
               const float* __restrict__ Bias, float* __restrict__ Out,
               int M, int N, int K, int PHPW, int PW, int R, int S,
               int Wd, int HW, int CHW, int RS) {
  constexpr int NT = WPM * WPN * 32;
  constexpr int WM = BM / WPM, WN = BN / WPN;    // warp tile
  constexpr int NM = WM / 16, NN = WN / 8;       // mma tiles per warp
  constexpr int PASSES = TF32 ? 1 : 3;           // tf32 / 3-term split fp32
  static_assert(BM % WPM == 0 && BN % WPN == 0, "bad warp split");
  static_assert(WM % 16 == 0 && WN % 8 == 0, "warp tile is m16n8 tiled");
  static_assert(BK % 8 == 0, "k-tile is a multiple of the mma k=8");
  // Row padding of +8, not +4: one mma fragment read spans 4 k rows x 8 lanes,
  // and a stride of BM+8 (== 8 or 24 mod 32 for every tile width used here)
  // spreads those 32 addresses over all 32 banks, where BM+4 collides 2-way.
  // Still a multiple of 4, so the vector weight staging stays 128-bit aligned.
  constexpr int SA = BM + 8;
  constexpr int SB = BN + 8;
  constexpr int MSTEP = NT / BK;               // A rows staged per step
  constexpr int NSA = BM > MSTEP ? BM / MSTEP : 1;
  constexpr int NF = BN / 4;                   // float4s per staged B row
  constexpr int BKSTEP = NT / NF;              // B k-rows staged per step
  constexpr int NSB = BK > BKSTEP ? BK / BKSTEP : 1;
  static_assert(NT % BK == 0 && NT % NF == 0, "bad staging shape");
  static_assert(BM % MSTEP == 0 || MSTEP > BM, "bad A staging shape");
  static_assert(BK % BKSTEP == 0 || BKSTEP > BK, "bad B staging shape");

  extern __shared__ float smem[];
  float* As = smem;                          // 2 x BK x SA
  float* Bs = As + 2 * BK * SA;              // 2 x BK x SB
  int* koff = (int*)(Bs + 2 * BK * SB);      // K

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  for (int k = tid; k < K; k += NT) {
    int c = k / RS, rem = k - c * RS, r = rem / S;
    koff[k] = c * HW + r * Wd + (rem - r * S);
  }

  // --- activation gather addresses (one patch row per staging step) --------
  const int ldk = tid % BK;
  const int ldm = tid / BK;
  const int alim = min(BM, M - m0) - ldm;   // step j is live iff j*MSTEP < alim
  int abase[NSA];
#pragma unroll
  for (int j = 0; j < NSA; ++j) {
    int m = m0 + ldm + j * MSTEP;
    int b = m / PHPW, rem = m - b * PHPW, ph = rem / PW;
    abase[j] = b * CHW + ph * R * Wd + (rem - ph * PW) * S;
  }
  // --- weight tile addresses (k-major, so a plain vector copy) -------------
  const int bk_ = tid / NF;            // k row inside the tile
  const int bn_ = (tid % NF) * 4;      // first column inside the tile
  const float* wk = Wk + (long)bk_ * N + n0 + bn_;
  const bool nok = bk_ < BK && n0 + bn_ + 3 < N;

  // --- mma fragment addressing --------------------------------------------
  // m16n8k8 fragments: lane (g = laneid>>2, t = laneid&3) holds A rows g, g+8
  // at k = t, t+4; B rows k = t, t+4 at column g; and accumulates C rows g,
  // g+8 at columns 2t, 2t+1.
  const int warp = tid / 32, lane = tid % 32;
  const int g = lane >> 2, t = lane & 3;
  const int wrow = (warp / WPN) * WM;
  const int wcol = (warp % WPN) * WN;
  float acc[NM][NN][4];
#pragma unroll
  for (int im = 0; im < NM; ++im)
#pragma unroll
    for (int in = 0; in < NN; ++in)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[im][in][e] = 0.f;

  float ar[NSA];
  float4 br[NSB];
  // mma fragments, double buffered: slice kb+8's shared loads (and its lo
  // split) issue while the tensor core is still chewing on slice kb.
  unsigned ah[2][NM][4], bh[2][NN][2];
  unsigned al[2][TF32 ? 1 : NM][4], bl[2][TF32 ? 1 : NN][2];

#define FK_STAGE(K0)                                                          \
  {                                                                           \
    const int kk = (K0) + ldk;                                                \
    const int koffv = kk < K ? koff[kk] : 0;                                  \
    _Pragma("unroll") for (int j = 0; j < NSA; ++j)                           \
        ar[j] = (kk < K && j * MSTEP < alim)                                  \
                    ? cvt<TF32>(X[abase[j] + koffv]) : 0.f;                   \
    _Pragma("unroll") for (int j = 0; j < NSB; ++j)                           \
        br[j] = (nok && bk_ + j * BKSTEP < BK && (K0) + bk_ + j * BKSTEP < K) \
                    ? *(const float4*)(wk + (long)((K0) + j * BKSTEP) * N)    \
                    : make_float4(0.f, 0.f, 0.f, 0.f);                        \
  }

#define FK_COMMIT(BUF)                                                        \
  {                                                                           \
    float* as = As + (BUF) * BK * SA;                                         \
    float* bs = Bs + (BUF) * BK * SB;                                         \
    _Pragma("unroll") for (int j = 0; j < NSA; ++j)                           \
        if (ldm + j * MSTEP < BM) as[ldk * SA + ldm + j * MSTEP] = ar[j];      \
    _Pragma("unroll") for (int j = 0; j < NSB; ++j)                           \
        if (bk_ + j * BKSTEP < BK)                                            \
          *(float4*)&bs[(bk_ + j * BKSTEP) * SB + bn_] = br[j];               \
  }

  // One mma k-slice (8 deep).  The tensor core reads only the top 19 bits of
  // each operand register, so a fragment never needs an explicit convert: in
  // TF32 mode the staged operands are already TF32-rounded, and in fp32 mode
  // the hi term is the raw register (truncated for free by the hardware) while
  // the lo term costs one mask and one subtract.  Summing
  // hi*hi + hi*lo + lo*hi leaves only the dropped lo*lo term and lo's own
  // truncation, each ~2^-20 of the product, so the dot product tracks fp32 to
  // ~1e-6 relative -- two orders inside the fp32 comparison bound.
#define FK_FRAG(AS, BS, KB, R)                                                \
  {                                                                           \
    _Pragma("unroll") for (int im = 0; im < NM; ++im) {                        \
      const float* p = (AS) + wrow + im * 16 + g;                             \
      float v[4] = {p[((KB) + t) * SA], p[((KB) + t) * SA + 8],                \
                    p[((KB) + t + 4) * SA], p[((KB) + t + 4) * SA + 8]};       \
      _Pragma("unroll") for (int e = 0; e < 4; ++e) {                          \
        ah[R][im][e] = __float_as_uint(v[e]);                                  \
        if (!TF32)                                                            \
          al[R][im][e] = __float_as_uint(                                      \
              v[e] - __uint_as_float(ah[R][im][e] & 0xFFFFE000u));             \
      }                                                                       \
    }                                                                         \
    _Pragma("unroll") for (int in = 0; in < NN; ++in) {                        \
      const float* q = (BS) + wcol + in * 8 + g;                              \
      float v[2] = {q[((KB) + t) * SB], q[((KB) + t + 4) * SB]};               \
      _Pragma("unroll") for (int e = 0; e < 2; ++e) {                          \
        bh[R][in][e] = __float_as_uint(v[e]);                                  \
        if (!TF32)                                                            \
          bl[R][in][e] = __float_as_uint(                                      \
              v[e] - __uint_as_float(bh[R][in][e] & 0xFFFFE000u));             \
      }                                                                       \
    }                                                                         \
  }

/* One pass over all tiles per product term, not three back-to-back on the same
   accumulator: a tile's three terms are serially dependent, so interleaving
   tiles is what keeps the tensor pipe fed. */
#define FK_MMAS(R)                                                            \
  {                                                                           \
    _Pragma("unroll") for (int im = 0; im < NM; ++im)                          \
    _Pragma("unroll") for (int in = 0; in < NN; ++in)                          \
      mma_tf32(acc[im][in], ah[R][im], bh[R][in]);                            \
    if (PASSES == 3) {                                                        \
      _Pragma("unroll") for (int im = 0; im < NM; ++im)                        \
      _Pragma("unroll") for (int in = 0; in < NN; ++in)                        \
        mma_tf32(acc[im][in], ah[R][im], bl[R][in]);                          \
      _Pragma("unroll") for (int im = 0; im < NM; ++im)                        \
      _Pragma("unroll") for (int in = 0; in < NN; ++in)                        \
        mma_tf32(acc[im][in], al[R][im], bh[R][in]);                          \
    }                                                                         \
  }

#define FK_COMPUTE(BUF)                                                       \
  {                                                                           \
    const float* as = As + (BUF) * BK * SA;                                   \
    const float* bs = Bs + (BUF) * BK * SB;                                   \
    FK_FRAG(as, bs, 0, 0)                                                     \
    _Pragma("unroll") for (int kb = 0; kb < BK; kb += 8) {                     \
      if (kb + 8 < BK) FK_FRAG(as, bs, kb + 8, ((kb / 8) + 1) & 1)             \
      FK_MMAS((kb / 8) & 1)                                                    \
    }                                                                         \
  }

// One k-tile: prefetch the next tile's globals, wait for this buffer's fill,
// multiply, then park the prefetch in the other buffer.
#define FK_BODY(BUF, K0)                                                      \
  {                                                                           \
    const bool more = (K0) + BK < K;                                          \
    if (more) FK_STAGE((K0) + BK)                                             \
    __syncthreads();                                                          \
    FK_COMPUTE(BUF)                                                           \
    if (more) FK_COMMIT((BUF) ^ 1)                                            \
  }

  __syncthreads();   // koff ready
  FK_STAGE(0)
  FK_COMMIT(0)

  for (int k0 = 0; k0 < K; k0 += 2 * BK) {
    FK_BODY(0, k0)
    if (k0 + BK < K) FK_BODY(1, k0 + BK)
  }
#undef FK_STAGE
#undef FK_COMMIT
#undef FK_FRAG
#undef FK_MMAS
#undef FK_COMPUTE
#undef FK_BODY

  // Epilogue: accumulator lane (g, t) owns rows g, g+8 and columns 2t, 2t+1 of
  // each m16n8 tile, so each pair of accumulators is one 64-bit store.
#pragma unroll
  for (int in = 0; in < NN; ++in) {
    const int col = n0 + wcol + in * 8 + 2 * t;
    if (col >= N) continue;   // N need not be a multiple of BN
    float bv[2] = {0.f, 0.f};
    if (Bias != nullptr) *(float2*)bv = *(const float2*)(Bias + col);
#pragma unroll
    for (int im = 0; im < NM; ++im) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = m0 + wrow + im * 16 + g + half * 8;
        if (row < M) {
          float o[2] = {acc[im][in][half * 2] + bv[0],
                        acc[im][in][half * 2 + 1] + bv[1]};
          *(float2*)(Out + (long)row * N + col) = *(float2*)o;
        }
      }
    }
  }
}

template <int BM, int BN, int BK, int WPM, int WPN>
static void launch(const float* x, const float* w, const float* bias, float* out,
                   int M, int N, int K, int PHPW, int PW, int R, int S,
                   int Wd, int HW, int CHW, bool tf32, cudaStream_t stream) {
  constexpr int NT = WPM * WPN * 32;
  dim3 grid(FK_DIVUP(N, BN), FK_DIVUP(M, BM));
  size_t shm = (size_t)(2 * BK * (BM + 8) + 2 * BK * (BN + 8)) * sizeof(float)
             + (size_t)K * sizeof(int);
  if (shm > 48 * 1024) {   // opt in to the larger dynamic shared window once
    static bool done[2] = {false, false};
    if (!done[tf32]) {
      cudaFuncSetAttribute(
          tf32 ? (const void*)&pe_kernel<BM, BN, BK, WPM, WPN, true>
               : (const void*)&pe_kernel<BM, BN, BK, WPM, WPN, false>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
      done[tf32] = true;
    }
  }
  if (tf32)
    pe_kernel<BM, BN, BK, WPM, WPN, true><<<grid, NT, shm, stream>>>(
        x, w, bias, out, M, N, K, PHPW, PW, R, S, Wd, HW, CHW, R * S);
  else
    pe_kernel<BM, BN, BK, WPM, WPN, false><<<grid, NT, shm, stream>>>(
        x, w, bias, out, M, N, K, PHPW, PW, R, S, Wd, HW, CHW, R * S);
}

//       id   BM   BN  BK  WPM WPN   (threads = WPM*WPN*32)
//       id   BM   BN  BK  WPM WPN   (threads = WPM*WPN*32)
#define FK_CFGS(F) \
  F(0,  16,  64, 16, 1, 4)        \
  F(1,  64,  64, 32, 4, 4)        

torch::Tensor patch_embed_forward(torch::Tensor x, torch::Tensor wk,
                                  c10::optional<torch::Tensor> bias,
                                  int64_t R, int64_t S, bool flatten,
                                  int64_t cfg, int64_t prec) {
  const int B = x.size(0), Cin = x.size(1), Hd = x.size(2), Wd = x.size(3);
  const int N = wk.size(1);
  const int PH = Hd / (int)R, PW = Wd / (int)S;
  const int K = Cin * (int)R * (int)S;
  const int M = B * PH * PW;
  const float* bp = bias.has_value() ? bias->data_ptr<float>() : nullptr;

  auto out = torch::empty(flatten ? std::vector<int64_t>{B, (long)PH * PW, N}
                                  : std::vector<int64_t>{B, PH, PW, N},
                          x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* xp = x.data_ptr<float>();
  const float* wp = wk.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int HW = Hd * Wd, CHW = Cin * HW, PHPW = PH * PW;
  // prec is decided by the caller: it also picks which staged weight (plain or
  // TF32-rounded) is passed in, so the policy lives in one place.
  TORCH_CHECK(prec == 0 || prec == 1, "prec must be 0 (fp32) or 1 (tf32)");
  const bool tf32 = prec == 1;

  if (cfg < 0) cfg = (K <= 256) ? 0 : 1;
  switch (cfg) {
#define FK_CASE(ID, bm, bn, bk, wm, wn)                                     \
  case ID:                                                                  \
    launch<bm, bn, bk, wm, wn>(xp, wp, bp, op, M, N, K, PHPW, PW, (int)R,     \
                               (int)S, Wd, HW, CHW, tf32, stream);            \
    break;
    FK_CFGS(FK_CASE)
#undef FK_CASE
    default:
      TORCH_CHECK(false, "bad cfg");
  }
  return out;
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor patch_embed_forward(torch::Tensor x, torch::Tensor wk,
                                  c10::optional<torch::Tensor> bias,
                                  int64_t R, int64_t S, bool flatten,
                                  int64_t cfg, int64_t prec);
"""

_EXT = None
_EXT_FAILED = False
_CFG_OVERRIDE = int(os.environ.get("FK_PE_CFG", "-1"))


def _arch_list() -> str:
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    return f"{major}.{minor}{'a' if major >= 9 else ''}"


def _build_ext():
    from torch.utils.cpp_extension import load_inline

    arch = _arch_list()
    if arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    tag = hashlib.sha1(_CUDA_SRC.encode()).hexdigest()[:10]
    return load_inline(
        name=f"fk_oasis_patch_embed_{tag}",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["patch_embed_forward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def _ext():
    """The compiled extension, or None if it cannot be built here.

    A build failure degrades to the baseline conv2d path rather than breaking
    the operator.
    """
    global _EXT, _EXT_FAILED
    if _EXT is None and not _EXT_FAILED:
        try:
            _EXT = _build_ext()
        except Exception:
            _EXT_FAILED = True
    return _EXT


def _to_tf32(t: torch.Tensor) -> torch.Tensor:
    """Round to TF32 (10 explicit mantissa bits), ties to even.

    Matches the kernel's activation rounding and cuDNN's operand convert, so a
    pre-rounded weight and a TF32-rounded activation reproduce the reference
    tensor-core dot product.
    """
    i = t.view(torch.int32)
    low = i & 0x1FFF
    up = torch.where(low == 0x1000, (i >> 13) & 1, (low > 0x1000).to(torch.int32))
    return (((i >> 13) + up) << 13).view(torch.float32)


# cuDNN computes this fp32 convolution in TF32 (torch.backends.cudnn.allow_tf32
# defaults to True) whenever it picks a tensor-core kernel: ~2.6e-4 absolute
# away from an exact fp32 dot product, i.e. outside the fp32 comparison bound in
# either direction.  Mirror that choice so the output stays interchangeable with
# the reference: cuDNN's tensor-core kernels need a 4-element aligned input
# channel count, and it only prefers them once the GEMM is big enough to
# amortize them; below either threshold its result is exact fp32.
def _use_tf32(in_chans: int, patches: int) -> bool:
    return in_chans % 4 == 0 and patches >= 384


class OasisPatchEmbed(nn.Module):
    def __init__(
        self,
        img_height: int = 256,
        img_width: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.img_size = (img_height, img_width)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_height // patch_size, img_width // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None
        self._wcache: dict[bool, torch.Tensor] = {}
        self._wkey = None
        self._fast = False
        self._fast_dtype = None

    # -- dispatch ---------------------------------------------------------
    def _eligible(self, x: torch.Tensor) -> bool:
        """Whether the fused kernel covers this call.

        Anything it does not (non-fp32, odd embed_dim, CPU tensors, a failed
        build) falls through to the conv2d path; none of the captured shapes
        land there.
        """
        w = self.proj.weight
        if not (x.is_cuda and x.dim() == 4 and w.dim() == 4):
            return False
        if x.dtype is not torch.float32 or w.dtype is not torch.float32:
            return False
        b = self.proj.bias
        if b is not None and b.dtype is not torch.float32:
            return False
        if x.shape[1] != w.shape[1] or w.shape[0] % 4:
            return False
        if w.shape[2] != self.patch_size[0] or w.shape[3] != self.patch_size[1]:
            return False
        return _ext() is not None

    def _weight_k_major(self, tf32: bool) -> torch.Tensor:
        """Weight as [K, Cout] (k-major), TF32-rounded on demand; cached.

        Pre-staging the constant weight makes the kernel's B tile a contiguous
        vector copy instead of a transpose-on-load, and keeps the TF32 rounding
        off the per-call path.
        """
        w = self.proj.weight
        key = (w.data_ptr(), w.shape, w._version)
        if self._wkey != key:
            self._wcache = {}
            self._wkey = key
        wk = self._wcache.get(tf32)
        if wk is None:
            wk = w.reshape(w.shape[0], -1).t().contiguous()
            if tf32:
                wk = _to_tf32(wk)
            self._wcache[tf32] = wk
        return wk

    def forward(self, x: torch.Tensor, random_sample: bool = False) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        if not random_sample and (height, width) != self.img_size:
            raise AssertionError(
                f"Input image size ({height}*{width}) doesn't match model {self.img_size}.",
            )
        if x.dtype is not self._fast_dtype:   # re-check only when dtype moves
            self._fast = self._eligible(x)
            self._fast_dtype = x.dtype
        if self._fast:
            r, s = self.patch_size
            patches = x.shape[0] * (height // r) * (width // s)
            if patches > 0:
                tf32 = _use_tf32(x.shape[1], patches)
                out = _ext().patch_embed_forward(
                    x if x.is_contiguous() else x.contiguous(),
                    self._weight_k_major(tf32), self.proj.bias, r, s,
                    self.flatten, _CFG_OVERRIDE, int(tf32))
                return self.norm(out) if self.norm is not None else out

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        else:
            x = x.permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x
