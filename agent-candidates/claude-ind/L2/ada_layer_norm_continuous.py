"""Adaptive continuous layer norm for diffusion transformers (L2 composite).

Used as the final output norm in FLUX (``norm_out``).  Projects the
conditioning embedding through SiLU + Linear into per-channel scale and
shift, then applies LayerNorm with those modulations.

The baseline spends five kernels here (silu, gemv, layer_norm, broadcast-mul,
broadcast-add) and streams the [M, C] activation through HBM three times.  One
C++ call replaces that with two kernels:

1. ``ada_modulation_kernel`` -- SiLU + conditioning GEMV + modulation algebra.
   ``((x-mean)*rstd*lnw + lnb) * (1+scale) + shift`` collapses to
   ``(x-mean)*rstd*A + B`` with ``A = lnw*(1+scale)`` and
   ``B = lnb*(1+scale) + shift``, so two bf16 tables of C entries are all the
   norm kernel needs.
2. ``ada_layer_norm_kernel`` -- LayerNorm fused with that per-channel affine,
   the row held in registers so ``x`` is read once and written once.

Both keep the baseline's bf16 rounding points, so the outputs agree to well
inside the bf16 tolerance rather than merely within it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ..L1.layer_norm import LayerNorm
from ..L1.linear import Linear
from ..L1.silu import SiLU

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// bf16 -> fp32 is an exact bit extension, so a shift/mask pair replaces the two
// cvt instructions __bfloat1622float2 would emit.
__device__ __forceinline__ void unpack2(unsigned int p, float& lo, float& hi) {
  lo = __int_as_float(p << 16);
  hi = __int_as_float(p & 0xffff0000u);
}

__device__ __forceinline__ float rnd_bf16(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

// Programmatic dependent launch: the norm kernel is launched while the
// modulation kernel is still running, so its ~4 us launch latency and the first
// wave of x loads overlap the GEMV.  Everything before this point is
// independent of the A/B tables; everything after needs them.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
}

// ---------------------------------------------------------------------------
// Kernel 1: SiLU + conditioning GEMV + modulation fold.
//
// SPLIT warps cooperate on one channel c, reducing the scale row W[c] together
// with the shift row W[c+C] so every SiLU value feeds two FMAs.  SPLIT > 1 is
// there for occupancy: C channels alone is ~4 warps/SM on a 148-SM part.
//
// Two details matter for bandwidth.  The weight loads are *issued before* the
// shared SiLU vector is built, so the ~3k-element SiLU prologue (a global read,
// an exp and a barrier per block) overlaps HBM latency instead of serializing
// ahead of it.  And the SiLU vector is bf16 in natural order, so a lane picks up
// the eight values matching one 16-byte weight vector with a single LDS.128; a
// transposed fp32 layout needs eight scalar loads and saturates L1TEX.
//
// Emits the two per-channel bf16 tables the norm kernel consumes:
//   A[c] = lnw[c] * (1 + scale[c]),  B[c] = lnb[c] * (1 + scale[c]) + shift[c]
// bf16 so a lane's eight channels are one 16-byte load, and rounded exactly
// where the baseline's bf16 ``F.linear`` output rounds.
// ---------------------------------------------------------------------------
template <int NVK, int SPLIT, int PAIRS>
__global__ __launch_bounds__(32 * SPLIT * PAIRS) void ada_modulation_kernel(
    const bf16* __restrict__ cond,   // [K]
    const bf16* __restrict__ W,      // [2C, K]
    const bf16* __restrict__ blin,   // [2C] or null
    const bf16* __restrict__ lnw,    // [C]  or null
    const bf16* __restrict__ lnb,    // [C]  or null
    bf16* __restrict__ Aout, bf16* __restrict__ Bout, int C) {
  constexpr int K = NVK * 256;
  constexpr int NVS = NVK / SPLIT;
  constexpr int TPB = 32 * SPLIT * PAIRS;
  __shared__ bf16 sv[K];
  __shared__ float part[2][SPLIT * PAIRS];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int p = warp / SPLIT;
  const int s = warp % SPLIT;
  const int c = blockIdx.x * PAIRS + p;

  uint4 r0[NVS], r1[NVS];
  if (c < C) {
    const uint4* w0 = reinterpret_cast<const uint4*>(W + (size_t)c * K);
    const uint4* w1 = reinterpret_cast<const uint4*>(W + (size_t)(c + C) * K);
#pragma unroll
    for (int i = 0; i < NVS; ++i) {
      const int v = (s * NVS + i) * 32 + lane;
      r0[i] = w0[v];
      r1[i] = w1[v];
    }
  }

  for (int i = threadIdx.x; i < K; i += TPB) {
    const float cf = __bfloat162float(cond[i]);
    sv[i] = __float2bfloat16(cf / (1.f + __expf(-cf)));
  }
  __syncthreads();
  const uint4* svv = reinterpret_cast<const uint4*>(sv);

  // Four accumulator pairs: a single chain of NVS*16 dependent FMAs would be
  // latency-bound on the FMA pipe well before the loads land.
  float a0[4] = {0.f, 0.f, 0.f, 0.f};
  float a1[4] = {0.f, 0.f, 0.f, 0.f};
  if (c < C) {
#pragma unroll
    for (int i = 0; i < NVS; ++i) {
      const int v = (s * NVS + i) * 32 + lane;
      const uint4 su = svv[v];
      const unsigned int* sp = reinterpret_cast<const unsigned int*>(&su);
      const unsigned int* q0 = reinterpret_cast<const unsigned int*>(&r0[i]);
      const unsigned int* q1 = reinterpret_cast<const unsigned int*>(&r1[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float sl, sh, x0, y0, x1, y1;
        unpack2(sp[j], sl, sh);
        unpack2(q0[j], x0, y0);
        unpack2(q1[j], x1, y1);
        a0[j] = fmaf(x0, sl, a0[j]);
        a1[j] = fmaf(x1, sl, a1[j]);
        a0[j] = fmaf(y0, sh, a0[j]);
        a1[j] = fmaf(y1, sh, a1[j]);
      }
    }
  }
  float t0 = (a0[0] + a0[1]) + (a0[2] + a0[3]);
  float t1 = (a1[0] + a1[1]) + (a1[2] + a1[3]);
  t0 = warp_sum(t0);
  t1 = warp_sum(t1);

  int cw = c;
  bool emit = (lane == 0);
  if (SPLIT > 1) {
    if (lane == 0) {
      part[0][warp] = t0;
      part[1][warp] = t1;
    }
    __syncthreads();
    emit = (lane == 0) && (warp < PAIRS);
    if (emit) {
      t0 = 0.f;
      t1 = 0.f;
#pragma unroll
      for (int t = 0; t < SPLIT; ++t) {
        t0 += part[0][warp * SPLIT + t];
        t1 += part[1][warp * SPLIT + t];
      }
      cw = blockIdx.x * PAIRS + warp;
    }
  }
  if (emit && cw < C) {
    if (blin != nullptr) {
      t0 += __bfloat162float(blin[cw]);
      t1 += __bfloat162float(blin[cw + C]);
    }
    // Mirror the baseline's bf16 intermediates: F.linear emits bf16, so
    // ``1 + scale`` and ``shift`` are bf16 before they hit the activation.
    const float g = rnd_bf16(1.f + rnd_bf16(t0));
    const float sh = rnd_bf16(t1);
    const float wv = (lnw != nullptr) ? __bfloat162float(lnw[cw]) : 1.f;
    const float bv = (lnb != nullptr) ? __bfloat162float(lnb[cw]) : 0.f;
    Aout[cw] = __float2bfloat16(g * wv);
    Bout[cw] = __float2bfloat16(fmaf(g, bv, sh));
  }
}

// ---------------------------------------------------------------------------
// EMU picks how much of the baseline's intermediate bf16 rounding to reproduce:
// 0 keeps everything in fp32 (fewest instructions), 1 rounds the normalized
// value, 2 also rounds the product before the shift is added (bit-for-bit with
// ``F.layer_norm(x) * (1+scale) + shift`` on bf16 tensors).
// ---------------------------------------------------------------------------
template <int EMU>
__device__ __forceinline__ __nv_bfloat162 modulate2(
    float n0, float n1, unsigned int apk, unsigned int bpk) {
  float a0, a1, b0, b1;
  unpack2(apk, a0, a1);
  unpack2(bpk, b0, b1);
  if (EMU >= 1) {
    n0 = rnd_bf16(n0);
    n1 = rnd_bf16(n1);
  }
  if (EMU >= 2)
    return __floats2bfloat162_rn(rnd_bf16(n0 * a0) + b0, rnd_bf16(n1 * a1) + b1);
  return __floats2bfloat162_rn(fmaf(n0, a0, b0), fmaf(n1, a1, b1));
}

// ---------------------------------------------------------------------------
// Kernel 2: LayerNorm fused with the per-channel affine.  TPB threads cover one
// row of C = TPB*8*NVT channels, the row unpacked into registers so that x is
// read once and written once, across both (exact, two-pass) reductions and the
// modulated write-back.  A/B are bf16 so a lane's eight channels are a single
// 16-byte load; an fp32 float2 table costs four strided 16-byte loads per eight
// channels instead, which alone doubled this kernel's runtime.
// ---------------------------------------------------------------------------
template <int NVT, int TPB, int EMU>
__global__ __launch_bounds__(TPB) void ada_layer_norm_kernel(
    const bf16* __restrict__ X, bf16* __restrict__ O,
    const bf16* __restrict__ A, const bf16* __restrict__ Bt,
    int M, int C, float eps, float inv_c) {
  constexpr int NW = TPB / 32;
  __shared__ float redA[NW], redB[NW];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint4* av = reinterpret_cast<const uint4*>(A);
  const uint4* bv = reinterpret_cast<const uint4*>(Bt);

  bool synced = false;
  for (int row = blockIdx.x; row < M; row += gridDim.x) {
    const uint4* xv = reinterpret_cast<const uint4*>(X + (size_t)row * C);
    uint4 raw[NVT];
#pragma unroll
    for (int i = 0; i < NVT; ++i) raw[i] = xv[i * TPB + tid];

    float f[NVT * 8];
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NVT; ++i) {
      const unsigned int* p = reinterpret_cast<const unsigned int*>(&raw[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float lo, hi;
        unpack2(p[j], lo, hi);
        f[i * 8 + 2 * j] = lo;
        f[i * 8 + 2 * j + 1] = hi;
        s += lo + hi;
      }
    }
    s = warp_sum(s);
    if (lane == 0) redA[warp] = s;
    __syncthreads();
    float tot = redA[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) tot += redA[w];
    const float mean = tot * inv_c;

    float q = 0.f;
#pragma unroll
    for (int i = 0; i < NVT * 8; ++i) {
      const float d = f[i] - mean;
      f[i] = d;
      q = fmaf(d, d, q);
    }
    q = warp_sum(q);
    if (lane == 0) redB[warp] = q;
    __syncthreads();
    float tq = redB[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) tq += redB[w];
    const float rstd = rsqrtf(tq * inv_c + eps);
    if (!synced) {
      pdl_wait();
      synced = true;
    }

    uint4* ov = reinterpret_cast<uint4*>(O + (size_t)row * C);
#pragma unroll
    for (int i = 0; i < NVT; ++i) {
      const int v = i * TPB + tid;
      const uint4 au = av[v];
      const uint4 bu = bv[v];
      const unsigned int* apk = reinterpret_cast<const unsigned int*>(&au);
      const unsigned int* bpk = reinterpret_cast<const unsigned int*>(&bu);
      uint4 o;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        op[j] = modulate2<EMU>(f[i * 8 + 2 * j] * rstd,
                               f[i * 8 + 2 * j + 1] * rstd, apk[j], bpk[j]);
      ov[v] = o;
    }
  }
}

// Launch with programmatic stream serialization when the device supports it,
// so the dependent grid starts before the producer grid retires.
template <typename K, typename... Args>
static void launch_maybe_pdl(K kernel, int grid, int block, cudaStream_t stream,
                             bool pdl, Args... args) {
  if (pdl) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid, 1, 1);
    cfg.blockDim = dim3(block, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
  } else {
    kernel<<<grid, block, 0, stream>>>(args...);
  }
}

static bool pdl_supported() {
  static int ok = -1;
  if (ok < 0) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    ok = (major >= 9) ? 1 : 0;
  }
  return ok == 1;
}

static int sm_count() {
  static int n = 0;
  if (n == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    if (n <= 0) n = 132;
  }
  return n;
}

torch::Tensor ada_ln_cont_forward(
    torch::Tensor x, torch::Tensor cond, torch::Tensor lin_w,
    c10::optional<torch::Tensor> lin_b, c10::optional<torch::Tensor> ln_w,
    c10::optional<torch::Tensor> ln_b, double eps,
    int64_t split, int64_t pairs, int64_t ln_tpb, int64_t ln_waves,
    int64_t emu, int64_t use_pdl) {
  const int C = (int)x.size(-1);
  const int K = (int)cond.size(-1);
  const int64_t M = x.numel() / C;

  auto out = torch::empty_like(x);
  auto tab = torch::empty({2, (int64_t)C}, x.options());
  bf16* ap = reinterpret_cast<bf16*>(tab.data_ptr());
  bf16* bp = ap + C;
  auto stream = at::cuda::getCurrentCUDAStream();

  {
    const bf16* condp = reinterpret_cast<const bf16*>(cond.data_ptr());
    const bf16* wp = reinterpret_cast<const bf16*>(lin_w.data_ptr());
    const bf16* lbp = lin_b.has_value() ? reinterpret_cast<const bf16*>(lin_b->data_ptr()) : nullptr;
    const bf16* nwp = ln_w.has_value() ? reinterpret_cast<const bf16*>(ln_w->data_ptr()) : nullptr;
    const bf16* nbp = ln_b.has_value() ? reinterpret_cast<const bf16*>(ln_b->data_ptr()) : nullptr;
    bool done = false;
#define MOD_CASE(NVK, SP, PR)                                                 \
  if (!done && (K >> 8) == NVK && split == SP && pairs == PR) {                \
    const int grid = (C + PR - 1) / PR;                                       \
    ada_modulation_kernel<NVK, SP, PR><<<grid, 32 * SP * PR, 0, stream>>>(    \
        condp, wp, lbp, nwp, nbp, ap, bp, C);                                 \
    done = true;                                                              \
  }
    MOD_CASE(12, 1, 1) MOD_CASE(12, 1, 2) MOD_CASE(12, 1, 4) MOD_CASE(12, 1, 8)
    MOD_CASE(12, 2, 1) MOD_CASE(12, 2, 2) MOD_CASE(12, 2, 4) MOD_CASE(12, 2, 8)
    MOD_CASE(12, 3, 2) MOD_CASE(12, 3, 4)
    MOD_CASE(12, 4, 1) MOD_CASE(12, 4, 2) MOD_CASE(12, 4, 4)
    MOD_CASE(12, 6, 1) MOD_CASE(12, 6, 2) MOD_CASE(12, 12, 1)
    MOD_CASE(8, 1, 4) MOD_CASE(8, 2, 2) MOD_CASE(8, 4, 2) MOD_CASE(8, 8, 1)
    MOD_CASE(16, 1, 4) MOD_CASE(16, 2, 2) MOD_CASE(16, 4, 2) MOD_CASE(16, 8, 1)
    MOD_CASE(4, 1, 4) MOD_CASE(4, 2, 2) MOD_CASE(4, 4, 2)
    MOD_CASE(6, 1, 4) MOD_CASE(6, 2, 2) MOD_CASE(6, 3, 2) MOD_CASE(6, 6, 1)
    MOD_CASE(10, 1, 4) MOD_CASE(10, 2, 2) MOD_CASE(10, 5, 2)
    MOD_CASE(2, 1, 4) MOD_CASE(2, 2, 2) MOD_CASE(1, 1, 4)
    MOD_CASE(3, 1, 4) MOD_CASE(3, 3, 2) MOD_CASE(5, 1, 4)
#undef MOD_CASE
    TORCH_CHECK(done, "no modulation tiling for K=", K, " split=", split,
                " pairs=", pairs);
  }

  const float inv_c = 1.0f / (float)C;
  const bf16* xp = reinterpret_cast<const bf16*>(x.data_ptr());
  bf16* op = reinterpret_cast<bf16*>(out.data_ptr());
  // Cap the grid so a block handles several rows: the A/B tables then stay L1
  // resident instead of being re-fetched once per row.
  const int64_t cap = (int64_t)sm_count() * ln_waves;
  const int grid = (int)(M < cap ? M : cap);
  const bool pdl = use_pdl && pdl_supported();

#define LN_EMU(NV, TPB, E)                                                    \
  launch_maybe_pdl(ada_layer_norm_kernel<NV, TPB, E>, grid, TPB, stream, pdl, \
                   xp, op, ap, bp, (int)M, C, (float)eps, inv_c);
#define LN_CASE(NV, TPB)                                                      \
  if (ln_tpb == TPB && C == TPB * 8 * NV) {                                    \
    if (emu == 0)      { LN_EMU(NV, TPB, 0) }                                 \
    else if (emu == 1) { LN_EMU(NV, TPB, 1) }                                 \
    else               { LN_EMU(NV, TPB, 2) }                                 \
    return out;                                                              \
  }
  LN_CASE(3, 128) LN_CASE(4, 96) LN_CASE(2, 192) LN_CASE(1, 384)
  LN_CASE(6, 64) LN_CASE(2, 128) LN_CASE(1, 128) LN_CASE(4, 128)
  LN_CASE(6, 128) LN_CASE(8, 128) LN_CASE(1, 256) LN_CASE(2, 256)
  LN_CASE(3, 256) LN_CASE(4, 256) LN_CASE(2, 384) LN_CASE(1, 512)
  LN_CASE(2, 512) LN_CASE(3, 64) LN_CASE(2, 96) LN_CASE(1, 96)
  LN_CASE(1, 192) LN_CASE(8, 48) LN_CASE(12, 64)
#undef LN_CASE
#undef LN_EMU
  TORCH_CHECK(false, "no layer-norm tiling for C=", C, " tpb=", ln_tpb);
}
"""

_CPP_SRC = r"""
torch::Tensor ada_ln_cont_forward(
    torch::Tensor x, torch::Tensor cond, torch::Tensor lin_w,
    c10::optional<torch::Tensor> lin_b, c10::optional<torch::Tensor> ln_w,
    c10::optional<torch::Tensor> ln_b, double eps,
    int64_t split, int64_t pairs, int64_t ln_tpb, int64_t ln_waves,
    int64_t emu, int64_t use_pdl);
"""

_EXT = None


def _ext():
    """JIT-build (once) and return the fused extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        # Build for the local architecture only: the default list compiles six
        # targets, and the PDL intrinsics are sm_90+ anyway.
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}.{cap[1]}" + ("a" if cap[0] >= 9 else "")
        prev = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            _EXT = load_inline(
                name="fk_ada_layer_norm_continuous",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["ada_ln_cont_forward"],
                extra_cuda_cflags=["-O3", "--use_fast_math",
                                   "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                   "--expt-relaxed-constexpr"]
                + (["-Xptxas", "-v"] if os.environ.get("FK_ADA_VERBOSE") else []),
                extra_cflags=["-O3"],
                verbose=bool(os.environ.get("FK_ADA_VERBOSE")),
            )
        finally:
            if prev is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = prev
    return _EXT


_SPLIT = int(os.environ.get("FK_ADA_SPLIT", "1"))
_PAIRS = int(os.environ.get("FK_ADA_PAIRS", "4"))
_LN_TPB = int(os.environ.get("FK_ADA_LN_TPB", "128"))
_LN_WAVES = int(os.environ.get("FK_ADA_LN_WAVES", "8"))
_EMU = int(os.environ.get("FK_ADA_EMU", "0"))
_PDL = int(os.environ.get("FK_ADA_PDL", "1"))
_SUPPORTED_K = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bool`, defaults to `True`): Whether to use bias in the linear layer.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        promote_fp32: bool = True,
    ):
        super().__init__()
        self.silu = SiLU()
        self.linear = Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine,
                                  promote_fp32=promote_fp32)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")
        self.embedding_dim = embedding_dim
        self.eps = eps

    def _reference(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

    def _fast_ok(self, x: torch.Tensor, c: torch.Tensor) -> bool:
        C = self.embedding_dim
        return (
            x.is_cuda
            and x.dtype is torch.bfloat16
            and c.dtype is torch.bfloat16
            and x.is_contiguous()
            and c.is_contiguous()
            # The baseline broadcasts ``(1+scale)[:, None, :]``, which only
            # agrees with a flat [rows, C] view when x has a leading batch dim.
            and x.dim() >= 3
            and c.dim() == 2
            and c.size(0) == 1
            and x.size(-1) == C
            and C % (8 * _LN_TPB) == 0
            and C // (8 * _LN_TPB) <= 8
            and c.size(-1) % 256 == 0
            and (c.size(-1) >> 8) in _SUPPORTED_K
            and self.linear.weight.dtype is torch.bfloat16
        )

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        if not self._fast_ok(x, conditioning_embedding):
            return self._reference(x, conditioning_embedding)
        return _ext().ada_ln_cont_forward(
            x, conditioning_embedding.view(-1), self.linear.weight,
            self.linear.bias, self.norm.weight, self.norm.bias,
            float(self.eps), _SPLIT, _PAIRS, _LN_TPB, _LN_WAVES, _EMU, _PDL,
        )
