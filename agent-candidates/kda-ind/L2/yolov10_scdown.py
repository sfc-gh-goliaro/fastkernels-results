"""YOLOv10 SCDown (spatial channel downsampling) block.

The reference block is ``cv2(cv1(x))``, five separate torch ops:

    1x1 conv -> BatchNorm -> SiLU -> depthwise kxk stride-s conv -> BatchNorm

In inference the two BatchNorms are affine, so each folds into the convolution
weight ahead of it, leaving ``conv -> silu -> conv``.  For the shape every
captured variant uses -- eval mode, a 16-bit CUDA tensor, 4-D contiguous NCHW,
``3x3`` stride 2 -- this module evaluates

    y[n,c,ho,wo] = b2[c] + sum_ij w2[c,i,j]
                          * silu( b1[c] + sum_t w1[c,t] * x[n,t,2*ho+i-1,2*wo+j-1] )

with Triton, by one of two routes chosen per output geometry from measured
latency:

``fused``
    One launch, no intermediate tensor.  Cheapest on traffic, but it recomputes
    the pointwise convolution about three times over.

``split``
    Two launches: a pointwise ``conv+BN+SiLU`` GEMM over the contiguous pixel
    axis, then the depthwise ``conv+BN``.  Pays for a full-resolution
    intermediate, but each kernel is far more instruction-efficient, and the
    pointwise convolution is evaluated exactly once.

Three properties of the expression drive both kernels:

* Out-of-range taps must contribute exactly zero *after* the activation.  The
  depthwise convolution's zero padding applies to the post-SiLU tensor and
  ``silu(b1)`` is not zero, so masking the gather instead of the tap
  contribution would leak ``w2 * silu(b1)`` into every border pixel.

* Triton only coalesces loads whose fast axis is stride 1.  The depthwise stride
  is 2, so a direct nine-tap gather degenerates to one 32-byte sector per lane.
  Both kernels therefore read two *contiguous* column strips per input row and
  recover the three column taps from them with ``tl.split``.

* Indices are deliberately left unclamped.  A false ``tl.load`` predicate emits
  no memory transaction, so clamping buys no safety, and it makes the address
  opaque to the stride analysis that the previous point depends on.

Everything else -- training mode, other dtypes, CPU tensors, non-contiguous or
non-4-D input, other kernel extents or strides -- runs on the folded torch path,
which is also the development oracle for both kernels.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolov10_conv import YOLOConv

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - Triton ships with torch on this box
    _HAVE_TRITON = False


# Dtypes the kernels are compiled and validated for.  fp32 stays on the torch
# path: a tf32 ``tl.dot`` would change the numerics for no latency win here.
_KERNEL_DTYPES = (torch.float16, torch.bfloat16)
# The strip decomposition is specific to a 3x3 stride-2 pad-1 depthwise
# convolution, which is what every captured variant of this block uses.
_KERNEL_GEOMETRY = (3, 2, 1)
# Index arithmetic inside the kernels is 32-bit, so every tensor they address has
# to stay inside that range -- including the output and the split's intermediate,
# which can exceed the input since c2 >= c1.
_MAX_ELEMENTS = 2**31 - 1

# Triton launches on the *current* CUDA device, not on the device the tensors
# live on, so a mismatch would read foreign pointers.  Checking that costs a call
# per forward, which is only worth paying when more than one device is visible.
_MULTI_DEVICE: bool | None = None


def _launch_device_ok(x: torch.Tensor) -> bool:
    global _MULTI_DEVICE
    if _MULTI_DEVICE is None:
        _MULTI_DEVICE = torch.cuda.device_count() > 1
    return not _MULTI_DEVICE or x.device.index == torch.cuda.current_device()


def _fold_conv_bn(conv: nn.Module, bn: nn.Module, dtype: torch.dtype):
    """Return ``(weight, bias)`` equivalent to ``bn(conv(x))`` for one block.

    Folding runs in fp32 regardless of the parameter dtype -- ``running_var`` and
    ``eps`` are small enough that an fp16 reciprocal square root loses real
    accuracy -- and the result is rounded once, into *dtype*.
    """
    weight = conv.weight.detach().float()
    scale = bn.weight.detach().float() * torch.rsqrt(
        bn.running_var.detach().float() + bn.eps
    )
    bias = bn.bias.detach().float() - scale * bn.running_mean.detach().float()
    if conv.bias is not None:
        bias = bias + scale * conv.bias.detach().float()
    weight = weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))
    return weight.to(dtype).contiguous(), bias.to(dtype).contiguous()


# ---------------------------------------------------------------------------
# One-launch shared-memory CUDA C++ route
#
# Both Triton routes evaluate the pointwise convolution about three times per
# output pixel, because Triton cannot shift a register tile along the pixel axis
# and so has to re-gather overlapping column strips. Here each CTA stages its
# output tile's *halo* of pointwise results in shared memory once --
# (2*TH+1) x (2*TW+1) positions for a TH x TW output tile, only 1.13x more than
# the non-overlapping minimum at 8x8 -- so that redundancy nearly vanishes.
#
# Per CTA (one batch item, TC output channels, TH x TW output pixels):
#   1. stage the folded pointwise weight tile and the halo slice of x in shared
#      memory;
#   2. compute the pointwise halo with nvcuda::wmma 16x16x16 fp16 tensor cores
#      accumulating in fp32, add b1, apply SiLU, round *once* into an fp16 shared
#      tile;
#   3. __syncthreads();
#   4. reduce the 3x3 stride-2 depthwise convolution out of that tile in fp32,
#      add b2, store.
#
# Out-of-range taps contribute exactly zero *after* the activation, matching the
# reference's zero padding of the post-SiLU tensor: the depthwise pass skips taps
# whose input row/column lies outside [0,H) x [0,W) rather than relying on a
# padded value.
# ---------------------------------------------------------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WM 16
#define WN 16
#define WK 16

// TC  output channels per CTA (multiple of 16)
// TH  output rows per CTA, TW output cols per CTA
// RH = 2*TH+1, RW = 2*TW+1 halo extent; RWP = RW rounded up to a multiple of 16
template <int TC, int TH, int TW, int C1MAX, int NWARP>
__global__ __launch_bounds__(NWARP * 32) void scdown_fused(
    const __half* __restrict__ x,
    const __half* __restrict__ w1,      // [C2, C1]
    const __half* __restrict__ b1,      // [C2]
    const __half* __restrict__ w2,      // [C2, 9]
    const __half* __restrict__ b2,      // [C2]
    __half* __restrict__ out,
    int N, int C1, int C2, int H, int W, int HO, int WO) {
  constexpr int RH = 2 * TH + 1;
  constexpr int RW = 2 * TW + 1;
  constexpr int RWP = ((RW + WN - 1) / WN) * WN;   // padded halo width
  constexpr int HALO = RH * RWP;

  extern __shared__ char smem_raw[];
  // w1 tile [TC][C1MAX], x halo [C1MAX][HALO], activations [TC][HALO]
  __half* sw1 = reinterpret_cast<__half*>(smem_raw);
  __half* sx = sw1 + TC * C1MAX;
  __half* sact = sx + C1MAX * HALO;
  // One 16x16 fp32 staging tile per warp. wmma::store_matrix_sync is a
  // warp-collective store: handing it a per-thread local array would scatter the
  // fragment across 32 private arrays and spill ~1 KB/thread to local memory.
  float* sacc = reinterpret_cast<float*>(sact + TC * HALO);

  const int tid = threadIdx.x;
  const int nthread = NWARP * 32;
  const int warp = tid / 32;

  // Decode the CTA's tile.
  const int tiles_w = (WO + TW - 1) / TW;
  const int tiles_h = (HO + TH - 1) / TH;
  const int ctiles = C2 / TC;                      // C2 is a multiple of TC
  int b = blockIdx.x;
  const int tw_i = b % tiles_w; b /= tiles_w;
  const int th_i = b % tiles_h; b /= tiles_h;
  const int c_i = b % ctiles;   b /= ctiles;
  const int n = b;
  if (n >= N) return;

  const int oh0 = th_i * TH;
  const int ow0 = tw_i * TW;
  const int c0 = c_i * TC;
  const int hi0 = 2 * oh0 - 1;                     // first halo input row
  const int wi0 = 2 * ow0 - 1;                     // first halo input col

  // 1a. w1 tile.
  for (int i = tid; i < TC * C1; i += nthread) {
    const int cc = i / C1, k = i - cc * C1;
    sw1[cc * C1MAX + k] = w1[(c0 + cc) * C1 + k];
  }
  // 1b. x halo, laid out [k][r * RWP + c] so the wmma B operand is contiguous
  //     along the halo-pixel axis.
  for (int i = tid; i < C1 * HALO; i += nthread) {
    const int k = i / HALO, p = i - k * HALO;
    const int r = p / RWP, cc = p - r * RWP;
    const int ih = hi0 + r, iw = wi0 + cc;
    __half v = __float2half(0.f);
    if (cc < RW && ih >= 0 && ih < H && iw >= 0 && iw < W) {
      v = x[((static_cast<long long>(n) * C1 + k) * H + ih) * W + iw];
    }
    sx[k * HALO + p] = v;
  }
  __syncthreads();

  // 2. Pointwise halo via tensor cores. Each warp owns a (16 channels x 16 halo
  //    pixels) output tile; tiles are handed out round-robin.
  constexpr int MT = TC / WM;
  constexpr int NT = HALO / WN;
  for (int t = warp; t < MT * NT; t += NWARP) {
    const int mt = t / NT, nt = t - mt * NT;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < C1; k += WK) {
      wmma::fragment<wmma::matrix_a, WM, WN, WK, __half, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, WM, WN, WK, __half, wmma::row_major> fb;
      wmma::load_matrix_sync(fa, sw1 + (mt * WM) * C1MAX + k, C1MAX);
      wmma::load_matrix_sync(fb, sx + k * HALO + nt * WN, HALO);
      wmma::mma_sync(acc, fa, fb, acc);
    }
    // Bias + SiLU, then one fp16 rounding into the activation tile.
    float* tmp = sacc + warp * (WM * WN);
    wmma::store_matrix_sync(tmp, acc, WN, wmma::mem_row_major);
    const int lane = tid & 31;
    for (int e = lane; e < WM * WN; e += 32) {
      const int cc = mt * WM + e / WN;
      const int p = nt * WN + e % WN;
      const float v = tmp[e] + __half2float(b1[c0 + cc]);
      sact[cc * HALO + p] = __float2half(v / (1.0f + __expf(-v)));
    }
    __syncwarp();
  }
  __syncthreads();

  // 3. Depthwise 3x3 stride-2 out of the shared activations, fp32 accumulate.
  const int npix = TH * TW;
  for (int i = tid; i < TC * npix; i += nthread) {
    const int cc = i / npix, p = i - cc * npix;
    const int lh = p / TW, lw = p - lh * TW;
    const int oh = oh0 + lh, ow = ow0 + lw;
    if (oh >= HO || ow >= WO) continue;
    const int c = c0 + cc;
    float acc = __half2float(b2[c * 1]);
    #pragma unroll
    for (int ki = 0; ki < 3; ++ki) {
      const int ih = 2 * oh + ki - 1;
      if (ih < 0 || ih >= H) continue;
      #pragma unroll
      for (int kj = 0; kj < 3; ++kj) {
        const int iw = 2 * ow + kj - 1;
        if (iw < 0 || iw >= W) continue;
        const int r = ih - hi0, cq = iw - wi0;
        acc = fmaf(__half2float(w2[c * 9 + ki * 3 + kj]),
                   __half2float(sact[cc * HALO + r * RWP + cq]), acc);
      }
    }
    out[((static_cast<long long>(n) * C2 + c) * HO + oh) * WO + ow] =
        __float2half(acc);
  }
}

template <int TC, int TH, int TW, int C1MAX, int NWARP>
static void launch(torch::Tensor x, torch::Tensor w1, torch::Tensor b1,
                   torch::Tensor w2, torch::Tensor b2, torch::Tensor out) {
  constexpr int RH = 2 * TH + 1;
  constexpr int RW = 2 * TW + 1;
  constexpr int RWP = ((RW + 16 - 1) / 16) * 16;
  constexpr int HALO = RH * RWP;
  const size_t smem = sizeof(__half) *
      (static_cast<size_t>(TC) * C1MAX + static_cast<size_t>(C1MAX) * HALO
       + static_cast<size_t>(TC) * HALO)
      + sizeof(float) * static_cast<size_t>(NWARP) * 16 * 16;

  const int N = x.size(0), C1 = x.size(1), H = x.size(2), W = x.size(3);
  const int C2 = out.size(1), HO = out.size(2), WO = out.size(3);
  const int tiles_w = (WO + TW - 1) / TW, tiles_h = (HO + TH - 1) / TH;
  const int grid = N * (C2 / TC) * tiles_h * tiles_w;

  auto kern = scdown_fused<TC, TH, TW, C1MAX, NWARP>;
  int smem_max = 0;
  cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                         x.device().index());
  TORCH_CHECK(static_cast<int>(smem) <= smem_max,
              "scdown tile needs ", smem, " B of shared memory, device allows ",
              smem_max);
  // Opt in every call: cheap, and a cached flag would wrongly persist a failure.
  TORCH_CHECK(cudaFuncSetAttribute(
                  kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  static_cast<int>(smem)) == cudaSuccess,
              "cudaFuncSetAttribute failed for ", smem, " B");
  kern<<<grid, NWARP * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w1.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(b1.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(b2.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      N, C1, C2, H, W, HO, WO);
}

void scdown(torch::Tensor x, torch::Tensor w1, torch::Tensor b1,
            torch::Tensor w2, torch::Tensor b2, torch::Tensor out,
            int64_t tc, int64_t th, int64_t tw, int64_t nwarp) {
  const int c1 = x.size(1);
  const int c1max = c1 <= 64 ? 64 : 128;
#define DISPATCH(TC_, TH_, TW_, NW_)                                          \
  if (tc == TC_ && th == TH_ && tw == TW_ && nwarp == NW_) {                   \
    if (c1max == 64) { launch<TC_, TH_, TW_, 64, NW_>(x, w1, b1, w2, b2, out); }\
    else { launch<TC_, TH_, TW_, 128, NW_>(x, w1, b1, w2, b2, out); }          \
    return;                                                                   \
  }
  DISPATCH(32, 8, 8, 8)
  DISPATCH(32, 8, 8, 4)
  DISPATCH(64, 8, 8, 8)
  DISPATCH(32, 4, 8, 4)
  DISPATCH(32, 4, 8, 8)
  DISPATCH(16, 8, 8, 4)
  DISPATCH(64, 4, 8, 8)
#undef DISPATCH
  TORCH_CHECK(false, "unsupported scdown tile ", tc, "/", th, "/", tw, "/", nwarp);
}
"""

_CPP_SRC = """
void scdown(torch::Tensor x, torch::Tensor w1, torch::Tensor b1,
            torch::Tensor w2, torch::Tensor b2, torch::Tensor out,
            int64_t tc, int64_t th, int64_t tw, int64_t nwarp);
"""

# Tile configurations the extension compiles a dispatch arm for.
CONFIGS = ((32, 8, 8, 8), (32, 8, 8, 4), (64, 8, 8, 8), (32, 4, 8, 4),
           (32, 4, 8, 8), (16, 8, 8, 4), (64, 4, 8, 8))

_lock = threading.Lock()
_module = None
_failed = False


def _cuda_extension():
    """Build (once) and return the compiled extension, or None if unavailable.

    Inlined here rather than kept in a sibling module on purpose: `validate.py`
    passes `--standalone`, under which a relative import of another candidate
    module resolves to that operator's *baseline*, which for a private helper does
    not exist. The deliverable has to be one file.
    """
    global _module, _failed
    if _module is not None or _failed:
        return _module
    with _lock:
        if _module is not None or _failed:
            return _module
        try:
            from torch.utils.cpp_extension import load_inline
            build_dir = os.environ.get("FK_SCDOWN_BUILD_DIR")
            _module = load_inline(
                name="scdown_sm100",
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["scdown"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                                   "-gencode=arch=compute_100,code=sm_100"],
                build_directory=build_dir,
                verbose=False,
            )
        except Exception:
            _failed = True
            _module = None
    return _module



def _mark_fold_stale(module: nn.Module, args, output) -> None:  # noqa: ARG001
    """Forward hook: a BatchNorm that ran in training mode invalidates any fold.

    ``YOLOSCDown.train()`` cannot see ``cv1.bn.train()`` -- that call never
    reaches the parent -- so the BatchNorm flags itself here and the parent
    notices on its next forward.  Set on the BatchNorm rather than captured in a
    closure so the module stays picklable and free of reference cycles.
    """
    if module.training:
        module._fold_stale = True


if _HAVE_TRITON:

    @triton.jit
    def _scdown_kernel(
        x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
        H, W, HO, WO, NWT,
        C1: tl.constexpr, C2: tl.constexpr,
        BLOCK_C: tl.constexpr, QW: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fully fused: one program per (channel tile, output-row segment, batch).

        ``w1_ptr`` is the folded ``[C2, C1]`` pointwise weight and ``w2_ptr`` the
        folded ``[C2, 9]`` depthwise weight.

        For output row ``oh`` the three input rows are ``2*oh + i - 1``.  Within a
        row, the three column taps of output column ``ow`` are input columns
        ``2*ow - 1``, ``2*ow`` and ``2*ow + 1``.  Strip A spans ``2*QW``
        contiguous columns starting at the first of those, so reshaped to
        ``[BLOCK_C, QW, 2]`` its even half is tap 0 and its odd half is tap 1;
        strip B is A shifted two columns, so its even half is tap 2.

        The channel-tile axis is ``program_id(0)`` so programs that share an
        output row -- and therefore read the same ``x`` -- are launched
        consecutively and stay co-resident for L2 reuse.
        """
        pid_c = tl.program_id(0)
        pid_r = tl.program_id(1)
        n = tl.program_id(2)

        oh = pid_r // NWT
        wt = pid_r % NWT
        ow = wt * QW + tl.arange(0, QW)
        ow_ok = ow < WO

        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C2
        kk = tl.arange(0, BLOCK_K)
        k_ok = kk < C1

        w1 = tl.load(w1_ptr + c[:, None] * C1 + kk[None, :],
                     mask=c_ok[:, None] & k_ok[None, :], other=0.0)
        b1 = tl.load(b1_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
        b2 = tl.load(b2_ptr + c, mask=c_ok, other=0.0).to(tl.float32)

        plane = kk * (H * W)
        col_a = wt * QW * 2 - 1 + tl.arange(0, 2 * QW)
        ok_a = (col_a >= 0) & (col_a < W)
        ok_b = ((col_a + 2) >= 0) & ((col_a + 2) < W)
        # Hoisted as fp32 multipliers rather than applied as a per-tap
        # ``tl.where`` in the loop: the source report attributed 22.1M FSEL +
        # 14.7M FSETP + 16.0M PLOP3 -- about 21% of all lane slots -- to that
        # select, and as a multiplier it folds into the existing FMA.  Measured
        # ~10% faster on the N=4 shapes.
        f0 = (ow_ok & (ow * 2 - 1 >= 0)).to(tl.float32)
        f1 = (ow_ok & (ow * 2 < W)).to(tl.float32)
        f2 = (ow_ok & (ow * 2 + 1 < W)).to(tl.float32)

        acc = tl.zeros([BLOCK_C, QW], dtype=tl.float32)
        x_base = x_ptr + n * (C1 * H * W)

        for i in tl.static_range(3):
            hi = oh * 2 + i - 1
            row_ok = (hi >= 0) & (hi < H)
            rowf = row_ok.to(tl.float32)          # scalar
            rowp = x_base + hi * W

            strip = tl.load(rowp + plane[:, None] + col_a[None, :],
                            mask=k_ok[:, None] & (row_ok & ok_a)[None, :],
                            other=0.0)
            t = tl.dot(w1, strip, out_dtype=tl.float32) + b1[:, None]
            t = t * tl.sigmoid(t)
            t0, t1 = tl.split(tl.reshape(t, [BLOCK_C, QW, 2]))

            strip = tl.load(rowp + plane[:, None] + col_a[None, :] + 2,
                            mask=k_ok[:, None] & (row_ok & ok_b)[None, :],
                            other=0.0)
            t = tl.dot(w1, strip, out_dtype=tl.float32) + b1[:, None]
            t = t * tl.sigmoid(t)
            t2, _ = tl.split(tl.reshape(t, [BLOCK_C, QW, 2]))

            # ``* rowf`` scales the [BLOCK_C] weight vector, not the
            # [BLOCK_C, QW] tile: an out-of-range input row must contribute
            # nothing, and this is the cheapest place to enforce it.
            w20 = tl.load(w2_ptr + c * 9 + (i * 3), mask=c_ok,
                          other=0.0).to(tl.float32) * rowf
            w21 = tl.load(w2_ptr + c * 9 + (i * 3 + 1), mask=c_ok,
                          other=0.0).to(tl.float32) * rowf
            w22 = tl.load(w2_ptr + c * 9 + (i * 3 + 2), mask=c_ok,
                          other=0.0).to(tl.float32) * rowf
            # Zero the *tap contribution*, not the gathered x: the padding
            # belongs to the post-activation tensor.
            acc += w20[:, None] * (t0 * f0[None, :])
            acc += w21[:, None] * (t1 * f1[None, :])
            acc += w22[:, None] * (t2 * f2[None, :])

        tl.store(
            out_ptr + n * (C2 * HO * WO) + c[:, None] * (HO * WO)
            + (oh * WO + ow)[None, :],
            (acc + b2[:, None]).to(out_ptr.dtype.element_ty),
            mask=c_ok[:, None] & ow_ok[None, :])

    @triton.jit
    def _pointwise_kernel(
        x_ptr, w1_ptr, b1_ptr, t_ptr, HW,
        C1: tl.constexpr, C2: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """``t[n,c,p] = silu(b1[c] + sum_k w1[c,k] * x[n,k,p])``.

        A plain GEMM whose N axis is the contiguous pixel axis, so the ``x`` tile
        is stride 1 on its fast axis and the channel contraction runs as a
        pipelined loop rather than one wide tile.
        """
        pid_c = tl.program_id(0)
        pid_p = tl.program_id(1)
        n = tl.program_id(2)
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C2
        p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_ok = p < HW
        acc = tl.zeros([BLOCK_C, BLOCK_P], dtype=tl.float32)
        x_base = x_ptr + n * (C1 * HW)
        for kb in tl.range(0, C1, BLOCK_K):
            kk = kb + tl.arange(0, BLOCK_K)
            k_ok = kk < C1
            w1 = tl.load(w1_ptr + c[:, None] * C1 + kk[None, :],
                         mask=c_ok[:, None] & k_ok[None, :], other=0.0)
            xt = tl.load(x_base + kk[:, None] * HW + p[None, :],
                         mask=k_ok[:, None] & p_ok[None, :], other=0.0)
            acc = tl.dot(w1, xt, acc)
        t = acc + tl.load(b1_ptr + c, mask=c_ok, other=0.0).to(tl.float32)[:, None]
        t = t * tl.sigmoid(t)
        tl.store(t_ptr + n * (C2 * HW) + c[:, None] * HW + p[None, :],
                 t.to(t_ptr.dtype.element_ty),
                 mask=c_ok[:, None] & p_ok[None, :])

    @triton.jit
    def _depthwise_kernel(
        t_ptr, w2_ptr, b2_ptr, out_ptr, H, W, HO, WO, NWT,
        C2: tl.constexpr, BLOCK_C: tl.constexpr, QW: tl.constexpr,
    ):
        """``out[n,c,oh,ow] = b2[c] + sum_ij w2[c,i,j] * t[n,c,2*oh+i-1,2*ow+j-1]``.

        The same two-contiguous-strip decomposition as the fused kernel, minus the
        pointwise convolution -- ``t`` already holds the post-SiLU values.
        """
        pid_r = tl.program_id(0)
        pid_c = tl.program_id(1)
        n = tl.program_id(2)
        oh = pid_r // NWT
        wt = pid_r % NWT
        ow = wt * QW + tl.arange(0, QW)
        ow_ok = ow < WO
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_ok = c < C2

        col_a = wt * QW * 2 - 1 + tl.arange(0, 2 * QW)
        ok_a = (col_a >= 0) & (col_a < W)
        ok_b = ((col_a + 2) >= 0) & ((col_a + 2) < W)
        tap0 = ow_ok & (ow * 2 - 1 >= 0)
        tap1 = ow_ok & (ow * 2 < W)
        tap2 = ow_ok & (ow * 2 + 1 < W)

        plane = t_ptr + n * (C2 * H * W) + c[:, None] * (H * W)
        acc = tl.zeros([BLOCK_C, QW], dtype=tl.float32)

        for i in tl.static_range(3):
            hi = oh * 2 + i - 1
            row_ok = (hi >= 0) & (hi < H)
            rowp = plane + hi * W

            strip = tl.load(rowp + col_a[None, :],
                            mask=c_ok[:, None] & (row_ok & ok_a)[None, :],
                            other=0.0).to(tl.float32)
            t0, t1 = tl.split(tl.reshape(strip, [BLOCK_C, QW, 2]))
            strip = tl.load(rowp + col_a[None, :] + 2,
                            mask=c_ok[:, None] & (row_ok & ok_b)[None, :],
                            other=0.0).to(tl.float32)
            t2, _ = tl.split(tl.reshape(strip, [BLOCK_C, QW, 2]))

            w20 = tl.load(w2_ptr + c * 9 + (i * 3), mask=c_ok,
                          other=0.0).to(tl.float32)
            w21 = tl.load(w2_ptr + c * 9 + (i * 3 + 1), mask=c_ok,
                          other=0.0).to(tl.float32)
            w22 = tl.load(w2_ptr + c * 9 + (i * 3 + 2), mask=c_ok,
                          other=0.0).to(tl.float32)
            acc += tl.where((row_ok & tap0)[None, :], w20[:, None] * t0, 0.0)
            acc += tl.where((row_ok & tap1)[None, :], w21[:, None] * t1, 0.0)
            acc += tl.where((row_ok & tap2)[None, :], w22[:, None] * t2, 0.0)

        b2 = tl.load(b2_ptr + c, mask=c_ok, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + n * (C2 * HO * WO) + c[:, None] * (HO * WO)
            + (oh * WO + ow)[None, :],
            (acc + b2[:, None]).to(out_ptr.dtype.element_ty),
            mask=c_ok[:, None] & ow_ok[None, :])


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


# Per-geometry route and launch metadata, from a single-process sweep under the
# harness timing protocol on B200 (sm_100); see profile/bench_kernel.py and
# profile/tune_split.py.  Keyed by (batch, out channels, out rows, out cols).
#
# fused: (BLOCK_C, QW, num_warps)
# split: (pw BLOCK_C, pw BLOCK_P, pw BLOCK_K, pw warps, dw BLOCK_C, dw QW, dw warps)
# Both routes were swept over their own configuration spaces (48 pointwise, 18
# depthwise, 12 fused) on all four benched geometries.  Measured microseconds:
#
#   geometry              folded torch   fused   pointwise  depthwise   split
#   (4, 128, 40, 40)             37.82   21.47       11.20      15.36   21.50
#   (1, 128, 40, 40)             25.60   13.34        9.15       9.20   15.36
#   (4, 256, 20, 20)             33.68   21.49       11.20      13.28   21.47
#   (1, 256, 20, 20)             25.34   13.34        9.18       9.18   15.39
#
# The split's kernels are individually much cheaper -- it evaluates the pointwise
# convolution once where the fused kernel evaluates it three times -- but the
# second launch costs a whole extra timed-region floor (~5 us: ~2.9 us of region
# floor plus ~2.1 us of launch), which cancels the saving exactly.  So the route
# is `fused` everywhere: the only case where the split even ties is
# (4, 256, 20, 20), by 0.02 us, which is noise.  The split stays in the module
# because the margin is thin enough that a different shape could flip it, and
# because a per-geometry route is how that gets decided.
_ROUTE_TABLE: dict[tuple[int, int, int, int], str] = {
    (4, 128, 40, 40): "fused",
    (1, 128, 40, 40): "fused",
    (4, 256, 20, 20): "fused",
    (1, 256, 20, 20): "fused",
}
# Retuned after the masking change, and restricted to configurations Triton
# reports **zero spills** for -- AC-9 requires no local-memory traffic, and the
# previous (128, 8, 8) point spilled (64 registers, 12 800 local loads).
# Measured us / registers / spills:
#   (4, 128, 40, 40)  19.57  128 regs  0 spills
#   (1, 128, 40, 40)  13.31   80 regs  0 spills
#   (4, 256, 20, 20)  19.46  127 regs  0 spills
#   (1, 256, 20, 20)  13.26   96 regs  0 spills
_FUSED_TABLE: dict[tuple[int, int, int, int], tuple[int, int, int]] = {
    (4, 128, 40, 40): (128, 8, 4),
    (1, 128, 40, 40): (64, 8, 4),
    (4, 256, 20, 20): (256, 8, 8),
    (1, 256, 20, 20): (64, 8, 4),
}
_SPLIT_TABLE: dict[tuple[int, int, int, int], tuple[int, ...]] = {
    (4, 128, 40, 40): (128, 64, 64, 4, 32, 16, 4),
    (1, 128, 40, 40): (64, 64, 32, 8, 32, 16, 8),
    (4, 256, 20, 20): (128, 128, 64, 8, 64, 16, 8),
    (1, 256, 20, 20): (64, 64, 64, 4, 32, 16, 8),
}


def _fused_config(c2: int, ho: int, wo: int) -> tuple[int, int, int]:
    """``(BLOCK_C, QW, num_warps)`` for the fused kernel.

    ``QW = 8`` won on every benched shape, by a wide margin on those whose output
    width does not divide the tile: at ``QW = 16`` and ``WO = 20`` the second
    column tile computes sixteen output columns for four useful ones, and each
    ``2*QW``-wide strip spans 32 columns of a 40-wide row.  ``2*QW = 16`` is also
    exactly the smallest ``tl.dot`` N axis that still reaches the fifth-generation
    tensor cores on sm_100.
    """
    return min(128, max(16, _next_pow2(c2))), 8, 8


def _split_config(c2: int, ho: int, wo: int) -> tuple[int, ...]:
    """Launch metadata for the two-kernel route."""
    return (64, 256, 64, 8, min(128, max(16, _next_pow2(c2))), 32, 4)


# CUDA route tiles: (TC output channels, TH output rows, TW output cols, warps).
# Must be one of the arms the extension compiles (see CONFIGS).
_CUDA_TABLE: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}


def _cuda_config(c2: int, ho: int, wo: int) -> tuple[int, int, int, int]:
    return (32, 8, 8, 8)


def _cuda_usable(c1: int, c2: int, tc: int) -> bool:
    """The kernel needs whole channel tiles and a pointwise K it can tile by 16."""
    return c1 <= 128 and c1 % 16 == 0 and c2 % tc == 0


class YOLOSCDown(nn.Module):
    """Drop-in replacement for the reference SCDown block.

    ``cv1``/``cv2`` are the reference ``YOLOConv`` blocks, so the parameter names
    and ``state_dict`` layout are unchanged.  The folded copies live in a plain
    dict attribute keyed by ``(device, dtype)`` and are therefore invisible to
    ``state_dict``; they are dropped whenever new weights are loaded, the module
    is moved or cast, or either BatchNorm runs in training mode.
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = YOLOConv(c1, c2, 1, 1)
        self.cv2 = YOLOConv(c2, c2, k=k, s=s, g=c2, act=False)
        self.c1 = c1
        self.c2 = c2
        self._folded: dict = {}
        self._plans: dict = {}
        self._kernel_ready = self._kernel_supported()
        self.register_load_state_dict_post_hook(_forget_folded_weights)
        for bn in (self.cv1.bn, self.cv2.bn):
            bn._fold_stale = False
            bn.register_forward_hook(_mark_fold_stale)

    # ------------------------------------------------------------------
    # Fast-path eligibility, decided once from the convolution attributes
    # ------------------------------------------------------------------
    def _kernel_supported(self) -> bool:
        """Whether the Triton routes can serve this block's configuration."""
        if not _HAVE_TRITON:
            return False
        a, b = self.cv1.conv, self.cv2.conv
        k, stride, pad = _KERNEL_GEOMETRY
        return (
            # cv1 must be a plain dense 1x1 with no spatial effect.
            tuple(a.weight.shape) == (self.c2, self.c1, 1, 1)
            and a.stride == (1, 1) and a.padding == (0, 0)
            and a.dilation == (1, 1) and a.groups == 1
            # cv2 must be the fully depthwise 3x3 stride-2 pad-1 the strip
            # decomposition is derived for.
            and tuple(b.weight.shape) == (self.c2, 1, k, k)
            and b.stride == (stride, stride) and b.padding == (pad, pad)
            and b.dilation == (1, 1) and b.groups == self.c2
            # The gather tile is [BLOCK_K, 2*QW] with BLOCK_K >= C1 masked; keep
            # the gate to the widths the tiling is exercised for.
            and self.c1 % 16 == 0
        )

    def _kernel_eligible(self, x: torch.Tensor) -> bool:
        if not (
            self._kernel_ready
            and x.is_cuda
            and x.dtype in _KERNEL_DTYPES
            and x.is_contiguous()
            and x.shape[1] == self.c1
            # A raw Triton forward records no autograd graph, so it must not
            # stand in for the reference when gradients are being tracked.
            and not torch.is_grad_enabled()
        ):
            return False
        n, _, h, w = x.shape
        k, stride, pad = _KERNEL_GEOMETRY
        ho = (h + 2 * pad - k) // stride + 1
        wo = (w + 2 * pad - k) // stride + 1
        # Every tensor either route indexes, not just x: c2 >= c1, so the output
        # and the split's intermediate can overflow 32-bit offsets while the
        # input is still small.
        return (
            max(x.numel(), n * self.c2 * ho * wo, n * self.c2 * h * w,
                self.c2 * self.c1, self.c2 * k * k) <= _MAX_ELEMENTS
            and _launch_device_ok(x)
        )

    # ------------------------------------------------------------------
    # Folded-weight cache
    # ------------------------------------------------------------------
    def _apply(self, *args, **kwargs):
        # ``_apply`` covers .to()/.half()/.float()/.cuda(); plain attributes are
        # not visited by it, so the cache has to be dropped by hand.
        self._folded = {}
        return super()._apply(*args, **kwargs)

    def train(self, mode: bool = True):
        # A training pass updates the BatchNorm running statistics, which makes
        # any fold taken before it stale.
        self._folded = {}
        return super().train(mode)

    def _drop_stale_fold(self) -> None:
        """Drop the fold if either BatchNorm has run in training mode since it.

        ``train()`` above covers a transition made through this module, but
        ``cv1.bn.train()`` never reaches it, so each BatchNorm flags itself from a
        forward hook and this notices on the next call.  Two attribute reads.
        """
        for block in (self.cv1, self.cv2):
            bn = getattr(block, "bn", None)
            if bn is not None and getattr(bn, "_fold_stale", False):
                bn._fold_stale = False
                self._folded = {}

    def _foldable(self, x: torch.Tensor) -> bool:
        """True when the folded form is equivalent to running the submodules.

        ``x.dim() == 4`` is part of the test: dropping the BatchNorms also drops
        the rank check they impose, and a 3-D input has to keep failing the way
        the reference block fails it rather than being silently accepted as an
        unbatched convolution.

        The BatchNorm submodules are checked in their own right, not just through
        ``self.training``: one of them can be put in training mode on its own, and
        folded running statistics are wrong the moment it is.  Autocast is
        excluded because it would cast the reference convolutions but not a fold
        that has already been rounded to a fixed dtype.
        """
        return (
            not self.training
            and x.dim() == 4
            and hasattr(self.cv1, "bn")
            and hasattr(self.cv2, "bn")
            and not self.cv1.bn.training
            and not self.cv2.bn.training
            and x.device == self.cv1.conv.weight.device
            and not torch.is_autocast_enabled()
        )

    @torch.no_grad()
    def _folded_weights(self, dtype: torch.dtype, device: torch.device):
        """Folded weights for *dtype* on *device*, computed on first use.

        Returns ``(w1, b1, w2, b2, w1_flat, w2_flat)``; the flat views are what
        the kernels index and share storage with the 4-D convolution weights.
        """
        key = (device.type, device.index, dtype)
        cached = self._folded.get(key)
        if cached is None:
            w1, b1 = _fold_conv_bn(self.cv1.conv, self.cv1.bn, dtype)
            w2, b2 = _fold_conv_bn(self.cv2.conv, self.cv2.bn, dtype)
            cached = (w1, b1, w2, b2,
                      w1.view(w1.shape[0], -1), w2.view(w2.shape[0], -1))
            self._folded[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _folded_forward(self, x: torch.Tensor) -> torch.Tensor:
        w1, b1, w2, b2 = self._folded_weights(x.dtype, x.device)[:4]
        a, b = self.cv1.conv, self.cv2.conv
        t = F.conv2d(x, w1, b1, a.stride, a.padding, a.dilation, a.groups)
        t = F.silu(t)
        return F.conv2d(t, w2, b2, b.stride, b.padding, b.dilation, b.groups)

    def _plan(self, n: int, h: int, w: int):
        """Cached route and launch metadata for one output geometry."""
        k, stride, pad = _KERNEL_GEOMETRY
        ho = (h + 2 * pad - k) // stride + 1
        wo = (w + 2 * pad - k) // stride + 1
        key = (n, self.c2, ho, wo)
        plan = self._plans.get(key)
        if plan is None:
            route = _ROUTE_TABLE.get(key, "fused")
            if route == "cuda":
                cfg = _CUDA_TABLE.get(key) or _cuda_config(self.c2, ho, wo)
                if not (_cuda_usable(self.c1, self.c2, cfg[0])
                        and _cuda_extension() is not None):
                    route = "fused"
                else:
                    plan = ("cuda", ho, wo, cfg)
            if route == "cuda":
                pass
            elif route == "split":
                cfg = _SPLIT_TABLE.get(key) or _split_config(self.c2, ho, wo)
                pw_c, pw_p, pw_k, pw_warps, dw_c, dw_qw, dw_warps = cfg
                nwt = -(-wo // dw_qw)
                plan = ("split", ho, wo,
                        (-(-self.c2 // pw_c), -(-(h * w) // pw_p), n),
                        (pw_c, pw_p, min(pw_k, _next_pow2(self.c1)), pw_warps),
                        (ho * nwt, -(-self.c2 // dw_c), n),
                        (dw_c, dw_qw, dw_warps, nwt))
            else:
                block_c, qw, warps = (_FUSED_TABLE.get(key)
                                      or _fused_config(self.c2, ho, wo))
                nwt = -(-wo // qw)
                plan = ("fused", ho, wo,
                        (-(-self.c2 // block_c), ho * nwt, n),
                        (block_c, qw, warps, nwt, _next_pow2(self.c1)))
            self._plans[key] = plan
        return plan

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        plan = self._plan(n, h, w)
        route, ho, wo = plan[0], plan[1], plan[2]
        if ho <= 0 or wo <= 0:
            return self._folded_forward(x)

        _, b1, _, b2, w1f, w2f = self._folded_weights(x.dtype, x.device)
        out = torch.empty((n, self.c2, ho, wo), dtype=x.dtype, device=x.device)

        if route == "cuda":
            tc, th, tw, nwarp = plan[3]
            _cuda_extension().scdown(x, w1f, b1, w2f, b2, out, tc, th, tw, nwarp)
            return out

        if route == "split":
            _, _, _, pw_grid, pw_cfg, dw_grid, dw_cfg = plan
            pw_c, pw_p, pw_k, pw_warps = pw_cfg
            dw_c, dw_qw, dw_warps, nwt = dw_cfg
            # Allocated per forward rather than kept as module or global state:
            # a cached workspace would have to track device, dtype and shape, and
            # the caching allocator already makes this cheap.
            mid = torch.empty((n, self.c2, h, w), dtype=x.dtype, device=x.device)
            _pointwise_kernel[pw_grid](
                x, w1f, b1, mid, h * w,
                C1=self.c1, C2=self.c2,
                BLOCK_C=pw_c, BLOCK_P=pw_p, BLOCK_K=pw_k,
                num_warps=pw_warps, num_stages=3,
            )
            _depthwise_kernel[dw_grid](
                mid, w2f, b2, out, h, w, ho, wo, nwt,
                C2=self.c2, BLOCK_C=dw_c, QW=dw_qw,
                num_warps=dw_warps, num_stages=1,
            )
            return out

        _, _, _, grid, cfg = plan
        block_c, qw, warps, nwt, block_k = cfg
        _scdown_kernel[grid](
            x, w1f, b1, w2f, b2, out,
            h, w, ho, wo, nwt,
            C1=self.c1, C2=self.c2,
            BLOCK_C=block_c, QW=qw, BLOCK_K=block_k,
            num_warps=warps, num_stages=1,
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._drop_stale_fold()
        if not self._foldable(x):
            # Training mode needs batch statistics, and an already-fused
            # YOLOConv has no ``bn`` left to fold; either way, defer.
            return self.cv2(self.cv1(x))
        if self._kernel_eligible(x):
            try:
                return self._triton_forward(x)
            except Exception:
                # A compilation or launch failure must degrade, not propagate.
                # Remember it so the retry cost is paid once.
                self._kernel_ready = False
        return self._folded_forward(x)


def _forget_folded_weights(module: YOLOSCDown, incompatible_keys) -> None:  # noqa: ARG001
    """``load_state_dict`` post-hook: the cached fold no longer matches."""
    module._folded = {}
