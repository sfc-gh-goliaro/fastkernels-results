// Hand-written CUDA for the YOLOv10 CIB block's depthwise stages, over the
// zero-padded workspace the Triton pipeline already uses:
//
//   workspace[n][c][pos],  pos = VLO + h*PITCH + PAD + w   for the valid pixel
//   (h, w);  every other position holds a permanent zero.
//
// Why C++ and not Triton.  In Triton each of the K*K taps is a separate element
// load off a freshly built address tile, so the 7x7 stage issues 49 loads per
// output and measures 7.2 us at N=4 for ~0.1 us of arithmetic.  Here a CTA
// stages PG whole plane strips through shared memory once, and each thread
// register-blocks WT consecutive output columns of one row, so every loaded
// pixel feeds all K taps of its row: K*(WT+K-1) shared reads per WT outputs
// instead of K*K per output, and each pixel is read from global exactly once
// per CTA instead of once per tap.
//
// Parallelism is the thing to get right, not reuse.  This problem is ~20 MMAC
// total, so whatever the mapping, the machine is latency-bound rather than
// throughput-bound: an earlier version with WT = W = 20 (one thread per output
// row) had only 20 K threads = ~4 warps/SM and measured 8.5 us -- *slower* than
// Triton -- purely because a single warp per scheduler cannot cover its own FFMA
// latency.  Small WT with many more threads is what makes the reuse pay.
//
// The strip is staged as **float**, not half.  Converting at staging time costs
// one cvt per pixel per CTA; converting inside the task loop cost one cvt per
// pixel per *tap row* (K*(WT+K-1) per WT outputs), and cvt issues on the
// low-throughput XU pipe, so it -- not the FFMAs -- was the binding cost: 6.3 M
// cvt at 16/clk/SM is 1.5 us on its own.  Float staging also turns the inner
// reads into 8 B LDS.64 instead of 2 B LDS.U16, cutting the load count 4x.
//
// Bounds.  The halo reads span positions [LO, LO+SPAN) of each plane; the host
// computes both from the same PADR/POS0 invariant the Triton kernels use and
// refuses the CUDA path (falling back to Triton) if they do not fit inside PS.
// Pad positions are never written, so the border stays zero for buffer reuse --
// matching the Triton kernels, which write an explicit zero there.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float silu(float a) {
  return __fdividef(a, 1.0f + __expf(-a));
}

// One CTA owns PG plane strips; BD threads walk PG*H*NCB (plane, row, column
// block) tasks.  Shared: PG strips of SPAN halfs, then PG*K*K weights and PG
// biases as floats so the inner loop never converts.
template <int K, int WT, int BD>
__global__ __launch_bounds__(BD) void dw_pad_kernel(
    const __half *__restrict__ gin, const __half *__restrict__ gwd,
    const float *__restrict__ gbd, __half *__restrict__ gout,
    int C, int H, int W, int PITCH, int PS, int VLO, int PAD,
    int nplanes, int PG, int SPAN, int LO) {
  constexpr int KP = K / 2;
  constexpr int KK = K * K;
  constexpr int RW = WT + K - 1;    // input columns a WT-wide output block needs
  constexpr int RW2 = (RW + 1) / 2;  // ... read as 8 B float2 pairs

  extern __shared__ char smem[];
  float *sh = reinterpret_cast<float *>(smem);          // PG * SPAN floats
  float *sw = sh + (size_t)PG * SPAN;
  float *sb = sw + (size_t)PG * KK;

  const int pbase = blockIdx.x * PG;

  // --- stage PG plane strips: 16 B coalesced load, cvt once, 32 B store -----
  const int vec_per_plane = SPAN >> 3;  // SPAN is a multiple of 8 by construction
  for (int v = threadIdx.x; v < PG * vec_per_plane; v += BD) {
    const int i = v / vec_per_plane;
    const int j = v - i * vec_per_plane;
    const int p = pbase + i;
    if (p >= nplanes) continue;
    const int n = p / C, c = p - n * C;
    const __half2 *src = reinterpret_cast<const __half2 *>(
        gin + ((size_t)n * C + c) * PS + LO) + 4 * j;
    float *dst = sh + (size_t)i * SPAN + 8 * j;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 f = __half22float2(src[q]);
      reinterpret_cast<float2 *>(dst)[q] = f;
    }
  }
  // --- weights + bias, one float each, converted once ----------------------
  for (int v = threadIdx.x; v < PG * (KK + 1); v += BD) {
    const int i = v / (KK + 1);
    const int k = v - i * (KK + 1);
    const int p = pbase + i;
    if (p >= nplanes) continue;
    const int c = p - (p / C) * C;
    if (k == KK) sb[i] = gbd[c];
    else sw[(size_t)i * KK + k] = __half2float(gwd[(size_t)k * C + c]);
  }
  __syncthreads();

  const int ncb = (W + WT - 1) / WT;      // column blocks per row
  const int tpp = H * ncb;                // tasks per plane
  for (int task = threadIdx.x; task < PG * tpp; task += BD) {
    const int i = task / tpp;
    const int r = task - i * tpp;
    const int p = pbase + i;
    if (p >= nplanes) break;
    const int h = r / ncb;
    const int w0 = (r - h * ncb) * WT;

    const int n = p / C, c = p - n * C;
    const float *strip = sh + (size_t)i * SPAN;
    const float *wrow = sw + (size_t)i * KK;

    float acc[WT];
#pragma unroll
    for (int j = 0; j < WT; ++j) acc[j] = sb[i];
    // absolute position of (row h-KP, column w0-KP), rebased into the strip.
    // WT, PITCH and (VLO + PAD - KP - LO) are all even (checked on the host), so
    // `base` is even and the row reads below are 8 B aligned.
    const int base = VLO + (h - KP) * PITCH + PAD + w0 - KP - LO;
#pragma unroll
    for (int dh = 0; dh < K; ++dh) {
      float row[RW2 * 2];
      const float2 *src =
          reinterpret_cast<const float2 *>(strip + base + dh * PITCH);
#pragma unroll
      for (int t = 0; t < RW2; ++t) {
        const float2 f = src[t];
        row[2 * t] = f.x;
        row[2 * t + 1] = f.y;
      }
#pragma unroll
      for (int dw = 0; dw < K; ++dw) {
        const float wv = wrow[dh * K + dw];
#pragma unroll
        for (int j = 0; j < WT; ++j) acc[j] += wv * row[j + dw];
      }
    }
    __half *dst = gout + ((size_t)n * C + c) * PS + VLO + h * PITCH + PAD + w0;
#pragma unroll
    for (int j = 0; j < WT; ++j)
      if (w0 + j < W) dst[j] = __float2half(silu(acc[j]));
  }
}

template <int K, int WT>
bool launch_bd(int bd, const __half *in, const __half *wd, const float *bdias,
               __half *out, int C, int H, int W, int PITCH, int PS, int VLO,
               int PAD, int nplanes, int PG, int SPAN, int LO, cudaStream_t s) {
  const size_t shb = (size_t)PG * (SPAN + K * K + 1) * 4;
  const int grid = (nplanes + PG - 1) / PG;
#define DW_CASE(B)                                                            \
  case B:                                                                     \
    dw_pad_kernel<K, WT, B><<<grid, B, shb, s>>>(                             \
        in, wd, bdias, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN,   \
        LO);                                                                  \
    return true;
  switch (bd) {
    DW_CASE(64) DW_CASE(128) DW_CASE(256) DW_CASE(512)
    default: return false;
  }
#undef DW_CASE
}

template <int K>
bool launch_wt(int wt, int bd, const __half *in, const __half *wd,
               const float *b, __half *out, int C, int H, int W, int PITCH,
               int PS, int VLO, int PAD, int nplanes, int PG, int SPAN, int LO,
               cudaStream_t s) {
  switch (wt) {
    case 2: return launch_bd<K, 2>(bd, in, wd, b, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
    case 4: return launch_bd<K, 4>(bd, in, wd, b, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
    case 5: return launch_bd<K, 5>(bd, in, wd, b, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
    case 10: return launch_bd<K, 10>(bd, in, wd, b, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
    case 20: return launch_bd<K, 20>(bd, in, wd, b, out, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
    default: return false;
  }
}

}  // namespace

// Depthwise KxK + bias + SiLU, padded workspace in and out.  Geometry is
// validated on the Python side; K must be 3 or 7.
void cib_dw_pad(int64_t in, int64_t wd, int64_t bd, int64_t out,
                int64_t C, int64_t H, int64_t W, int64_t PITCH, int64_t PS,
                int64_t VLO, int64_t PAD, int64_t K, int64_t nplanes,
                int64_t PG, int64_t SPAN, int64_t LO, int64_t WT,
                int64_t THREADS) {
  cudaStream_t s = at::cuda::getCurrentCUDAStream();
  const __half *i = reinterpret_cast<const __half *>(in);
  const __half *w = reinterpret_cast<const __half *>(wd);
  const float *b = reinterpret_cast<const float *>(bd);
  __half *o = reinterpret_cast<__half *>(out);
  if (K == 7)
    launch_wt<7>(WT, THREADS, i, w, b, o, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
  else
    launch_wt<3>(WT, THREADS, i, w, b, o, C, H, W, PITCH, PS, VLO, PAD, nplanes, PG, SPAN, LO, s);
}
