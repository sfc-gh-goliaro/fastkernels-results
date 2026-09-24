// Fused depthwise-conv (BN folded) + bias + SiLU for the YOLOv10 RepVGG-DW block.
//
// silu(BN(dw7x7(x)) + BN(dw3x3(x))) collapses, in eval mode, into a single
// depthwise 7x7 convolution with a per-channel bias followed by SiLU: both BNs
// are affine at inference and the 3x3 kernel is the centre of a 7x7 one.  The
// baseline needs six latency-bound launches for it; this is one kernel.
//
// 0.8 MB of data over 148 SMs leaves ~2 warps per scheduler, so the kernel is
// bound by exposed latency, not bandwidth.  Every choice here serves that:
//   * a block stages P (channel-)planes in shared memory with a zero halo and
//     each thread owns one TH x TW output tile, so every weight is reused
//     TH*TW times out of registers;
//   * all shape constants are template parameters: indices fold to immediates,
//     and the global loads for the payload and the weights are issued in the
//     first instructions of the kernel so their latency is absorbed by the
//     halo zeroing and the single barrier that follows;
//   * only the halo cells actually read are zeroed (the row stride is padded
//     past them for bank reasons), which keeps that pass to a few stores;
//   * taps and weight rows are read from shared as float4/float2, and the taps
//     for row ir+1 are issued before row ir is consumed, so shared-load
//     latency overlaps the FFMAs.  SW is chosen to make those vector loads
//     bank-conflict-free: with lane -> (tile row, strip), the banks come out
//     lane-ordered when SW % 32 == W, i.e. SW = 52 for a 20-wide plane.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CDIV(a, b) (((a) + (b) - 1) / (b))

template <int BYTES>
struct Chunk {  // widest power-of-two access for a BYTES-sized packet
  static constexpr int v = (BYTES % 16 == 0) ? 16 : (BYTES % 8 == 0) ? 8
                          : (BYTES % 4 == 0) ? 4 : 2;
};

template <typename T, int N>
__device__ __forceinline__ void copy_pkt(T* dst, const T* src) {
  constexpr int CH = Chunk<N * (int)sizeof(T)>::v;
  char* d = reinterpret_cast<char*>(dst);
  const char* s = reinterpret_cast<const char*>(src);
#pragma unroll
  for (int o = 0; o < N * (int)sizeof(T); o += CH) {
    if (CH == 16) *reinterpret_cast<uint4*>(d + o) = *reinterpret_cast<const uint4*>(s + o);
    else if (CH == 8) *reinterpret_cast<uint2*>(d + o) = *reinterpret_cast<const uint2*>(s + o);
    else if (CH == 4) *reinterpret_cast<uint*>(d + o) = *reinterpret_cast<const uint*>(s + o);
    else *reinterpret_cast<ushort*>(d + o) = *reinterpret_cast<const ushort*>(s + o);
  }
}

// VV consecutive floats at a time out of shared memory.
template <int VV, int N>
__device__ __forceinline__ void load_row(float (&v)[N], const float* p) {
#pragma unroll
  for (int j = 0; j < N; j += VV) {
    if (VV == 4) {
      const float4 t = *reinterpret_cast<const float4*>(p + j);
      v[j] = t.x; v[j + 1] = t.y; v[j + 2] = t.z; v[j + 3] = t.w;
    } else if (VV == 2) {
      const float2 t = *reinterpret_cast<const float2*>(p + j);
      v[j] = t.x; v[j + 1] = t.y;
    } else {
      v[j] = p[j];
    }
  }
}

template <typename T, int KS, int H, int W, int TH, int TW, int SW, int P>
__global__ __launch_bounds__(P*(H / TH) * (W / TW)) void dw_conv_bias_silu_k(
    const T* __restrict__ x, const float* __restrict__ wt,
    const float* __restrict__ bias, T* __restrict__ out, int C) {
  constexpr int PAD = KS / 2;
  constexpr int KK = KS * KS;
  constexpr int KSP = (KS + 3) / 4 * 4;          // weight row stride (float4)
  constexpr int TAPS = TW + KS - 1;
  constexpr int VV = (TW % 4 == 0 && SW % 4 == 0) ? 4
                     : (TW % 2 == 0 && SW % 2 == 0) ? 2 : 1;
  constexpr int VW = CDIV(TAPS, VV) * VV;        // taps per row, VV-rounded
  constexpr int ROWS = TH + KS - 1;              // input rows a tile touches
  constexpr int SPR = W / TW;                    // strips per output row
  constexpr int TPP = (H / TH) * SPR;            // threads per plane
  constexpr int NT = P * TPP;
  constexpr int SH = H + 2 * PAD;
  constexpr int PLANE_IN = SH * SW;
  constexpr int HW = H * W;
  constexpr int VEC = 4;                         // elements per global load
  constexpr int WE = W + 2 * PAD;                // padded width actually read
  constexpr int NZP = 2 * PAD * WE + H * 2 * PAD;  // halo cells per plane
  constexpr int NCHUNK = P * HW / VEC, CPT = CDIV(NCHUNK, NT);
  constexpr int NWT = P * KS * KSP, WPT = CDIV(NWT, NT);
  static_assert(SW >= W - TW + VW, "shared row stride too small");
  static_assert(H % TH == 0 && W % TW == 0 && W % VEC == 0, "bad tiling");

  __shared__ __align__(16) float sx[P * PLANE_IN];
  __shared__ __align__(16) float sw[P * KS * KSP];
  __shared__ float sb[P];

  const int tid = threadIdx.x;
  const int p0 = blockIdx.x * P;
  const int cbase = p0 % C;
  const long long gbase = (long long)p0 * HW;

  // 1) Issue every global load up front.
  T xin[CPT][VEC];
#pragma unroll
  for (int u = 0; u < CPT; ++u) {
    const int i = tid + u * NT;
    if (NCHUNK % NT == 0 || i < NCHUNK)
      copy_pkt<T, VEC>(xin[u], x + gbase + (long long)i * VEC);
  }
  float win[WPT];
#pragma unroll
  for (int u = 0; u < WPT; ++u) {
    const int i = tid + u * NT;
    if (NWT % NT == 0 || i < NWT) {
      const int s = i / (KS * KSP);
      const int rk = i - s * KS * KSP;
      const int r = rk / KSP, k = rk - r * KSP;
      win[u] = k < KS ? wt[(cbase + s) * KK + r * KS + k] : 0.f;
    }
  }
  const float bin = tid < P ? bias[cbase + tid] : 0.f;

  // 2) Zero just the halo cells the tap loads can reach.
#pragma unroll
  for (int i = tid; i < P * NZP; i += NT) {
    const int s = i / NZP;
    int h = i - s * NZP;
    int idx;
    if (h < PAD * WE) {
      const int r = h / WE;
      idx = r * SW + (h - r * WE);
    } else if (h - PAD * WE < PAD * WE) {
      h -= PAD * WE;
      const int r = h / WE;
      idx = (PAD + H + r) * SW + (h - r * WE);
    } else {
      h -= 2 * PAD * WE;
      const int r = h / (2 * PAD), c = h - r * (2 * PAD);
      idx = (PAD + r) * SW + (c < PAD ? c : W + c);
    }
    sx[s * PLANE_IN + idx] = 0.f;
  }

  // 3) Land the payload / weights and synchronise once.
#pragma unroll
  for (int u = 0; u < CPT; ++u) {
    const int i = tid + u * NT;
    if (NCHUNK % NT == 0 || i < NCHUNK) {
      const int s = i / (HW / VEC);
      const int rc = i - s * (HW / VEC);
      const int row = rc / (W / VEC);
      float* d = sx + s * PLANE_IN + (row + PAD) * SW + (rc - row * (W / VEC)) * VEC + PAD;
#pragma unroll
      for (int j = 0; j < VEC; ++j) d[j] = static_cast<float>(xin[u][j]);
    }
  }
#pragma unroll
  for (int u = 0; u < WPT; ++u) {
    const int i = tid + u * NT;
    if (NWT % NT == 0 || i < NWT) sw[i] = win[u];
  }
  if (tid < P) sb[tid] = bin;
  __syncthreads();

  // 4) Convolve.
  const int slot = tid / TPP;
  const int rem = tid - slot * TPP;
  const int oh0 = (rem / SPR) * TH;
  const int ow0 = (rem - rem / SPR * SPR) * TW;
  const float* sp = sx + slot * PLANE_IN + oh0 * SW + ow0;

  float wr[KS][KSP];
#pragma unroll
  for (int r = 0; r < KS; ++r)
    load_row<4>(wr[r], sw + slot * KS * KSP + r * KSP);

  float acc[TH][TW];
  const float bb = sb[slot];
#pragma unroll
  for (int t = 0; t < TH; ++t)
#pragma unroll
    for (int j = 0; j < TW; ++j) acc[t][j] = bb;

  float v[2][VW];
  load_row<VV>(v[0], sp);
#pragma unroll
  for (int ir = 0; ir < ROWS; ++ir) {
    if (ir + 1 < ROWS) load_row<VV>(v[(ir + 1) & 1], sp + (ir + 1) * SW);
#pragma unroll
    for (int r = 0; r < KS; ++r) {
      const int t = ir - r;
      if (t >= 0 && t < TH) {
#pragma unroll
        for (int kx = 0; kx < KS; ++kx) {
#pragma unroll
          for (int j = 0; j < TW; ++j)
            acc[t][j] = fmaf(wr[r][kx], v[ir & 1][j + kx], acc[t][j]);
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TH; ++t) {
    T res[TW];
#pragma unroll
    for (int j = 0; j < TW; ++j) {
      const float vv = acc[t][j];
      res[j] = static_cast<T>(vv * (1.f / (1.f + __expf(-vv))));
    }
    copy_pkt<T, TW>(out + gbase + (long long)slot * HW + (oh0 + t) * W + ow0, res);
  }
}

template <typename T, int KS, int H, int W, int TH, int TW, int SW, int P>
static void launch(const torch::Tensor& x, const torch::Tensor& w,
                   const torch::Tensor& b, torch::Tensor& out) {
  const int C = x.size(1);
  const int NC = x.size(0) * C;
  auto stream = at::cuda::getCurrentCUDAStream();
  dw_conv_bias_silu_k<T, KS, H, W, TH, TW, SW, P>
      <<<NC / P, P * (H / TH) * (W / TW), 0, stream>>>(
          x.data_ptr<T>(), w.data_ptr<float>(), b.data_ptr<float>(),
          out.data_ptr<T>(), C);
}

// One output row-strip of 4 per thread, one plane per block, shared row stride
// 52 (bank-conflict-free float4 taps).  Picked by sweeping TH/TW/SW/P against
// the benchmark harness on a B200; the alternatives cost 1-12% geomean.
#define FK_LAUNCH(T) launch<T, 7, 20, 20, 1, 4, 52, 1>(x, w, b, out)

void dw_conv_bias_silu(const torch::Tensor& x, const torch::Tensor& w,
                       const torch::Tensor& b, torch::Tensor out) {
  TORCH_CHECK(x.dim() == 4 && out.dim() == 4, "expected NCHW");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "expected contiguous");
  TORCH_CHECK(w.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
              "weight/bias must be float32");
  TORCH_CHECK(x.size(2) == 20 && x.size(3) == 20, "unsupported spatial size");
  const c10::cuda::CUDAGuard guard(x.device());
  if (x.scalar_type() == at::kHalf) {
    FK_LAUNCH(at::Half);
  } else if (x.scalar_type() == at::kBFloat16) {
    FK_LAUNCH(at::BFloat16);
  } else if (x.scalar_type() == at::kFloat) {
    FK_LAUNCH(float);
  } else {
    TORCH_CHECK(false, "unsupported dtype");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dw_conv_bias_silu", &dw_conv_bias_silu,
        "fused depthwise 7x7 conv + bias + SiLU");
}
