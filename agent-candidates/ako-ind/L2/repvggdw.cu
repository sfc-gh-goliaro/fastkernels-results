// Single-launch fused depthwise block for YOLOv10 RepVGGDW.
//
//   out[n,c,y,x] = silu( bias[c] + sum_{ky,kx} W[c,ky,kx] * x[n,c,y+ky-3,x+kx-3] )
//
// W/bias already carry both branches (the 7x7 and the zero-padded 3x3) with
// their BatchNorms folded in, so a single launch reads x exactly once: each CTA
// stages zero-padded (n,c) planes in shared memory, holds that channel's taps
// and bias in registers, and writes the activated fp16 result.  No intermediate
// tensors, no separate add/activation launch.
//
// Two kernel families:
//
//  * ``dw_fused_kernel``  -- one plane per thread-group, fp32 shared, fp32
//    accumulation.  This is round 1's kernel; it is the numerically widest path
//    and the fallback whenever the pair kernel's guards do not hold.
//
//  * ``dw_pair_kernel``   -- the working horse.  One CTA owns *two* adjacent
//    channels of the same image; shared memory holds the two planes interleaved
//    as ``__half2`` so a single LDS.32 feeds two planes, and the 49 taps are
//    ``__half2`` pairs so a single HFMA2 advances two planes.  That halves both
//    the shared-load count and the dependent-FMA count per output pixel, which
//    is what this kernel is actually charged for: ncu reports 2% memory / 18.7%
//    SM throughput at 0.99 waves/SM, i.e. one wave with nothing saturated, so
//    only the per-CTA instruction chain length matters.  It also halves the
//    cold tap bytes (fp16 taps, 26.6 KB instead of 53 KB) and drops one of the
//    two staging barriers.
//
// Staging (``NEWSTG`` / the pair kernel): the plane is padded 4 on the left and
// 3 on the top/bottom, so each shared row is a whole number of 16 B words *and*
// the interior starts 16 B-aligned.  Only the halo is zeroed -- those cells are
// exactly the ones the interior fill does not write -- so the zero and the fill
// are disjoint and ONE ``__syncthreads`` suffices.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define TAP_STRIDE 52    // fp32 taps per channel:      49 + 3 pad -> 13 x 128b
#define TAP_STRIDE_P 52  // half2 taps per channel pair: 49 + 3 pad -> 13 x 128b

namespace {

// Direction item (e): taps in __constant__ so the warp-uniform tap reads become
// LDC through the uniform datapath instead of 13 LDG + 52 live registers.  One
// buffer is shared by the whole process, so the host copy is keyed on an owner
// pointer.  Sized for C <= 512.
#define CTAP_MAX_PAIRS 256
__constant__ __half2 g_ctaps[CTAP_MAX_PAIRS * TAP_STRIDE_P];

__device__ __forceinline__ unsigned as_u(__half2 v) {
  return *reinterpret_cast<const unsigned *>(&v);
}
__device__ __forceinline__ __half2 as_h2(unsigned v) {
  return *reinterpret_cast<const __half2 *>(&v);
}

// silu(v) = v / (1 + exp(-v)), two pixels at a time.
//
// The obvious half2 form (``v * h2rcp(1 + h2exp(-v))``) costs ~9 SASS
// instructions per output: each intrinsic expands to two MUFU plus HSET2
// special-case handling, and the reciprocal adds a second MUFU pair.  Going
// through the half-angle identity instead --
//   sigmoid(v) = (1 + tanh(v/2)) / 2   =>   silu(v) = h + h*tanh(h),  h = v/2
// -- makes the whole activation one HMUL2, one hardware ``tanh.approx.f16x2``
// (a single instruction on sm_75+) and one HFMA2: 1.5 per output.  It is also
// right in the limits (tanh -> +-1 gives silu -> v and silu -> 0).
__device__ __forceinline__ __half2 silu2_tanh(__half2 v) {
  const __half2 h = __hmul2(v, __float2half2_rn(0.5f));
  unsigned t;
  asm("tanh.approx.f16x2 %0, %1;" : "=r"(t) : "r"(as_u(h)));
  return __hfma2(h, as_h2(t), h);
}
__device__ __forceinline__ __half2 silu2_exp(__half2 v) {
  const __half2 one = __float2half2_rn(1.f);
  return __hmul2(v, h2rcp(__hadd2(one, h2exp(__hneg2(v)))));
}
__device__ __forceinline__ __half2 silu2_f32(__half2 v) {
  float a = __low2float(v), b = __high2float(v);
  a *= __frcp_rn(1.f + __expf(-a));
  b *= __frcp_rn(1.f + __expf(-b));
  return __floats2half2_rn(a, b);
}

// ==========================================================================
// Round-1 kernel: one plane per thread group, fp32 shared + fp32 accumulate.
// ==========================================================================
// H,W    : plane geometry (compile time, so all index math folds to multiplies)
// P      : (n,c) planes handled by one CTA
// RT,CT  : output rows / cols computed by one thread
// TAPSM  : 0 = taps in registers (13x128-bit loads), 1 = staged in shared.
//          Measured: (1) costs 0.07-0.26 us more on every shape even though it
//          drops ~47 registers/thread.  Kept for the record only.
// NEWSTG : 0 = zero the whole padded plane, barrier, fill interior, barrier.
//          1 = zero only the halo and fill the interior: disjoint writes, so
//          one barrier and 400 fewer shared stores per plane.
template <int H, int W, int P, int RT, int CT, int TAPSM, int NEWSTG>
__global__ __launch_bounds__(P *((H / RT) * (W / CT))) void dw_fused_kernel(
    const __half *__restrict__ x, const float *__restrict__ wt,
    const float *__restrict__ bs, __half *__restrict__ out, int NC, int C) {
  static_assert(H % RT == 0 && W % CT == 0, "tile must divide the plane");
  static_assert(W % 4 == 0, "vectorized staging needs W divisible by 4");

  constexpr int LP = NEWSTG ? 4 : 3;         // left pad
  constexpr int PH = H + 6;                  // padded rows
  constexpr int SW = ((LP + W + 3 + 3) / 4) * 4;  // row stride, 16B multiple
  constexpr int PLANE = PH * SW;
  constexpr int TX = W / CT, TY = H / RT;
  constexpr int TPP = TX * TY;               // threads per plane
  constexpr int NT = P * TPP;                // block size
  constexpr int HW = H * W;
  constexpr int NTAP = TAPSM ? P * TAP_STRIDE : 1;
  constexpr int TITER = (NTAP + NT - 1) / NT;

  __shared__ float sm[P * PLANE + NTAP];
  float *const stap = sm + P * PLANE;

  const int tid = threadIdx.x;
  const int p = tid / TPP;
  const int g = blockIdx.x * P + p;   // flat (n,c) plane of this thread
  const bool live = g < NC;
  const int c = live ? g - (g / C) * C : 0;

  // Taps + bias first: they miss L2 on every call, so overlap the round trip
  // with the staging below.
  float4 w4[TAPSM ? 1 : 13];
  if constexpr (!TAPSM) {
    const float4 *wp4 =
        reinterpret_cast<const float4 *>(wt + (size_t)c * TAP_STRIDE);
#pragma unroll
    for (int k = 0; k < 13; ++k) w4[k] = wp4[k];
  }
  const float bv = bs[c];

  if constexpr (TAPSM) {  // one coalesced pass for the whole CTA's taps
#pragma unroll
    for (int k = 0; k < TITER; ++k) {
      const int i = tid + k * NT;
      if (TITER * NT == NTAP || i < NTAP) {
        const int pp = i / TAP_STRIDE;
        const int gg = blockIdx.x * P + pp;
        const int cc = gg < NC ? gg - (gg / C) * C : 0;
        stap[i] = wt[(size_t)cc * TAP_STRIDE + (i - pp * TAP_STRIDE)];
      }
    }
  }

  if constexpr (!NEWSTG) {
    constexpr int H2ROW = W / 2;             // half2 per input row
    constexpr int NZ = (P * PLANE) / 4;      // float4 to zero
    constexpr int NL = P * H * H2ROW;        // half2 to stage
    constexpr int ZITER = (NZ + NT - 1) / NT;
    constexpr int LITER = (NL + NT - 1) / NT;
    {  // zero the padded plane(s)
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 *s4 = reinterpret_cast<float4 *>(sm);
#pragma unroll
      for (int k = 0; k < ZITER; ++k) {
        const int i = tid + k * NT;
        if (ZITER * NT == NZ || i < NZ) s4[i] = z;
      }
    }
    __syncthreads();
    {  // stage the interior, half2 at a time (coalesced across the CTA)
#pragma unroll
      for (int k = 0; k < LITER; ++k) {
        const int i = tid + k * NT;
        if (LITER * NT == NL || i < NL) {
          const int pp = i / (H * H2ROW);
          const int q = i - pp * (H * H2ROW);
          const int r = q / H2ROW;
          const int c2 = q - r * H2ROW;
          const int gg = blockIdx.x * P + pp;
          if (gg < NC) {
            const __half2 v = reinterpret_cast<const __half2 *>(
                x + (size_t)gg * HW + r * W)[c2];
            float *d = &sm[pp * PLANE + (r + 3) * SW + 2 * c2 + LP];
            d[0] = __low2float(v);
            d[1] = __high2float(v);
          }
        }
      }
    }
  } else {
    constexpr int RW = SW / 4;               // uint4 (float4) per shared row
    constexpr int NCH = W / 4;               // 4-pixel chunks per input row
    constexpr int NZ = P * (6 * RW + H * 2);
    constexpr int NDATA = P * H * NCH;
    constexpr int ZITER = (NZ + NT - 1) / NT;
    constexpr int DITER = (NDATA + NT - 1) / NT;
    float4 *const s4 = reinterpret_cast<float4 *>(sm);
    {  // halo only: the cells the interior fill does not write
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int k = 0; k < ZITER; ++k) {
        int i = tid + k * NT;
        if (ZITER * NT == NZ || i < NZ) {
          const int pp = i / (6 * RW + H * 2);
          i -= pp * (6 * RW + H * 2);
          int row, cw;
          if (i < 6 * RW) {
            const int u = i / RW;
            row = u < 3 ? u : u + H;
            cw = i - u * RW;
          } else {
            const int j = i - 6 * RW;
            row = 3 + (j >> 1);
            cw = (j & 1) ? RW - 1 : 0;
          }
          s4[pp * (PH * RW) + row * RW + cw] = z;
        }
      }
    }
    {  // interior: 4 pixels (one 8B global read) -> one 16B shared store
#pragma unroll
      for (int k = 0; k < DITER; ++k) {
        const int i = tid + k * NT;
        if (DITER * NT == NDATA || i < NDATA) {
          const int pp = i / (H * NCH);
          const int j = i - pp * (H * NCH);
          const int r = j / NCH, cc = j - r * NCH;
          const int gg = blockIdx.x * P + pp;
          if (gg < NC) {
            const uint2 v = *reinterpret_cast<const uint2 *>(
                x + (size_t)gg * HW + r * W + 4 * cc);
            const __half2 lo = as_h2(v.x), hi = as_h2(v.y);
            s4[pp * (PH * RW) + (r + 3) * RW + 1 + cc] =
                make_float4(__low2float(lo), __high2float(lo), __low2float(hi),
                            __high2float(hi));
          }
        }
      }
    }
  }
  __syncthreads();
  if (!live) return;

  const int t = tid - p * TPP;
  const int ty = t / TX;
  const int y0 = ty * RT, x0 = (t - ty * TX) * CT;

  float acc[RT][CT];
#pragma unroll
  for (int j = 0; j < RT; ++j)
#pragma unroll
    for (int i = 0; i < CT; ++i) acc[j][i] = bv;

  const float *sp = &sm[p * PLANE + y0 * SW + x0 + (LP - 3)];
#pragma unroll
  for (int r = 0; r < RT + 6; ++r) {
    float in[CT + 6];
#pragma unroll
    for (int i = 0; i < CT + 6; ++i) in[i] = sp[r * SW + i];
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      const int ky = r - j;
      if (ky >= 0 && ky < 7) {
#pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
          const int k = ky * 7 + kx;
          float wv;
          if constexpr (TAPSM) {
            wv = stap[p * TAP_STRIDE + k];
          } else {
            const float4 v = w4[k >> 2];
            wv = (k & 3) == 0 ? v.x
                 : (k & 3) == 1 ? v.y
                 : (k & 3) == 2 ? v.z
                                : v.w;
          }
#pragma unroll
          for (int i = 0; i < CT; ++i) acc[j][i] += wv * in[i + kx];
        }
      }
    }
  }

  __half *op = out + (size_t)g * HW + y0 * W + x0;
#pragma unroll
  for (int j = 0; j < RT; ++j)
#pragma unroll
    for (int i = 0; i < CT; ++i) {
      const float v = acc[j][i];
      op[j * W + i] = __float2half(v * __frcp_rn(1.f + __expf(-v)));
    }
}

// ==========================================================================
// Pair kernel: two adjacent channels per CTA, packed into half2 lanes.
// ==========================================================================
// One CTA owns planes (g0, g0+1) = channels (2*cp, 2*cp+1) of one image.
// Shared memory holds them interleaved -- ``sm[row][col]`` is the ``__half2``
// (plane0 pixel, plane1 pixel) -- so one LDS.32 and one HFMA2 serve both.
//
// Store: the accumulator holds one pixel of *each* plane, so the two halves go
// to different rows of ``out``; ``__byte_perm`` regroups CT accumulators into
// contiguous per-plane words so each plane still gets wide stores.
template <int CT>
__device__ __forceinline__ void store_pair_row(__half *pa, __half *pb,
                                               const __half2 (&s)[CT]) {
  if constexpr (CT % 4 == 0) {
#pragma unroll
    for (int i = 0; i < CT; i += 4) {
      const unsigned s0 = as_u(s[i]), s1 = as_u(s[i + 1]);
      const unsigned s2 = as_u(s[i + 2]), s3 = as_u(s[i + 3]);
      uint2 wa, wb;
      wa.x = __byte_perm(s0, s1, 0x5410);
      wa.y = __byte_perm(s2, s3, 0x5410);
      wb.x = __byte_perm(s0, s1, 0x7632);
      wb.y = __byte_perm(s2, s3, 0x7632);
      *reinterpret_cast<uint2 *>(pa + i) = wa;
      *reinterpret_cast<uint2 *>(pb + i) = wb;
    }
  } else if constexpr (CT % 2 == 0) {
#pragma unroll
    for (int i = 0; i < CT; i += 2) {
      const unsigned s0 = as_u(s[i]), s1 = as_u(s[i + 1]);
      *reinterpret_cast<unsigned *>(pa + i) = __byte_perm(s0, s1, 0x5410);
      *reinterpret_cast<unsigned *>(pb + i) = __byte_perm(s0, s1, 0x7632);
    }
  } else {
#pragma unroll
    for (int i = 0; i < CT; ++i) {
      pa[i] = __low2half(s[i]);
      pb[i] = __high2half(s[i]);
    }
  }
}

// WIDE: read each shared row as 128-bit chunks instead of one LDS.32 per
// element.  The tile's own column origin x0 = tx*CT is 16B-aligned whenever
// CT % 4 == 0, and the row stride is a multiple of 4 half2, so a thread can pull
// its whole (CT+7)-element window as ceil((CT+7)/4) LDS.128 -- 3 instructions
// instead of 10 for CT=4.  Same bytes, same LSU cycles, a third of the issue
// slots, which is what a kernel stalling 72% of cycles is actually short of.
template <int H, int W, int P, int RT, int CT, int SILU, int WIDE = 0,
          int CTAP = 0>
__global__ __launch_bounds__(P *((H / RT) * (W / CT))) void dw_pair_kernel(
    const __half *__restrict__ x, const __half2 *__restrict__ wt,
    const __half2 *__restrict__ bs, __half *__restrict__ out, int C) {
  static_assert(H % RT == 0 && W % CT == 0, "tile must divide the plane");
  static_assert(W % 4 == 0, "staging moves 4 pixels at a time");
  static_assert(!WIDE || CT % 4 == 0, "128-bit row loads need CT % 4 == 0");

  constexpr int LP = 4;                            // 16B-aligned interior
  constexpr int PH = H + 6;
  constexpr int SW = ((LP + W + 3 + 3) / 4) * 4;   // 16B-aligned rows
  constexpr int RW = SW / 4;                       // uint4 per shared row
  constexpr int TX = W / CT, TY = H / RT;
  constexpr int TPP = TX * TY;                     // threads per pair
  constexpr int NT = P * TPP;
  constexpr int HW = H * W;
  constexpr int NCH = W / 4;
  constexpr int NZ = P * (6 * RW + H * 2);         // halo uint4 stores
  constexpr int NDATA = P * H * NCH;               // interior uint4 stores
  constexpr int ZITER = (NZ + NT - 1) / NT;
  constexpr int DITER = (NDATA + NT - 1) / NT;

  __shared__ __half2 sm[P * PH * SW];
  uint4 *const s4 = reinterpret_cast<uint4 *>(sm);

  // The grid is 2D -- x over channel pairs, y over images -- so the channel
  // pair index is a pure blockIdx expression.  A flat 1D grid would need
  // ``g0 % C``, i.e. a *runtime* integer division, which costs ~20 SASS
  // instructions in a kernel whose whole body is only ~600.
  const int pp0 = threadIdx.x / TPP;               // which pair this thread owns
  const int cp = blockIdx.x * P + pp0;             // channel-pair index
  const int g0 = blockIdx.y * C + 2 * cp;          // first plane of that pair

  // 52 half2 = 13 x 128-bit, issued before the staging so the DRAM round trip
  // (taps are evicted before every benched call) overlaps it.
  uint4 tr[CTAP ? 1 : 13];
  if constexpr (!CTAP) {
    const uint4 *wp =
        reinterpret_cast<const uint4 *>(wt + (size_t)cp * TAP_STRIDE_P);
#pragma unroll
    for (int k = 0; k < 13; ++k) tr[k] = wp[k];
  }
  const __half2 bv = bs[cp];

  const int tid = threadIdx.x;
  {  // halo only -> disjoint from the interior fill -> one barrier
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int k = 0; k < ZITER; ++k) {
      int i = tid + k * NT;
      if (ZITER * NT == NZ || i < NZ) {
        const int pz = i / (6 * RW + H * 2);
        i -= pz * (6 * RW + H * 2);
        int row, cw;
        if (i < 6 * RW) {
          const int u = i / RW;
          row = u < 3 ? u : u + H;
          cw = i - u * RW;
        } else {
          const int j = i - 6 * RW;
          row = 3 + (j >> 1);
          cw = (j & 1) ? RW - 1 : 0;
        }
        s4[pz * (PH * RW) + row * RW + cw] = z;
      }
    }
  }
  {  // 4 pixels of both planes: two 8B reads -> one interleaved 16B store
#pragma unroll
    for (int k = 0; k < DITER; ++k) {
      const int i = tid + k * NT;
      if (DITER * NT == NDATA || i < NDATA) {
        const int pd = i / (H * NCH);
        const int jd = i - pd * (H * NCH);
        const int r = jd / NCH, cc = jd - r * NCH;
        const int gd = blockIdx.y * C + 2 * (blockIdx.x * P + pd);
        const __half *pa = x + (size_t)gd * HW + r * W + 4 * cc;
        const uint2 a = *reinterpret_cast<const uint2 *>(pa);
        const uint2 b = *reinterpret_cast<const uint2 *>(pa + HW);
        uint4 v;
        v.x = __byte_perm(a.x, b.x, 0x5410);
        v.y = __byte_perm(a.x, b.x, 0x7632);
        v.z = __byte_perm(a.y, b.y, 0x5410);
        v.w = __byte_perm(a.y, b.y, 0x7632);
        s4[pd * (PH * RW) + (r + 3) * RW + 1 + cc] = v;
      }
    }
  }
  __syncthreads();

  const int t = tid - pp0 * TPP;
  const int ty = t / TX;
  const int y0 = ty * RT, x0 = (t - ty * TX) * CT;

  __half2 acc[RT][CT];
#pragma unroll
  for (int j = 0; j < RT; ++j)
#pragma unroll
    for (int i = 0; i < CT; ++i) acc[j][i] = bv;

  constexpr int NW = (CT + 6 + (LP - 3) + 3) / 4;   // 128-bit chunks per row
  const __half2 *sp = &sm[pp0 * (PH * SW) + y0 * SW + x0 + (LP - 3)];
#pragma unroll
  for (int r = 0; r < RT + 6; ++r) {
    __half2 in[CT + 6];
    if constexpr (WIDE) {
      // x0 = tx*CT is a multiple of 4 half2 only when CT % 4 == 0, which the
      // static_assert above guarantees; form the uint4 view only here so no
      // over-aligned pointer is ever built for the narrow configs.
      const uint4 *sp4 =
          reinterpret_cast<const uint4 *>(&sm[pp0 * (PH * SW) + y0 * SW + x0]);
      uint4 raw[NW];
#pragma unroll
      for (int k = 0; k < NW; ++k) raw[k] = sp4[r * (SW / 4) + k];
#pragma unroll
      for (int i = 0; i < CT + 6; ++i) {
        const int j = i + (LP - 3);               // element index inside raw
        const uint4 v = raw[j >> 2];
        in[i] = as_h2((j & 3) == 0 ? v.x
                      : (j & 3) == 1 ? v.y
                      : (j & 3) == 2 ? v.z
                                     : v.w);
      }
    } else {
#pragma unroll
      for (int i = 0; i < CT + 6; ++i) in[i] = sp[r * SW + i];
    }
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      const int ky = r - j;
      if (ky >= 0 && ky < 7) {
#pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
          const int k = ky * 7 + kx;
          __half2 wv;
          if constexpr (CTAP) {
            wv = g_ctaps[cp * TAP_STRIDE_P + k];
          } else {
            const uint4 v = tr[k >> 2];
            wv = as_h2((k & 3) == 0 ? v.x
                       : (k & 3) == 1 ? v.y
                       : (k & 3) == 2 ? v.z
                                      : v.w);
          }
#pragma unroll
          for (int i = 0; i < CT; ++i)
            acc[j][i] = __hfma2(wv, in[i + kx], acc[j][i]);
        }
      }
    }
  }

  __half *op = out + (size_t)g0 * HW + y0 * W + x0;
#pragma unroll
  for (int j = 0; j < RT; ++j) {
    __half2 s[CT];
#pragma unroll
    for (int i = 0; i < CT; ++i)
      s[i] = SILU == 0 ? silu2_tanh(acc[j][i])
             : SILU == 1 ? silu2_f32(acc[j][i])
                         : silu2_exp(acc[j][i]);
    store_pair_row<CT>(op + j * W, op + j * W + HW, s);
  }
}

// ==========================================================================
// Structural alternative: no shared memory at all.
// ==========================================================================
// Reads x straight from global with __ldg and lets L1/L2 absorb the 49x halo
// re-read.  This deletes both staging barriers and every STS/LDS from the chain,
// which is the one thing a one-wave latency-bound kernel might want; the price
// is one predicated LDG per input element instead of one LDS per element plus
// amortized staging.  fp32 accumulate, one plane per thread group.
template <int H, int W, int RT, int CT>
__global__ __launch_bounds__((H / RT) * (W / CT)) void dw_global_kernel(
    const __half *__restrict__ x, const float *__restrict__ wt,
    const float *__restrict__ bs, __half *__restrict__ out, int C) {
  constexpr int TX = W / CT, HW = H * W;
  const int c = blockIdx.x;
  const int g = blockIdx.y * C + c;

  float4 w4[13];
  {
    const float4 *wp =
        reinterpret_cast<const float4 *>(wt + (size_t)c * TAP_STRIDE);
#pragma unroll
    for (int k = 0; k < 13; ++k) w4[k] = wp[k];
  }
  const float bv = bs[c];

  const int ty = threadIdx.x / TX;
  const int y0 = ty * RT, x0 = (threadIdx.x - ty * TX) * CT;

  float acc[RT][CT];
#pragma unroll
  for (int j = 0; j < RT; ++j)
#pragma unroll
    for (int i = 0; i < CT; ++i) acc[j][i] = bv;

  const __half *xp = x + (size_t)g * HW;
#pragma unroll
  for (int r = 0; r < RT + 6; ++r) {
    const int iy = y0 + r - 3;
    const bool ry = (unsigned)iy < (unsigned)H;
    float in[CT + 6];
#pragma unroll
    for (int i = 0; i < CT + 6; ++i) {
      const int ix = x0 + i - 3;
      in[i] = (ry && (unsigned)ix < (unsigned)W)
                  ? __half2float(__ldg(xp + iy * W + ix))
                  : 0.f;
    }
#pragma unroll
    for (int j = 0; j < RT; ++j) {
      const int ky = r - j;
      if (ky >= 0 && ky < 7) {
#pragma unroll
        for (int kx = 0; kx < 7; ++kx) {
          const int k = ky * 7 + kx;
          const float4 v = w4[k >> 2];
          const float wv = (k & 3) == 0 ? v.x
                           : (k & 3) == 1 ? v.y
                           : (k & 3) == 2 ? v.z
                                          : v.w;
#pragma unroll
          for (int i = 0; i < CT; ++i) acc[j][i] += wv * in[i + kx];
        }
      }
    }
  }

  __half *op = out + (size_t)g * HW + y0 * W + x0;
#pragma unroll
  for (int j = 0; j < RT; ++j)
#pragma unroll
    for (int i = 0; i < CT; ++i) {
      const float v = acc[j][i];
      op[j * W + i] = __float2half(v * __frcp_rn(1.f + __expf(-v)));
    }
}

// ==========================================================================
// Launch / dispatch
// ==========================================================================
template <int H, int W, int P, int RT, int CT, int TAPSM, int NEWSTG>
inline void launch_scalar(const __half *x, const float *w, const float *b,
                          __half *o, int NC, int C, cudaStream_t s) {
  constexpr int NT = P * ((H / RT) * (W / CT));
  dw_fused_kernel<H, W, P, RT, CT, TAPSM, NEWSTG>
      <<<(NC + P - 1) / P, NT, 0, s>>>(x, w, b, o, NC, C);
}

template <int H, int W, int P, int RT, int CT, int SILU, int WIDE, int CTAP>
inline void launch_pair(const __half *x, const __half2 *w, const __half2 *b,
                        __half *o, int NC, int C, cudaStream_t s) {
  constexpr int NT = P * ((H / RT) * (W / CT));
  if constexpr (CTAP) {
    static const void *owner = nullptr;   // one buffer per process
    if (owner != (const void *)w) {
      C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
          g_ctaps, w, (size_t)(C / 2) * TAP_STRIDE_P * sizeof(__half2), 0,
          cudaMemcpyDeviceToDevice, s));
      owner = (const void *)w;
    }
  }
  dw_pair_kernel<H, W, P, RT, CT, SILU, WIDE, CTAP>
      <<<dim3(C / (2 * P), NC / C), NT, 0, s>>>(x, w, b, o, C);
}

template <int H, int W, int RT, int CT>
inline void launch_global(const __half *x, const float *w, const float *b,
                          __half *o, int NC, int C, cudaStream_t s) {
  constexpr int NT = (H / RT) * (W / CT);
  dw_global_kernel<H, W, RT, CT><<<dim3(C, NC / C), NT, 0, s>>>(x, w, b, o, C);
}

// Compiled configs.  13 is what ships on both captured shapes; the rest are the
// attribution points behind that choice, kept so the measurements in
// ITERATIONS.md can be reproduced without re-deriving them:
//
//   0,2   round-1 scalar kernel (fp32 shared + fp32 accumulate), the fallback
//         auto-picks when the pair layout is unusable
//   5     scalar kernel with the halo-only single-barrier staging -- isolates
//         that change from the half2 packing (worth ~0.1 us on its own)
//   11,13 pair kernel, 1x4 and 2x2 tiles; 13 is fastest on both shapes
//   22    pair kernel with 2 pairs per CTA -- fewer, fatter CTAs
//   16,31 pair kernel with the fp32 and h2exp SiLU -- isolate tanh.approx
//   45    pair kernel with taps in __constant__: measured 0.2-0.3 us *faster*
//         than register taps, deliberately NOT shipped (one buffer is shared
//         process-wide, so several block instances would re-copy 26 KB per
//         forward, and the reported time is already in its floor bucket)
//   50    no-shared-memory variant: reads x from global with __ldg, no barrier
//         at all -- 20-30% slower, the 49x halo re-read costs more than the
//         barriers save
#define CFG_LIST                    \
  CFG_S(0, 1, 1, 5, 0, 0)           \
  CFG_S(2, 1, 1, 4, 0, 0)           \
  CFG_S(5, 1, 1, 5, 0, 1)           \
  CFG_P(11, 1, 1, 4, 0, 0, 0)       \
  CFG_P(13, 1, 2, 2, 0, 0, 0)       \
  CFG_P(22, 2, 1, 5, 0, 0, 0)       \
  CFG_P(16, 1, 1, 4, 1, 0, 0)       \
  CFG_P(31, 2, 1, 5, 2, 0, 0)       \
  CFG_P(45, 1, 1, 4, 0, 0, 1)       \
  CFG_G(50, 1, 5)

#define CFG_MAX_ID 50

#ifndef CFG_AUTO_SMALL
#define CFG_AUTO_SMALL 2  // 256 planes: 1x4 tiles, 100 thr/plane
#endif
#ifndef CFG_AUTO_LARGE
#define CFG_AUTO_LARGE 0  // 1024 planes: 1x5 tiles, 80 thr/plane
#endif
#ifndef CFG_AUTO_SMALL_P
#define CFG_AUTO_SMALL_P 13
#endif
#ifndef CFG_AUTO_LARGE_P
#define CFG_AUTO_LARGE_P 13
#endif

bool is_pair_cfg(int64_t cfg) { return cfg >= 10 && cfg < 50; }

// Pairs per CTA of each pair config (NC must be divisible by 2*P).
int pair_P(int64_t cfg) {
  switch (cfg) {
#define CFG_S(i, P, RT, CT, TAPSM, NEWSTG)
#define CFG_P(i, P, RT, CT, SILU, WIDE, CTAP) \
  case i:                                     \
    return P;
#define CFG_G(i, RT, CT) \
  case i:                \
    return 1;
    CFG_LIST
#undef CFG_S
#undef CFG_P
#undef CFG_G
    default:
      return 0;
  }
}

void dispatch(int64_t cfg, const __half *xp, const float *wp, const float *bp,
              const __half2 *w2, const __half2 *b2, __half *op, int NC, int C,
              cudaStream_t s) {
  switch (cfg) {
#define CFG_S(id, P, RT, CT, TAPSM, NEWSTG)                                 \
  case id:                                                                  \
    launch_scalar<20, 20, P, RT, CT, TAPSM, NEWSTG>(xp, wp, bp, op, NC, C, s); \
    return;
#define CFG_P(id, P, RT, CT, SILU, WIDE, CTAP)                               \
  case id:                                                                  \
    launch_pair<20, 20, P, RT, CT, SILU, WIDE, CTAP>(xp, w2, b2, op, NC, C,  \
                                                     s);                     \
    return;
#define CFG_G(id, RT, CT)                                       \
  case id:                                                      \
    launch_global<20, 20, RT, CT>(xp, wp, bp, op, NC, C, s);      \
    return;
    CFG_LIST
#undef CFG_S
#undef CFG_P
#undef CFG_G
    default:
      TORCH_CHECK(false, "unknown config id ", cfg);
  }
}

at::Tensor run(const at::Tensor &x, const at::Tensor &w, const at::Tensor &b,
               const at::Tensor &w2, const at::Tensor &b2, int64_t cfg) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 4 &&
                  x.is_contiguous(),
              "fast path needs a contiguous fp16 NCHW cuda tensor");
  const int N = x.size(0), C = x.size(1);
  TORCH_CHECK(x.size(2) == 20 && x.size(3) == 20,
              "fast path is specialized for 20x20");
  TORCH_CHECK(w.is_cuda() && b.is_cuda() && w.scalar_type() == at::kFloat &&
                  b.scalar_type() == at::kFloat && w.is_contiguous() &&
                  b.is_contiguous() && w.numel() == (int64_t)C * TAP_STRIDE &&
                  b.numel() == C,
              "taps/bias must be contiguous fp32 cuda tensors of [C,",
              TAP_STRIDE, "] / [C]");
  const int NC = N * C;
  // The pair kernel needs an even channel count (pairs never straddle an
  // image) and its own fp16 tap/bias layout; Python leaves w2/b2 empty when the
  // taps are too large for fp16 accumulation to be safe.
  const bool pair_ok =
      (C % 2 == 0) && w2.defined() && b2.defined() && w2.numel() > 0 &&
      w2.is_cuda() && b2.is_cuda() && w2.scalar_type() == at::kHalf &&
      b2.scalar_type() == at::kHalf && w2.is_contiguous() &&
      b2.is_contiguous() && w2.numel() == (int64_t)C * TAP_STRIDE_P &&
      b2.numel() == C;
  if (cfg < 0) {
    if (pair_ok && C % (2 * pair_P(NC <= 256 ? CFG_AUTO_SMALL_P
                                             : CFG_AUTO_LARGE_P)) == 0)
      cfg = NC <= 256 ? CFG_AUTO_SMALL_P : CFG_AUTO_LARGE_P;
    else
      cfg = NC <= 256 ? CFG_AUTO_SMALL : CFG_AUTO_LARGE;
  } else {
    TORCH_CHECK(!is_pair_cfg(cfg) || (pair_ok && C % (2 * pair_P(cfg)) == 0),
                "pair config requested but unusable for this shape");
  }
  at::Tensor out = at::empty_like(x);
  dispatch(cfg, reinterpret_cast<const __half *>(x.data_ptr()),
           w.data_ptr<float>(), b.data_ptr<float>(),
           pair_ok ? reinterpret_cast<const __half2 *>(w2.data_ptr()) : nullptr,
           pair_ok ? reinterpret_cast<const __half2 *>(b2.data_ptr()) : nullptr,
           reinterpret_cast<__half *>(out.data_ptr()), NC, C,
           at::cuda::getCurrentCUDAStream());
  return out;
}

}  // namespace

// ---- cost attribution probes (dev only; never on the shipped hot path) ----
namespace {
__global__ void empty_kernel() {}
}  // namespace

at::Tensor repvgg_dw_probe_alloc(const at::Tensor &x) { return at::empty_like(x); }

at::Tensor repvgg_dw_probe_args5(const at::Tensor &x, const at::Tensor &w,
                                 const at::Tensor &b, const at::Tensor &w2,
                                 const at::Tensor &b2) {
  return at::empty_like(x);
}

at::Tensor repvgg_dw_probe_empty(const at::Tensor &x, const at::Tensor &w,
                                 const at::Tensor &b, const at::Tensor &w2,
                                 const at::Tensor &b2, int64_t nargs) {
  at::Tensor out = at::empty_like(x);
  empty_kernel<<<x.size(0) * x.size(1) / 2, 100, 0,
                 at::cuda::getCurrentCUDAStream()>>>();
  return out;
}

at::Tensor repvgg_dw_probe_empty_g(const at::Tensor &x, int64_t blocks,
                                   int64_t threads) {
  at::Tensor out = at::empty_like(x);
  empty_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>();
  return out;
}

at::Tensor repvgg_dw_probe_empty3(const at::Tensor &x, const at::Tensor &w2,
                                  const at::Tensor &b2) {
  at::Tensor out = at::empty_like(x);
  empty_kernel<<<x.size(0) * x.size(1) / 2, 100, 0,
                 at::cuda::getCurrentCUDAStream()>>>();
  return out;
}

// Pair-only entry point: three tensor arguments, no fp32 taps to marshal or
// validate.  This is what ships when the pair layout is available.
at::Tensor repvgg_dw_forward_pair(const at::Tensor &x, const at::Tensor &w2,
                                  const at::Tensor &b2) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 4 &&
                  x.is_contiguous() && x.size(2) == 20 && x.size(3) == 20,
              "pair fast path needs a contiguous fp16 [N,C,20,20] cuda tensor");
  const int NC = x.size(0) * x.size(1);
  at::Tensor out = at::empty_like(x);
  dispatch(NC <= 256 ? CFG_AUTO_SMALL_P : CFG_AUTO_LARGE_P,
           reinterpret_cast<const __half *>(x.data_ptr()), nullptr, nullptr,
           reinterpret_cast<const __half2 *>(w2.data_ptr()),
           reinterpret_cast<const __half2 *>(b2.data_ptr()),
           reinterpret_cast<__half *>(out.data_ptr()), NC, x.size(1),
           at::cuda::getCurrentCUDAStream());
  return out;
}

at::Tensor repvgg_dw_forward(const at::Tensor &x, const at::Tensor &w,
                             const at::Tensor &b, const at::Tensor &w2,
                             const at::Tensor &b2) {
  return run(x, w, b, w2, b2, -1);
}

at::Tensor repvgg_dw_forward_cfg(const at::Tensor &x, const at::Tensor &w,
                                 const at::Tensor &b, const at::Tensor &w2,
                                 const at::Tensor &b2, int64_t cfg) {
  return run(x, w, b, w2, b2, cfg);
}

int64_t repvgg_dw_num_cfgs() { return CFG_MAX_ID + 1; }

bool repvgg_dw_has_cfg(int64_t id) {
  switch (id) {
#define CFG_S(i, P, RT, CT, TAPSM, NEWSTG) \
  case i:                                  \
    return true;
#define CFG_P(i, P, RT, CT, SILU, WIDE, CTAP) \
  case i:                                     \
    return true;
#define CFG_G(i, RT, CT) \
  case i:                \
    return true;
    CFG_LIST
#undef CFG_S
#undef CFG_P
#undef CFG_G
    default:
      return false;
  }
}

int64_t repvgg_dw_tap_stride() { return TAP_STRIDE; }
int64_t repvgg_dw_pair_tap_stride() { return TAP_STRIDE_P; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &repvgg_dw_forward, "fused RepVGGDW depthwise block");
  m.def("forward_pair", &repvgg_dw_forward_pair, "fused block, pair kernel only");
  m.def("probe_alloc", &repvgg_dw_probe_alloc, "dev: output alloc only");
  m.def("probe_args5", &repvgg_dw_probe_args5, "dev: 5-arg marshal + alloc");
  m.def("probe_empty", &repvgg_dw_probe_empty, "dev: 5-arg + alloc + empty kernel");
  m.def("probe_empty3", &repvgg_dw_probe_empty3, "dev: 3-arg + alloc + empty kernel");
  m.def("probe_empty_g", &repvgg_dw_probe_empty_g, "dev: alloc + empty kernel, given grid");
  m.def("forward_cfg", &repvgg_dw_forward_cfg, "fused block, explicit config");
  m.def("num_cfgs", &repvgg_dw_num_cfgs, "config id upper bound");
  m.def("has_cfg", &repvgg_dw_has_cfg, "whether a config id is compiled");
  m.def("tap_stride", &repvgg_dw_tap_stride, "fp32 per-channel tap row stride");
  m.def("pair_tap_stride", &repvgg_dw_pair_tap_stride,
        "half2 per-channel-pair tap row stride");
}
