// Hand-written MXFP4 MoE inner loops (see ITERATIONS.md for the measurements that
// fix every choice here).
//
// Both gemms stream the trtllm-gen-shuffled packed e2m1 weights straight from HBM
// into registers, dequantize them there and feed mma.sync.m16n8k16.bf16 with the
// token tile in the M=16 dim and 8 weight rows in the N=8 dim.
//
// Dequant, 17 ops per 8 elements (measured 2.1 ops/element; the 23-op sequence it
// replaces cost 238 us on a 641 MB gemm1-shaped stream, this one 200 us):
//   * one PRMT byte-LUT lookup per 4 elements for the low bf16 bytes and one for the
//     high bytes.  The UE8M0 block scale is folded into the LUT itself -- the 8 e2m1
//     magnitudes are scaled once per 32-element block (4 mul.rn.bf16x2 + 4 prmt to
//     repack into byte tables), which is 2 ops/8 elements instead of the 4
//     mul.rn.bf16x2 that scaling every element costs.  Magnitude 0 stays exactly 0
//     under the multiply, which an exponent-add would not.
//   * the sign is OR'd into the *high-byte word* before the low/high interleave:
//     `w << 4` puts the even nibbles' sign bits on byte MSBs and `w` itself has the
//     odd ones, so one PRMT gathers the four signs a high-byte word needs and one
//     LOP3 masks them in.  That is 2 ops per 4 elements where sign-fixing the merged
//     bf16x2 words costs 2 per 2 elements.
//
// Layout: a warp owns 16 consecutive *physical* weight rows = 2 mma N-tiles.  The
// trtllm row shuffle maps physical mi -> logical 32*(mi/32) + (mi%32%8)*4 + (mi%32)/8,
// so a 16-row physical block holds 8 complete logical (up, gate) row pairs and tile 0
// vs tile 1 is exactly that pairing -- SwiGLU is register-local, no shuffles.
//
// K labelling: thread `lane` loads 16 bytes (32 nibbles = exactly one UE8M0 block) at
// byte offset 16*(lane%4) of a 64-byte group, for row 8*q + lane/4.  Nibble
// (word m, element i) of that window carries logical k = 32*(lane%4) + 8*m + i; the A
// fragments are pre-permuted into the matching mma slot order when the token tile is
// staged into shared memory, so the inner loop is 1 LDG.128 -> 4 dequant -> 8 mma with
// no shared memory for the weights at all.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdint>

// e2m1 magnitudes 0,.5,1,1.5,2,3,4,6 as bf16x2 pairs (low half first).
#define M01 0x3F000000u
#define M23 0x3FC03F80u
#define M45 0x40404000u
#define M67 0x40C04080u

// UE8M0 byte -> bf16x2 (value duplicated).  byte 0 is 2^-127, a bf16 subnormal.
__device__ __forceinline__ unsigned sbf16x2(unsigned b) {
  unsigned h = b ? (b << 7) : 0x40u;
  return h | (h << 16);
}

// Scale the 8 magnitudes once per 32-element block, repacked as byte LUTs.
__device__ __forceinline__ void scaled_lut(unsigned s2, unsigned &lo0, unsigned &lo1,
                                           unsigned &hi0, unsigned &hi1) {
  unsigned a = M01, b = M23, c = M45, d = M67;
  asm("mul.rn.bf16x2 %0,%0,%1;" : "+r"(a) : "r"(s2));
  asm("mul.rn.bf16x2 %0,%0,%1;" : "+r"(b) : "r"(s2));
  asm("mul.rn.bf16x2 %0,%0,%1;" : "+r"(c) : "r"(s2));
  asm("mul.rn.bf16x2 %0,%0,%1;" : "+r"(d) : "r"(s2));
  asm("prmt.b32 %0,%1,%2,0x6420;" : "=r"(lo0) : "r"(a), "r"(b));
  asm("prmt.b32 %0,%1,%2,0x7531;" : "=r"(hi0) : "r"(a), "r"(b));
  asm("prmt.b32 %0,%1,%2,0x6420;" : "=r"(lo1) : "r"(c), "r"(d));
  asm("prmt.b32 %0,%1,%2,0x7531;" : "=r"(hi1) : "r"(c), "r"(d));
}

// One packed uint (8 nibbles) -> four bf16x2 registers, already scaled.
__device__ __forceinline__ void dequant8(unsigned w, unsigned lo0, unsigned lo1,
                                         unsigned hi0, unsigned hi1, unsigned &r0,
                                         unsigned &r1, unsigned &r2, unsigned &r3) {
  unsigned c, ch, la, ha, lb, hb, A, sa, sb;
  asm("shl.b32 %0,%1,4;" : "=r"(A) : "r"(w));
  asm("and.b32 %0,%1,0x77777777;" : "=r"(c) : "r"(w));
  asm("shr.b32 %0,%1,16;" : "=r"(ch) : "r"(c));
  asm("prmt.b32 %0,%1,%2,%3;" : "=r"(la) : "r"(lo0), "r"(lo1), "r"(c));
  asm("prmt.b32 %0,%1,%2,%3;" : "=r"(ha) : "r"(hi0), "r"(hi1), "r"(c));
  asm("prmt.b32 %0,%1,%2,0x5140;" : "=r"(sa) : "r"(A), "r"(w));
  asm("lop3.b32 %0,%1,0x80808080,%0,0xEA;" : "+r"(ha) : "r"(sa));
  asm("prmt.b32 %0,%1,%2,%3;" : "=r"(lb) : "r"(lo0), "r"(lo1), "r"(ch));
  asm("prmt.b32 %0,%1,%2,%3;" : "=r"(hb) : "r"(hi0), "r"(hi1), "r"(ch));
  asm("prmt.b32 %0,%1,%2,0x7362;" : "=r"(sb) : "r"(A), "r"(w));
  asm("lop3.b32 %0,%1,0x80808080,%0,0xEA;" : "+r"(hb) : "r"(sb));
  asm("prmt.b32 %0,%1,%2,0x5140;" : "=r"(r0) : "r"(la), "r"(ha));
  asm("prmt.b32 %0,%1,%2,0x7362;" : "=r"(r1) : "r"(la), "r"(ha));
  asm("prmt.b32 %0,%1,%2,0x5140;" : "=r"(r2) : "r"(lb), "r"(hb));
  asm("prmt.b32 %0,%1,%2,0x7362;" : "=r"(r3) : "r"(lb), "r"(hb));
}

__device__ __forceinline__ void mma16816(float *d, const uint4 &a, unsigned b0,
                                         unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float swiglu(float yu, float yg) {
  float u = __bfloat162float(__float2bfloat16(yu));
  float g = __bfloat162float(__float2bfloat16(yg));
  g = fminf(g, 7.0f);
  u = fminf(fmaxf(u, -7.0f), 7.0f);
  return (u + 1.0f) * (g * (1.0f / (1.0f + expf(-1.702f * g))));
}

#define TILES 2                 // mma N-tiles per warp -> 16 physical rows
#define ROWS_WARP (TILES * 8)

// Stage a 16-token A tile chunk into shared memory in mma fragment order: fragment
// (group g, step j, lane t) is 16 bytes at ((g*8+j)*32+t)*16 holding
// {x[t/4][k0],x[t/4][k0+1], x[t/4+8][k0],x[t/4+8][k0+1],
//  x[t/4][k0+2],x[t/4][k0+3], x[t/4+8][k0+2],x[t/4+8][k0+3]},  k0 = 128*g+32*(t%4)+4*j.
__device__ __forceinline__ void stage_A(char *smem, const __nv_bfloat16 *src,
                                        const int *rows, int ntok, int k0, int knib,
                                        long stride, int tid, int nthread) {
  const int chunks = knib >> 3;
  // Only the live token slots are staged: an unused slot's A fragment feeds mma rows
  // whose accumulators are never stored, so stale shared memory there is harmless.
  for (int L = tid; L < ntok * chunks; L += nthread) {
    int i = L / chunks, kb = L - i * chunks;
    uint4 v = *(const uint4 *)(src + (long)rows[i] * stride + k0 + (kb << 3));
    int k = kb << 3, g = k >> 7, rem = k & 127, c = rem >> 5, jj = (rem & 31) >> 2;
    int t = (i & 7) * 4 + c, o = (i < 8) ? 0 : 4;
    char *p0 = smem + ((((g << 3) + jj) * 32 + t) << 4) + o;
    *(unsigned *)(p0) = v.x;
    *(unsigned *)(p0 + 8) = v.y;
    *(unsigned *)(p0 + 512) = v.z;
    *(unsigned *)(p0 + 520) = v.w;
  }
}

// One 128-nibble K group: dequantize TILES x 4 packed words and issue 8 mma each.
__device__ __forceinline__ void mma_group(float acc[TILES][4], const char *ab,
                                          const uint4 *wc, const unsigned *sb) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint4 af[4];
#pragma unroll
    for (int j = 0; j < 4; j++) af[j] = *(const uint4 *)(ab + (half * 4 + j) * 512);
#pragma unroll
    for (int q = 0; q < TILES; q++) {
      unsigned lo0, lo1, hi0, hi1;
      scaled_lut(sbf16x2(sb[q]), lo0, lo1, hi0, hi1);
      unsigned w0 = half ? wc[q].z : wc[q].x, w1 = half ? wc[q].w : wc[q].y;
      unsigned r0, r1, r2, r3, t0, t1, t2, t3;
      dequant8(w0, lo0, lo1, hi0, hi1, r0, r1, r2, r3);
      dequant8(w1, lo0, lo1, hi0, hi1, t0, t1, t2, t3);
      mma16816(acc[q], af[0], r0, r1);
      mma16816(acc[q], af[1], r2, r3);
      mma16816(acc[q], af[2], t0, t1);
      mma16816(acc[q], af[3], t2, t3);
    }
  }
}

// Accumulate TILES mma N-tiles over one staged K chunk.  DEPTH is the register
// prefetch depth; every shipped config uses 1 because depths 2/3/4/6 measured within
// 3% at mid M and *worse* at tiny M -- the exposed latency is in the arithmetic
// chain, not in the loads (see ITERATIONS.md).  The parameter is kept because that
// is easy to re-measure and easy to get wrong: the stage index has to be a
// compile-time constant, since a `g % DEPTH` index over a runtime-bounded loop puts
// the staging array in local memory (128 bytes of stack frame at DEPTH=3).
template <int DEPTH>
__device__ __forceinline__ void kloop(float acc[TILES][4], const char *ab0,
                                      const uint8_t *wp, const uint8_t *sp0,
                                      const int *soff, int g0, int ng, int WQ) {
  const int nfull = (ng / DEPTH) * DEPTH, last = g0 + ng - 1;
  uint4 st[DEPTH][TILES];
  unsigned ss[DEPTH][TILES];
#pragma unroll
  for (int d = 0; d < DEPTH; d++) {
    int g = min(g0 + d, last);  // clamped: never read past this expert's K
#pragma unroll
    for (int q = 0; q < TILES; q++) {
      st[d][q] = *(const uint4 *)(wp + q * WQ + g * 64);
      ss[d][q] = sp0[soff[q] + g * 512];
    }
  }
  for (int gb = g0; gb < g0 + nfull; gb += DEPTH) {
#pragma unroll
    for (int d = 0; d < DEPTH; ++d) {
      uint4 wc[TILES];
      unsigned sb[TILES];
#pragma unroll
      for (int q = 0; q < TILES; q++) { wc[q] = st[d][q]; sb[q] = ss[d][q]; }
      int gn = gb + d + DEPTH;
      if (gn < g0 + ng) {
#pragma unroll
        for (int q = 0; q < TILES; q++) {
          st[d][q] = *(const uint4 *)(wp + q * WQ + gn * 64);
          ss[d][q] = sp0[soff[q] + gn * 512];
        }
      }
      mma_group(acc, ab0 + ((gb + d - g0) << 3) * 32 * 16, wc, sb);
    }
  }
  for (int g = g0 + nfull; g < g0 + ng; ++g) {  // ng % DEPTH tail, unprefetched
    uint4 wc[TILES];
    unsigned sb[TILES];
#pragma unroll
    for (int q = 0; q < TILES; q++) {
      wc[q] = *(const uint4 *)(wp + q * WQ + g * 64);
      sb[q] = sp0[soff[q] + g * 512];
    }
    mma_group(acc, ab0 + ((g - g0) << 3) * 32 * 16, wc, sb);
  }
}

template <int WARPS, int NSTAGE, int DEPTH, int KSPLIT>
__global__ __launch_bounds__(WARPS * 32) void gemm1_kernel(
    const __nv_bfloat16 *__restrict__ x, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ sc, const float *__restrict__ bias,
    const int32_t *__restrict__ order, const int32_t *__restrict__ desc,
    const int32_t *__restrict__ nvalid, __nv_bfloat16 *__restrict__ act,
    float *__restrict__ zout, long zn, int KNIB, int NROWS, int NT, int IDIM,
    int TOPK, long XSTRIDE) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x, lane = tid & 31;
  // gemm2 accumulates into `zout` with atomics, so it has to start zeroed.  Doing it
  // here rather than with a torch zero_() removes a launch (~3 us of gap plus 2 us of
  // device time) from the tiny-M path, and it is ordered correctly because gemm2 is a
  // later launch on the same stream.
  if (zout != nullptr) {
    long nc = (long)gridDim.x * gridDim.y * blockDim.x;
    for (long i = ((long)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid;
         i < zn; i += nc)
      zout[i] = 0.0f;
  }
  const int wt = blockIdx.y;
  if (wt >= *nvalid) return;
  const int4 d = *(const int4 *)(desc + wt * 4);
  const int e = d.x, m0 = d.y, cnt = d.z, off = d.w;
  int ntok = cnt - m0;
  if (ntok > 16) ntok = 16;
  const int KC = KNIB / NSTAGE;  // nibbles staged per pass

  int *toks = (int *)(smem + 16 * KC * 2);
  if (tid < 16) toks[tid] = tid < ntok ? order[off + m0 + tid] / TOPK : 0;

  // KSPLIT warps share one 16-row group and split its K range; RG groups per CTA.
  const int RG = WARPS / KSPLIT, warp = tid >> 5, rg = warp % RG, ks = warp / RG;
  const int rbase = blockIdx.x * (RG * ROWS_WARP) + rg * ROWS_WARP;
  const bool live = rbase + ROWS_WARP <= NROWS;
  const int lrow = lane >> 2, c = lane & 3, hh = (rbase >> 4) & 1;
  const uint8_t *wp = w + (long)e * NROWS * (KNIB / 2) +
                      (long)(rbase + lrow) * (KNIB / 2) + c * 16;
  const uint8_t *sp0 = sc + (long)e * NROWS * (KNIB / 32) +
                       (long)(rbase >> 7) * NT * 512 + (((rbase >> 5) & 3) << 2) + c;
  int soff[TILES];
#pragma unroll
  for (int q = 0; q < TILES; q++) soff[q] = (16 * hh + 8 * q + lrow) * 16;
  const int WQ = 8 * (KNIB / 2);

  float acc[TILES][4];
#pragma unroll
  for (int q = 0; q < TILES; q++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[q][i] = 0.0f;

  const int ngs = (KC >> 7) / KSPLIT;  // K groups per warp per staging pass
  for (int s = 0; s < NSTAGE; ++s) {
    __syncthreads();
    stage_A(smem, x, toks, ntok, s * KC, KC, XSTRIDE, tid, WARPS * 32);
    __syncthreads();
    if (live)
      kloop<DEPTH>(acc, smem + ((ks * ngs) << 3) * 32 * 16 + lane * 16, wp, sp0, soff,
                   s * (KC >> 7) + ks * ngs, ngs, WQ);
  }
  if (KSPLIT > 1) {
    // Partial sums of the K splits meet in shared memory.  Dead warps take part
    // harmlessly: their row group's owner is dead too and never reads the slots.
    float *red = (float *)(smem + 16 * KC * 2 + 64);
    float *slot = red + ((rg * (KSPLIT - 1) + (ks ? ks - 1 : 0)) * 32 + lane) * TILES * 4;
    if (ks) {
#pragma unroll
      for (int q = 0; q < TILES; q++)
#pragma unroll
        for (int i = 0; i < 4; i++) slot[q * 4 + i] = acc[q][i];
    }
    __syncthreads();
    if (ks == 0) {
#pragma unroll
      for (int j = 0; j < KSPLIT - 1; j++) {
        const float *p = red + ((rg * (KSPLIT - 1) + j) * 32 + lane) * TILES * 4;
#pragma unroll
        for (int q = 0; q < TILES; q++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[q][i] += p[q * 4 + i];
      }
    }
  }
  if (!live || ks != 0) return;

  const float *bp = bias + (long)e * NROWS + rbase;
  const int jb = (rbase >> 5) * 16 + 4 * c + hh;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int slot = lrow + 8 * h;
    if (slot >= ntok) continue;
    __nv_bfloat16 *ap = act + (long)(off + m0 + slot) * IDIM + jb;
#pragma unroll
    for (int u = 0; u < 2; ++u)
      ap[2 * u] = __float2bfloat16(swiglu(acc[0][2 * h + u] + bp[2 * c + u],
                                          acc[1][2 * h + u] + bp[8 + 2 * c + u]));
  }
}

template <int WARPS, int NSTAGE, int DEPTH, int KSPLIT>
__global__ __launch_bounds__(WARPS * 32) void gemm2_kernel(
    const __nv_bfloat16 *__restrict__ actin, const uint8_t *__restrict__ w,
    const uint8_t *__restrict__ sc, const float *__restrict__ bias,
    const __nv_bfloat16 *__restrict__ tw, const int32_t *__restrict__ order,
    const int32_t *__restrict__ desc, const int32_t *__restrict__ nvalid,
    float *__restrict__ out, int KNIB, int NROWS, int NT, int HU, int TOPK) {
  extern __shared__ __align__(16) char smem[];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wt = blockIdx.y;
  if (wt >= *nvalid) return;
  const int4 d = *(const int4 *)(desc + wt * 4);
  const int e = d.x, m0 = d.y, cnt = d.z, off = d.w;
  int ntok = cnt - m0;
  if (ntok > 16) ntok = 16;
  const int KC = KNIB / NSTAGE;

  int *toks = (int *)(smem + 16 * KC * 2);
  if (tid < 16) toks[tid] = off + m0 + tid;

  // KSPLIT warps share one 16-row group and split its K range; RG groups per CTA.
  const int RG = WARPS / KSPLIT, warp = tid >> 5, rg = warp % RG, ks = warp / RG;
  const int rbase = blockIdx.x * (RG * ROWS_WARP) + rg * ROWS_WARP;
  const bool live = rbase + ROWS_WARP <= HU;
  const int lrow = lane >> 2, c = lane & 3, hh = (rbase >> 4) & 1;
  const uint8_t *wp = w + (long)e * NROWS * (KNIB / 2) +
                      (long)(rbase + lrow) * (KNIB / 2) + c * 16;
  const uint8_t *sp0 = sc + (long)e * NROWS * (KNIB / 32) +
                       (long)(rbase >> 7) * NT * 512 + (((rbase >> 5) & 3) << 2) + c;
  int soff[TILES];
#pragma unroll
  for (int q = 0; q < TILES; q++) soff[q] = (16 * hh + 8 * q + lrow) * 16;
  const int WQ = 8 * (KNIB / 2);

  float acc[TILES][4];
#pragma unroll
  for (int q = 0; q < TILES; q++)
#pragma unroll
    for (int i = 0; i < 4; i++) acc[q][i] = 0.0f;

  const int ngs = (KC >> 7) / KSPLIT;  // K groups per warp per staging pass
  for (int s = 0; s < NSTAGE; ++s) {
    __syncthreads();
    stage_A(smem, actin, toks, ntok, s * KC, KC, KNIB, tid, WARPS * 32);
    __syncthreads();
    if (live)
      kloop<DEPTH>(acc, smem + ((ks * ngs) << 3) * 32 * 16 + lane * 16, wp, sp0, soff,
                   s * (KC >> 7) + ks * ngs, ngs, WQ);
  }
  if (KSPLIT > 1) {
    // Partial sums of the K splits meet in shared memory.  Dead warps take part
    // harmlessly: their row group's owner is dead too and never reads the slots.
    float *red = (float *)(smem + 16 * KC * 2 + 64);
    float *slot = red + ((rg * (KSPLIT - 1) + (ks ? ks - 1 : 0)) * 32 + lane) * TILES * 4;
    if (ks) {
#pragma unroll
      for (int q = 0; q < TILES; q++)
#pragma unroll
        for (int i = 0; i < 4; i++) slot[q * 4 + i] = acc[q][i];
    }
    __syncthreads();
    if (ks == 0) {
#pragma unroll
      for (int j = 0; j < KSPLIT - 1; j++) {
        const float *p = red + ((rg * (KSPLIT - 1) + j) * 32 + lane) * TILES * 4;
#pragma unroll
        for (int q = 0; q < TILES; q++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc[q][i] += p[q * 4 + i];
      }
    }
  }
  if (!live || ks != 0) return;

  const float *bp = bias + (long)e * NROWS + rbase;
  const int nb = (rbase >> 5) * 32 + 8 * c + 2 * hh;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int slot = lrow + 8 * h;
    if (slot >= ntok) continue;
    int pr = order[off + m0 + slot];
    float wgt = __bfloat162float(tw[pr]);
    float *op = out + (long)(pr / TOPK) * HU + nb;
#pragma unroll
    for (int u = 0; u < 2; ++u)
#pragma unroll
      for (int q = 0; q < TILES; q++)
        atomicAdd(op + 4 * u + q,
                  wgt * __bfloat162float(__float2bfloat16(acc[q][2 * h + u] +
                                                          bp[8 * q + 2 * c + u])));
  }
}

// Mid-M: 16 warps x 16 rows = 256 rows per CTA, one staging pass (96 KB, 2 CTAs/SM =
// 32 warps).  Tiny M: 2 warps = 32 rows per CTA so 4 work tiles still fill the machine,
// with K staged in 4 chunks (24 KB) to keep 8 CTAs resident.
#define WBIG 16
#define NSBIG 1
#define DBIG 1
#define DSML 1
// Tiny-M shapes: KSPLIT warps per 16-row group, chosen so that (K groups)/KSPLIT is
// an integer -- gemm1 has 24 groups at K=3072, gemm2 12 at K=1536.
// KSPLIT=4 measured better than 8 (26.7 us vs 28.7 us for the whole M=1 path).
#define W1SML 8
#define NS1SML 2
#define KS1SML 4
#define W2SML 8
#define NS2SML 1
#define KS2SML 4

void gemm1_launch(torch::Tensor x, torch::Tensor w, torch::Tensor sc,
                  torch::Tensor bias, torch::Tensor order, torch::Tensor desc,
                  torch::Tensor nvalid, torch::Tensor act, torch::Tensor out,
                  int64_t tmax, int64_t topk, bool small) {
  int KNIB = (int)x.size(1), NROWS = (int)w.size(1);
  int NT = (KNIB / 32 + 3) / 4, IDIM = (int)act.size(1);
  const int W = small ? W1SML : WBIG, NS = small ? NS1SML : NSBIG;
  const int KS = small ? KS1SML : 1;
  int smem = 16 * (KNIB / NS) * 2 + 64 + (KS > 1 ? (W / KS) * (KS - 1) * 32 * TILES * 16 : 0);
  static bool done[2] = {false, false};
  if (!done[small]) {
    cudaFuncSetAttribute(small ? (const void *)gemm1_kernel<W1SML, NS1SML, DSML, KS1SML>
                               : (const void *)gemm1_kernel<WBIG, NSBIG, DBIG, 1>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    done[small] = true;
  }
  const int rows_cta = (W / KS) * ROWS_WARP;
  dim3 grid((NROWS + rows_cta - 1) / rows_cta, (unsigned)tmax);
  auto fn = small ? gemm1_kernel<W1SML, NS1SML, DSML, KS1SML>
                  : gemm1_kernel<WBIG, NSBIG, DBIG, 1>;
  fn<<<grid, W * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16 *)x.data_ptr(), (const uint8_t *)w.data_ptr(),
      (const uint8_t *)sc.data_ptr(), (const float *)bias.data_ptr(),
      (const int32_t *)order.data_ptr(), (const int32_t *)desc.data_ptr(),
      (const int32_t *)nvalid.data_ptr(), (__nv_bfloat16 *)act.data_ptr(),
      (float *)out.data_ptr(), out.numel(), KNIB, NROWS, NT, IDIM, (int)topk,
      (long)x.stride(0));
}

void gemm2_launch(torch::Tensor actin, torch::Tensor w, torch::Tensor sc,
                  torch::Tensor bias, torch::Tensor tw, torch::Tensor order,
                  torch::Tensor desc, torch::Tensor nvalid, torch::Tensor out,
                  int64_t tmax, int64_t topk, bool small) {
  int KNIB = (int)actin.size(1), NROWS = (int)w.size(1), HU = (int)out.size(1);
  int NT = (KNIB / 32 + 3) / 4;
  const int W = small ? W2SML : WBIG, NS = small ? NS2SML : NSBIG;
  const int KS = small ? KS2SML : 1;
  int smem = 16 * (KNIB / NS) * 2 + 64 + (KS > 1 ? (W / KS) * (KS - 1) * 32 * TILES * 16 : 0);
  static bool done[2] = {false, false};
  if (!done[small]) {
    cudaFuncSetAttribute(small ? (const void *)gemm2_kernel<W2SML, NS2SML, DSML, KS2SML>
                               : (const void *)gemm2_kernel<WBIG, NSBIG, DBIG, 1>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    done[small] = true;
  }
  const int rows_cta = (W / KS) * ROWS_WARP;
  dim3 grid((HU + rows_cta - 1) / rows_cta, (unsigned)tmax);
  auto fn = small ? gemm2_kernel<W2SML, NS2SML, DSML, KS2SML>
                  : gemm2_kernel<WBIG, NSBIG, DBIG, 1>;
  fn<<<grid, W * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16 *)actin.data_ptr(), (const uint8_t *)w.data_ptr(),
      (const uint8_t *)sc.data_ptr(), (const float *)bias.data_ptr(),
      (const __nv_bfloat16 *)tw.data_ptr(), (const int32_t *)order.data_ptr(),
      (const int32_t *)desc.data_ptr(), (const int32_t *)nvalid.data_ptr(),
      (float *)out.data_ptr(), KNIB, NROWS, NT, HU, (int)topk);
}

__global__ void dq_test_kernel(const unsigned *packed, const unsigned char *sbytes,
                               __nv_bfloat16 *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned lo0, lo1, hi0, hi1, r[4];
  scaled_lut(sbf16x2(sbytes[i]), lo0, lo1, hi0, hi1);
  dequant8(packed[i], lo0, lo1, hi0, hi1, r[0], r[1], r[2], r[3]);
  *(uint4 *)(out + 8 * i) = *(uint4 *)r;
}

void dq_test(torch::Tensor packed, torch::Tensor sbytes, torch::Tensor out) {
  int n = (int)packed.numel();
  dq_test_kernel<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const unsigned *)packed.data_ptr(), (const unsigned char *)sbytes.data_ptr(),
      (__nv_bfloat16 *)out.data_ptr(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm1", &gemm1_launch, "");
  m.def("gemm2", &gemm2_launch, "");
  m.def("dq_test", &dq_test, "");
}
