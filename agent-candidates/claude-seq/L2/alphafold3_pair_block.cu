// Fused AlphaFold3 PairBlock -- one kernel launch, one 16-CTA cluster.
//
// The captured shape is tiny (N_res=16, c_z=128): the baseline's ~120 eager
// kernels carry only a few microseconds of real work, so it is entirely launch
// bound. Everything runs here in a single kernel.
//
// Decomposition
//   CTA r owns row r of the pair matrix (16 tokens) for the two triangle
//   multiplications and the starting-node attention, then switches to column r
//   for the ending-node attention and the transition. The transition is token
//   local, so that switch costs no extra barrier.
//   Cross-CTA data (the b matrix of the triangle products, z for the transpose,
//   the triangle bias) travels through a small global scratch buffer; the
//   barrier is cluster.sync() (~0.4us) rather than a device-wide barrier (~2us).
//
// GEMMs
//   mma.sync m16n8k16, fp32 accumulate. Weights are pre-swizzled on the host
//   into mma B-fragment order, so a plain cp.async.bulk copy lands a 128x128
//   tile in shared memory and each mma reads its operand with one conflict-free
//   8-byte shared load. The kernel consumes the 34 tiles in a fixed order, so
//   one kernel-wide NBUF-deep prefetch pipeline hides the copy latency across
//   GEMM boundaries.
//
// Numerics
//   bf16 rounding is applied at exactly the points the eager baseline rounds
//   (every nn.Linear output, every sigmoid/silu input, every residual add), and
//   layer norm / softmax reduce in fp32 like the baseline.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstring>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

#define NRES   16
#define NTOK   256
#define CZ     128
#define NHEAD  4
#define DHEAD  32
#define THID   512
#define NWARP  16                // one n-tile per warp per 128-row chunk
#define NTHR   (NWARP * 32)
#define LDA    (CZ + 8)          // padded row stride, 128-wide activations
#define LDQ    (4 * CZ + 8)      // q|k|v|gate
#define LDH    (THID + 8)        // transition hidden
#define TILEB  32768             // one weight tile: 128 rows x 128 k, bf16
// one 128-row chunk = 16 n-tiles = one per warp, so NWARP must be 16
#define NBUF   4                 // weight tiles in flight
#define NTILE  34                // total weight tiles consumed, in order
#define EPSLN  1e-5f

// ---------------------------------------------------------------- primitives
__device__ __forceinline__ uint32_t su32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(unsigned long long* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(su32(b)));
}
__device__ __forceinline__ void mbar_arrive_tx(unsigned long long* b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(su32(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, int parity) {
  asm volatile("{ .reg .pred p;\n"
               "LW_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra LW_%=;\n}" :: "r"(su32(b)), "r"(parity));
}
__device__ __forceinline__ void bulk_cp(void* dst, const void* src, int bytes,
                                        unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1], %2, [%3];"
               :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sigf(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rb(float x) { return b2f(f2b(x)); }  // round to bf16
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  uint32_t r;
  memcpy(&r, &v, 4);
  return r;
}
__device__ __forceinline__ float blo(uint32_t v) { return __int_as_float(v << 16); }
__device__ __forceinline__ float bhi(uint32_t v) { return __int_as_float(v & 0xffff0000u); }
__device__ __forceinline__ bf16 lo16(uint32_t v) {
  return __ushort_as_bfloat16((unsigned short)(v & 0xffffu));
}
__device__ __forceinline__ bf16 hi16(uint32_t v) {
  return __ushort_as_bfloat16((unsigned short)(v >> 16));
}

// -------------------------------------------------------------- shared state
struct Smem {
  __align__(16) char wbuf[NBUF][TILEB];
  __align__(16) bf16 zr[NRES][LDA];    // residual stream for the owned tokens
  __align__(16) bf16 xl[NRES][LDA];    // layer-norm output
  __align__(16) bf16 ar[NRES][LDA];
  __align__(16) bf16 br[NRES][LDA];
  __align__(16) bf16 gt[NRES][LDA];    // sigmoid gate
  __align__(16) bf16 pr[NRES][LDA];    // triangle product / gated attn out
  // q|k|v|gate (attention) and the SwiGLU hidden state never overlap in time
  union {
    __align__(16) bf16 qkvg[NRES][LDQ];
    __align__(16) bf16 hid[NRES][LDH];
  };
  __align__(16) bf16 tbs[NRES][NRES][NHEAD];
  __align__(16) bf16 tbw[2][NHEAD][CZ];   // the two 4x128 linear_z weights
  __align__(8) unsigned long long mbar[NBUF];
  float satt[NWARP][NRES];
  float msk[NRES];
};

struct Args {
  const bf16* __restrict__ z;
  const bf16* __restrict__ mask;
  bf16* __restrict__ out;
  const char* __restrict__ w;      // fragment-swizzled weight tiles, in order
  const float* __restrict__ ln;    // 7 x (128 weight + 128 bias), fp32
  bf16* __restrict__ ws;           // global scratch
  int mask_trans;
};

// 4x128 triangle-bias projections sit just past the swizzled tiles
#define TB_WEIGHT_OFF ((size_t)NTILE * TILEB)

// layer-norm blocks, in packing order
#define LN_MO_IN  0
#define LN_MO_OUT 1
#define LN_MI_IN  2
#define LN_MI_OUT 3
#define LN_ATT_S  4
#define LN_ATT_E  5
#define LN_TRANS  6

// --------------------------------------------------------- weight tile pipeline
__device__ __forceinline__ void tile_issue(Smem* sm, const Args& ar, int t) {
  if (t >= NTILE) return;
  const int b = t % NBUF;
  mbar_arrive_tx(&sm->mbar[b], TILEB);
  bulk_cp(sm->wbuf[b], ar.w + (size_t)t * TILEB, TILEB, &sm->mbar[b]);
}

// --------------------------------------------------------------- mma helpers
__device__ __forceinline__ void ld_afrag(const bf16* As, int lda, int koff,
                                         int g, int q, uint32_t* a) {
  const bf16* p = As + koff + (q << 1);
  a[0] = *(const uint32_t*)(p + g * lda);
  a[1] = *(const uint32_t*)(p + (g + 8) * lda);
  a[2] = *(const uint32_t*)(p + g * lda + 8);
  a[3] = *(const uint32_t*)(p + (g + 8) * lda + 8);
}
__device__ __forceinline__ uint2 ld_bfrag(const char* W, int nt, int kt, int lane) {
  return *(const uint2*)(W + ((((nt << 3) + kt) << 5) + lane) * 8);
}
// A is 16 x 128, i.e. a single m-tile: its eight k-tile fragments are 32
// registers, so they are loaded once per GEMM and reused for every output
// column block. Only the B fragments then touch shared memory per mma.
__device__ __forceinline__ void ld_a_all(const bf16* As, int lda, int koff,
                                         int lane, uint32_t af[8][4]) {
  const int g = lane >> 2, q = lane & 3;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) ld_afrag(As, lda, koff + kt * 16, g, q, af[kt]);
}
// ---------------------------------------------------------------- layer norm
// 16 tokens x 128 channels, two tokens per warp; fp32 reduction, bf16 out,
// matching the baseline's promote_fp32 LayerNorm.
__device__ __forceinline__ void layernorm16(const bf16* in, int ldi, bf16* out, int ldo,
                                            const float* w, const float* b,
                                            int warp, int lane) {
  for (int t = warp; t < NRES; t += NWARP) {
    const bf16* p = in + t * ldi;
    float v0 = b2f(p[lane]), v1 = b2f(p[lane + 32]);
    float v2 = b2f(p[lane + 64]), v3 = b2f(p[lane + 96]);
    float s = v0 + v1 + v2 + v3;
    float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      s += __shfl_xor_sync(0xffffffffu, s, o);
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    float mean = s * (1.f / CZ);
    float rstd = rsqrtf(fmaxf(ss * (1.f / CZ) - mean * mean, 0.f) + EPSLN);
    bf16* o2 = out + t * ldo;
    o2[lane]      = f2b((v0 - mean) * rstd * w[lane]      + b[lane]);
    o2[lane + 32] = f2b((v1 - mean) * rstd * w[lane + 32] + b[lane + 32]);
    o2[lane + 64] = f2b((v2 - mean) * rstd * w[lane + 64] + b[lane + 64]);
    o2[lane + 96] = f2b((v3 - mean) * rstd * w[lane + 96] + b[lane + 96]);
  }
}

// ------------------------------------------------------------------- the GEMM
// Two flavours, both with A cached in registers:
//   gemm2 -- two output-column chunks per step. Each warp owns the same n-tile
//            in both chunks, so a pair of linears that combine elementwise
//            (a_p * sigmoid(a_g), silu(linear_a) * linear_b, ...) meet in
//            registers, and the two mma chains are independent, which is what
//            this kernel needs: with 16 warps there is little else to hide
//            mma/shared latency behind.
//   gemm1 -- one chunk, KC k-chunks accumulated (used by the narrow outputs).
enum { EPI2_TRIMUL, EPI2_QKVG, EPI2_TRANS };
enum { EPI_G, EPI_MULZ, EPI_ATTO, EPI_TRANS2 };

#define WAIT_TILE(t) \
  do { if (threadIdx.x == 0) mbar_wait(&sm->mbar[(t) % NBUF], ((t) / NBUF) & 1); } while (0)

template <int NPAIR, int EPI>
__device__ void gemm2(Smem* sm, const Args& ar, const bf16* As, int lda,
                      int warp, int lane, int& tile) {
  const int g = lane >> 2, q = lane & 3;
  uint32_t af[8][4];
  ld_a_all(As, lda, 0, lane, af);
  for (int p = 0; p < NPAIR; ++p) {
    float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
    float e0[4] = {0.f, 0.f, 0.f, 0.f}, e1[4] = {0.f, 0.f, 0.f, 0.f};
    const int t0 = tile, t1 = tile + 1;
    const char* W0 = sm->wbuf[t0 % NBUF];
    const char* W1 = sm->wbuf[t1 % NBUF];
    if (threadIdx.x == 0) {
      mbar_wait(&sm->mbar[t0 % NBUF], (t0 / NBUF) & 1);
      mbar_wait(&sm->mbar[t1 % NBUF], (t1 / NBUF) & 1);
    }
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < 8; kt += 2) {
      uint2 b0 = ld_bfrag(W0, warp, kt, lane);
      uint2 b1 = ld_bfrag(W1, warp, kt, lane);
      uint2 c0 = ld_bfrag(W0, warp, kt + 1, lane);
      uint2 c1 = ld_bfrag(W1, warp, kt + 1, lane);
      mma16816(a0, af[kt], b0.x, b0.y);
      mma16816(a1, af[kt], b1.x, b1.y);
      mma16816(e0, af[kt + 1], c0.x, c0.y);
      mma16816(e1, af[kt + 1], c1.x, c1.y);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) { a0[i] += e0[i]; a1[i] += e1[i]; }
    __syncthreads();
    if (threadIdx.x == 0) { tile_issue(sm, ar, t0 + NBUF); tile_issue(sm, ar, t1 + NBUF); }
    tile += 2;

    const int col = warp * 8 + (q << 1);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = g + half * 8;
      const float p0 = a0[half * 2], p1 = a0[half * 2 + 1];
      const float g0 = a1[half * 2], g1 = a1[half * 2 + 1];
      if (EPI == EPI2_TRIMUL) {
        // mask * sigmoid(linear_*_g) * linear_*_p
        const float m = sm->msk[row];
        const uint32_t v = pack2(rb(m * rb(sigf(rb(g0)))) * rb(p0),
                                 rb(m * rb(sigf(rb(g1)))) * rb(p1));
        *(uint32_t*)(p == 0 ? &sm->ar[row][col] : &sm->br[row][col]) = v;
      } else if (EPI == EPI2_QKVG) {
        if (p == 0) {   // q (pre-scaled by 1/sqrt(c_hidden)) and k
          *(uint32_t*)&sm->qkvg[row][col] =
              pack2(rb(p0) / 5.656854249492381f, rb(p1) / 5.656854249492381f);
          *(uint32_t*)&sm->qkvg[row][CZ + col] = pack2(g0, g1);
        } else {        // v and the output gate
          *(uint32_t*)&sm->qkvg[row][2 * CZ + col] = pack2(p0, p1);
          *(uint32_t*)&sm->qkvg[row][3 * CZ + col] = pack2(sigf(rb(g0)), sigf(rb(g1)));
        }
      } else {          // EPI2_TRANS: silu(linear_a) * linear_b
        const float s0 = rb(p0), s1 = rb(p1);
        *(uint32_t*)&sm->hid[row][p * 128 + col] =
            pack2(rb(s0 * sigf(s0)) * rb(g0), rb(s1 * sigf(s1)) * rb(g1));
      }
    }
    __syncthreads();
  }
}

template <int KC, int EPI>
__device__ void gemm1(Smem* sm, const Args& ar, const bf16* As, int lda,
                      int warp, int lane, int& tile) {
  const int g = lane >> 2, q = lane & 3;
  uint32_t af[8][4];
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, ace[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kc = 0; kc < KC; ++kc, ++tile) {
    ld_a_all(As, lda, kc * 128, lane, af);
    const char* W = sm->wbuf[tile % NBUF];
    WAIT_TILE(tile);
    __syncthreads();
#pragma unroll
    for (int kt = 0; kt < 8; kt += 2) {
      uint2 b = ld_bfrag(W, warp, kt, lane);
      uint2 c = ld_bfrag(W, warp, kt + 1, lane);
      mma16816(acc, af[kt], b.x, b.y);
      mma16816(ace, af[kt + 1], c.x, c.y);
    }
    __syncthreads();
    if (threadIdx.x == 0) tile_issue(sm, ar, tile + NBUF);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] += ace[i];
  const int col = warp * 8 + (q << 1);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = g + half * 8;
    const float x0 = rb(acc[half * 2]), x1 = rb(acc[half * 2 + 1]);
    if (EPI == EPI_G) {                       // sigmoid(linear_g)
      *(uint32_t*)&sm->gt[row][col] = pack2(sigf(x0), sigf(x1));
    } else if (EPI == EPI_MULZ) {             // z += linear_z(x) * gate
      *(uint32_t*)&sm->zr[row][col] =
          pack2(b2f(sm->zr[row][col])     + rb(x0 * b2f(sm->gt[row][col])),
                b2f(sm->zr[row][col + 1]) + rb(x1 * b2f(sm->gt[row][col + 1])));
    } else if (EPI == EPI_ATTO) {             // z += linear_o(o)
      *(uint32_t*)&sm->zr[row][col] =
          pack2(b2f(sm->zr[row][col]) + x0, b2f(sm->zr[row][col + 1]) + x1);
    } else {                                  // EPI_TRANS2: z += out * mask
      const float m = ar.mask_trans ? sm->msk[row] : 1.f;
      *(uint32_t*)&sm->zr[row][col] =
          pack2(b2f(sm->zr[row][col])     + rb(x0 * m),
                b2f(sm->zr[row][col + 1]) + rb(x1 * m));
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------ attention
// Triangle attention over the 16 tokens the CTA owns. Reads q|k|v|gate from
// sm->qkvg, writes the gated output to sm->pr.
__device__ void attention(Smem* sm, int warp, int lane) {
  const int ki = lane & 15, half = lane >> 4;
  for (int pair = warp; pair < NHEAD * NRES; pair += NWARP) {
    const int h = pair >> 4, qi = pair & 15;
    const uint32_t* qp = (const uint32_t*)&sm->qkvg[qi][h * DHEAD + half * 16];
    const uint32_t* kp = (const uint32_t*)&sm->qkvg[ki][CZ + h * DHEAD + half * 16];
    // four independent chains: this loop is latency, not throughput, bound
    float dd[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      const uint32_t qv = qp[d], kv = kp[d];
      dd[d & 3] += blo(qv) * blo(kv) + bhi(qv) * bhi(kv);
    }
    float dot = (dd[0] + dd[1]) + (dd[2] + dd[3]);
    dot += __shfl_xor_sync(0xffffffffu, dot, 16);
    float sc = rb(dot);
    sc = rb(sc + rb(1e9f * (sm->msk[ki] - 1.f)));      // mask bias
    sc = rb(sc + b2f(sm->tbs[qi][ki][h]));             // triangle bias
    float mx = sc;
#pragma unroll
    for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    float e = __expf(sc - mx), se = e;
#pragma unroll
    for (int o = 8; o; o >>= 1) se += __shfl_xor_sync(0xffffffffu, se, o);
    if (half == 0) sm->satt[warp][ki] = rb(e / se);
    __syncwarp();
    float av[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < NRES; ++k)
      av[k & 3] += sm->satt[warp][k] * b2f(sm->qkvg[k][2 * CZ + h * DHEAD + lane]);
    const float acc = (av[0] + av[1]) + (av[2] + av[3]);
    const int col = h * DHEAD + lane;
    sm->pr[qi][col] = f2b(rb(acc) * b2f(sm->qkvg[qi][3 * CZ + col]));
    __syncwarp();
  }
}

// ------------------------------------------------------------- triangle bias
// tb[j, h] = sum_c xl[j, c] * linear_z.weight[h, c]; one warp per two tokens.
__device__ void tri_bias(Smem* sm, bf16* gtb, int which, int rank,
                         int warp, int lane) {
  for (int j = warp; j < NRES; j += NWARP) {
    const float x0 = b2f(sm->xl[j][lane]),      x1 = b2f(sm->xl[j][lane + 32]);
    const float x2 = b2f(sm->xl[j][lane + 64]), x3 = b2f(sm->xl[j][lane + 96]);
#pragma unroll
    for (int h = 0; h < NHEAD; ++h) {
      const bf16* w = &sm->tbw[which][h][0];
      float s = x0 * b2f(w[lane])      + x1 * b2f(w[lane + 32])
              + x2 * b2f(w[lane + 64]) + x3 * b2f(w[lane + 96]);
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      if (lane == 0) gtb[(rank * NRES + j) * NHEAD + h] = f2b(s);
    }
  }
}

// ------------------------------------------------------ small shared/global IO
#define FOR_ROW16(i) for (int i = tid; i < NRES * 16; i += NTHR)
#define RC(i) const int j = (i) >> 4, c2 = ((i) & 15) << 3
#define CP16(dst, src) *(uint4*)(dst) = *(const uint4*)(src)

// ------------------------------------------------------------------ the kernel
extern "C" __global__ void __cluster_dims__(16, 1, 1)
__launch_bounds__(NTHR) pair_block_kernel(__grid_constant__ const Args ar) {
  extern __shared__ char raw[];
  Smem* sm = reinterpret_cast<Smem*>(raw);
  cg::cluster_group cl = cg::this_cluster();

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r = blockIdx.x;
  int tile = 0;

  bf16* gB1 = ar.ws;
  bf16* gA2 = gB1 + NTOK * CZ;
  bf16* gB2 = gA2 + NTOK * CZ;
  bf16* gZ  = gB2 + NTOK * CZ;
  bf16* gT1 = gZ + NTOK * CZ;
  bf16* gT2 = gT1 + NTOK * NHEAD;

  if (tid == 0) {
#pragma unroll
    for (int b = 0; b < NBUF; ++b) mbar_init(&sm->mbar[b]);
#pragma unroll
    for (int b = 0; b < NBUF; ++b) tile_issue(sm, ar, b);
  }
  {
    // the two small triangle-bias projections live in shared for the whole run
    const uint4* tw = (const uint4*)(ar.w + TB_WEIGHT_OFF);
    for (int i = tid; i < 2 * NHEAD * CZ / 8; i += NTHR)
      ((uint4*)&sm->tbw[0][0][0])[i] = tw[i];
    const bf16* src = ar.z + (size_t)r * NRES * CZ;
    FOR_ROW16(i) { RC(i); CP16(&sm->zr[j][c2], src + j * CZ + c2); }
    if (tid < NRES) sm->msk[tid] = b2f(ar.mask[r * NRES + tid]);
  }
  __syncthreads();

  const float* lnp = ar.ln;
#define LNW(i) (lnp + (size_t)(i) * 2 * CZ)
#define LNB(i) (lnp + (size_t)(i) * 2 * CZ + CZ)

  // ======================= triangle multiplication, outgoing
  layernorm16(&sm->zr[0][0], LDA, &sm->xl[0][0], LDA, LNW(LN_MO_IN), LNB(LN_MO_IN), warp, lane);
  __syncthreads();
  gemm2<2, EPI2_TRIMUL>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  gemm1<1, EPI_G>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  FOR_ROW16(i) { RC(i);
    CP16(gB1 + (r * NRES + j) * CZ + c2, &sm->br[j][c2]); }
  __threadfence();
  cl.sync();
  // p[k, c] = sum_j a[r, j, c] * b[k, j, c]
  for (int i = tid; i < NRES * 64; i += NTHR) {
    const int k = i >> 6, c2 = (i & 63) << 1;
    float s0 = 0.f, s1 = 0.f, u0 = 0.f, u1 = 0.f;
#pragma unroll
    for (int j = 0; j < NRES; j += 2) {
      uint32_t av = *(const uint32_t*)&sm->ar[j][c2];
      uint32_t bv = *(const uint32_t*)(gB1 + (k * NRES + j) * CZ + c2);
      uint32_t aw = *(const uint32_t*)&sm->ar[j + 1][c2];
      uint32_t bw = *(const uint32_t*)(gB1 + (k * NRES + j + 1) * CZ + c2);
      s0 += blo(av) * blo(bv);
      s1 += bhi(av) * bhi(bv);
      u0 += blo(aw) * blo(bw);
      u1 += bhi(aw) * bhi(bw);
    }
    *(uint32_t*)&sm->pr[k][c2] = pack2(s0 + u0, s1 + u1);
  }
  __syncthreads();
  layernorm16(&sm->pr[0][0], LDA, &sm->pr[0][0], LDA, LNW(LN_MO_OUT), LNB(LN_MO_OUT), warp, lane);
  __syncthreads();
  gemm1<1, EPI_MULZ>(sm, ar, &sm->pr[0][0], LDA, warp, lane, tile);

  // ======================= triangle multiplication, incoming
  layernorm16(&sm->zr[0][0], LDA, &sm->xl[0][0], LDA, LNW(LN_MI_IN), LNB(LN_MI_IN), warp, lane);
  __syncthreads();
  gemm2<2, EPI2_TRIMUL>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  gemm1<1, EPI_G>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  FOR_ROW16(i) { RC(i);
    CP16(gA2 + (r * NRES + j) * CZ + c2, &sm->ar[j][c2]);
    CP16(gB2 + (r * NRES + j) * CZ + c2, &sm->br[j][c2]); }
  __threadfence();
  cl.sync();
  // column r of a is reused by every output token -> stage it into sm->ar
  FOR_ROW16(i) { RC(i);
    CP16(&sm->ar[j][c2], gA2 + (j * NRES + r) * CZ + c2); }
  __syncthreads();
  // p[j2, c] = sum_i a[i, r, c] * b[i, j2, c]
  for (int i = tid; i < NRES * 64; i += NTHR) {
    const int j2 = i >> 6, c2 = (i & 63) << 1;
    float s0 = 0.f, s1 = 0.f, u0 = 0.f, u1 = 0.f;
#pragma unroll
    for (int ii = 0; ii < NRES; ii += 2) {
      uint32_t av = *(const uint32_t*)&sm->ar[ii][c2];
      uint32_t bv = *(const uint32_t*)(gB2 + (ii * NRES + j2) * CZ + c2);
      uint32_t aw = *(const uint32_t*)&sm->ar[ii + 1][c2];
      uint32_t bw = *(const uint32_t*)(gB2 + ((ii + 1) * NRES + j2) * CZ + c2);
      s0 += blo(av) * blo(bv);
      s1 += bhi(av) * bhi(bv);
      u0 += blo(aw) * blo(bw);
      u1 += bhi(aw) * bhi(bw);
    }
    *(uint32_t*)&sm->pr[j2][c2] = pack2(s0 + u0, s1 + u1);
  }
  __syncthreads();
  layernorm16(&sm->pr[0][0], LDA, &sm->pr[0][0], LDA, LNW(LN_MI_OUT), LNB(LN_MI_OUT), warp, lane);
  __syncthreads();
  gemm1<1, EPI_MULZ>(sm, ar, &sm->pr[0][0], LDA, warp, lane, tile);

  // ======================= triangle attention, starting node
  layernorm16(&sm->zr[0][0], LDA, &sm->xl[0][0], LDA, LNW(LN_ATT_S), LNB(LN_ATT_S), warp, lane);
  __syncthreads();
  tri_bias(sm, gT1, 0, r, warp, lane);
  __threadfence();
  cl.sync();
  for (int i = tid; i < NTOK * NHEAD / 8; i += NTHR)
    ((uint4*)&sm->tbs[0][0][0])[i] = ((const uint4*)gT1)[i];
  gemm2<2, EPI2_QKVG>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  attention(sm, warp, lane);
  __syncthreads();
  gemm1<1, EPI_ATTO>(sm, ar, &sm->pr[0][0], LDA, warp, lane, tile);
  FOR_ROW16(i) { RC(i);
    CP16(gZ + (r * NRES + j) * CZ + c2, &sm->zr[j][c2]); }
  __threadfence();
  cl.sync();

  // ======================= triangle attention, ending node (column r)
  FOR_ROW16(i) { RC(i);
    CP16(&sm->zr[j][c2], gZ + (j * NRES + r) * CZ + c2); }
  if (tid < NRES) sm->msk[tid] = b2f(ar.mask[tid * NRES + r]);
  __syncthreads();
  layernorm16(&sm->zr[0][0], LDA, &sm->xl[0][0], LDA, LNW(LN_ATT_E), LNB(LN_ATT_E), warp, lane);
  __syncthreads();
  tri_bias(sm, gT2, 1, r, warp, lane);
  __threadfence();
  cl.sync();
  for (int i = tid; i < NTOK * NHEAD / 8; i += NTHR)
    ((uint4*)&sm->tbs[0][0][0])[i] = ((const uint4*)gT2)[i];
  gemm2<2, EPI2_QKVG>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  attention(sm, warp, lane);
  __syncthreads();
  gemm1<1, EPI_ATTO>(sm, ar, &sm->pr[0][0], LDA, warp, lane, tile);

  // ======================= SwiGLU transition (token local)
  layernorm16(&sm->zr[0][0], LDA, &sm->xl[0][0], LDA, LNW(LN_TRANS), LNB(LN_TRANS), warp, lane);
  __syncthreads();
  gemm2<4, EPI2_TRANS>(sm, ar, &sm->xl[0][0], LDA, warp, lane, tile);
  gemm1<4, EPI_TRANS2>(sm, ar, &sm->hid[0][0], LDH, warp, lane, tile);

  FOR_ROW16(i) { RC(i);
    CP16(ar.out + (j * NRES + r) * CZ + c2, &sm->zr[j][c2]); }
}

// -------------------------------------------------------------------- launcher
void pair_block(at::Tensor z, at::Tensor mask, at::Tensor out, at::Tensor w,
                at::Tensor ln, at::Tensor ws, int64_t mask_trans) {
  Args a;
  a.z = (const bf16*)z.data_ptr();
  a.mask = (const bf16*)mask.data_ptr();
  a.out = (bf16*)out.data_ptr();
  a.w = (const char*)w.data_ptr();
  a.ln = ln.data_ptr<float>();
  a.ws = (bf16*)ws.data_ptr();
  a.mask_trans = (int)mask_trans;

  static int smem = -1;
  if (smem < 0) {
    smem = (int)sizeof(Smem);
    cudaFuncSetAttribute((void*)pair_block_kernel,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute((void*)pair_block_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  pair_block_kernel<<<NRES, NTHR, smem, c10::cuda::getCurrentCUDAStream()>>>(a);
}

int64_t smem_bytes() { return (int64_t)sizeof(Smem); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pair_block", &pair_block);
  m.def("smem_bytes", &smem_bytes);
}
