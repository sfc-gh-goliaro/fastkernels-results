// Fused AlphaFold3 diffusion-transformer stack (rationale in the .py docstring).
//
// One kernel launch runs the whole `no_blocks`-deep stack.  The captured shapes
// are tiny in the token dimension (16 tokens x 768 channels for the self-attention
// config) and enormous in weights (24 blocks x 16.5 MB), so this is a pure
// weight-streaming problem wrapped around a strictly sequential dependency chain.
// Everything here follows from that:
//
//   * one launch with G resident CTAs and grid barriers instead of kernel
//     boundaries -- a barrier measured 1.2 us on this B200 against 4.5 us for a
//     launch in the bench's own timing loop;
//   * everything that depends only on `s` / `z` / `mask` -- both AdaLN
//     conditioners, both output gates, the pair bias -- is hoisted into a
//     barrier-free prologue that runs all blocks at once (21% of the weights);
//   * the per-block chain is cut into four stages, so four barriers per block;
//   * activations are 24 KB, so every CTA keeps the whole token tensor in shared
//     memory and *recomputes* the LayerNorm / elementwise steps rather than
//     synchronizing on them.  Only the four reductions (q/k/v/g, linear_o,
//     SwiGLU hidden, linear_out) are split across CTAs, two of them closing with
//     fp32 atomics so the split costs no extra barrier.
//
// Weights are pre-packed host-side straight into `mma.m16n8k16` B-fragment order,
// so a warp's operand load is one contiguous 256 B transaction per tile.  Every
// intermediate is rounded to bf16 exactly where the reference composition rounds
// it: the last matmul sums 1536 mixed-sign terms, so a rounding point in the
// wrong place shows up far above bf16 noise in the match ratio.
//
// The geometry is a compile-time template.  At these sizes a runtime `/ C` in an
// elementwise loop costs more than the memory it addresses (measured: 1.09 ms of
// the first working version's 2.56 ms), and the mma loops need compile-time trip
// counts to keep their operand loads in flight.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16 = __nv_bfloat16;

#define NWARP    16
#define THREADS  (NWARP * 32)
#define PAD      8          /* bf16 padding on shared-memory row strides */
#define PARTMAX  6          /* widest n-tile group any stage reduces at once  */

// ---------------------------------------------------------------- primitives
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }
// Round through bf16 while staying in fp32 registers: used at exactly the points
// where the reference composition materializes a bf16 tensor.
__device__ __forceinline__ float rb(float x) {
  return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ uint32_t lds32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}
__device__ __forceinline__ uint64_t ldg64(const void* p) {
  return *reinterpret_cast<const uint64_t*>(p);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A-fragment of one 16x16 tile out of a row-major shared tile [.][ld].  The four
// 32-bit shared loads land on 32 distinct banks whenever (ld*2/4) % 32 == 4,
// which is why every shared row stride here is padded by PAD = 8 bf16.
__device__ __forceinline__ void ld_afrag(uint32_t (&a)[4], const bf16* A, int ld,
                                         int kt) {
  const int lane = threadIdx.x & 31;
  const bf16* p = A + (lane >> 2) * ld + kt * 16 + ((lane & 3) << 1);
  a[0] = lds32(p);
  a[1] = lds32(p + 8 * ld);
  a[2] = lds32(p + 8);
  a[3] = lds32(p + 8 * ld + 8);
}

// B-fragment out of a row-major shared tile B[n][ld]; the "col-major" operand of
// mma.*.row.col is just [N][K] row-major.
__device__ __forceinline__ void ld_bfrag_s(uint32_t& b0, uint32_t& b1,
                                           const bf16* B, int ld, int nt, int kt) {
  const int lane = threadIdx.x & 31;
  const bf16* p = B + (nt * 8 + (lane >> 2)) * ld + kt * 16 + ((lane & 3) << 1);
  b0 = lds32(p);
  b1 = lds32(p + 8);
}

// ------------------------------------------------------------------- barrier
struct Sync { unsigned* cnt; unsigned* gen; };

__device__ __forceinline__ unsigned ldv(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void stv(unsigned* p, unsigned v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// One __syncthreads on each side and a single fence, executed by the one thread
// that actually needs release ordering; the spin uses ld.acquire so the matching
// acquire fence is free.  The two dropped __syncthreads and the dropped second
// fence were 1/3 of all warp-issue stalls (barrier 30%, membar 11%).
__device__ __forceinline__ void gbar(Sync s, int ncta, unsigned& gen) {
  __syncthreads();
  if (threadIdx.x < 32) {
    if (threadIdx.x == 0) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned t = atomicAdd(s.cnt, 1u);
      if (t == (unsigned)(ncta - 1)) {
        stv(s.cnt, 0u);
        asm volatile("st.release.gpu.u32 [%0], %1;" :: "l"(s.gen), "r"(gen + 1)
                     : "memory");
      }
    }
    unsigned v;
    do {
      asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(s.gen) : "memory");
    } while (v == gen);
  }
  __syncthreads();
  gen++;
}

// Weight section ids (the host fills Params::off with per-block element offsets).
enum {
  W_LN1 = 0, W_GS1, W_BG1, W_AO, W_BAO, W_LN2, W_GS2, W_BG2, W_GC, W_BGC,
  W_LNZ, W_Z, W_QKVG, W_BQ, W_O, W_SG, W_OUT, W_NSECT
};

struct Params {
  const bf16* a; const bf16* s; const bf16* z; const bf16* mask;
  bf16* out;
  const bf16* W; long wstride;
  long off[W_NSECT];
  bf16* cond;      // [NB][6][N][C]  g1, b1, gate_o, g2, b2, gate_ctb
  bf16* zb;        // [NB][H][N][N]  pair bias, head-major
  bf16* qkvg;      // [4][H][N][D]
  bf16* hid;       // [N][F]
  float* acc;      // [4][N][C]      two ping-ponged fp32 reduction buffers x2
  unsigned* cnt; unsigned* gen;
  int NB, dbg;
  float eps, qdiv, inf;
  int has_mask;
};

// Four bf16 in one 8-byte access; every row stride here keeps 8-byte alignment.
struct V4 { bf16 x[4]; };
// The same eight bytes seen as two bf16x2 SIMD lanes.  Every elementwise step the
// reference performs on bf16 tensors -- gate * update, residual add, gamma *
// (norm + beta) -- is exactly one `mul.rn.bf16x2` / `add.rn.bf16x2`, which rounds
// once just like fp32-then-round does, at a quarter of the instructions.
struct __align__(8) H2 { __nv_bfloat162 h[2]; };

// ----------------------------------------------------------------- utilities
// Packed B-operand address of tile (nt, kt) for this lane.
template <int KTILES>
__device__ __forceinline__ const bf16* bfrag(const bf16* W, int nt, int kt) {
  return W + ((size_t)nt * KTILES + kt) * 128 + ((threadIdx.x & 31) << 2);
}

// Per-row mean / rstd over K columns of a row-major bf16 tile, in fp32.
template <int NR, int K>
__device__ void row_stats(const bf16* X, int ld, float eps, float* mu, float* rs) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
  for (int r = w; r < NR; r += NWARP) {
    const V4* p = (const V4*)(X + (size_t)r * ld);
    float s1 = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = lane; i < K / 4; i += 32) {
      V4 v = p[i];
#pragma unroll
      for (int j = 0; j < 4; j++) { float t = b2f(v.x[j]); s1 += t; s2 += t * t; }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    if (lane == 0) {
      float m = s1 / K;
      mu[r] = m;
      rs[r] = rsqrtf(fmaxf(s2 / K - m * m, 0.f) + eps);
    }
  }
}

// One warp owns one token row, so every LayerNorm statistic is a warp shuffle --
// no shared-memory round trip and no __syncthreads for the reduction.  These two
// helpers are the whole elementwise half of a block: separating the residual add,
// the row statistics and the AdaLN into three passes cost 228 us of 820 us,
// almost all of it re-reading `a` and the conditioners.
#define ROWLOOP(NR) \
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31; \
  _Pragma("unroll 1") for (int r = w; r < (NR); r += NWARP)

__device__ __forceinline__ void warp_stats(float s1, float s2, int K, float eps,
                                           float& mu, float& rs) {
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  }
  mu = s1 / K;
  rs = rsqrtf(fmaxf(s2 / K - mu * mu, 0.f) + eps);
}

// g * (bf16(layer_norm(x)) + b) for one 4-element group, in two bf16x2 ops.
__device__ __forceinline__ H2 adaln4(const float (&v)[4], float mu, float rs,
                                     H2 gv, H2 bv) {
  H2 o;
  o.h[0] = __hmul2(gv.h[0], __hadd2(__floats2bfloat162_rn((v[0] - mu) * rs,
                                                          (v[1] - mu) * rs),
                                    bv.h[0]));
  o.h[1] = __hmul2(gv.h[1], __hadd2(__floats2bfloat162_rn((v[2] - mu) * rs,
                                                          (v[3] - mu) * rs),
                                    bv.h[1]));
  return o;
}

// Y = g * (layer_norm(X) + b), X already in shared memory.
template <int N, int C, int LDA>
__device__ void stats_adaln(bf16* Y, const bf16* X, const bf16* g, const bf16* bb,
                            float eps) {
  constexpr int NV = C / 4;
  ROWLOOP(N) {
    const H2* xp = (const H2*)(X + (size_t)r * LDA);
    float s1 = 0.f, s2 = 0.f;
    float v[(NV + 31) / 32][4];
#pragma unroll
    for (int i = lane, u = 0; i < NV; i += 32, u++) {
      H2 t = xp[i];
      float2 f0 = __bfloat1622float2(t.h[0]), f1 = __bfloat1622float2(t.h[1]);
      v[u][0] = f0.x; v[u][1] = f0.y; v[u][2] = f1.x; v[u][3] = f1.y;
      s1 += (f0.x + f0.y) + (f1.x + f1.y);
      s2 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, s2))));
    }
    float mu, rs; warp_stats(s1, s2, C, eps, mu, rs);
    const H2* gp = (const H2*)(g + (size_t)r * C);
    const H2* bp = (const H2*)(bb + (size_t)r * C);
    H2* yp = (H2*)(Y + (size_t)r * LDA);
#pragma unroll
    for (int i = lane, u = 0; i < NV; i += 32, u++)
      yp[i] = adaln4(v[u], mu, rs, gp[i], bp[i]);
  }
}

// X += gate * bf16(acc) [* mask]; then, from the same registers, the next AdaLN.
template <int N, int C, int LDA, bool MASKED, bool NEXT>
__device__ void resid_adaln(bf16* X, bf16* Y, const float* acc, const bf16* gate,
                            const float* mk, const bf16* g, const bf16* bb,
                            float eps) {
  constexpr int NV = C / 4;
  ROWLOOP(N) {
    const __nv_bfloat162 mv = MASKED ? __bfloat162bfloat162(f2b(mk[r]))
                                     : __nv_bfloat162();
    H2* xp = (H2*)(X + (size_t)r * LDA);
    const float4* ap = (const float4*)(acc + (size_t)r * C);
    const H2* gp = (const H2*)(gate + (size_t)r * C);
    float s1 = 0.f, s2 = 0.f;
    float v[(NV + 31) / 32][4];
#pragma unroll
    for (int i = lane, u = 0; i < NV; i += 32, u++) {
      float4 av = ap[i];
      H2 gv = gp[i], xv = xp[i], o;
      __nv_bfloat162 t0 = __hmul2(__floats2bfloat162_rn(av.x, av.y), gv.h[0]);
      __nv_bfloat162 t1 = __hmul2(__floats2bfloat162_rn(av.z, av.w), gv.h[1]);
      if (MASKED) { t0 = __hmul2(t0, mv); t1 = __hmul2(t1, mv); }
      o.h[0] = __hadd2(xv.h[0], t0);
      o.h[1] = __hadd2(xv.h[1], t1);
      xp[i] = o;
      float2 f0 = __bfloat1622float2(o.h[0]), f1 = __bfloat1622float2(o.h[1]);
      v[u][0] = f0.x; v[u][1] = f0.y; v[u][2] = f1.x; v[u][3] = f1.y;
      s1 += (f0.x + f0.y) + (f1.x + f1.y);
      s2 = fmaf(f0.x, f0.x, fmaf(f0.y, f0.y, fmaf(f1.x, f1.x, fmaf(f1.y, f1.y, s2))));
    }
    if (!NEXT) continue;
    float mu, rs; warp_stats(s1, s2, C, eps, mu, rs);
    const H2* g2p = (const H2*)(g + (size_t)r * C);
    const H2* b2p = (const H2*)(bb + (size_t)r * C);
    H2* yp = (H2*)(Y + (size_t)r * LDA);
#pragma unroll
    for (int i = lane, u = 0; i < NV; i += 32, u++)
      yp[i] = adaln4(v[u], mu, rs, g2p[i], b2p[i]);
  }
}

// -------------------------------------------------------- shared-memory plan
template <int N, int C, int S, int H, int D, int F, int CZ, int HPG, int KG>
struct SmPlan {
  static constexpr int LDA = C + PAD;
  static constexpr int LDS = S + PAD;
  static constexpr int LDZ = CZ + PAD;
  static constexpr int LDD = D + PAD;
  static constexpr int LDN = N + PAD;
  static constexpr int LDO = HPG * D + PAD;
  static constexpr int LDH = KG + PAD;
  static constexpr int NTMAX = PARTMAX;
  static constexpr size_t A_BYTES = (size_t)N * LDA * 2;
  static constexpr size_t PART = (size_t)NWARP * NTMAX * N * 8 * 4;
  static constexpr size_t BASE =
      ((2 * A_BYTES + PART + (size_t)3 * N * 4) + 15) & ~(size_t)15;
  static constexpr size_t U1 = (size_t)N * LDS * 2;
  static constexpr size_t U2 = (size_t)16 * LDZ * 2;
  static constexpr size_t U3 = (size_t)HPG * N * LDD * 2 * 3
                             + (size_t)HPG * D * LDN * 2
                             + (size_t)4 * N * LDN * 2
                             + (size_t)N * LDO * 2;
  static constexpr size_t U4 = (size_t)N * LDH * 2;
  static constexpr size_t UNION =
      (U1 > U2 ? U1 : U2) > (U3 > U4 ? U3 : U4) ? (U1 > U2 ? U1 : U2)
                                               : (U3 > U4 ? U3 : U4);
  static constexpr size_t TOTAL = BASE + UNION + 16;
};

// ###########################################################################
// Self-attention configuration (AttentionPairBias inside the block).
// ###########################################################################
template <int N, int C, int S, int H, int D, int F, int CZ, int G, int HPG, int HG4>
__global__ __launch_bounds__(THREADS) void dit_self(Params p) {
  using P = SmPlan<N, C, S, H, D, F, CZ, HPG, F / HG4>;
  static_assert(N == 16, "one m-tile: the captured self-attention shape is 16 tokens");
  static_assert(H * D == C, "h*c_hidden == c_a");
  static_assert(4 * (C / 8) % G == 0 && 2 * (F / 8) % G == 0, "column split");
  static_assert(H % HPG == 0 && H % 8 == 0 && D % 16 == 0 && S % 16 == 0, "tiles");
  constexpr int LDA = P::LDA, NTMAX = P::NTMAX, ACCMAX = 4;
  constexpr int NT1 = 4 * (C / 8) / G;          // q,k,v,g columns per CTA
  constexpr int CG2 = G / (H / HPG);            // linear_o column groups
  constexpr int NT2 = (C / 8) / CG2;
  constexpr int NT3 = 2 * (F / 8) / G;          // SwiGLU (a,b) pairs per CTA
  constexpr int CG4 = G / HG4;
  constexpr int NT4 = (C / 8) / CG4;
  constexpr int KG = F / HG4;
  constexpr int NKQ = C / 16, NKO = C / 16, NKH = F / 16, NKS = S / 16,
                NKZ = CZ / 16;
  static_assert(NT1 <= ACCMAX && NT2 <= ACCMAX && NT3 <= ACCMAX && NT4 <= ACCMAX, "NT");
  static_assert(NTMAX >= ACCMAX, "part buffer");

  extern __shared__ char smem_raw[];
  bf16* smX = (bf16*)smem_raw;
  bf16* smY = (bf16*)(smem_raw + P::A_BYTES);
  float* part = (float*)(smem_raw + 2 * P::A_BYTES);
  float* mu = (float*)(smem_raw + 2 * P::A_BYTES + P::PART);
  float* rs = mu + N;
  float* mk = rs + N;
  char* U = smem_raw + P::BASE;

  const int cta = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  Sync sy{p.cnt, p.gen};
  unsigned gen = ldv(p.gen);
  float acc[ACCMAX][4];

  // ================================================== prologue: zero the accs
  for (int i = tid + cta * THREADS; i < 4 * N * C; i += G * THREADS) p.acc[i] = 0.f;

  // ===================================== prologue: AdaLN conditioners + gates
  if (!(p.dbg & 1)) {
    constexpr int NTP = 6, n8 = C / 8;   // fatter groups: more loads in flight
    constexpr int gA = 2 * n8 / NTP, gB = n8 / NTP, GPB = 2 * (gA + gB);
    static_assert(2 * n8 % NTP == 0 && n8 % NTP == 0, "prologue groups");
    const int total = p.NB * GPB, per = (total + G - 1) / G;
    const int lo = cta * per, hi = min(lo + per, total);
    bf16* SN = (bf16*)U;
    int cur = -1;
    for (int gi = lo; gi < hi; gi++) {
      const int blk = gi / GPB, g = gi % GPB;
      int sec, nt0;
      if (g < gA) { sec = 0; nt0 = g * NTP; }
      else if (g < gA + gB) { sec = 1; nt0 = (g - gA) * NTP; }
      else if (g < 2 * gA + gB) { sec = 2; nt0 = (g - gA - gB) * NTP; }
      else { sec = 3; nt0 = (g - 2 * gA - gB) * NTP; }
      const int kind = (sec == 0) ? 0 : (sec == 2 ? 2 : 1);
      if (blk * 4 + kind != cur) {
        cur = blk * 4 + kind;
        __syncthreads();
        for (int r = w; r < N; r += NWARP) {
          const V4* sp = (const V4*)(p.s + (size_t)r * S);
          V4* dp = (V4*)(SN + (size_t)r * P::LDS);
          for (int i = lane; i < S / 4; i += 32) dp[i] = sp[i];
        }
        __syncthreads();
        if (kind != 1) {
          row_stats<N, S>(SN, P::LDS, p.eps, mu, rs);
          __syncthreads();
          const bf16* wl = p.W + (size_t)blk * p.wstride
                           + p.off[kind == 0 ? W_LN1 : W_LN2];
          for (int r = w; r < N; r += NWARP) {
            const float m = mu[r], sd = rs[r];
            V4* dp = (V4*)(SN + (size_t)r * P::LDS);
            const V4* wp = (const V4*)wl;
            for (int i = lane; i < S / 4; i += 32) {
              V4 v = dp[i], wv = wp[i], o;
#pragma unroll
              for (int j = 0; j < 4; j++)
                o.x[j] = f2b((b2f(v.x[j]) - m) * sd * b2f(wv.x[j]));
              dp[i] = o;
            }
          }
          __syncthreads();
        }
      }
      const bf16* Wb = p.W + (size_t)blk * p.wstride;
      const bf16* Wm = Wb + p.off[sec == 0 ? W_GS1 : sec == 1 ? W_AO
                                                   : sec == 2 ? W_GS2 : W_GC];
      {
        float a3[NTP][4];
#pragma unroll
        for (int n = 0; n < NTP; n++)
          a3[n][0] = a3[n][1] = a3[n][2] = a3[n][3] = 0.f;
        for (int kt = w; kt < NKS; kt += NWARP) {
          uint32_t af[4];
          ld_afrag(af, SN, P::LDS, kt);
          uint64_t bv[NTP];
#pragma unroll
          for (int n = 0; n < NTP; n++) bv[n] = ldg64(bfrag<NKS>(Wm, nt0 + n, kt));
          asm volatile("" ::: "memory");   // keep the loads above in flight
#pragma unroll
          for (int n = 0; n < NTP; n++)
            mma16816(a3[n], af, (uint32_t)(bv[n] & 0xffffffffull),
                     (uint32_t)(bv[n] >> 32));
        }
        __syncthreads();
        float* mine = part + (size_t)w * NTMAX * N * 8;
        const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
        for (int n = 0; n < NTP; n++) {
          float* q = mine + (size_t)n * N * 8;
          q[r0 * 8 + c0] = a3[n][0]; q[r0 * 8 + c0 + 1] = a3[n][1];
          q[(r0 + 8) * 8 + c0] = a3[n][2]; q[(r0 + 8) * 8 + c0 + 1] = a3[n][3];
        }
        __syncthreads();
      }
      const bf16* bias = Wb + p.off[sec == 0 ? W_BG1 : sec == 1 ? W_BAO
                                                 : sec == 2 ? W_BG2 : W_BGC];
      bf16* cb = p.cond + (size_t)blk * 6 * N * C;
      for (int i = tid; i < NTP * N * 8; i += THREADS) {
        const int n = i >> 7, r = (i >> 3) & (N - 1), c = i & 7;
        const int col = (nt0 + n) * 8 + c;
        float v = part[i];
        for (int ww = 1; ww < NWARP; ww++) v += part[(size_t)ww * NTMAX * N * 8 + i];
        if (sec == 0 || sec == 2) {
          const int base = (sec == 0) ? 0 : 3;
          if (col < C) cb[(size_t)(base * N + r) * C + col] =
              f2b(sigm(rb(v + b2f(bias[col]))));
          else cb[(size_t)((base + 1) * N + r) * C + col - C] = f2b(v);
        } else {
          const int base = (sec == 1) ? 2 : 5;
          cb[(size_t)(base * N + r) * C + col] = f2b(sigm(rb(v + b2f(bias[col]))));
        }
      }
      __syncthreads();
    }
  }

  // ============================================= prologue: pair bias from z
  if (!(p.dbg & 2)) {
    constexpr int MT = N * N / 16, NTZ = H / 8;
    bf16* ZL = (bf16*)U;
    for (int t = cta; t < p.NB * MT; t += G) {
      const int blk = t / MT, mt = t % MT;
      __syncthreads();
      for (int r = w; r < 16; r += NWARP) {
        const V4* sp = (const V4*)(p.z + (size_t)(mt * 16 + r) * CZ);
        V4* dp = (V4*)(ZL + (size_t)r * P::LDZ);
        for (int i = lane; i < CZ / 4; i += 32) dp[i] = sp[i];
      }
      __syncthreads();
      row_stats<16, CZ>(ZL, P::LDZ, p.eps, mu, rs);
      __syncthreads();
      const bf16* Wb = p.W + (size_t)blk * p.wstride;
      const bf16* wl = Wb + p.off[W_LNZ];
      for (int r = w; r < 16; r += NWARP) {
        const float m = mu[r], sd = rs[r];
        V4* dp = (V4*)(ZL + (size_t)r * P::LDZ);
        const V4* wp = (const V4*)wl;
        for (int i = lane; i < CZ / 4; i += 32) {
          V4 v = dp[i], wv = wp[i], o;
#pragma unroll
          for (int j = 0; j < 4; j++)
            o.x[j] = f2b((b2f(v.x[j]) - m) * sd * b2f(wv.x[j]));
          dp[i] = o;
        }
      }
      __syncthreads();
      const bf16* Wz = Wb + p.off[W_Z];
      float az[NTZ][4];
#pragma unroll
      for (int n = 0; n < NTZ; n++) az[n][0] = az[n][1] = az[n][2] = az[n][3] = 0.f;
      for (int kt = w; kt < NKZ; kt += NWARP) {
        uint32_t af[4];
        ld_afrag(af, ZL, P::LDZ, kt);
#pragma unroll
        for (int n = 0; n < NTZ; n++) {
          uint64_t bv = ldg64(bfrag<NKZ>(Wz, n, kt));
          mma16816(az[n], af, (uint32_t)(bv & 0xffffffffull), (uint32_t)(bv >> 32));
        }
      }
      __syncthreads();
      {
        float* mine = part + (size_t)w * NTMAX * N * 8;
        const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
        for (int n = 0; n < NTZ; n++) {
          float* q = mine + (size_t)n * 16 * 8;
          q[r0 * 8 + c0] = az[n][0]; q[r0 * 8 + c0 + 1] = az[n][1];
          q[(r0 + 8) * 8 + c0] = az[n][2]; q[(r0 + 8) * 8 + c0 + 1] = az[n][3];
        }
      }
      __syncthreads();
      bf16* zb = p.zb + (size_t)blk * H * N * N;
      for (int i = tid; i < NTZ * 16 * 8; i += THREADS) {
        const int n = i >> 7, r = (i >> 3) & 15, c = i & 7;
        float v = part[i];
        for (int ww = 1; ww < NWARP; ww++) v += part[(size_t)ww * NTMAX * N * 8 + i];
        zb[(size_t)(n * 8 + c) * N * N + mt * 16 + r] = f2b(v);
      }
    }
  }

  // --------------------------------------------------- token tensor -> shared
  for (int r = w; r < N; r += NWARP) {
    const V4* sp = (const V4*)(p.a + (size_t)r * C);
    V4* dp = (V4*)(smX + (size_t)r * LDA);
    for (int i = lane; i < C / 4; i += 32) dp[i] = sp[i];
  }
  if (tid < N) mk[tid] = p.has_mask ? b2f(p.mask[tid]) : 1.f;
  __syncthreads();
  gbar(sy, G, gen);
  // The first block's AdaLN; every later one rides along in the residual pass
  // that produced its input.
  if (!(p.dbg & 4)) {
    stats_adaln<N, C, LDA>(smY, smX, p.cond, p.cond + (size_t)N * C, p.eps);
    __syncthreads();
  }

  const int cg2 = cta % CG2, hg2 = cta / CG2;
  const int cg4 = cta % CG4, hg4 = cta / CG4;

  for (int b = 0; b < ((p.dbg & 4) ? 0 : p.NB); b++) {
    const bf16* Wb = p.W + (size_t)b * p.wstride;
    const bf16* cb = p.cond + (size_t)b * 6 * N * C;
    float* A2 = p.acc + (size_t)(b & 1) * N * C;
    float* A4 = p.acc + (size_t)(2 + (b & 1)) * N * C;
    float* Z2 = p.acc + (size_t)((b + 1) & 1) * N * C;
    float* Z4 = p.acc + (size_t)(2 + ((b + 1) & 1)) * N * C;

    // ---------------------------------------------------------------- stage 1
    for (int i = tid + cta * THREADS; i < N * C; i += G * THREADS) {
      Z2[i] = 0.f; Z4[i] = 0.f;
    }
    if (!(p.dbg & 16)) {
      const bf16* Wq = Wb + p.off[W_QKVG];
      constexpr int nt0 = 0;
      (void)nt0;
#pragma unroll
      for (int n = 0; n < NT1; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      const int base = cta * NT1;
      int kt = w;
      for (; kt + NWARP < NKQ; kt += 2 * NWARP) {
        uint32_t a0[4], a1[4];
        ld_afrag(a0, smY, LDA, kt);
        ld_afrag(a1, smY, LDA, kt + NWARP);
        uint64_t v0[NT1], v1[NT1];
#pragma unroll
        for (int n = 0; n < NT1; n++) {
          v0[n] = ldg64(bfrag<NKQ>(Wq, base + n, kt));
          v1[n] = ldg64(bfrag<NKQ>(Wq, base + n, kt + NWARP));
        }
        asm volatile("" ::: "memory");   // keep the loads above in flight
#pragma unroll
        for (int n = 0; n < NT1; n++) {
          mma16816(acc[n], a0, (uint32_t)(v0[n] & 0xffffffffull), (uint32_t)(v0[n] >> 32));
          mma16816(acc[n], a1, (uint32_t)(v1[n] & 0xffffffffull), (uint32_t)(v1[n] >> 32));
        }
      }
      for (; kt < NKQ; kt += NWARP) {
        uint32_t a0[4];
        ld_afrag(a0, smY, LDA, kt);
#pragma unroll
        for (int n = 0; n < NT1; n++) {
          uint64_t v = ldg64(bfrag<NKQ>(Wq, base + n, kt));
          mma16816(acc[n], a0, (uint32_t)(v & 0xffffffffull), (uint32_t)(v >> 32));
        }
      }
      __syncthreads();
      float* mine = part + (size_t)w * NTMAX * N * 8;
      const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
      for (int n = 0; n < NT1; n++) {
        float* q = mine + (size_t)n * N * 8;
        q[r0 * 8 + c0] = acc[n][0]; q[r0 * 8 + c0 + 1] = acc[n][1];
        q[(r0 + 8) * 8 + c0] = acc[n][2]; q[(r0 + 8) * 8 + c0 + 1] = acc[n][3];
      }
      __syncthreads();
    }
    if (!(p.dbg & 32)) {
      const bf16* bq = Wb + p.off[W_BQ];
      const int base = cta * NT1 * 8;
      for (int i = tid; i < NT1 * N * 8; i += THREADS) {
        const int n = i >> 7, r = (i >> 3) & (N - 1), c = i & 7;
        float v = part[i];
        for (int ww = 1; ww < NWARP; ww++) v += part[(size_t)ww * NTMAX * N * 8 + i];
        const int col = base + n * 8 + c;
        const int t = col / C, hd = col - t * C, hh = hd / D, d = hd - hh * D;
        constexpr int DP = P::LDD, LDN = P::LDN, QSZ = H * N * DP;
        if (t == 0)
          p.qkvg[(size_t)(hh * N + r) * DP + d] =
              f2b(rb(rb(v + b2f(bq[hd])) / p.qdiv));
        else if (t == 1) p.qkvg[QSZ + (size_t)(hh * N + r) * DP + d] = f2b(v);
        else if (t == 3)  // the attention gate is sigmoid(linear_g(a1))
          p.qkvg[2 * QSZ + (size_t)(hh * N + r) * DP + d] = f2b(sigm(rb(v)));
        else p.qkvg[3 * QSZ + (size_t)(hh * D + d) * LDN + r] = f2b(v);
      }
    }
    gbar(sy, G, gen);

    // ---------------------------------------------------------------- stage 2
    if (!(p.dbg & 64)) {
      constexpr int LDD = P::LDD, LDN = P::LDN, LDO = P::LDO;
      bf16* Qs = (bf16*)U;
      bf16* Ks = Qs + HPG * N * LDD;
      bf16* Gs = Ks + HPG * N * LDD;
      bf16* Vt = Gs + HPG * N * LDD;
      bf16* Ps = Vt + HPG * D * LDN;
      bf16* OH = Ps + 4 * N * LDN;
      // Q/K/G are stored [H][N][LDD] and V pre-transposed to [H][D][LDN] by
      // stage 1, with the same padding the shared tiles use, so each of these is
      // one fully-coalesced contiguous run.
      constexpr int QSZ = H * N * LDD, VSZ = H * D * LDN;
      constexpr int QV = HPG * N * LDD / 4, VV = HPG * D * LDN / 4;
      {
        const V4* qg = (const V4*)(p.qkvg + (size_t)hg2 * HPG * N * LDD);
        const V4* kg = (const V4*)(p.qkvg + QSZ + (size_t)hg2 * HPG * N * LDD);
        const V4* gg = (const V4*)(p.qkvg + 2 * QSZ + (size_t)hg2 * HPG * N * LDD);
        const V4* vg = (const V4*)(p.qkvg + 3 * QSZ + (size_t)hg2 * HPG * D * LDN);
        V4* qd = (V4*)Qs; V4* kd = (V4*)Ks; V4* gd = (V4*)Gs; V4* vd = (V4*)Vt;
        (void)VSZ;
        for (int i = tid; i < QV; i += THREADS) { qd[i] = qg[i]; kd[i] = kg[i]; gd[i] = gg[i]; }
        for (int i = tid; i < VV; i += THREADS) vd[i] = vg[i];
      }
      __syncthreads();
      if (w < HPG) {
        constexpr int NQ = N / 8, NDV = D / 8, NKD = D / 16;
        const bf16* Qh = Qs + (size_t)w * N * LDD;
        const bf16* Kh = Ks + (size_t)w * N * LDD;
        const bf16* Gh = Gs + (size_t)w * N * LDD;
        const bf16* Vh = Vt + (size_t)w * D * LDN;
        const bf16* zbh = p.zb + (size_t)((b * H) + hg2 * HPG + w) * N * N;
        bf16* Pw = Ps + (size_t)w * N * LDN;
        const int r0 = lane >> 2, c0 = (lane & 3) << 1;
        float sc[NQ][4];
#pragma unroll
        for (int n = 0; n < NQ; n++) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
        for (int kt = 0; kt < NKD; kt++) {
          uint32_t af[4];
          ld_afrag(af, Qh, LDD, kt);
#pragma unroll
          for (int n = 0; n < NQ; n++) {
            uint32_t b0, b1;
            ld_bfrag_s(b0, b1, Kh, LDD, n, kt);
            mma16816(sc[n], af, b0, b1);
          }
        }
        float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
        for (int n = 0; n < NQ; n++)
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const int col = n * 8 + c0 + j;
            const float mb = p.has_mask ? p.inf * (mk[col] - 1.f) : 0.f;
            float v = rb(rb(rb(sc[n][j]) + mb) + b2f(zbh[r0 * N + col]));
            float u = rb(rb(rb(sc[n][2 + j]) + mb) + b2f(zbh[(r0 + 8) * N + col]));
            sc[n][j] = v; sc[n][2 + j] = u;
            mx0 = fmaxf(mx0, v); mx1 = fmaxf(mx1, u);
          }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int n = 0; n < NQ; n++)
#pragma unroll
          for (int j = 0; j < 2; j++) {
            sc[n][j] = expf(sc[n][j] - mx0); s0 += sc[n][j];
            sc[n][2 + j] = expf(sc[n][2 + j] - mx1); s1 += sc[n][2 + j];
          }
        s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
        s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
        s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
        const float i0 = 1.f / s0, i1 = 1.f / s1;
#pragma unroll
        for (int n = 0; n < NQ; n++)
#pragma unroll
          for (int j = 0; j < 2; j++) {
            Pw[r0 * LDN + n * 8 + c0 + j] = f2b(sc[n][j] * i0);
            Pw[(r0 + 8) * LDN + n * 8 + c0 + j] = f2b(sc[n][2 + j] * i1);
          }
        __syncwarp();
        float ov[NDV][4];
#pragma unroll
        for (int n = 0; n < NDV; n++) ov[n][0] = ov[n][1] = ov[n][2] = ov[n][3] = 0.f;
        {
          uint32_t af[4];
          ld_afrag(af, Pw, LDN, 0);
#pragma unroll
          for (int n = 0; n < NDV; n++) {
            uint32_t b0, b1;
            ld_bfrag_s(b0, b1, Vh, LDN, n, 0);
            mma16816(ov[n], af, b0, b1);
          }
        }
#pragma unroll
        for (int n = 0; n < NDV; n++)
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const int d = n * 8 + c0 + j;
            OH[r0 * LDO + w * D + d] = f2b(rb(rb(ov[n][j]) * b2f(Gh[r0 * LDD + d])));
            OH[(r0 + 8) * LDO + w * D + d] =
                f2b(rb(rb(ov[n][2 + j]) * b2f(Gh[(r0 + 8) * LDD + d])));
          }
      }
      __syncthreads();
      constexpr int NKI = HPG * D / 16;
      const int kt0 = hg2 * NKI;
      const bf16* Wo = Wb + p.off[W_O];
      const int base = cg2 * NT2;
#pragma unroll
      for (int n = 0; n < NT2; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      for (int kt = kt0 + w; kt < kt0 + NKI; kt += NWARP) {
        uint32_t af[4];
        ld_afrag(af, OH, LDO, kt - kt0);
#pragma unroll
        for (int n = 0; n < NT2; n++) {
          uint64_t v = ldg64(bfrag<NKO>(Wo, base + n, kt));
          mma16816(acc[n], af, (uint32_t)(v & 0xffffffffull), (uint32_t)(v >> 32));
        }
      }
      __syncthreads();
      {
        float* mine = part + (size_t)w * NTMAX * N * 8;
        const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
        for (int n = 0; n < NT2; n++) {
          float* q = mine + (size_t)n * N * 8;
          q[r0 * 8 + c0] = acc[n][0]; q[r0 * 8 + c0 + 1] = acc[n][1];
          q[(r0 + 8) * 8 + c0] = acc[n][2]; q[(r0 + 8) * 8 + c0 + 1] = acc[n][3];
        }
      }
      __syncthreads();
      for (int i = tid; i < NT2 * N * 8; i += THREADS) {
        const int n = i >> 7, r = (i >> 3) & (N - 1), c = i & 7;
        float v = part[i];
        for (int ww = 1; ww < NWARP; ww++) v += part[(size_t)ww * NTMAX * N * 8 + i];
        atomicAdd(&A2[(size_t)r * C + (base + n) * 8 + c], v);
      }
    }
    gbar(sy, G, gen);

    // a2 = a + gate_o * linear_o(o), then straight into the transition's AdaLN
    if (!(p.dbg & 128)) {
      resid_adaln<N, C, LDA, false, true>(smX, smY, A2, cb + (size_t)2 * N * C, mk,
                                          cb + (size_t)3 * N * C,
                                          cb + (size_t)4 * N * C, p.eps);
      __syncthreads();
    }
    if (!(p.dbg & 512)) {
      const bf16* Wsg = Wb + p.off[W_SG];
      const int base = cta * NT3;
#pragma unroll
      for (int n = 0; n < NT3; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      int kt = w;
      for (; kt + NWARP < NKQ; kt += 2 * NWARP) {
        uint32_t a0[4], a1[4];
        ld_afrag(a0, smY, LDA, kt);
        ld_afrag(a1, smY, LDA, kt + NWARP);
        uint64_t v0[NT3], v1[NT3];
#pragma unroll
        for (int n = 0; n < NT3; n++) {
          v0[n] = ldg64(bfrag<NKQ>(Wsg, base + n, kt));
          v1[n] = ldg64(bfrag<NKQ>(Wsg, base + n, kt + NWARP));
        }
        asm volatile("" ::: "memory");   // keep the loads above in flight
#pragma unroll
        for (int n = 0; n < NT3; n++) {
          mma16816(acc[n], a0, (uint32_t)(v0[n] & 0xffffffffull), (uint32_t)(v0[n] >> 32));
          mma16816(acc[n], a1, (uint32_t)(v1[n] & 0xffffffffull), (uint32_t)(v1[n] >> 32));
        }
      }
      for (; kt < NKQ; kt += NWARP) {
        uint32_t a0[4];
        ld_afrag(a0, smY, LDA, kt);
#pragma unroll
        for (int n = 0; n < NT3; n++) {
          uint64_t v = ldg64(bfrag<NKQ>(Wsg, base + n, kt));
          mma16816(acc[n], a0, (uint32_t)(v & 0xffffffffull), (uint32_t)(v >> 32));
        }
      }
      __syncthreads();
      float* mine = part + (size_t)w * NTMAX * N * 8;
      const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
      for (int n = 0; n < NT3; n++) {
        float* q = mine + (size_t)n * N * 8;
        q[r0 * 8 + c0] = acc[n][0]; q[r0 * 8 + c0 + 1] = acc[n][1];
        q[(r0 + 8) * 8 + c0] = acc[n][2]; q[(r0 + 8) * 8 + c0 + 1] = acc[n][3];
      }
      __syncthreads();
    }
    if (!(p.dbg & 1024)) {
      const int base = cta * NT3 * 4;
      for (int i = tid; i < NT3 * N * 4; i += THREADS) {
        const int n = i >> 6, r = (i >> 2) & (N - 1), u = i & 3;
        const int o0 = n * N * 8 + r * 8 + 2 * u;
        float va = part[o0], vb = part[o0 + 1];
        for (int ww = 1; ww < NWARP; ww++) {
          const float* q = part + (size_t)ww * NTMAX * N * 8 + o0;
          va += q[0]; vb += q[1];
        }
        float ha = rb(va), hb = rb(vb);
        p.hid[(size_t)r * F + base + n * 4 + u] = f2b(rb(rb(ha * sigm(ha)) * hb));
      }
    }
    gbar(sy, G, gen);

    // ---------------------------------------------------------------- stage 4
    if (!(p.dbg & 2048)) {
      constexpr int LDH = P::LDH, NKI = KG / 16;
      bf16* HS = (bf16*)U;
      for (int r = w; r < N; r += NWARP) {
        const V4* sp = (const V4*)(p.hid + (size_t)r * F + hg4 * KG);
        V4* dp = (V4*)(HS + (size_t)r * LDH);
        for (int i = lane; i < KG / 4; i += 32) dp[i] = sp[i];
      }
      __syncthreads();
      const bf16* Wout = Wb + p.off[W_OUT];
      const int kt0 = hg4 * NKI, base = cg4 * NT4;
#pragma unroll
      for (int n = 0; n < NT4; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
      int kt = kt0 + w;
      for (; kt + NWARP < kt0 + NKI; kt += 2 * NWARP) {
        uint32_t a0[4], a1[4];
        ld_afrag(a0, HS, LDH, kt - kt0);
        ld_afrag(a1, HS, LDH, kt + NWARP - kt0);
        uint64_t v0[NT4], v1[NT4];
#pragma unroll
        for (int n = 0; n < NT4; n++) {
          v0[n] = ldg64(bfrag<NKH>(Wout, base + n, kt));
          v1[n] = ldg64(bfrag<NKH>(Wout, base + n, kt + NWARP));
        }
        asm volatile("" ::: "memory");   // keep the loads above in flight
#pragma unroll
        for (int n = 0; n < NT4; n++) {
          mma16816(acc[n], a0, (uint32_t)(v0[n] & 0xffffffffull), (uint32_t)(v0[n] >> 32));
          mma16816(acc[n], a1, (uint32_t)(v1[n] & 0xffffffffull), (uint32_t)(v1[n] >> 32));
        }
      }
      for (; kt < kt0 + NKI; kt += NWARP) {
        uint32_t a0[4];
        ld_afrag(a0, HS, LDH, kt - kt0);
#pragma unroll
        for (int n = 0; n < NT4; n++) {
          uint64_t v = ldg64(bfrag<NKH>(Wout, base + n, kt));
          mma16816(acc[n], a0, (uint32_t)(v & 0xffffffffull), (uint32_t)(v >> 32));
        }
      }
      __syncthreads();
      {
        float* mine = part + (size_t)w * NTMAX * N * 8;
        const int r0 = lane >> 2, c0 = (lane & 3) << 1;
#pragma unroll
        for (int n = 0; n < NT4; n++) {
          float* q = mine + (size_t)n * N * 8;
          q[r0 * 8 + c0] = acc[n][0]; q[r0 * 8 + c0 + 1] = acc[n][1];
          q[(r0 + 8) * 8 + c0] = acc[n][2]; q[(r0 + 8) * 8 + c0 + 1] = acc[n][3];
        }
      }
      __syncthreads();
      for (int i = tid; i < NT4 * N * 8; i += THREADS) {
        const int n = i >> 7, r = (i >> 3) & (N - 1), c = i & 7;
        float v = part[i];
        for (int ww = 1; ww < NWARP; ww++) v += part[(size_t)ww * NTMAX * N * 8 + i];
        atomicAdd(&A4[(size_t)r * C + (base + n) * 8 + c], v);
      }
    }
    gbar(sy, G, gen);

    // a_next = a2 + mask * gate_ctb * linear_out(h), then the next block's AdaLN
    if (!(p.dbg & 4096)) {
      const bf16* nb = cb + (size_t)6 * N * C;
      if (b + 1 < p.NB)
        resid_adaln<N, C, LDA, true, true>(smX, smY, A4, cb + (size_t)5 * N * C,
                                           mk, nb, nb + (size_t)N * C, p.eps);
      else
        resid_adaln<N, C, LDA, true, false>(smX, smY, A4, cb + (size_t)5 * N * C,
                                            mk, nb, nb, p.eps);
      __syncthreads();
    }
  }

  for (int i = tid + cta * THREADS; i < N * C; i += G * THREADS) {
    const int r = i / C, c = i - r * C;
    p.out[i] = smX[r * LDA + c];
  }
}

// ###########################################################################
// Host side
// ###########################################################################
#define CFG_B 16, 768, 384, 16, 48, 1536, 128
#define GEO_B 128, 4, 4

at::Tensor dit_self_fwd(const at::Tensor& a, const at::Tensor& s, const at::Tensor& z,
                        const at::Tensor& mask, const at::Tensor& W,
                        const at::Tensor& offs, const at::Tensor& cond,
                        const at::Tensor& zb, const at::Tensor& qkvg,
                        const at::Tensor& hid, const at::Tensor& acc,
                        const at::Tensor& sync, int64_t N, int64_t C, int64_t S,
                        int64_t H, int64_t D, int64_t F, int64_t CZ, int64_t NB,
                        int64_t wstride, double eps, double qdiv, double inf,
                        int64_t has_mask, int64_t dbg) {
  TORCH_CHECK(N == 16 && C == 768 && S == 384 && H == 16 && D == 48 && F == 1536
              && CZ == 128, "unsupported geometry");
  at::Tensor out = at::empty_like(a);
  Params p{};
  p.a = (const bf16*)a.data_ptr();
  p.s = (const bf16*)s.data_ptr();
  p.z = (const bf16*)z.data_ptr();
  p.mask = has_mask ? (const bf16*)mask.data_ptr() : nullptr;
  p.out = (bf16*)out.data_ptr();
  p.W = (const bf16*)W.data_ptr();
  p.wstride = wstride;
  const int64_t* o = offs.data_ptr<int64_t>();
  for (int i = 0; i < W_NSECT; i++) p.off[i] = o[i];
  p.cond = (bf16*)cond.data_ptr();
  p.zb = (bf16*)zb.data_ptr();
  p.qkvg = (bf16*)qkvg.data_ptr();
  p.hid = (bf16*)hid.data_ptr();
  p.acc = acc.data_ptr<float>();
  p.cnt = (unsigned*)sync.data_ptr();
  p.gen = p.cnt + 1;
  p.NB = NB; p.dbg = dbg;
  p.eps = eps; p.qdiv = qdiv; p.inf = inf; p.has_mask = has_mask;

  using PB = SmPlan<16, 768, 384, 16, 48, 1536, 128, 4, 1536 / 4>;
  constexpr int smb = (int)PB::TOTAL, GB = 128;
  static int done = 0;
  if (!done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute((void*)dit_self<CFG_B, GEO_B>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
    done = 1;
  }
  dit_self<CFG_B, GEO_B><<<GB, THREADS, smb, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t dit_self_smem() {
  return (int64_t)SmPlan<16, 768, 384, 16, 48, 1536, 128, 4, 1536 / 4>::TOTAL;
}
int64_t dit_self_grid() { return 128; }
// Element count of the q/k/v/g staging buffer: Q,K,G as [H][N][D+PAD] and V
// pre-transposed to [H][D][N+PAD], padded so stage 2's copies are contiguous.
int64_t dit_self_qkvg(int64_t N, int64_t H, int64_t D) {
  return 3 * H * N * (D + PAD) + H * D * (N + PAD);
}
int64_t dit_max_smem() {
  int v = 0;
  cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  return v;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dit_self_fwd", &dit_self_fwd);
  m.def("dit_self_smem", &dit_self_smem);
  m.def("dit_self_grid", &dit_self_grid);
  m.def("dit_self_qkvg", &dit_self_qkvg);
  m.def("dit_max_smem", &dit_max_smem);
}
